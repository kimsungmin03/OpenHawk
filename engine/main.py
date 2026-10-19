"""
main.py
-------
AI Hawk-Eye Lite 추론 엔진 메인 진입점.
동영상 클립을 분석하여 [착지 순간 감지 -> 베이스라인 물리 In/Out 판정 -> 호크아이 스냅샷 생성 -> 결과 JSON 반환]
을 단일 파이프라인으로 수행합니다.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

# Windows 콘솔 인코딩 호환성 보장
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2

from core.baseline_roi import BaselineROI
from core.tracker import ShuttlecockTracker
from core.impact_detector import ImpactDetector
from core.visualizer import HawkEyeVisualizer
from core.commentary import CommentaryEngine


def _read_frame_at(video_path: str, target_idx: int, fallback_frame: Optional[Any] = None) -> Optional[Any]:
    """전체 프레임을 메모리에 적재하지 않고 특정 인덱스의 프레임 1장만 정밀 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback_frame
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        # 코덱에 따라 seek가 부정확할 경우 순차 스킵으로 폴백
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr = 0
        last_valid = fallback_frame
        while curr <= target_idx:
            ret, f = cap.read()
            if not ret:
                break
            last_valid = f
            curr += 1
        return last_valid
    finally:
        cap.release()


def analyze_clip(
    video_path: str,
    roi_coords: Optional[str] = None,
    output_dir: str = "engine/outputs",
    save_snapshot: bool = True,
    enable_ai: bool = True,
    tracker_type: str = "auto",
    model_path: Optional[str] = None,
    tracker: Optional[ShuttlecockTracker] = None,
    commentary_engine: Optional[CommentaryEngine] = None,
) -> Dict[str, Any]:
    """
    비디오 클립을 분석하여 셔틀콕 착지 프레임 검출, In/Out 판정 및 AI 해설 생성 수행

    Args:
        video_path: 분석할 동영상 파일 경로
        roi_coords: "x,y,w,h" 형태의 고정 베이스라인 좌표 (없으면 자동 탐지)
        output_dir: 스냅샷 저장 디렉토리
        save_snapshot: 결과 스냅샷 이미지 생성 여부
        enable_ai: 생성형 AI 2차 검증 및 해설 모듈 구동 여부
        tracker: 상주 워커에서 재사용할 ShuttlecockTracker 인스턴스 (선택)
        commentary_engine: 상주 워커에서 재사용할 CommentaryEngine 인스턴스 (선택)

    Returns:
        dict: 최종 판정 결과 데이터
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 1. 베이스라인 ROI 초기화
    baseline = BaselineROI()
    if roi_coords:
        parts = [int(v.strip()) for v in roi_coords.split(",")]
        if len(parts) == 4:
            baseline._update_boundaries((parts[0], parts[1], parts[2], parts[3]))
            print(f"[ENGINE] 수동 지정 베이스라인 ROI 적용: {parts}")

    if tracker is not None:
        tracker.reset()
    else:
        tracker = ShuttlecockTracker(tracker_type=tracker_type, model_path=model_path)

    last_frame = None
    frame_idx = 0

    # 2. 비디오 프레임 순회 및 셔틀콕 추적 (RAM 절약을 위해 전체 프레임 버퍼 미사용)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_idx / fps) * 1000.0

        # 첫 프레임에서 베이스라인이 미설정 상태라면 자동 검출
        if baseline.line_rect is None and frame_idx == 0:
            success = baseline.auto_detect_baseline(frame)
            if success:
                print(
                    f"[ENGINE] 베이스라인 자동 검출 성공: outer_y={baseline.outer_y}, "
                    f"line_h={baseline.pixel_height}px, slope={baseline.slope:.4f}"
                )
            else:
                print("[ENGINE] 베이스라인 자동 검출 실패. 기본 하단 영역을 적용합니다.")

        # 셔틀콕 탐색
        tracker.process_frame(frame, frame_idx=frame_idx, timestamp_ms=timestamp_ms)
        last_frame = frame
        frame_idx += 1

    cap.release()

    if not tracker.trajectory:
        return {
            "status": "FAIL",
            "message": "영상에서 셔틀콕의 이동 궤적을 감지하지 못했습니다.",
            "decision": "UNCERTAIN",
            "physical_decision": "UNCERTAIN",
            "final_decision": "UNCERTAIN",
        }

    # 3. 착지 순간(Impact Frame) 분석
    detector = ImpactDetector(baseline)
    impact = detector.detect_impact(tracker.trajectory)

    if impact is None:
        return {
            "status": "FAIL",
            "message": "착지 순간을 명확히 특정할 수 없습니다.",
            "decision": "UNCERTAIN",
            "physical_decision": "UNCERTAIN",
            "final_decision": "UNCERTAIN",
        }

    impact_idx = impact["impact_frame_idx"]
    impact_frame = _read_frame_at(video_path, impact_idx, fallback_frame=last_frame)

    verdict = dict(impact["baseline_verdict"])
    verdict["confidence"] = impact["confidence"]

    # 4. 생성형 AI (VLM & LLM) 2차 검증 및 해설 브리핑 생성
    #    텍스트 배지에 의한 확증 편향을 없애기 위해 배지 없는 검증 캔버스를 먼저 전달
    commentary_info = None
    if enable_ai and impact_frame is not None:
        verification_canvas = HawkEyeVisualizer.render_canvas(
            impact_frame=impact_frame,
            impact_result=impact,
            trajectory=tracker.trajectory,
            draw_badge=False,
        )
        engine = commentary_engine or CommentaryEngine()
        commentary_info = engine.analyze_and_comment(
            snapshot_path=None,
            physical_result=verdict,
            verification_frame=verification_canvas,
        )

    final_decision = (
        commentary_info.get("final_decision", verdict["decision"])
        if commentary_info
        else verdict["decision"]
    )
    match_status = (
        commentary_info.get("match_status", "CONFIRMED")
        if commentary_info
        else "CONFIRMED"
    )

    # 5. 호크아이 판정 스냅샷 생성 (VLM 번복 시 최종 판정에 맞춰 배지 표시)
    snapshot_path = None
    if save_snapshot and impact_frame is not None:
        snapshot_impact = dict(impact)
        snapshot_verdict = dict(verdict)
        snapshot_verdict["decision"] = final_decision
        snapshot_impact["baseline_verdict"] = snapshot_verdict
        snapshot_path = HawkEyeVisualizer.create_judgment_snapshot(
            impact_frame=impact_frame,
            impact_result=snapshot_impact,
            trajectory=tracker.trajectory,
            output_dir=output_dir,
        )

    result = {
        "status": "SUCCESS",
        "video_file": os.path.basename(video_path),
        "total_frames": frame_idx,
        "impact_frame_idx": impact_idx,
        "timestamp_ms": round(impact["timestamp_ms"], 1),
        "contact_point": impact["contact_point"],
        "physical_decision": verdict["decision"],
        "final_decision": final_decision,
        "match_status": match_status,
        "status_detail": verdict["status_detail"],
        "distance_cm": verdict["distance_cm"],
        "confidence": impact["confidence"],
        "snapshot_path": snapshot_path,
        "commentary": {
            "one_line_summary": commentary_info.get("one_line_summary", "") if commentary_info else "",
            "detailed": commentary_info.get("commentary", "") if commentary_info else "",
            "vlm_decision": commentary_info.get("vlm_decision", verdict["decision"]) if commentary_info else verdict["decision"],
            "vlm_confidence": commentary_info.get("vlm_confidence", impact["confidence"]) if commentary_info else impact["confidence"],
            "engine_type": commentary_info.get("engine_type", "NONE") if commentary_info else "NONE",
        },
        "baseline_info": {
            "outer_boundary_y": verdict["outer_boundary_y"],
            "inner_boundary_y": verdict["inner_boundary_y"],
            "slope": verdict.get("slope", 0.0),
            "cm_per_pixel": verdict["cm_per_pixel"],
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="AI Hawk-Eye Lite 추론 및 해설 엔진")
    parser.add_argument("--video", type=str, required=True, help="분석할 동영상 파일 경로")
    parser.add_argument("--roi", type=str, default=None, help="베이스라인 수동 지정 x,y,w,h")
    parser.add_argument("--output-dir", type=str, default="engine/outputs", help="스냅샷 저장 폴더")
    parser.add_argument("--tracker", type=str, default="auto", choices=["auto", "yolo", "hsv"], help="트래커 유형 (auto, yolo, hsv)")
    parser.add_argument("--model", type=str, default=None, help="YOLO 가중치 파일 경로 (.pt)")
    parser.add_argument("--no-ai", action="store_true", help="생성형 AI 해설 생성 비활성화")
    parser.add_argument("--json", action="store_true", help="결과를 순수 JSON 포맷으로만 출력 (서버 연동용)")
    args = parser.parse_args()

    try:
        result = analyze_clip(
            video_path=args.video,
            roi_coords=args.roi,
            output_dir=args.output_dir,
            enable_ai=not args.no_ai,
            tracker_type=args.tracker,
            model_path=args.model,
        )

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            final_dec = result.get("final_decision", "UNCERTAIN")
            comm = result.get("commentary", {})

            print("\n" + "=" * 60)
            print(f"[AI Hawk-Eye Lite 최종 판독: {final_dec}] ({result.get('match_status')})")
            print("=" * 60)
            print(f" * 비디오 파일    : {result.get('video_file')}")
            print(f" * 1차 물리 판정  : {result.get('physical_decision')} ({result.get('status_detail')})")
            print(f" * 최종 AI 판정   : {final_dec}")
            print(f" * 경계 오차      : {result.get('distance_cm')} cm")
            print(f" * 판정 신뢰도    : {int(result.get('confidence', 0) * 100)}%")
            print(f" * 착지 프레임    : {result.get('impact_frame_idx')} 번 (약 {result.get('timestamp_ms')} ms)")
            print(f" * 착지 접촉 좌표 : {result.get('contact_point')}")
            print(f" * 판독 스냅샷    : {result.get('snapshot_path')}")
            print("-" * 60)
            print(f" [AI 한줄 요약]   : {comm.get('one_line_summary')}")
            print(f" [AI 해설 브리핑] : {comm.get('detailed')}")
            print(f" [해설 생성 엔진] : {comm.get('engine_type')}")
            print("=" * 60 + "\n")

    except Exception as e:
        error_res = {"status": "ERROR", "message": str(e)}
        if args.json:
            print(json.dumps(error_res, ensure_ascii=False))
        else:
            print(f"[ERROR] 분석 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
