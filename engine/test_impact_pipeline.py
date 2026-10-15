"""
test_impact_pipeline.py
-----------------------
2단계: 베이스라인 판정 로직 결합 테스트 스크립트
추적된 좌표 (x, y)의 Y축 변화율(dy/dt)과 바닥 베이스라인 경계선(outer_y)을 비교 분석하여
정밀 착지 순간(Impact Frame)을 검출하고 최종 IN/OUT 판정을 산출합니다.
"""

import os
import sys
import argparse
from typing import Dict, Any

# Windows 콘솔 utf-8 보장
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# core 모듈 import 경로 설정
engine_dir = os.path.dirname(os.path.abspath(__file__))
if engine_dir not in sys.path:
    sys.path.insert(0, engine_dir)

from core.baseline_roi import BaselineROI
from core.tracker import ShuttlecockTracker
from core.impact_detector import ImpactDetector
from core.visualizer import HawkEyeVisualizer


def test_landing_algorithm(
    video_path: str,
    output_dir: str = "engine/outputs",
    tracker_type: str = "auto",
) -> Dict[str, Any]:
    """
    비디오 클립에서 Y축 변화율과 바닥 베이스라인을 결합한 착지 알고리즘을 검증합니다.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오를 찾을 수 없습니다: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print("🏸 [2단계] Y축 변화율 기반 착지 검출 및 베이스라인 판정 테스트")
    print("=" * 70)
    print(f" * 비디오 파일   : {os.path.basename(video_path)}")
    print(f" * 해상도 / FPS  : {width}x{height} @ {fps:.1f} fps")
    print("-" * 70)

    baseline = BaselineROI()
    tracker = ShuttlecockTracker(tracker_type=tracker_type)
    frames_buffer = []
    frame_idx = 0

    # 1. 프레임 순회 및 궤적 기록
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_idx / fps) * 1000.0

        # 베이스라인 자동 검출 (첫 프레임)
        if baseline.line_rect is None and frame_idx == 0:
            success = baseline.auto_detect_baseline(frame)
            if success:
                print(f" [검출] 베이스라인 외곽 경계 y={baseline.outer_y}px, 내곽 y={baseline.inner_y}px (1px당 {baseline.cm_per_pixel:.3f}cm)")
            else:
                print(" [주의] 베이스라인 자동 검출 실패 -> 화면 하단 기본 영역 적용")

        tracker.process_frame(frame, frame_idx=frame_idx, timestamp_ms=timestamp_ms)
        frames_buffer.append(frame)
        frame_idx += 1

    cap.release()

    trajectory = tracker.get_trajectory()
    if not trajectory:
        print("[오류] 셔틀콕 이동 궤적을 감지하지 못했습니다.")
        return {"status": "FAIL"}

    print(f" [추적] 총 {len(trajectory)}개 프레임에서 셔틀콕 위치(x, y) 확보 완료")

    # 2. Y축 변화율 (수직 속도 dy) 시계열 분석 및 출력
    y_vals = [pt["center"][1] for pt in trajectory]
    frame_nums = [pt["frame_idx"] for pt in trajectory]
    dy_vals = np.diff(y_vals).tolist() if len(y_vals) > 1 else [0.0]

    print("\n📈 [Y축 좌표 및 속도 변화율 (dy) 분석 샘플 - 주요 낙하 구간]")
    print(f"{'프레임':^8} | {'X좌표':^8} | {'Y좌표':^8} | {'수직변화율(dy)':^14} | {'상태':^12}")
    print("-" * 60)

    # 낙하 정점 주변 데이터 출력
    for i in range(len(trajectory)):
        f_num = trajectory[i]["frame_idx"]
        cx, cy = trajectory[i]["center"]
        dy_str = f"{dy_vals[i-1]:+.1f} px/f" if i > 0 else "  -  "
        
        status_tag = "공중 비행"
        if i > 0 and dy_vals[i-1] > 0:
            status_tag = "수직 낙하중 (↓)"
        elif i > 0 and dy_vals[i-1] <= 0:
            status_tag = "착지/바운드 (정점)"

        # 간격 맞춰 일부 프레임과 충돌 구간을 중점 표시
        if i % 3 == 0 or (i > 0 and dy_vals[i-1] <= 0 and dy_vals[i-2] > 0):
            print(f"{f_num:^8} | {cx:^8.1f} | {cy:^8.1f} | {dy_str:^14} | {status_tag:^12}")

    # 3. 착지 순간(Impact Frame) 및 물리 판정 산출
    detector = ImpactDetector(baseline)
    impact = detector.detect_impact(trajectory)

    if impact is None:
        print("\n[오류] 착지 순간을 특정할 수 없습니다.")
        return {"status": "FAIL"}

    verdict = impact["baseline_verdict"]
    impact_frame_idx = impact["impact_frame_idx"]
    contact_x, contact_y = impact["contact_point"]
    decision = verdict["decision"]
    distance_cm = verdict["distance_cm"]
    status_detail = verdict["status_detail"]

    print("\n🎯 [최종 착지 및 베이스라인 판정 결과]")
    print("=" * 70)
    print(f" * 착지 판정 프레임   : {impact_frame_idx} 번 프레임 (영상 시간: {impact['timestamp_ms']:.1f} ms)")
    print(f" * 셔틀콕 접촉점 좌표 : ({contact_x:.1f}, {contact_y:.1f}) px")
    print(f" * 베이스라인 경계선  : outer_y = {verdict['outer_boundary_y']} px")
    print(f" * 1차 물리 판정      : >> [{decision}] << ({status_detail})")
    print(f" * 라인 경계 이격 거리: {distance_cm:.2f} cm (IN: 0.0cm 걸침, OUT: 외곽 초과 거리)")
    print(f" * 알고리즘 신뢰도    : {impact['confidence'] * 100:.1f}%")
    print("=" * 70)

    # 4. 판정 스냅샷 이미지 저장
    impact_frame = frames_buffer[min(impact_frame_idx, len(frames_buffer) - 1)]
    snapshot_path = HawkEyeVisualizer.create_judgment_snapshot(
        impact_frame=impact_frame,
        impact_result=impact,
        trajectory=trajectory,
        output_dir=output_dir,
    )
    print(f"📸 호크아이 정밀 판정 스냅샷 저장: {snapshot_path}")

    return {
        "status": "SUCCESS",
        "video": os.path.basename(video_path),
        "impact_frame_idx": impact_frame_idx,
        "contact_point": impact["contact_point"],
        "decision": decision,
        "distance_cm": distance_cm,
        "status_detail": status_detail,
        "confidence": impact["confidence"],
        "snapshot_path": snapshot_path,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Hawk-Eye 2단계 착지 알고리즘 검증")
    parser.add_argument("--video", type=str, default="engine/assets/videos/sample_in.mp4", help="테스트 비디오")
    parser.add_argument("--tracker", type=str, default="auto", choices=["auto", "yolo", "hsv"], help="트래커 유형")
    args = parser.parse_args()

    # IN / OUT 샘플이 있으면 둘 다 테스트
    if args.video == "all":
        in_video = "engine/assets/videos/sample_in.mp4"
        out_video = "engine/assets/videos/sample_out.mp4"
        print("\n>>> [1/2] IN 샘플 영상 검증 시작 <<<")
        test_landing_algorithm(in_video, tracker_type=args.tracker)
        print("\n>>> [2/2] OUT 샘플 영상 검증 시작 <<<")
        test_landing_algorithm(out_video, tracker_type=args.tracker)
    else:
        test_landing_algorithm(args.video, tracker_type=args.tracker)


if __name__ == "__main__":
    main()
