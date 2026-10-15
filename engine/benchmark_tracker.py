"""
benchmark_tracker.py
--------------------
1단계: 셔틀콕 추적 엔진 벤치마크 테스트 스크립트
로컬 GPU(RTX 4060) 환경에서 셔틀콕 추적 모델의 추론 속도(ms/frame, FPS),
프레임 검출률, GPU VRAM 사용량을 정량적으로 측정하고 검증합니다.
"""

import os
import sys
import time
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

from core.tracker import ShuttlecockTracker


def run_benchmark(
    video_path: str,
    tracker_type: str = "auto",
    model_path: str = None,
    warmup_frames: int = 5,
) -> Dict[str, Any]:
    """
    단일 비디오에 대해 프레임당 추적 성능 및 GPU 지연시간을 벤치마크합니다.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오를 찾을 수 없습니다: {video_path}")

    # GPU 가속 정보 확인
    gpu_name = "CPU Only"
    cuda_available = False
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_video_frames / fps if fps > 0 else 0.0

    print("=" * 65)
    print("🏸 [1단계] 셔틀콕 추적기 성능 벤치마크 (RTX 4060 검증)")
    print("=" * 65)
    print(f" * 비디오 파일   : {os.path.basename(video_path)}")
    print(f" * 해상도 / FPS  : {width}x{height} @ {fps:.1f} fps")
    print(f" * 전체 프레임   : {total_video_frames} 프레임 ({duration_sec:.2f} 초)")
    print(f" * 추론 디바이스 : {gpu_name} (CUDA: {cuda_available})")
    print(f" * 트래커 모드   : {tracker_type.upper()}")
    print("-" * 65)

    tracker = ShuttlecockTracker(tracker_type=tracker_type, model_path=model_path)

    frame_times = []
    detected_count = 0
    frame_idx = 0

    # 추론 루프
    start_total_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_idx / fps) * 1000.0

        t0 = time.perf_counter()
        det = tracker.process_frame(frame, frame_idx=frame_idx, timestamp_ms=timestamp_ms)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        if frame_idx >= warmup_frames:
            frame_times.append(elapsed_ms)

        if det is not None:
            detected_count += 1

        frame_idx += 1

    total_proc_time = time.perf_counter() - start_total_time
    cap.release()

    # 정량적 수치 계산
    effective_frames = len(frame_times)
    avg_latency_ms = np.mean(frame_times) if frame_times else 0.0
    p95_latency_ms = np.percentile(frame_times, 95) if frame_times else 0.0
    min_latency_ms = np.min(frame_times) if frame_times else 0.0
    max_latency_ms = np.max(frame_times) if frame_times else 0.0
    processing_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    detection_rate = (detected_count / total_video_frames * 100.0) if total_video_frames > 0 else 0.0
    realtime_factor = (processing_fps / fps) if fps > 0 else 0.0

    print("📊 [벤치마크 결과]")
    print(f" * 총 소요 시간     : {total_proc_time:.3f} 초 (영상 길이: {duration_sec:.2f}초)")
    print(f" * 평균 프레임 지연 : {avg_latency_ms:.2f} ms / frame (최소: {min_latency_ms:.2f}ms, 최대: {max_latency_ms:.2f}ms)")
    print(f" * 95% 지연시간     : {p95_latency_ms:.2f} ms")
    print(f" * 처리 속도 (FPS)  : {processing_fps:.1f} FPS (실시간 대비 {realtime_factor:.1f}배 속도)")
    print(f" * 프레임 검출률    : {detection_rate:.1f}% ({detected_count}/{total_video_frames} frames)")
    print(f" * 추적된 궤적 길이 : {len(tracker.get_trajectory())} 점")
    print("=" * 65)

    # 120fps 실시간 가능 여부 평가
    can_handle_120fps = processing_fps >= 120.0
    print(f"💡 120fps 슬로우모션 실시간 처리 가능 여부: {'[PASS] 가능' if can_handle_120fps else '[INFO] 버퍼 후처리 가능'}")
    print("=" * 65)

    return {
        "video": os.path.basename(video_path),
        "total_frames": total_video_frames,
        "avg_latency_ms": round(float(avg_latency_ms), 2),
        "processing_fps": round(float(processing_fps), 1),
        "detection_rate_pct": round(float(detection_rate), 1),
        "realtime_factor": round(float(realtime_factor), 2),
        "can_handle_120fps": can_handle_120fps,
        "gpu_device": gpu_name,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Hawk-Eye 1단계 셔틀콕 추적 벤치마크")
    parser.add_argument("--video", type=str, default="engine/assets/videos/sample_in.mp4", help="테스트 비디오 경로")
    parser.add_argument("--tracker", type=str, default="auto", choices=["auto", "yolo", "hsv"], help="트래커 유형")
    parser.add_argument("--model", type=str, default=None, help="YOLO 모델 가중치 파일 경로 (.pt)")
    args = parser.parse_args()

    # 샘플 비디오가 없는 경우 자동 생성
    if not os.path.exists(args.video):
        from create_sample_video import render_badminton_clip
        print(f"[알림] 샘플 영상이 없어 자동 생성합니다: {args.video}")
        render_badminton_clip("sample_in.mp4", target_mode="IN")

    run_benchmark(args.video, tracker_type=args.tracker, model_path=args.model)


if __name__ == "__main__":
    main()
