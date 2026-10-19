"""
worker.py
---------
Node.js 백엔드와 표준 입출력(stdin/stdout JSON-RPC)으로 통신하는 상주형(Persistent) AI 추론 워커.
서버 기동 시 YOLOv8 모델(CUDA)과 Gemini VLM 클라이언트를 메모리에 미리 로드하여
매 요청마다 프로세스를 재생성할 때 발생하는 Cold-Start 지연(2~4초)을 제거합니다.
"""

import json
import os
import sys
import traceback

# Windows 인코딩 호환성 보장
if sys.stdin.encoding and sys.stdin.encoding.lower() != "utf-8":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# engine 디렉토리를 모듈 탐색 경로에 추가
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

# 일반 print() 로그가 JSON 응답 스트림(stdout)을 오염시키지 않도록
# 실제 응답 파이프를 별도 보관하고 기본 stdout은 stderr로 리다이렉트
_rpc_stdout = sys.stdout
sys.stdout = sys.stderr

import numpy as np
from core.tracker import ShuttlecockTracker
from core.commentary import CommentaryEngine
from main import analyze_clip


def send_rpc_message(payload: dict):
    """Node.js 부모 프로세스로 단일 라인 JSON 메시지 전송"""
    line = json.dumps(payload, ensure_ascii=False)
    _rpc_stdout.write(line + "\n")
    _rpc_stdout.flush()


def main():
    print("[Worker] 상주형 AI 추론 워커 초기화 시작...")
    tracker = ShuttlecockTracker(tracker_type="auto")
    commentary_engine = CommentaryEngine()

    # YOLO 모델 워밍업 (첫 요청 지연 방지)
    if tracker.active_backend == "yolo" and tracker.yolo_model is not None:
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            tracker.yolo_model.predict(dummy, device=tracker.device, verbose=False)
            print(f"[Worker] YOLOv8 GPU/CPU 워밍업 완료 (device={tracker.device})")
        except Exception as e:
            print(f"[Worker] 워밍업 경고: {e}")

    # 준비 완료 신호 전송
    send_rpc_message({
        "type": "READY",
        "backend": tracker.active_backend,
        "device": tracker.device,
    })

    # 요청 수신 루프
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            cmd = req.get("cmd", "analyze")

            if cmd == "ping":
                send_rpc_message({"id": req_id, "ok": True, "result": {"status": "PONG"}})
                continue

            if cmd == "shutdown":
                send_rpc_message({"id": req_id, "ok": True, "result": {"status": "BYE"}})
                break

            video_path = req.get("video_path")
            roi_coords = req.get("roi_coords")
            output_dir = req.get("output_dir", "engine/outputs")
            enable_ai = req.get("enable_ai", True)

            result = analyze_clip(
                video_path=video_path,
                roi_coords=roi_coords,
                output_dir=output_dir,
                save_snapshot=True,
                enable_ai=enable_ai,
                tracker=tracker,
                commentary_engine=commentary_engine,
            )

            send_rpc_message({
                "id": req_id,
                "ok": True,
                "result": result,
            })

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            send_rpc_message({
                "id": req_id,
                "ok": False,
                "error": str(e),
            })


if __name__ == "__main__":
    main()
