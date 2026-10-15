"""
test_e2e.py
-----------
AI Hawk-Eye Lite 전체 5단계 End-to-End 통합 파이프라인 검증 스크립트.
[검증 흐름]
1. 백엔드 API 게이트웨이 헬스체크 (GET /api/health)
2. 비디오 클립 멀티파트 업로드 및 고속 추론 (POST /api/judge/upload)
3. 호크아이 스냅샷 생성 및 2차 AI 해설 브리핑 수신 확인
4. 사용자 불복(오심 제기) 피드백 제출 (POST /api/feedback)
5. Active Learning 오심 격리 데이터셋 아카이빙 확인 (dataset/hard-negatives/<id>/)
6. 관리자 리뷰 큐(GET /api/feedback/review-queue) 반영 확인
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error

# Windows 콘솔 인코딩 호환성
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


SERVER_URL = "http://localhost:4000"


def log_step(step_num: int, title: str):
    print(f"\n[Step {step_num}] {title}")
    print("-" * 55)


def http_get(url: str):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, data: dict):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_multipart(url: str, file_path: str):
    boundary = "----WebKitFormBoundary" + str(int(time.time() * 1000))
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def run_e2e_test():
    print("=" * 60)
    print("[AI Hawk-Eye Lite] 전체 5단계 E2E 통합 검증 시작")
    print("=" * 60)

    # 1. 헬스체크
    log_step(1, "백엔드 API 게이트웨이 헬스체크")
    try:
        health = http_get(f"{SERVER_URL}/api/health")
        print(f" -> 서버 상태: {health.get('status')} ({health.get('service')})")
        assert health.get("status") == "ONLINE", "서버가 온라인 상태가 아닙니다."
    except Exception as e:
        print(f"[FAIL] 서버 연결 실패: {e}")
        print(" -> server/ 디렉토리에서 npm start 가 실행 중인지 확인하세요.")
        sys.exit(1)

    # 2. 비디오 업로드 및 AI 판정
    log_step(2, "비디오 클립 업로드 및 초고속 AI 판정 파이프라인")
    test_video = "engine/assets/videos/sample_out.mp4"
    if not os.path.exists(test_video):
        print(f"[ERROR] 테스트 비디오 파일이 없습니다: {test_video}")
        sys.exit(1)

    print(f" -> 비디오 업로드 전송: {test_video} ({os.path.getsize(test_video)} bytes)")
    start_t = time.time()
    upload_res = http_post_multipart(f"{SERVER_URL}/api/judge/upload", test_video)
    elapsed = round(time.time() - start_t, 2)

    assert upload_res.get("success") is True, f"업로드 실패: {upload_res}"
    record = upload_res["data"]
    record_id = record["id"]

    print(f" -> 판독 완료! (소요 시간: {elapsed}초)")
    print(f" -> 최종 판정     : {record['final_decision']} ({record['match_status']})")
    print(f" -> 경계 오차     : {record['distance_cm']} cm")
    print(f" -> 신뢰도        : {int(record['confidence'] * 100)}%")
    print(f" -> 착지 프레임   : #{record['impact_frame_idx']}")
    print(f" -> 호크아이 스냅샷: {record['snapshot_url']}")

    # 3. AI 해설 브리핑 확인
    log_step(3, "생성형 AI 해설 브리핑 검증")
    comm = record.get("commentary", {})
    print(f" -> [한줄 요약] : {comm.get('one_line_summary')}")
    print(f" -> [상세 해설] : {comm.get('detailed')}")
    print(f" -> [해설 엔진] : {comm.get('engine_type')}")
    assert len(comm.get("one_line_summary", "")) > 0, "한 줄 요약이 비어있습니다."

    # 4. 사용자 불복(오심 제기) 피드백 제출
    log_step(4, "동호인 사용자 불복(오심 제기) 피드백 제출")
    feedback_payload = {
        "recordId": record_id,
        "agreed": False,  # 불복!
        "user_decision": "IN",
        "comment": "라인에 코르크가 스친 것 같습니다. 재심 요청합니다.",
    }
    fb_res = http_post_json(f"{SERVER_URL}/api/feedback", feedback_payload)
    assert fb_res.get("success") is True, f"피드백 등록 실패: {fb_res}"
    print(f" -> 피드백 등록 메시지: {fb_res.get('message')}")

    # 5. Active Learning 오심 격리 아카이브 검증
    log_step(5, "Active Learning 오심 케이스 자동 격리 아카이빙 검증")
    archive_dir = os.path.join("dataset", "hard-negatives", record_id)
    print(f" -> 아카이브 디렉토리 확인: {archive_dir}")

    time.sleep(0.5)  # 파일 기록 대기
    assert os.path.exists(archive_dir), f"아카이브 디렉토리가 생성되지 않았습니다: {archive_dir}"

    case_info_file = os.path.join(archive_dir, "case_info.json")
    assert os.path.exists(case_info_file), "case_info.json 메타데이터 파일이 없습니다."
    print(" -> [PASS] case_info.json 메타데이터 확인 완료")

    # 격리된 스냅샷 파일 확인
    archived_files = os.listdir(archive_dir)
    print(f" -> 격리 보관된 파일 목록: {archived_files}")
    assert any(f.endswith(".jpg") for f in archived_files), "격리된 스냅샷 이미지가 없습니다."

    # 6. 관리자 리뷰 큐 반영 확인
    log_step(6, "관리자 리뷰 큐(review-queue) 실시간 반영 확인")
    queue_res = http_get(f"{SERVER_URL}/api/feedback/review-queue")
    queue_items = queue_res.get("data", [])
    print(f" -> 현재 리뷰 큐 대기 건수: {len(queue_items)}건")

    matched = [item for item in queue_items if item["id"] == record_id]
    assert len(matched) == 1, "제출한 불복 케이스가 리뷰 큐에 정상 등록되지 않았습니다."
    print(" -> [PASS] 관리자 리뷰 큐에 등록된 케이스 검증 완료!")

    print("\n" + "=" * 60)
    print("[SUCCESS] AI Hawk-Eye Lite 전체 5단계 통합 테스트 100% 통과!")
    print("=" * 60)
    print(" * 스마트폰 브라우저 접속 : http://localhost:4000/")
    print(" * 관리자 리뷰 큐 접속     : http://localhost:4000/admin.html")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_e2e_test()
