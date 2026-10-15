"""
create_sample_video.py
----------------------
배드민턴 베이스라인 판독 검증을 위한 가상 경기 클립 2종 생성기
1) sample_in.mp4  : 베이스라인 선 위에 아슬아슬하게 걸쳐 떨어지는 샷 (IN 판정)
2) sample_out.mp4 : 베이스라인 외곽선을 2~3cm 벗어나 떨어지는 샷 (OUT 판정)
"""

import os
import cv2
import numpy as np


def render_badminton_clip(
    output_filename: str,
    target_mode: str = "IN",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_sec: float = 2.5,
):
    """
    포물선 궤적과 착지 충돌 후 감속 모션을 포함한 고품질 시뮬레이션 클립 렌더링
    """
    output_dir = "engine/assets/videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(fps * duration_sec)
    impact_frame_target = int(fps * 1.6)  # 1.6초 시점에 착지

    # 베이스라인 규격 정의 (화면 하단)
    baseline_outer_y = int(height * 0.80)  # 외곽선 y = 576
    line_thickness = 12  # 약 4cm에 해당하는 픽셀 두께
    baseline_inner_y = baseline_outer_y - line_thickness

    # 착지 목표 좌표 결정
    if target_mode == "IN":
        # 라인 위(중간)에 정확히 착지
        landing_x = int(width * 0.62)
        landing_y = baseline_inner_y + int(line_thickness * 0.5)
    else:
        # 라인 외곽선에서 약 18픽셀(약 2.5cm) 바깥에 착지
        landing_x = int(width * 0.62)
        landing_y = baseline_outer_y + 18

    p_start = (int(width * 0.2), int(height * 0.15))
    p_peak = (int(width * 0.45), int(height * 0.25))
    p_impact = (landing_x, landing_y)

    print(f"[생성 중] {target_mode} 테스트 클립 렌더링 -> {output_path}")

    shuttle_pos = p_start

    for f_idx in range(total_frames):
        # 1. 코트 바닥 (배드민턴 녹색 매트: BGR 55, 115, 45)
        frame = np.full((height, width, 3), (55, 115, 45), dtype=np.uint8)

        # 2. 베이스라인 및 사이드라인 그리기
        # 베이스라인 (백 바운더리 라인)
        cv2.rectangle(frame, (100, baseline_inner_y), (width - 100, baseline_outer_y), (255, 255, 255), -1)

        # 좌우 사이드라인
        cv2.line(frame, (100, 50), (100, baseline_outer_y), (255, 255, 255), 8)
        cv2.line(frame, (width - 100, 50), (width - 100, baseline_outer_y), (255, 255, 255), 8)

        # 3. 셔틀콕 위치 계산
        if f_idx <= impact_frame_target:
            # 착지 전 비행 (2차 베지어 곡선)
            t = f_idx / float(impact_frame_target)
            cx = int((1 - t)**2 * p_start[0] + 2 * (1 - t) * t * p_peak[0] + t**2 * p_impact[0])
            cy = int((1 - t)**2 * p_start[1] + 2 * (1 - t) * t * p_peak[1] + t**2 * p_impact[1])
            shuttle_pos = (cx, cy)
        else:
            # 착지 후: 바닥에서 살짝 미끄러지며 정지
            after_frames = f_idx - impact_frame_target
            slide_dx = min(15, after_frames * 2)
            shuttle_pos = (p_impact[0] + slide_dx, p_impact[1])

        # 4. 셔틀콕 렌더링 (원근감 고려)
        progress = min(1.0, f_idx / float(impact_frame_target))
        shuttle_radius = int(8 + progress * 7)

        # 흰색 셔틀콕 깃털
        cv2.circle(frame, shuttle_pos, shuttle_radius, (255, 255, 255), -1)
        cv2.circle(frame, shuttle_pos, shuttle_radius, (210, 210, 210), 1)

        # 코르크 (앞부분)
        cork_pos = (shuttle_pos[0] + int(shuttle_radius * 0.4), shuttle_pos[1] + int(shuttle_radius * 0.4))
        cv2.circle(frame, cork_pos, max(2, shuttle_radius // 3), (35, 80, 150), -1)

        writer.write(frame)

    writer.release()
    print(f"[완료] {output_path} 생성 성공.")


if __name__ == "__main__":
    render_badminton_clip("sample_in.mp4", target_mode="IN")
    render_badminton_clip("sample_out.mp4", target_mode="OUT")
