"""
visualizer.py
-------------
배드민턴 호크아이 스타일의 착지 순간 하이라이트 영상 및 스냅샷 생성기.
- 착지 지점 2.5배 확대 인셋(돋보기) 오버레이
- 대형 IN / OUT 판정 뱃지 및 물리 거리(cm) 표시
- 전체 비행 궤적 선 렌더링
"""

from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import cv2
import numpy as np


class HawkEyeVisualizer:
    """
    판정 결과 시각화 및 스냅샷 생성 클래스
    """

    @staticmethod
    def create_judgment_snapshot(
        impact_frame: np.ndarray,
        impact_result: Dict[str, Any],
        trajectory: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "engine/outputs",
    ) -> str:
        """
        호크아이 스타일 판독 스냅샷 이미지를 생성하고 파일로 저장

        Args:
            impact_frame: 착지 순간의 원본 BGR 프레임
            impact_result: ImpactDetector의 분석 결과 dict
            trajectory: 전체 궤적 리스트
            output_dir: 결과 저장 디렉토리

        Returns:
            str: 저장된 이미지 파일 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        h, w = impact_frame.shape[:2]
        canvas = impact_frame.copy()

        verdict = impact_result["baseline_verdict"]
        decision = verdict["decision"]
        dist_cm = verdict["distance_cm"]
        cx, cy = int(impact_result["contact_point"][0]), int(impact_result["contact_point"][1])

        # 1. 베이스라인 외곽선 강조 (적색 실선)
        outer_y = verdict["outer_boundary_y"]
        cv2.line(canvas, (0, outer_y), (w, outer_y), (0, 0, 255), 2)
        cv2.putText(canvas, f"OUTER BASELINE (y={outer_y})", (20, outer_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 2. 궤적 선 렌더링 (주황색)
        if trajectory:
            pts = [pt["center"] for pt in trajectory]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (0, 165, 255), 2)

        # 3. 착지 접촉점 십자선 (Crosshair)
        cross_size = 15
        contact_color = (0, 255, 0) if decision == "IN" else (0, 0, 255)
        cv2.drawMarker(canvas, (cx, cy), contact_color, cv2.MARKER_CROSS, cross_size, 2)
        cv2.circle(canvas, (cx, cy), 6, contact_color, 2)

        # 4. 착지점 돋보기 인셋 뷰 (Magnifier Inset)
        crop_size = 70  # 140x140 영역 크롭
        x1 = max(0, cx - crop_size)
        y1 = max(0, cy - crop_size)
        x2 = min(w, cx + crop_size)
        y2 = min(h, cy + crop_size)

        patch = impact_frame[y1:y2, x1:x2]
        if patch.size > 0:
            zoom_size = 200
            zoom_patch = cv2.resize(patch, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

            # 돋보기 내 십자선 표시 (중심점 위치 환산)
            center_zoom_x = int((cx - x1) * (zoom_size / float(x2 - x1)))
            center_zoom_y = int((cy - y1) * (zoom_size / float(y2 - y1)))
            cv2.drawMarker(zoom_patch, (center_zoom_x, center_zoom_y), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

            # 돋보기 위치: 화면 우측 상단
            inset_x = w - zoom_size - 25
            inset_y = 25

            # 돋보기 테두리
            cv2.rectangle(canvas, (inset_x - 3, inset_y - 3),
                          (inset_x + zoom_size + 3, inset_y + zoom_size + 3), (255, 255, 255), 3)
            canvas[inset_y:inset_y + zoom_size, inset_x:inset_x + zoom_size] = zoom_patch

            cv2.putText(canvas, "IMPACT ZOOM (2.5x)", (inset_x + 10, inset_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. 호크아이 대형 판정 뱃지 (상단 좌측)
        badge_w, badge_h = 240, 75
        badge_x, badge_y = 20, 20
        badge_bg = (0, 160, 0) if decision == "IN" else (0, 0, 210)  # 초록(IN) vs 빨강(OUT)

        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_bg, -1)
        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (255, 255, 255), 2)

        cv2.putText(canvas, f"DECISION: {decision}", (badge_x + 15, badge_y + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        sub_text = f"Margin: {dist_cm} cm" if decision == "OUT" else f"Status: {verdict['status_detail']}"
        cv2.putText(canvas, sub_text, (badge_x + 15, badge_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        # 6. 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hawkeye_{decision.lower()}_{timestamp}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, canvas)
        return save_path
