"""
baseline_roi.py
---------------
배드민턴 베이스라인 관심 영역(ROI) 정의 및 물리적 좌표/거리 환산 모듈.
국제 배드민턴 연맹(BWF) 규격: 라인 두께 4cm(40mm)를 기준으로 픽셀-물리거리 매핑을 수행합니다.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class BaselineROI:
    """
    베이스라인(백 바운더리 라인)의 위치, 경계선 및 물리 척도를 관리하는 클래스
    """

    # BWF 규격 라인 폭 (cm)
    STANDARD_LINE_WIDTH_CM = 4.0

    def __init__(
        self,
        line_rect: Optional[Tuple[int, int, int, int]] = None,
        inner_y: Optional[int] = None,
        outer_y: Optional[int] = None,
    ):
        """
        Args:
            line_rect: (x, y, w, h) 라인 전체를 감싸는 바운딩 박스
            inner_y: 코트 안쪽(네트 방향) 라인 경계 y좌표 (None이면 line_rect[1])
            outer_y: 코트 바깥쪽 라인 외곽 경계 y좌표 (None이면 line_rect[1] + line_rect[3])
        """
        self.line_rect = line_rect
        self.inner_y = inner_y
        self.outer_y = outer_y
        self.pixel_height = None
        self.cm_per_pixel = 0.1  # 기본 추정치 (1px당 1mm)

        if line_rect is not None:
            self._update_boundaries(line_rect, inner_y, outer_y)

    def _update_boundaries(
        self,
        rect: Tuple[int, int, int, int],
        inner_y: Optional[int] = None,
        outer_y: Optional[int] = None,
    ):
        """경계 좌표 및 물리 환산 계수 갱신"""
        self.line_rect = rect
        x, y, w, h = rect
        self.inner_y = inner_y if inner_y is not None else y
        self.outer_y = outer_y if outer_y is not None else (y + h)

        self.pixel_height = max(1, abs(self.outer_y - self.inner_y))
        # 4.0cm / 라인 픽셀 두께 = 1픽셀당 센티미터
        self.cm_per_pixel = self.STANDARD_LINE_WIDTH_CM / float(self.pixel_height)

    def auto_detect_baseline(self, frame: np.ndarray, search_ratio: float = 0.5) -> bool:
        """
        영상 하단부(search_ratio 비율)에서 흰색 수평선(베이스라인)을 자동 탐지합니다.

        Args:
            frame: 분석할 원본 비디오 프레임
            search_ratio: 영상 하단 몇 % 영역을 탐색할지 (기본 0.5 = 하단 50%)

        Returns:
            bool: 탐지 성공 여부
        """
        h, w = frame.shape[:2]
        start_y = int(h * (1.0 - search_ratio))
        roi_img = frame[start_y:h, 0:w]

        # 흰색 라인 분리를 위한 HSV 변환
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 수평 성분을 강조하기 위한 형태학적 연산 (수평 커널)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)

        # 컨투어 추출
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        max_width = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 가로로 긴 형태 (화면 폭의 20% 이상)
            if cw > w * 0.2 and ch >= 3:
                if cw > max_width:
                    max_width = cw
                    best_candidate = (x, start_y + y, cw, ch)

        if best_candidate is not None:
            self._update_boundaries(best_candidate)
            return True

        # 자동 탐지 실패 시 화면 하단 기본 영역 설정
        default_y = int(h * 0.8)
        default_h = max(6, int(h * 0.02))
        self._update_boundaries((0, default_y, w, default_h))
        return False

    def select_interactively(self, frame: np.ndarray, window_name: str = "Select Baseline Line"):
        """마우스 드래그로 베이스라인 영역을 지정합니다."""
        guide = frame.copy()
        cv2.putText(
            guide,
            "Drag baseline line (Space/Enter: Confirm, c: Cancel)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        rect = cv2.selectROI(window_name, guide, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if rect[2] > 0 and rect[3] > 0:
            self._update_boundaries((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
            return True
        return False

    def judge_point(self, x: float, y: float, contact_radius_px: float = 4.0) -> Dict[str, Any]:
        """
        주어진 셔틀콕 착지 접촉점 (x, y)에 대한 In/Out 물리 판정 수행

        배드민턴 규정:
        - 라인 위(In): 셔틀콕의 접촉점이 베이스라인의 외곽선(outer_y) 안쪽에 닿음
        - 아웃(Out): 셔틀콕의 접촉점이 베이스라인 외곽선(outer_y)을 완전히 벗어남

        Args:
            x, y: 셔틀콕 접촉점 픽셀 좌표
            contact_radius_px: 셔틀콕 코르크 지름(약 2.5~2.8cm)에 대응하는 접촉 유효 반경(px)

        Returns:
            dict: {
                "decision": "IN" | "OUT",
                "status_detail": "ON_LINE" | "COURT_INSIDE" | "LINE_OUT",
                "distance_cm": float,  # 라인 경계로부터의 거리 (cm, Out이면 양수, In이면 0 또는 음수)
                "outer_boundary_y": int,
                "contact_y": float
            }
        """
        if self.outer_y is None or self.inner_y is None:
            raise ValueError("베이스라인 경계가 설정되지 않았습니다.")

        # 셔틀콕 코르크가 라인 쪽으로 닿는 가장 바깥쪽 지점
        # 카메라 뷰에서 아래쪽으로 떨어지므로, y좌표가 클수록 코트 바깥쪽
        effective_y = y - contact_radius_px

        # 1. 라인 외곽선(outer_y)을 넘어섰는지 확인
        pixel_diff = effective_y - self.outer_y

        if pixel_diff <= 0:
            # 라인 안쪽 또는 라인 위 착지 -> IN
            dist_cm = 0.0
            if y >= self.inner_y:
                status = "ON_LINE"
            else:
                status = "COURT_INSIDE"
            decision = "IN"
        else:
            # 외곽선을 완전히 벗어남 -> OUT
            dist_cm = round(pixel_diff * self.cm_per_pixel, 2)
            status = "LINE_OUT"
            decision = "OUT"

        return {
            "decision": decision,
            "status_detail": status,
            "distance_cm": dist_cm,
            "outer_boundary_y": self.outer_y,
            "inner_boundary_y": self.inner_y,
            "contact_x": round(x, 1),
            "contact_y": round(y, 1),
            "cm_per_pixel": round(self.cm_per_pixel, 4),
        }

    def draw_baseline(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """프레임 위에 베이스라인 경계선 및 영역 오버레이"""
        out = frame.copy()
        if self.line_rect is None:
            return out

        x, y, w, h = self.line_rect
        # 라인 외곽선 (Outer Boundary) - 빨간색 점선 느낌의 강조선
        cv2.line(out, (x, self.outer_y), (x + w, self.outer_y), (0, 0, 255), 2)
        # 라인 내곽선 (Inner Boundary) - 노란색
        cv2.line(out, (x, self.inner_y), (x + w, self.inner_y), (0, 255, 255), 1)

        # 라인 라벨
        cv2.putText(
            out,
            f"BASELINE (outer_y={self.outer_y})",
            (x + 10, self.outer_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return out
