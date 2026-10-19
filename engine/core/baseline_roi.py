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

    # BWF 규격 라인 폭 (cm) 및 셔틀콕 코르크 접촉 반경 (cm)
    STANDARD_LINE_WIDTH_CM = 4.0
    STANDARD_CORK_RADIUS_CM = 1.3

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
        # 기울어진 카메라 대응을 위한 직선 방정식 계수 (y = slope * x + intercept)
        self.slope: float = 0.0
        self.outer_intercept: Optional[float] = None
        self.inner_intercept: Optional[float] = None

        if line_rect is not None:
            self._update_boundaries(line_rect, inner_y, outer_y)

    def _update_boundaries(
        self,
        rect: Tuple[int, int, int, int],
        inner_y: Optional[int] = None,
        outer_y: Optional[int] = None,
        slope: float = 0.0,
        true_thickness_px: Optional[float] = None,
    ):
        """경계 좌표 및 물리 환산 계수 갱신"""
        self.line_rect = rect
        x, y, w, h = rect
        self.inner_y = int(inner_y if inner_y is not None else y)
        self.outer_y = int(outer_y if outer_y is not None else (y + h))

        raw_height = max(1.0, float(abs(self.outer_y - self.inner_y)))
        self.pixel_height = int(round(true_thickness_px)) if true_thickness_px else int(raw_height)
        self.pixel_height = max(1, self.pixel_height)

        # 4.0cm / 라인 픽셀 두께 = 1픽셀당 센티미터
        self.cm_per_pixel = self.STANDARD_LINE_WIDTH_CM / float(self.pixel_height)

        center_x = x + w / 2.0
        self.slope = float(slope)
        self.outer_intercept = float(self.outer_y) - self.slope * center_x
        self.inner_intercept = float(self.inner_y) - self.slope * center_x

    def get_outer_y_at(self, x: float) -> float:
        """주어진 x 좌표에서의 베이스라인 외곽 경계 y값 반환 (기울기 반영)"""
        if self.outer_intercept is not None:
            return self.slope * float(x) + self.outer_intercept
        return float(self.outer_y or 0)

    def get_inner_y_at(self, x: float) -> float:
        """주어진 x 좌표에서의 베이스라인 내곽 경계 y값 반환 (기울기 반영)"""
        if self.inner_intercept is not None:
            return self.slope * float(x) + self.inner_intercept
        return float(self.inner_y or 0)

    def auto_detect_baseline(self, frame: np.ndarray, search_ratio: float = 0.5) -> bool:
        """
        영상 하단부(search_ratio 비율)에서 흰색 베이스라인을 자동 탐지합니다.
        카메라가 약간 기울어진 경우에도 열(column)별 상/하단 경계를 추출하여
        직선 방정식(y = ax + b)과 실제 라인 두께를 정밀하게 추정합니다.

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
        best_cnt = None
        best_candidate = None
        max_width = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 가로로 긴 형태 (화면 폭의 20% 이상)
            if cw > w * 0.2 and ch >= 3:
                if cw > max_width:
                    max_width = cw
                    best_cnt = cnt
                    best_candidate = (x, start_y + y, cw, ch)

        if best_candidate is not None and best_cnt is not None:
            bx, by, bw, bh = best_candidate
            # 선택된 컨투어 영역 내에서 열(column)별 상/하단 y좌표 정밀 분석
            cnt_mask = np.zeros_like(opened)
            cv2.drawContours(cnt_mask, [best_cnt], -1, 255, -1)

            col_xs = []
            top_ys = []
            bot_ys = []
            thicknesses = []

            for col in range(bx, bx + bw, 2):
                ys = np.where(cnt_mask[:, col] > 0)[0]
                if len(ys) >= 2:
                    y_top = float(start_y + ys[0])
                    y_bot = float(start_y + ys[-1] + 1)
                    thickness = y_bot - y_top
                    if 3.0 <= thickness <= h * 0.12:
                        col_xs.append(float(col))
                        top_ys.append(y_top)
                        bot_ys.append(y_bot)
                        thicknesses.append(thickness)

            if len(col_xs) >= 10:
                slope, outer_b = np.polyfit(col_xs, bot_ys, 1)
                # 비정상적인 급경사(15도 초과) 방지 클램핑
                slope = float(np.clip(slope, -0.27, 0.27))
                median_thickness = float(np.median(thicknesses))
                center_x = bx + bw / 2.0
                center_outer_y = int(round(slope * center_x + outer_b))
                center_inner_y = int(round(center_outer_y - median_thickness))
                self._update_boundaries(
                    best_candidate,
                    inner_y=center_inner_y,
                    outer_y=center_outer_y,
                    slope=slope,
                    true_thickness_px=median_thickness,
                )
            else:
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

    def judge_point(
        self,
        x: float,
        y: float,
        contact_radius_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        주어진 셔틀콕 착지 접촉점 (x, y)에 대한 In/Out 물리 판정 수행

        배드민턴 규정:
        - 라인 위(In): 셔틀콕의 접촉점이 베이스라인의 외곽선(outer_y) 안쪽에 닿음
        - 아웃(Out): 셔틀콕의 접촉점이 베이스라인 외곽선(outer_y)을 완전히 벗어남

        Args:
            x, y: 셔틀콕 코르크 착지 중심점 픽셀 좌표
            contact_radius_px: 셔틀콕 코르크 접촉 유효 반경(px). None이면 cm_per_pixel 기반 자동 산출

        Returns:
            dict: 물리 판정 상세 결과
        """
        if self.outer_y is None or self.inner_y is None:
            raise ValueError("베이스라인 경계가 설정되지 않았습니다.")

        # x 좌표에서의 기울기 보정된 외곽/내곽 경계 y좌표
        local_outer_y = self.get_outer_y_at(x)
        local_inner_y = self.get_inner_y_at(x)

        # 해상도/라인 두께에 비례하는 코르크 접촉 반경(px) 동적 계산
        if contact_radius_px is None:
            # 코르크 지면 접촉 유효 반경(약 0.8~1.2cm)을 픽셀로 환산
            dynamic_radius = 1.0 / max(0.05, self.cm_per_pixel)
            contact_radius_px = float(np.clip(dynamic_radius, 2.0, 10.0))

        # 셔틀콕 코르크가 라인 쪽으로 닿는 가장 안쪽 지점
        # 카메라 뷰에서 아래쪽으로 떨어지므로, y좌표가 클수록 코트 바깥쪽
        effective_y = y - contact_radius_px

        # 기울기 보정 수직 거리 (cos(theta))
        cos_theta = 1.0 / np.hypot(1.0, self.slope)
        pixel_diff = (effective_y - local_outer_y) * cos_theta

        if pixel_diff <= 0:
            # 라인 안쪽 또는 라인 위 착지 -> IN
            dist_cm = 0.0
            if (y + contact_radius_px) >= local_inner_y:
                status = "ON_LINE"
            else:
                status = "COURT_INSIDE"
            decision = "IN"
        else:
            # 외곽선을 완전히 벗어남 -> OUT
            dist_cm = round(float(pixel_diff * self.cm_per_pixel), 2)
            status = "LINE_OUT"
            decision = "OUT"

        return {
            "decision": decision,
            "status_detail": status,
            "distance_cm": dist_cm,
            "outer_boundary_y": int(round(local_outer_y)),
            "inner_boundary_y": int(round(local_inner_y)),
            "slope": round(self.slope, 5),
            "contact_x": round(x, 1),
            "contact_y": round(y, 1),
            "contact_point": (round(x, 1), round(y, 1)),
            "cm_per_pixel": round(self.cm_per_pixel, 4),
        }

    def draw_baseline(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """프레임 위에 베이스라인 경계선 및 영역 오버레이 (기울기 반영)"""
        out = frame.copy()
        if self.line_rect is None:
            return out

        x, y, w, h = self.line_rect
        x1, x2 = x, x + w
        outer_y1 = int(round(self.get_outer_y_at(x1)))
        outer_y2 = int(round(self.get_outer_y_at(x2)))
        inner_y1 = int(round(self.get_inner_y_at(x1)))
        inner_y2 = int(round(self.get_inner_y_at(x2)))

        # 라인 외곽선 (Outer Boundary) - 빨간색 강조선
        cv2.line(out, (x1, outer_y1), (x2, outer_y2), (0, 0, 255), 2)
        # 라인 내곽선 (Inner Boundary) - 노란색
        cv2.line(out, (x1, inner_y1), (x2, inner_y2), color, 1)

        # 라인 라벨
        cv2.putText(
            out,
            f"BASELINE (outer_y={self.outer_y})",
            (x1 + 10, outer_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return out
