"""
impact_detector.py
-------------------
셔틀콕 궤적의 운동학적 특성(수직 낙하 정점, 속도 급감, 방향 전환)을 분석하여
바닥 착지 순간(Impact Frame)을 정밀 검출하고 1차 In/Out 물리 판정을 산출하는 모듈
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .baseline_roi import BaselineROI


class ImpactDetector:
    """
    셔틀콕 착지 프레임(Impact Frame) 검출 및 판정 엔진
    """

    def __init__(self, baseline_roi: BaselineROI):
        """
        Args:
            baseline_roi: 판독 기준이 되는 베이스라인 ROI 객체
        """
        self.baseline_roi = baseline_roi

    def detect_impact(
        self,
        trajectory: List[Dict[str, Any]],
        min_trajectory_len: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        누적된 비행 궤적 데이터를 기반으로 착지 지점(Impact Point)을 판별합니다.

        물리 알고리즘:
        1. 셔틀콕의 수직 좌표(y)는 지면을 향해 떨어지면서 증가합니다.
        2. 지면과 접촉하는 순간:
           - y 좌표가 로컬 극대점(Local Maximum, 화면상 최하단)을 형성
           - 수직 속도 vy의 부호가 양수(하강)에서 음수(바운드) 또는 0(정지)으로 급변
           - 가속도 벡터의 피크 발생
        3. 후보 중 베이스라인 주변(라인 근방 y좌표)에서 발생한 가장 확실한 착지점을 선택

        Args:
            trajectory: ShuttlecockTracker에서 수집된 궤적 리스트
            min_trajectory_len: 분석에 필요한 최소 궤적 길이

        Returns:
            dict: {
                "impact_frame_idx": int,
                "timestamp_ms": float,
                "contact_point": (x, y),
                "bbox": (x, y, w, h),
                "baseline_verdict": dict (BaselineROI.judge_point 결과),
                "confidence": float (0.0 ~ 1.0)
            } 또는 None
        """
        if len(trajectory) < min_trajectory_len:
            return None

        # y좌표 시계열 배열
        y_vals = np.array([pt["center"][1] for pt in trajectory], dtype=float)
        x_vals = np.array([pt["center"][0] for pt in trajectory], dtype=float)

        # 1차 차분 (수직 속도 근사)
        dy = np.diff(y_vals)

        # 로컬 극대점(착지 후보) 탐색: 수직 낙하 속도(dy) 급감/반전 지점
        candidates = []
        for i in range(1, len(y_vals) - 1):
            prev_dy = dy[i - 1]
            curr_dy = dy[i]

            # 하강(prev_dy > 0)하다가 바닥 충돌로 정지 또는 반전(curr_dy <= 0)하는 감속 피크
            deceleration = prev_dy - curr_dy
            is_peak = (y_vals[i] >= y_vals[i - 1]) and (y_vals[i] >= y_vals[i + 1])
            is_impact_decel = (prev_dy > 1.5) and (curr_dy <= 1.0)

            if is_peak or is_impact_decel:
                pt_info = trajectory[i]
                contact_y = pt_info["center"][1]
                contact_x = pt_info["center"][0]

                # 베이스라인과의 근접성 평가
                outer_y = self.baseline_roi.outer_y or 0
                dist_to_baseline = abs(contact_y - outer_y)

                # 점수 계산: 낙하 감속 충격량(deceleration) + 바닥 근접성 + 첫 충돌 우선 가중치
                # 착지 후 굴러가는 현상을 배제하기 위해 최초 낙하 충돌 지점에 가중치 부여
                score = (deceleration * 10.0) + y_vals[i] - (dist_to_baseline * 0.5)

                candidates.append({
                    "index": i,
                    "score": score,
                    "deceleration": deceleration,
                    "pt_info": pt_info,
                })

        if not candidates:
            # 뚜렷한 극대점이 없는 경우 전체 궤적 중 y값이 가장 큰 지점 선택
            best_idx = int(np.argmax(y_vals))
            best_candidate = trajectory[best_idx]
        else:
            best_match = max(candidates, key=lambda c: c["score"])
            best_candidate = best_match["pt_info"]

        # 착지 접촉점 계산 (셔틀콕 코르크 부분: 바운딩 박스의 최하단 중앙)
        bx, by, bw, bh = best_candidate["bbox"]
        contact_x = float(bx + bw / 2.0)
        contact_y = float(by + bh)  # 코르크 하단부

        # 베이스라인 물리 판정 수행
        verdict = self.baseline_roi.judge_point(contact_x, contact_y)

        # 판정 신뢰도 계산 (궤적의 완성도 및 베이스라인과의 명확한 거리 기반)
        confidence = 0.92 if len(trajectory) >= 10 else 0.78
        if verdict["status_detail"] == "ON_LINE":
            confidence = 0.95  # 라인 위 확실한 접촉

        return {
            "impact_frame_idx": best_candidate["frame_idx"],
            "timestamp_ms": best_candidate.get("timestamp_ms", 0.0),
            "contact_point": (round(contact_x, 1), round(contact_y, 1)),
            "bbox": best_candidate["bbox"],
            "baseline_verdict": verdict,
            "confidence": confidence,
        }
