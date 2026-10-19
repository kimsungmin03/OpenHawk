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

        # 시계열 좌표 및 프레임 인덱스 배열
        y_vals = np.array([pt["center"][1] for pt in trajectory], dtype=float)
        x_vals = np.array([pt["center"][0] for pt in trajectory], dtype=float)
        f_idxs = np.array([pt["frame_idx"] for pt in trajectory], dtype=float)

        # 프레임 간격(dt)으로 정규화한 1차 차분 (수직 속도 vy: px/frame)
        dt = np.maximum(1.0, np.diff(f_idxs))
        vy = np.diff(y_vals) / dt

        # 로컬 극대점(착지 후보) 탐색: 수직 낙하 속도(vy) 급감/반전 지점
        candidates = []
        for i in range(1, len(y_vals) - 1):
            prev_vy = vy[i - 1]
            curr_vy = vy[i]

            # 하강(prev_vy > 0)하다가 바닥 충돌로 급감속(deceleration >= 3.0) 또는 반전하는 지점
            deceleration = prev_vy - curr_vy
            is_peak = (y_vals[i] >= y_vals[i - 1]) and (y_vals[i] >= y_vals[i + 1]) and (prev_vy > 0)
            is_impact_decel = (prev_vy > 1.5) and (curr_vy <= 1.5) and (deceleration >= 2.5)

            if is_peak or is_impact_decel:
                pt_info = trajectory[i]
                contact_x = float(pt_info["center"][0])
                contact_y = float(pt_info["center"][1])

                # 기울기가 반영된 해당 x좌표의 베이스라인 외곽선 y좌표
                local_outer_y = max(1.0, self.baseline_roi.get_outer_y_at(contact_x))
                dist_to_baseline_norm = abs(contact_y - local_outer_y) / local_outer_y
                y_norm = contact_y / local_outer_y

                # 해상도 독립적 점수 계산:
                # 1) 감속 충격량(deceleration)
                # 2) 베이스라인 근접성 및 하단 도달 비율(y_norm)
                # 3) 착지 후 미끄러짐(sliding) 프레임보다 '최초 충돌 프레임'을 우선시하는 시간 가중치
                order_penalty = (i / float(len(y_vals))) * 2.5
                score = (deceleration * 3.5) + (y_norm * 10.0) - (dist_to_baseline_norm * 8.0) - order_penalty

                candidates.append({
                    "index": i,
                    "score": score,
                    "deceleration": float(max(0.0, deceleration)),
                    "pt_info": pt_info,
                })

        if not candidates:
            # 뚜렷한 극대점이 없는 경우 전체 궤적 중 y값이 가장 큰 최초 지점 선택
            best_idx = int(np.argmax(y_vals))
            best_candidate = trajectory[best_idx]
            best_decel = 1.0
        else:
            best_match = max(candidates, key=lambda c: c["score"])
            best_candidate = best_match["pt_info"]
            best_decel = best_match["deceleration"]

        # 착지 접촉점 계산:
        # 셔틀콕 중심(cx, cy)을 기준 접촉점으로 전달하고, 코르크 반경 보정은 BaselineROI.judge_point에서 수행
        cx, cy = best_candidate["center"]
        contact_x = float(cx)
        contact_y = float(cy)

        # 베이스라인 물리 판정 수행
        verdict = self.baseline_roi.judge_point(contact_x, contact_y)

        # 동적 판정 신뢰도 계산 (궤적 밀도 + 감속 명확도 + 검출기 신뢰도 결합)
        traj_score = min(1.0, len(trajectory) / 25.0)
        decel_score = min(1.0, best_decel / 6.0)
        raw_det_conf = float(best_candidate.get("confidence", 0.88))
        det_conf = max(0.82, raw_det_conf) if traj_score >= 0.8 else raw_det_conf
        raw_conf = 0.45 * det_conf + 0.30 * traj_score + 0.25 * decel_score
        confidence = round(float(np.clip(raw_conf, 0.70, 0.99)), 2)

        return {
            "impact_frame_idx": best_candidate["frame_idx"],
            "timestamp_ms": best_candidate.get("timestamp_ms", 0.0),
            "contact_point": (round(contact_x, 1), round(contact_y, 1)),
            "bbox": best_candidate["bbox"],
            "baseline_verdict": verdict,
            "confidence": confidence,
        }
