"""
tracker.py
----------
영상 프레임 시퀀스에서 셔틀콕을 고속으로 검출하고,
비행 궤적(Trajectory)을 추적 및 평활화(Smoothing)하는 듀얼 모드 모듈
- 모드 1: YOLOv8 딥러닝 객체 검출 (Roboflow Shuttlecock / Ultralytics CUDA 가속)
- 모드 2: 초고속 HSV 색상 필터 및 윤곽선 분석 (Fallback 및 경량 CPU 환경)
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class ShuttlecockTracker:
    """
    셔틀콕 검출 및 궤적(Trajectory) 추적기 (YOLOv8 + OpenCV HSV 하이브리드 지원)
    """

    def __init__(
        self,
        tracker_type: str = "auto",
        model_path: Optional[str] = None,
        conf_threshold: float = 0.20,
        min_area: int = 15,
        max_area: int = 3000,
        max_jump_dist: float = 220.0,
    ):
        """
        Args:
            tracker_type: 'auto' | 'yolo' | 'hsv'
            model_path: YOLOv8 모델 가중치 파일 경로 (.pt). None이면 기본 경로 탐색
            conf_threshold: YOLO 객체 검출 신뢰도 임계값
            min_area: HSV 모드용 최소 면적 (픽셀)
            max_area: HSV 모드용 최대 면적 (픽셀)
            max_jump_dist: 프레임 간 최대 허용 이동 거리 (이상치 배제 필터)
        """
        self.tracker_type = tracker_type.lower()
        self.conf_threshold = conf_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_jump_dist = max_jump_dist

        # HSV 색상 파라미터 (흰색 셔틀콕)
        self.lower_white = np.array([0, 0, 175], dtype=np.uint8)
        self.upper_white = np.array([180, 65, 255], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 베이스라인과 셔틀콕이 겹칠 때 긴 수평 라인만 제거하기 위한 커널
        self.horiz_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))

        # 전체 궤적 기록 리스트
        self.trajectory: List[Dict[str, Any]] = []

        # YOLO 모델 초기화
        self.yolo_model = None
        self.yolo_classes: Optional[List[int]] = None
        self.device = "cpu"
        self.active_backend = "hsv"

        if self.tracker_type in ("auto", "yolo"):
            self._init_yolo(model_path)

        if self.tracker_type == "hsv" or self.yolo_model is None:
            self.active_backend = "hsv"

    def _init_yolo(self, model_path: Optional[str]):
        """YOLOv8 가중치 로드 및 GPU(CUDA) 가속 초기화"""
        try:
            from ultralytics import YOLO
            import torch

            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"

            # 1. 전달받은 경로 우선 확인
            candidate_paths = []
            if model_path:
                candidate_paths.append(model_path)

            # 2. engine/models 폴더 내 사전학습 가중치 탐색
            engine_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(engine_root, "models")
            candidate_paths.extend([
                os.path.join(models_dir, "shuttlecock_yolov8.pt"),
                os.path.join(models_dir, "best.pt"),
                os.path.join(models_dir, "yolov8n.pt"),
                "yolov8n.pt",  # Ultralytics 기본 나노 모델 (사전학습 가중치 부재 시 fallback)
            ])

            loaded_path = None
            for p in candidate_paths:
                if os.path.exists(p) or p == "yolov8n.pt":
                    loaded_path = p
                    break

            if loaded_path:
                print(f"[TRACKER] YOLOv8 로드 중: {loaded_path} (Device: {self.device})")
                self.yolo_model = YOLO(loaded_path)
                # 일반 COCO 80클래스 모델인 경우 sports ball(32) 클래스만 탐지하도록 필터링하여 선수/배경 오검출 방지
                names = getattr(self.yolo_model, "names", {}) or {}
                if isinstance(names, dict) and len(names) >= 80 and 32 in names:
                    self.yolo_classes = [32]
                else:
                    self.yolo_classes = None
                self.active_backend = "yolo"
                print(f"[TRACKER] YOLOv8 백엔드 활성화 완료 (가속: {self.device})")
            else:
                if self.tracker_type == "yolo":
                    print("[TRACKER] 경고: 지정된 YOLO 모델을 찾지 못했습니다. HSV 모드로 전환합니다.")
                self.active_backend = "hsv"

        except Exception as e:
            if self.tracker_type == "yolo":
                print(f"[TRACKER] YOLOv8 초기화 실패 ({e}). HSV 백엔드로 전환합니다.")
            self.yolo_model = None
            self.active_backend = "hsv"

    def reset(self):
        """추적 상태 초기화"""
        self.trajectory.clear()

    def _get_tracking_reference(self, frame_idx: int) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], int, float]:
        """
        직전 궤적 정보를 바탕으로 프레임 간격(frame_gap)과 속도 예측 좌표, 동적 허용 이동 반경을 반환합니다.
        """
        if not self.trajectory:
            return None, None, 1, self.max_jump_dist

        last_item = self.trajectory[-1]
        last_pos = last_item["center"]
        last_vel = last_item.get("velocity", (0.0, 0.0))
        frame_gap = max(1, int(frame_idx - last_item["frame_idx"]))

        # 등속 예측 위치 (최대 3프레임까지만 외삽)
        extrap_steps = min(frame_gap, 3)
        pred_pos = (
            last_pos[0] + last_vel[0] * extrap_steps,
            last_pos[1] + last_vel[1] * extrap_steps,
        )
        # 프레임 누락 시 허용 점프 반경을 완만하게 확장
        allowed_jump = self.max_jump_dist * (1.0 + 0.65 * min(frame_gap - 1, 5))
        return last_pos, pred_pos, frame_gap, allowed_jump

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: float = 0.0,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 프레임에서 셔틀콕을 탐지하고 궤적에 추가합니다.
        active_backend에 따라 YOLO 또는 HSV 검출을 수행합니다.
        """
        if self.active_backend == "yolo" and self.yolo_model is not None:
            detected = self._detect_yolo(frame, frame_idx, timestamp_ms, roi_rect)
            if detected is not None:
                return detected
            # YOLO 미검출 시 HSV 보조 검출
            return self._detect_hsv(frame, frame_idx, timestamp_ms, roi_rect)
        else:
            return self._detect_hsv(frame, frame_idx, timestamp_ms, roi_rect)

    def _detect_yolo(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: float,
        roi_rect: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Dict[str, Any]]:
        """YOLOv8 모델을 이용한 셔틀콕 객체 검출"""
        h_frame, w_frame = frame.shape[:2]

        if roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            rx = max(0, min(rx, w_frame - 1))
            ry = max(0, min(ry, h_frame - 1))
            rw = max(1, min(rw, w_frame - rx))
            rh = max(1, min(rh, h_frame - ry))
            crop = frame[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry
        else:
            crop = frame
            offset_x, offset_y = 0, 0

        # YOLO 추론 (verbose 비활성화)
        results = self.yolo_model.predict(
            crop,
            conf=self.conf_threshold,
            classes=self.yolo_classes,
            device=self.device,
            verbose=False,
        )

        candidates = []
        last_pos, pred_pos, frame_gap, allowed_jump = self._get_tracking_reference(frame_idx)

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                bw = int(x2 - x1)
                bh = int(y2 - y1)
                if bw * bh > self.max_area * 2.0:
                    continue

                cx = offset_x + int(x1 + bw / 2.0)
                cy = offset_y + int(y1 + bh / 2.0)
                orig_bbox = (offset_x + int(x1), offset_y + int(y1), bw, bh)

                dist = 0.0
                if last_pos is not None and pred_pos is not None:
                    raw_dist = float(np.hypot(cx - last_pos[0], cy - last_pos[1]))
                    pred_dist = float(np.hypot(cx - pred_pos[0], cy - pred_pos[1]))
                    dist = min(raw_dist, pred_dist)
                    if raw_dist > allowed_jump and pred_dist > allowed_jump:
                        continue

                candidates.append({
                    "center": (cx, cy),
                    "bbox": orig_bbox,
                    "confidence": conf,
                    "dist_from_last": dist,
                })

        if not candidates:
            return None

        # 이전 위치가 있으면 예측/최근방 후보, 없으면 최고 신뢰도 후보 선택
        if last_pos is not None:
            best = min(candidates, key=lambda c: c["dist_from_last"])
        else:
            best = max(candidates, key=lambda c: c["confidence"])

        vx, vy = 0.0, 0.0
        if last_pos is not None:
            vx = (best["center"][0] - last_pos[0]) / float(frame_gap)
            vy = (best["center"][1] - last_pos[1]) / float(frame_gap)

        detected_item = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "center": best["center"],
            "bbox": best["bbox"],
            "confidence": best.get("confidence", 0.9),
            "velocity": (round(vx, 2), round(vy, 2)),
            "backend": "yolo",
        }
        self.trajectory.append(detected_item)
        return detected_item

    def _extract_hsv_candidates(
        self,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        last_pos: Optional[Tuple[float, float]],
        pred_pos: Optional[Tuple[float, float]],
        allowed_jump: float,
    ) -> List[Dict[str, Any]]:
        """이진 마스크에서 셔틀콕 형태/면적/이동거리 조건을 만족하는 컨투어 후보 추출"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
                if 0.3 <= aspect_ratio <= 3.2:
                    cx = offset_x + bx + bw // 2
                    cy = offset_y + by + bh // 2
                    orig_bbox = (offset_x + bx, offset_y + by, bw, bh)

                    dist = 0.0
                    if last_pos is not None and pred_pos is not None:
                        raw_dist = float(np.hypot(cx - last_pos[0], cy - last_pos[1]))
                        pred_dist = float(np.hypot(cx - pred_pos[0], cy - pred_pos[1]))
                        dist = min(raw_dist, pred_dist)
                        if raw_dist > allowed_jump and pred_dist > allowed_jump:
                            continue

                    candidates.append({
                        "center": (cx, cy),
                        "bbox": orig_bbox,
                        "area": area,
                        "dist_from_last": dist,
                    })
        return candidates

    def _detect_hsv(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: float,
        roi_rect: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Dict[str, Any]]:
        """HSV 색상 및 윤곽선 분석 기반 고속 검출 (베이스라인 겹침 분리 지원)"""
        h_frame, w_frame = frame.shape[:2]

        if roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            rx = max(0, min(rx, w_frame - 1))
            ry = max(0, min(ry, h_frame - 1))
            rw = max(1, min(rw, w_frame - rx))
            rh = max(1, min(rh, h_frame - ry))
            crop = frame[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry
        else:
            crop = frame
            offset_x, offset_y = 0, 0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        # 긴 수평 흰색 코트 라인(베이스라인) 성분을 제거한 뒤,
        # 라인에 의해 상/하로 분할된 셔틀콕 윤곽선을 수직 클로징(MORPH_CLOSE)으로 다시 하나로 결합
        horiz_lines = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.horiz_line_kernel, iterations=1)
        clean_mask = cv2.subtract(raw_mask, horiz_lines)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        vert_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 19))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, vert_bridge, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_DILATE, self.kernel, iterations=1)

        last_pos, pred_pos, frame_gap, allowed_jump = self._get_tracking_reference(frame_idx)

        candidates = self._extract_hsv_candidates(
            clean_mask, offset_x, offset_y, last_pos, pred_pos, allowed_jump
        )
        if not candidates:
            # 보조: 원본 마스크로 재시도
            fallback_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_DILATE, self.kernel, iterations=1)
            candidates = self._extract_hsv_candidates(
                fallback_mask, offset_x, offset_y, last_pos, pred_pos, allowed_jump
            )

        if not candidates:
            return None

        if last_pos is not None:
            best = min(candidates, key=lambda c: c["dist_from_last"])
        else:
            best = max(candidates, key=lambda c: c["area"])

        vx, vy = 0.0, 0.0
        if last_pos is not None:
            vx = (best["center"][0] - last_pos[0]) / float(frame_gap)
            vy = (best["center"][1] - last_pos[1]) / float(frame_gap)

        detected_item = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "center": best["center"],
            "bbox": best["bbox"],
            "area": best["area"],
            "confidence": 0.92,
            "velocity": (round(vx, 2), round(vy, 2)),
            "backend": "hsv",
        }
        self.trajectory.append(detected_item)
        return detected_item

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """누적된 전체 궤적 반환"""
        return self.trajectory

    def draw_trajectory(
        self,
        frame: np.ndarray,
        max_points: int = 30,
        color: Tuple[int, int, int] = (0, 165, 255),
    ) -> np.ndarray:
        """프레임 위에 셔틀콕 궤적 선 및 현재 위치 렌더링"""
        out = frame.copy()
        pts = [item["center"] for item in self.trajectory[-max_points:]]

        for i in range(1, len(pts)):
            thickness = int(np.sqrt(max_points / float(i + 1)) * 2) + 1
            cv2.line(out, pts[i - 1], pts[i], color, thickness)

        if self.trajectory:
            latest = self.trajectory[-1]
            cx, cy = latest["center"]
            bx, by, bw, bh = latest["bbox"]
            box_color = (0, 255, 0) if latest.get("backend") == "yolo" else (255, 200, 0)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), box_color, 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        return out
