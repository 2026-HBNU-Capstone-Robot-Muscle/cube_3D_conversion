"""큐브 면 탐지와 캡처 트리거를 담당하는 모듈.

이 모듈은 카메라 장치, 색상 추출, 6면 캡처 순서를 알지 않는다. ``main.py``가
프레임을 전달하고, 이 모듈이 반환하는 ``FrameObservation``을 사용한다.

기본 트리거는 정지한 면을 한 번 캡처한 뒤 잠그며, 면이 사라지거나 충분히
이동해야 잠금이 해제된다. 로봇 신호 기반 트리거는 ``FaceTracker`` 변경 없이
``StationaryCaptureTrigger``를 교체하여 사용할 수 있다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Deque, Optional, Protocol

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # 의존성 설치 전에도 모듈 자체는 불러올 수 있게 한다.
    YOLO = None  # type: ignore[assignment, misc]


MODEL_FILENAME = "cube-detector.pt"


def find_default_model_path() -> Path:
    """기본 ``cube-detector.pt`` 위치를 찾는다.

    배포할 때는 이 파일과 같은 폴더에 모델을 두는 방식을 우선한다. 현재
    프로토타입 환경에서는 GitHub에서 받은 파일이 Downloads에 있을 수 있으므로
    그 위치를 보조 경로로 확인한다.
    """
    candidates = (
        Path(__file__).resolve().with_name(MODEL_FILENAME),
        Path.cwd() / MODEL_FILENAME,
        Path.home() / "Downloads" / MODEL_FILENAME,
    )
    for candidate in candidates:
        # 정상 YOLO 가중치(약 5.6MB)만 선택해 실수로 만든 빈 파일을 제외한다.
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"{MODEL_FILENAME}을 찾을 수 없습니다. 확인한 위치:\n{searched}"
    )


@dataclass(frozen=True)
class FaceDetection:
    """한 프레임에서 탐지된 최고 신뢰도 큐브 면 OBB.

    ``corners``는 모델이 출력한 순환 꼭짓점 순서의 ``float32`` 배열이며 모양은
    ``(4, 2)``이다. 시작점이 좌상단이라고 가정하면 안 되므로, ``color_extract``가
    원근 변환 전에 꼭짓점을 정렬해야 한다.
    """

    corners: np.ndarray
    center: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class TriggerState:
    """현재 프레임의 캡처 판단 결과."""

    should_capture: bool
    locked: bool
    stable_frames: int
    message: str


@dataclass(frozen=True)
class CaptureQualityResult:
    """자동 캡처를 허용할지 판단한 품질 측정 결과."""

    accepted: bool
    reasons: tuple[str, ...]
    polygon_area_px: float
    aspect_ratio: float
    sharpness: float


@dataclass(frozen=True)
class FrameObservation:
    """카메라 프레임 하나를 처리한 결과."""

    detection: Optional[FaceDetection]
    trigger: TriggerState
    quality: Optional[CaptureQualityResult]


class CaptureTrigger(Protocol):
    """교체 가능한 캡처 트리거 인터페이스.

    이후 로봇 연동 시 이 인터페이스를 구현하여 로봇이 정지 신호를 보낼 때
    ``should_capture=True``를 반환하면 된다.
    """

    def update(self, detection: Optional[FaceDetection]) -> TriggerState:
        """탐지 결과 하나를 받아 이번 프레임의 트리거 결과를 반환한다."""

    def reset(self) -> None:
        """새 스캔 세션 시작 등에서 내부 상태를 초기화한다."""


class StationaryCaptureTrigger:
    """면이 공간적으로 안정된 상태일 때 한 번만 캡처를 발생시킨다.

    인접 프레임끼리만 비교하지 않고, 최근 OBB 중심과 평균 중심의 최대 거리를
    사용한다. 따라서 천천히 계속 이동하는 면을 정지 상태로 오판하지 않는다.
    """

    def __init__(
        self,
        stable_frame_count: int = 12,
        stability_threshold_px: float = 5.0,
        hold_seconds: float = 0.6,
        unlock_motion_threshold_px: float = 35.0,
        missing_frames_to_unlock: int = 5,
    ) -> None:
        if stable_frame_count < 2:
            raise ValueError("stable_frame_count must be at least 2")
        if stability_threshold_px <= 0 or unlock_motion_threshold_px <= 0:
            raise ValueError("motion thresholds must be positive")
        if hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive")
        if missing_frames_to_unlock < 1:
            raise ValueError("missing_frames_to_unlock must be at least 1")

        self.stable_frame_count = stable_frame_count
        self.stability_threshold_px = stability_threshold_px
        self.hold_seconds = hold_seconds
        self.unlock_motion_threshold_px = unlock_motion_threshold_px
        self.missing_frames_to_unlock = missing_frames_to_unlock
        self._centers: Deque[tuple[float, np.ndarray]] = deque(maxlen=stable_frame_count)
        self._locked = False
        self._lock_center: Optional[np.ndarray] = None
        self._missing_frames = 0

    def reset(self) -> None:
        self._centers.clear()
        self._locked = False
        self._lock_center = None
        self._missing_frames = 0

    def update(self, detection: Optional[FaceDetection]) -> TriggerState:
        if detection is None:
            self._centers.clear()
            if self._locked:
                self._missing_frames += 1
                if self._missing_frames >= self.missing_frames_to_unlock:
                    self._unlock()
                    return self._state("면이 사라졌습니다. 다음 면을 인식할 준비가 됐습니다.")
                return self._state("캡처 잠금 상태: 면이 화면에서 사라지기를 기다리는 중")
            return self._state("큐브 면이 탐지되지 않았습니다.")

        center = np.asarray(detection.center, dtype=np.float32)
        now = time.monotonic()
        self._missing_frames = 0

        if self._locked:
            assert self._lock_center is not None
            moved = float(np.linalg.norm(center - self._lock_center))
            if moved >= self.unlock_motion_threshold_px:
                self._unlock()
                # 이동한 면을 새 정지 판정 구간의 첫 프레임으로 사용한다.
                self._centers.append((now, center))
                return self._state("면이 이동했습니다. 다음 면을 인식할 준비가 됐습니다.")
            return self._state("캡처 잠금 상태: 큐브 면을 이동하거나 화면에서 숨겨주세요.")

        self._centers.append((now, center))
        if len(self._centers) < self.stable_frame_count:
            return self._state(
                f"정지 여부 확인 중 ({len(self._centers)}/{self.stable_frame_count})"
            )

        timestamps, centers = zip(*self._centers)
        points = np.stack(centers)
        mean = points.mean(axis=0)
        max_deviation = float(np.linalg.norm(points - mean, axis=1).max())
        if max_deviation > self.stability_threshold_px:
            # 가장 최근 좌표부터 새 정지 판정 구간을 시작한다.
            self._centers.clear()
            self._centers.append((now, center))
            return self._state("면이 움직이고 있습니다.")

        held_seconds = now - timestamps[0]
        if held_seconds < self.hold_seconds:
            return self._state(
                f"정지 시간 확인 중 ({held_seconds:.1f}/{self.hold_seconds:.1f}초)"
            )

        if max_deviation <= self.stability_threshold_px:
            self._locked = True
            self._lock_center = center.copy()
            return TriggerState(
                should_capture=True,
                locked=True,
                stable_frames=len(self._centers),
                message="면이 정지했습니다. 지금 캡처하세요.",
            )

    def _unlock(self) -> None:
        self._locked = False
        self._lock_center = None
        self._missing_frames = 0
        self._centers.clear()

    def _state(self, message: str) -> TriggerState:
        return TriggerState(
            should_capture=False,
            locked=self._locked,
            stable_frames=len(self._centers),
            message=message,
        )


class CaptureQualityGate:
    """정지한 면이 자동 캡처에 충분한 품질인지 검사한다.

    OBB 종횡비는 실제 원근 기울기를 정밀하게 측정하는 값이 아니라, 너무 옆으로
    기울어진 면을 거르는 프로토타입용 정면성 추정값이다.
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        min_area_px: float = 12_000.0,
        corner_margin_px: int = 4,
        min_aspect_ratio: float = 0.55,
        min_laplacian_variance: float = 80.0,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if min_area_px <= 0 or corner_margin_px < 0:
            raise ValueError("area and corner margin must be non-negative")
        if not 0.0 < min_aspect_ratio <= 1.0:
            raise ValueError("min_aspect_ratio must be between 0 and 1")
        if min_laplacian_variance < 0:
            raise ValueError("min_laplacian_variance must be non-negative")

        self.min_confidence = min_confidence
        self.min_area_px = min_area_px
        self.corner_margin_px = corner_margin_px
        self.min_aspect_ratio = min_aspect_ratio
        self.min_laplacian_variance = min_laplacian_variance

    def evaluate(self, frame: np.ndarray, detection: FaceDetection) -> CaptureQualityResult:
        """프레임과 OBB를 검사해 캡처 허용 여부와 측정값을 반환한다."""
        height, width = frame.shape[:2]
        corners = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
        reasons: list[str] = []

        polygon_area = float(abs(cv2.contourArea(corners)))
        edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        shortest_edge = float(edges.min())
        longest_edge = float(edges.max())
        aspect_ratio = shortest_edge / longest_edge if longest_edge > 0 else 0.0

        if detection.confidence < self.min_confidence:
            reasons.append("탐지 신뢰도가 낮습니다")
        if polygon_area < self.min_area_px:
            reasons.append("면이 너무 작습니다")
        margin = self.corner_margin_px
        if (
            (corners[:, 0] < margin).any()
            or (corners[:, 0] > width - 1 - margin).any()
            or (corners[:, 1] < margin).any()
            or (corners[:, 1] > height - 1 - margin).any()
        ):
            reasons.append("면의 꼭짓점이 화면 가장자리에 닿았습니다")
        if aspect_ratio < self.min_aspect_ratio:
            reasons.append("면이 너무 기울어졌습니다")

        left = max(0, int(np.floor(corners[:, 0].min())))
        right = min(width, int(np.ceil(corners[:, 0].max())))
        top = max(0, int(np.floor(corners[:, 1].min())))
        bottom = min(height, int(np.ceil(corners[:, 1].max())))
        roi = frame[top:bottom, left:right]
        sharpness = 0.0
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_laplacian_variance:
            reasons.append("영상이 흐립니다")

        return CaptureQualityResult(
            accepted=not reasons,
            reasons=tuple(reasons),
            polygon_area_px=polygon_area,
            aspect_ratio=aspect_ratio,
            sharpness=sharpness,
        )


class FaceTracker:
    """YOLO OBB 추론을 수행하고 캡처 판단을 트리거에 위임한다.

    ``model_path``에는 cude-detector 저장소에서 받은 ``cube-detector.pt``의
    로컬 경로를 전달한다.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        trigger: Optional[CaptureTrigger] = None,
        quality_gate: Optional[CaptureQualityGate] = None,
        confidence_threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is required. Install it with: pip install ultralytics"
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.model_path = Path(model_path) if model_path is not None else find_default_model_path()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.trigger: CaptureTrigger = trigger or StationaryCaptureTrigger()
        self.quality_gate = quality_gate or CaptureQualityGate(
            min_confidence=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
        self.device = device

    def process(
        self,
        frame: np.ndarray,
        *,
        update_trigger: bool = True,
    ) -> FrameObservation:
        """OpenCV BGR 프레임에서 면을 탐지하고 필요할 때만 트리거를 갱신한다.

        ``update_trigger=False``는 사용자가 키를 눌러 촬영하는 수동 모드용이다.
        이 경우에도 YOLO 탐지와 품질 검사는 매 프레임 수행하지만, 정지 판정 이력과
        자동 캡처 잠금 상태는 변경하지 않는다.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image")

        prediction = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )[0]
        detection = self._best_detection(prediction)
        quality = (
            self.quality_gate.evaluate(frame, detection) if detection is not None else None
        )

        # 품질 미달 상태는 정지 판정 이력에 넣지 않아, 저품질 면이 잠금 상태를
        # 만들거나 자동 캡처되는 일을 방지한다.
        if update_trigger:
            trigger_input = detection if quality is None or quality.accepted else None
            trigger = self.trigger.update(trigger_input)
            if quality is not None and not quality.accepted:
                trigger = replace(
                    trigger,
                    message="캡처 대기: " + ", ".join(quality.reasons),
                )
        else:
            trigger = TriggerState(
                should_capture=False,
                locked=False,
                stable_frames=0,
                message="수동 촬영 대기: Space 키를 누르세요.",
            )
        return FrameObservation(
            detection=detection,
            trigger=trigger,
            quality=quality,
        )

    @staticmethod
    def _best_detection(prediction: object) -> Optional[FaceDetection]:
        obb = getattr(prediction, "obb", None)
        if obb is None or len(obb) == 0:
            return None

        confidences = obb.conf.cpu().numpy()
        index = int(np.argmax(confidences))
        corners = obb.xyxyxyxy[index].cpu().numpy().astype(np.float32)
        center_xy = corners.mean(axis=0)
        return FaceDetection(
            corners=corners,
            center=(float(center_xy[0]), float(center_xy[1])),
            confidence=float(confidences[index]),
        )

    def reset(self) -> None:
        """새 6면 스캔 세션을 위해 캡처 트리거 상태를 초기화한다."""
        self.trigger.reset()


__all__ = [
    "CaptureTrigger",
    "CaptureQualityGate",
    "CaptureQualityResult",
    "FaceDetection",
    "FaceTracker",
    "FrameObservation",
    "MODEL_FILENAME",
    "StationaryCaptureTrigger",
    "TriggerState",
    "find_default_model_path",
]
