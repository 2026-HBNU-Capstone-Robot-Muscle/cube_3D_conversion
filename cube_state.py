"""6개 큐브 면을 조립하고 Lab 기반 세션 캘리브레이션으로 색상을 분류한다."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np


FACE_ORDER = ("F", "R", "B", "L", "U", "D")
FACE_NAMES = {
    "F": "Front",
    "R": "Right",
    "B": "Back",
    "L": "Left",
    "U": "Up",
    "D": "Down",
}


def _validate_face_name(face_name: str) -> str:
    normalized = face_name.upper()
    if normalized not in FACE_ORDER:
        raise ValueError(f"face_name must be one of {FACE_ORDER}: {face_name}")
    return normalized


def bgr_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    """BGR 색상 배열 ``(..., 3)``을 OpenCV Lab 배열로 변환한다."""
    colors = np.asarray(colors_bgr)
    if colors.ndim < 1 or colors.shape[-1] != 3:
        raise ValueError("colors_bgr must have shape (..., 3)")
    clipped = np.ascontiguousarray(np.clip(colors, 0, 255).astype(np.uint8))
    flattened = clipped.reshape(-1, 1, 3)
    return cv2.cvtColor(flattened, cv2.COLOR_BGR2LAB).reshape(colors.shape)


def rotate_face_colors(colors_bgr: np.ndarray, rotation_quarter_turns: int) -> np.ndarray:
    """9칸 색상을 표준 방향으로 맞추기 위해 시계방향으로 회전한다."""
    if not 0 <= rotation_quarter_turns <= 3:
        raise ValueError("rotation_quarter_turns must be between 0 and 3")
    colors = np.asarray(colors_bgr, dtype=np.uint8)
    if colors.shape != (9, 3):
        raise ValueError("colors_bgr must have shape (9, 3)")
    return np.rot90(colors.reshape(3, 3, 3), -rotation_quarter_turns).reshape(9, 3)


@dataclass(frozen=True)
class FaceCapture:
    """한 면의 원시 색상·영상·기하 메타데이터."""

    face_name: str
    cell_colors_bgr: np.ndarray
    rotation_quarter_turns: int = 0
    timestamp: str = ""
    confidence: Optional[float] = None
    original_frame: Optional[np.ndarray] = None
    warped_face: Optional[np.ndarray] = None
    obb_corners: Optional[np.ndarray] = None
    corrected_corners: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "face_name", _validate_face_name(self.face_name))
        colors = np.asarray(self.cell_colors_bgr)
        if colors.shape != (9, 3):
            raise ValueError("cell_colors_bgr must have shape (9, 3)")
        if not np.isfinite(colors).all():
            raise ValueError("cell_colors_bgr must contain finite values")
        object.__setattr__(self, "cell_colors_bgr", np.clip(colors, 0, 255).astype(np.uint8))
        if not 0 <= self.rotation_quarter_turns <= 3:
            raise ValueError("rotation_quarter_turns must be between 0 and 3")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )

    @property
    def canonical_colors_bgr(self) -> np.ndarray:
        """사용자가 지정한 회전을 적용한 표준 방향의 3×3 BGR 색상."""
        return rotate_face_colors(self.cell_colors_bgr, self.rotation_quarter_turns)

    @property
    def center_bgr(self) -> np.ndarray:
        """원본 회전과 무관한 가운데 스티커 BGR 값."""
        return self.cell_colors_bgr[4]

    @property
    def center_lab(self) -> np.ndarray:
        return bgr_to_lab(self.center_bgr)


@dataclass(frozen=True)
class StickerClassification:
    """스티커 하나의 가장 가까운 기준색과 애매함 측정값."""

    face_name: str
    sticker_index: int
    label: str
    distance: float
    second_label: str
    second_distance: float

    @property
    def margin(self) -> float:
        """1·2순위 Lab 거리 차이. 작을수록 분류가 애매하다."""
        return self.second_distance - self.distance


@dataclass(frozen=True)
class ValidationReport:
    """54칸 색상 상태의 자동 검증 결과. 오류를 억지로 수정하지 않는다."""

    color_counts: Mapping[str, int]
    ambiguous_stickers: tuple[StickerClassification, ...]
    close_center_pairs: tuple[tuple[str, str, float], ...]
    expected_count_per_color: int = 9

    @property
    def count_errors(self) -> Mapping[str, int]:
        return {
            face: count
            for face, count in self.color_counts.items()
            if count != self.expected_count_per_color
        }

    @property
    def is_valid(self) -> bool:
        return not self.count_errors and not self.close_center_pairs


@dataclass(frozen=True)
class CubeStateResult:
    """세션 기준색, 54개 분류 결과, 검증 보고서를 담은 완성 상태."""

    captures: Mapping[str, FaceCapture]
    reference_colors_lab: Mapping[str, np.ndarray]
    stickers: tuple[StickerClassification, ...]
    validation: ValidationReport

    @property
    def labels_by_face(self) -> Mapping[str, tuple[str, ...]]:
        return {
            face: tuple(
                sticker.label
                for sticker in self.stickers
                if sticker.face_name == face
            )
            for face in FACE_ORDER
        }


class CubeSession:
    """F→R→B→L→U→D 슬롯을 관리하는 한 번의 큐브 스캔 세션."""

    def __init__(self, face_order: tuple[str, ...] = FACE_ORDER) -> None:
        normalized_order = tuple(_validate_face_name(face) for face in face_order)
        if len(normalized_order) != 6 or len(set(normalized_order)) != 6:
            raise ValueError("face_order must contain each of the six faces exactly once")
        self.face_order = normalized_order
        self._captures: dict[str, FaceCapture] = {}

    @property
    def captures(self) -> Mapping[str, FaceCapture]:
        return dict(self._captures)

    @property
    def next_face_name(self) -> Optional[str]:
        return next((face for face in self.face_order if face not in self._captures), None)

    @property
    def is_complete(self) -> bool:
        return len(self._captures) == len(self.face_order)

    def add_capture(self, capture: FaceCapture, *, replace_existing: bool = False) -> tuple[str, ...]:
        """현재 차례의 면을 저장하고, 기존 중심색과의 중복 의심 경고를 반환한다."""
        expected = self.next_face_name
        if capture.face_name not in self._captures and capture.face_name != expected:
            raise ValueError(f"next required face is {expected}, not {capture.face_name}")
        if capture.face_name in self._captures and not replace_existing:
            raise ValueError(f"face {capture.face_name} is already captured")

        warnings = self._duplicate_center_warnings(capture)
        self._captures[capture.face_name] = capture
        return warnings

    def duplicate_center_warnings(self, capture: FaceCapture) -> tuple[str, ...]:
        """저장 전, 기존 슬롯의 중심색과 유사한지 확인한다."""
        return self._duplicate_center_warnings(capture)

    def remove_capture(self, face_name: str) -> FaceCapture:
        """재촬영을 위해 저장된 면 하나를 제거하고 원래 데이터를 반환한다."""
        normalized = _validate_face_name(face_name)
        if normalized not in self._captures:
            raise KeyError(f"face {normalized} is not captured")
        return self._captures.pop(normalized)

    def _duplicate_center_warnings(self, capture: FaceCapture, threshold: float = 16.0) -> tuple[str, ...]:
        candidate_lab = capture.center_lab.astype(np.float32)
        warnings: list[str] = []
        for face_name, previous in self._captures.items():
            distance = float(np.linalg.norm(candidate_lab - previous.center_lab.astype(np.float32)))
            if distance < threshold:
                warnings.append(
                    f"{face_name} 면의 중심색과 유사합니다 (Lab 거리 {distance:.1f})"
                )
        return tuple(warnings)

    def classify(
        self,
        *,
        ambiguous_margin_threshold: float = 8.0,
        min_center_separation: float = 16.0,
    ) -> CubeStateResult:
        """6개 중심색을 기준색으로 하여 전체 54칸을 Lab 거리로 분류·검증한다."""
        if not self.is_complete:
            missing = [face for face in self.face_order if face not in self._captures]
            raise RuntimeError(f"cannot classify before all faces are captured: missing {missing}")
        if ambiguous_margin_threshold < 0 or min_center_separation < 0:
            raise ValueError("distance thresholds must be non-negative")

        references = {
            face: self._captures[face].center_lab.astype(np.float32)
            for face in self.face_order
        }
        labels = tuple(self.face_order)
        reference_matrix = np.stack([references[label] for label in labels])
        stickers: list[StickerClassification] = []

        for face in self.face_order:
            colors_lab = bgr_to_lab(self._captures[face].canonical_colors_bgr).astype(np.float32)
            for index, color_lab in enumerate(colors_lab):
                distances = np.linalg.norm(reference_matrix - color_lab, axis=1)
                ranked = np.argsort(distances)
                best, second = int(ranked[0]), int(ranked[1])
                stickers.append(
                    StickerClassification(
                        face_name=face,
                        sticker_index=index,
                        label=labels[best],
                        distance=float(distances[best]),
                        second_label=labels[second],
                        second_distance=float(distances[second]),
                    )
                )

        counts = Counter(sticker.label for sticker in stickers)
        full_counts = {label: counts.get(label, 0) for label in labels}
        ambiguous = tuple(
            sticker
            for sticker in stickers
            if sticker.margin < ambiguous_margin_threshold
        )
        close_pairs: list[tuple[str, str, float]] = []
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1 :]:
                distance = float(np.linalg.norm(references[left_label] - references[right_label]))
                if distance < min_center_separation:
                    close_pairs.append((left_label, right_label, distance))

        return CubeStateResult(
            captures=self.captures,
            reference_colors_lab=references,
            stickers=tuple(stickers),
            validation=ValidationReport(
                color_counts=full_counts,
                ambiguous_stickers=ambiguous,
                close_center_pairs=tuple(close_pairs),
            ),
        )


class CaptureArchive:
    """원본 프레임과 추출 결과를 면별 파일·JSON 메타데이터로 보존한다."""

    def __init__(self, root_directory: str | Path) -> None:
        self.root_directory = Path(root_directory)

    def save(self, capture: FaceCapture) -> Path:
        """한 면의 사용 가능한 이미지와 원시 BGR/Lab 데이터를 저장한다."""
        face_directory = self.root_directory / capture.face_name
        face_directory.mkdir(parents=True, exist_ok=True)
        if capture.original_frame is not None:
            self._write_image(face_directory / "original.png", capture.original_frame)
        if capture.warped_face is not None:
            self._write_image(face_directory / "warped.png", capture.warped_face)

        metadata = {
            "face_name": capture.face_name,
            "face_display_name": FACE_NAMES[capture.face_name],
            "timestamp": capture.timestamp,
            "confidence": capture.confidence,
            "rotation_quarter_turns": capture.rotation_quarter_turns,
            "cell_colors_bgr": capture.cell_colors_bgr.tolist(),
            "cell_colors_lab": bgr_to_lab(capture.cell_colors_bgr).tolist(),
            "obb_corners": self._array_or_none(capture.obb_corners),
            "corrected_corners": self._array_or_none(capture.corrected_corners),
        }
        metadata_path = face_directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata_path

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to save image: {path}")

    @staticmethod
    def _array_or_none(value: Optional[np.ndarray]) -> Optional[list[object]]:
        return None if value is None else np.asarray(value).tolist()


__all__ = [
    "CaptureArchive",
    "CubeSession",
    "CubeStateResult",
    "FACE_NAMES",
    "FACE_ORDER",
    "FaceCapture",
    "StickerClassification",
    "ValidationReport",
    "bgr_to_lab",
    "rotate_face_colors",
]
