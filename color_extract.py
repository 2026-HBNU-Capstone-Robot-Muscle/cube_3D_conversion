"""탐지된 큐브 면에서 3×3 스티커의 대표 BGR 색상을 추출하는 모듈.

이 모듈은 색상 이름을 분류하지 않는다. 조명 환경별 절대 BGR 값 대신,
``cube_state.py``가 이번 세션에서 캡처한 6개 중심 스티커를 기준색으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WARP_SIZE = 300
GRID_SIZE = 3
# 각 스티커 칸의 테두리 14%를 제외하고, 가운데 72%만 색상 샘플에 사용한다.
SAMPLE_INNER_RATIO = 0.72


@dataclass(frozen=True)
class FaceColorSamples:
    """원근 보정된 면과, 행 우선 순서로 추출한 9개 BGR 색상."""

    warped_face: np.ndarray
    cell_colors_bgr: np.ndarray
    cell_regions: tuple[tuple[int, int, int, int], ...]

    @property
    def center_bgr(self) -> np.ndarray:
        """세션 색상 캘리브레이션에 사용할 가운데 스티커의 BGR 색상."""
        return self.cell_colors_bgr[4]


def order_corners(corners: np.ndarray) -> np.ndarray:
    """OBB 꼭짓점의 시작점은 보존하고, 순환 방향만 이미지 좌표계 기준으로 통일한다.

    Ultralytics OBB의 ``xyxyxyxy``는 회전 사각형의 꼭짓점을 순환 순서로 반환한다.
    이때 좌상단을 강제로 시작점으로 고르면, 면이 카메라 앞에서 회전할 때 3×3
    결과가 90도 단위로 뒤집힐 수 있다. 따라서 입력 첫 꼭짓점은 그대로 유지한다.

    단, 라이브러리 버전이나 후처리 과정에서 순환 방향이 반대가 될 가능성에 대비해
    필요한 경우에만 첫 점을 유지한 채 나머지 순서를 뒤집는다.
    """
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("corners must contain only finite coordinates")

    # 이미지 좌표계(y가 아래로 증가)에서 양수 부호는 시계방향 순서다.
    signed_double_area = float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1))
        - np.sum(points[:, 1] * np.roll(points[:, 0], -1))
    )
    if abs(signed_double_area) < 1e-3:
        raise ValueError("corners must form a non-degenerate quadrilateral")
    if signed_double_area < 0:
        # points[0]은 보존하고, 이후 꼭짓점의 방향만 반전한다.
        points = np.concatenate((points[:1], points[:0:-1]), axis=0)
    return points


def warp_face(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int = WARP_SIZE,
) -> np.ndarray:
    """OBB의 사각형 영역을 정면에서 본 정사각형 BGR 이미지로 보정한다."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty BGR image")
    if output_size < 90 or output_size % GRID_SIZE != 0:
        raise ValueError("output_size must be at least 90 and divisible by 3")

    source = order_corners(corners)
    destination = np.array(
        [
            # [0, 0],
            # [output_size - 1, 0],
            # [output_size - 1, output_size - 1],
            # [0, output_size - 1],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
            [0, 0],
            [output_size - 1, 0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, transform, (output_size, output_size))


def split_grid_and_sample(
    warped_face: np.ndarray,
    inner_ratio: float = SAMPLE_INNER_RATIO,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """정사각형 면을 3×3으로 나누고 각 칸 중앙부의 중앙값 BGR을 반환한다.

    스티커 사이의 검은 틈, 격자선, 가장자리 그림자의 영향을 줄이기 위해 각
    칸 전체가 아니라 중앙 ``inner_ratio`` 영역만 사용한다. 평균보다 중앙값을
    사용하므로 손가락 반사광이나 작은 노이즈에 덜 민감하다.
    """
    if warped_face is None or warped_face.size == 0:
        raise ValueError("warped_face must be a non-empty BGR image")
    if warped_face.ndim != 3 or warped_face.shape[2] != 3:
        raise ValueError("warped_face must have shape (height, width, 3)")
    if not 0.2 <= inner_ratio <= 1.0:
        raise ValueError("inner_ratio must be between 0.2 and 1.0")

    height, width = warped_face.shape[:2]
    cell_width = width / GRID_SIZE
    cell_height = height / GRID_SIZE
    colors: list[np.ndarray] = []
    regions: list[tuple[int, int, int, int]] = []

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            cell_center_x = (column + 0.5) * cell_width
            cell_center_y = (row + 0.5) * cell_height
            sample_width = cell_width * inner_ratio
            sample_height = cell_height * inner_ratio

            left = max(0, int(round(cell_center_x - sample_width / 2)))
            top = max(0, int(round(cell_center_y - sample_height / 2)))
            right = min(width, int(round(cell_center_x + sample_width / 2)))
            bottom = min(height, int(round(cell_center_y + sample_height / 2)))
            patch = warped_face[top:bottom, left:right]
            if patch.size == 0:
                raise RuntimeError("sticker sampling region is empty")

            colors.append(np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8))
            regions.append((left, top, right, bottom))

    return np.stack(colors), tuple(regions)


def extract_face_colors(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int = WARP_SIZE,
    inner_ratio: float = SAMPLE_INNER_RATIO,
) -> FaceColorSamples:
    """카메라 프레임과 OBB 꼭짓점에서 원근 보정·3×3 색상 추출을 한 번에 수행한다."""
    warped_face = warp_face(frame, corners, output_size=output_size)
    colors, regions = split_grid_and_sample(warped_face, inner_ratio=inner_ratio)
    return FaceColorSamples(
        warped_face=warped_face,
        cell_colors_bgr=colors,
        cell_regions=regions,
    )


def draw_sampling_grid(
    warped_face: np.ndarray,
    cell_regions: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """추출 영역을 확인하기 위한 디버그 이미지를 생성한다."""
    preview = warped_face.copy()
    height, width = preview.shape[:2]
    for position in (width // GRID_SIZE, (width * 2) // GRID_SIZE):
        cv2.line(preview, (position, 0), (position, height), (255, 255, 255), 1)
    for position in (height // GRID_SIZE, (height * 2) // GRID_SIZE):
        cv2.line(preview, (0, position), (width, position), (255, 255, 255), 1)
    for left, top, right, bottom in cell_regions:
        cv2.rectangle(preview, (left, top), (right - 1, bottom - 1), (0, 255, 0), 1)
    return preview


__all__ = [
    "FaceColorSamples",
    "GRID_SIZE",
    "SAMPLE_INNER_RATIO",
    "WARP_SIZE",
    "draw_sampling_grid",
    "extract_face_colors",
    "order_corners",
    "split_grid_and_sample",
    "warp_face",
]
