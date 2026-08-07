"""OpenCV 기반 내장 카메라와 IP 스트림의 공통 입력 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional, Protocol, TypeAlias

import cv2
import numpy as np


CameraAddress: TypeAlias = int | str


class CameraError(RuntimeError):
    """카메라 열기 또는 프레임 수신 실패의 기반 예외."""


class CameraOpenError(CameraError):
    """카메라 소스를 열 수 없을 때 발생한다."""


class VideoCaptureLike(Protocol):
    """테스트용 가짜 캡처 객체도 사용할 수 있게 정의한 최소 OpenCV 인터페이스."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...

    def set(self, property_id: int, value: float) -> bool: ...


@dataclass(frozen=True)
class CameraFrame:
    """카메라에서 정상 수신한 BGR 프레임과 수신 메타데이터."""

    image: np.ndarray
    timestamp: float
    index: int


class CameraSource(Protocol):
    """main.py가 카메라 종류와 관계없이 사용하는 공통 인터페이스."""

    @property
    def is_open(self) -> bool: ...

    @property
    def description(self) -> str: ...

    def open(self) -> None: ...

    def read(self) -> Optional[CameraFrame]: ...

    def release(self) -> None: ...


def parse_camera_address(source: CameraAddress) -> CameraAddress:
    """문자열 카메라 번호는 정수로, IP 스트림 URL은 문자열로 반환한다."""
    if isinstance(source, int):
        if source < 0:
            raise ValueError("camera index must be non-negative")
        return source
    if not isinstance(source, str):
        raise TypeError("camera source must be an int or str")
    value = source.strip()
    if not value:
        raise ValueError("camera source must not be empty")
    return int(value) if value.isdecimal() else value


def _opencv_capture_factory(
    source: CameraAddress, backend: Optional[int]
) -> VideoCaptureLike:
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


CaptureFactory: TypeAlias = Callable[[CameraAddress, Optional[int]], VideoCaptureLike]


class OpenCVCameraSource:
    """OpenCV ``VideoCapture``를 감싼 카메라 소스.

    ``source``에는 ``0`` 같은 내장 카메라 인덱스 또는 휴대폰 앱이 제공하는
    ``http://...`` / ``rtsp://...`` 스트림 URL을 넣는다.
    """

    def __init__(
        self,
        # 시스템 내장 카메라
        # source: CameraAddress = 0,
        # 외부 카메라
        source: CameraAddress = 1,
        *,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[float] = 30.0,
        backend: Optional[int] = None,
        reconnect_attempts: int = 2,
        reconnect_interval_seconds: float = 0.25,
        capture_factory: CaptureFactory = _opencv_capture_factory,
    ) -> None:
        if width is not None and width <= 0:
            raise ValueError("width must be positive or None")
        if height is not None and height <= 0:
            raise ValueError("height must be positive or None")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive or None")
        if reconnect_attempts < 0 or reconnect_interval_seconds < 0:
            raise ValueError("reconnect settings must be non-negative")

        self.source = parse_camera_address(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_interval_seconds = reconnect_interval_seconds
        self._capture_factory = capture_factory
        self._capture: Optional[VideoCaptureLike] = None
        self._frame_index = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def description(self) -> str:
        if isinstance(self.source, int):
            return f"내장 카메라 {self.source}"
        return self.source

    def open(self) -> None:
        """카메라 또는 스트림에 연결하고 가능한 범위에서 해상도/FPS를 요청한다."""
        self.release()
        capture = self._capture_factory(self.source, self.backend)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(f"카메라를 열 수 없습니다: {self.description}")

        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        # 스트림은 요청값을 무시할 수 있으므로 실제 값 검증은 main.py에서 하지 않는다.
        self._capture = capture

    def read(self) -> Optional[CameraFrame]:
        """프레임 하나를 읽는다. 일시적 연결 실패는 재연결 후 다시 시도한다.

        모든 재시도가 실패하면 ``None``을 반환한다. main.py는 이때 사용자에게
        연결 상태를 표시하고 다음 루프에서 재시도하면 된다.
        """
        for attempt in range(self.reconnect_attempts + 1):
            if not self.is_open:
                try:
                    self.open()
                except CameraOpenError:
                    if attempt == self.reconnect_attempts:
                        return None
                    self._wait_before_reconnect()
                    continue

            assert self._capture is not None
            ok, image = self._capture.read()
            if ok and image is not None and image.size > 0:
                frame = CameraFrame(
                    image=image,
                    timestamp=time.time(),
                    index=self._frame_index,
                )
                self._frame_index += 1
                return frame

            self.release()
            if attempt < self.reconnect_attempts:
                self._wait_before_reconnect()
        return None

    def release(self) -> None:
        """카메라 핸들을 안전하게 해제한다. 여러 번 호출해도 된다."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _wait_before_reconnect(self) -> None:
        if self.reconnect_interval_seconds > 0:
            time.sleep(self.reconnect_interval_seconds)

    def __enter__(self) -> OpenCVCameraSource:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


__all__ = [
    "CameraAddress",
    "CameraError",
    "CameraFrame",
    "CameraOpenError",
    "CameraSource",
    "OpenCVCameraSource",
    "parse_camera_address",
]
