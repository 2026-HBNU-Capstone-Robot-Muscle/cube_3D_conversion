"""큐브 6면 자동 캡처와 상태 조립을 연결하는 프로토타입 실행 진입점."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional
from urllib.parse import quote
import webbrowser

import cv2
import numpy as np

from camera_source import OpenCVCameraSource, parse_camera_address
from color_extract import FaceColorSamples, draw_sampling_grid, extract_face_colors
from cube_state import (
    CaptureArchive,
    CubeSession,
    CubeStateResult,
    FACE_NAMES,
    FACE_ORDER,
    FaceCapture,
)
from face_tracker import CaptureQualityGate, FaceDetection, FaceTracker, FrameObservation


# 모든 안내는 최초 자세(F가 카메라 정면, U가 위쪽)를 기준으로 한다.
# OpenCV 기본 글꼴은 한글 렌더링을 지원하지 않아 화면에는 영어를, 콘솔에는 한글을 출력한다.
FACE_SCAN_GUIDANCE_EN = {
    "F": "Place FRONT face toward camera",
    "R": "Turn cube LEFT 90 deg: RIGHT face toward camera",
    "B": "Turn cube LEFT 90 deg again: BACK face toward camera",
    "L": "Turn cube LEFT 90 deg again: LEFT face toward camera",
    "U": "Return FRONT to camera, then tilt DOWN 90 deg: UP face toward camera",
    "D": "From UP view, tilt DOWN 180 deg: DOWN face toward camera",
}

FACE_SCAN_GUIDANCE_KO = {
    "F": "앞면이 카메라 정면을 향하게 하세요.",
    "R": "큐브 전체를 왼쪽으로 90도 돌려 오른쪽 면이 카메라를 향하게 하세요.",
    "B": "큐브 전체를 왼쪽으로 90도 더 돌려 뒷면이 카메라를 향하게 하세요.",
    "L": "큐브 전체를 왼쪽으로 90도 더 돌려 왼쪽 면이 카메라를 향하게 하세요.",
    "U": "앞면을 카메라 정면으로 되돌린 뒤, 아래로 90도 돌려 윗면을 보여주세요.",
    "D": "윗면 촬영 자세에서 아래로 180도 더 돌려 아랫면을 보여주세요.",
}


@dataclass
class PendingCapture:
    """자동 캡처 후 사용자 확인을 기다리는 면 데이터."""

    face_name: str
    original_frame: np.ndarray
    detection: FaceDetection
    samples: FaceColorSamples
    rotation_quarter_turns: int = 0
    duplicate_override_requested: bool = False
    notice: str = "Use 0-3 to rotate, A to accept, R to recapture"

    def build_face_capture(self) -> FaceCapture:
        return FaceCapture(
            face_name=self.face_name,
            cell_colors_bgr=self.samples.cell_colors_bgr,
            rotation_quarter_turns=self.rotation_quarter_turns,
            confidence=self.detection.confidence,
            original_frame=self.original_frame,
            warped_face=self.samples.warped_face,
            obb_corners=self.detection.corners,
            cell_regions=self.samples.cell_regions,
        )


def _draw_text(image: np.ndarray, text: str, line: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (15, 30 + line * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
    )


def draw_live_view(
    frame: np.ndarray,
    observation: Optional[FrameObservation],
    session: CubeSession,
    *,
    manual_capture: bool,
) -> np.ndarray:
    """카메라 화면에 OBB, 다음 면, 품질·정지 상태를 표시한다."""
    display = frame.copy()
    next_face = session.next_face_name
    if next_face is None:
        _draw_text(display, "All six faces captured", 0, (0, 255, 0))
        return display

    _draw_text(
        display,
        f"Next face: {FACE_NAMES[next_face]} ({next_face}) | {len(session.captures)}/6",
        0,
        (0, 255, 255),
    )
    _draw_text(display, FACE_SCAN_GUIDANCE_EN[next_face], 1, (255, 255, 255))
    if manual_capture:
        _draw_text(display, "SPACE: capture & recognize | Q: quit", 4, (210, 210, 210))
    else:
        _draw_text(display, "Automatic capture enabled | Q: quit", 4, (210, 210, 210))

    if observation is None or observation.detection is None:
        _draw_text(display, "Waiting for cube face", 2, (0, 0, 255))
        return display

    detection = observation.detection
    corners = detection.corners.astype(np.int32).reshape((-1, 1, 2))
    color = (0, 255, 0) if observation.trigger.locked else (0, 200, 255)
    cv2.polylines(display, [corners], True, color, 2)
    cx, cy = map(int, detection.center)
    cv2.circle(display, (cx, cy), 4, color, -1)
    _draw_text(display, f"OBB confidence: {detection.confidence:.2f}", 2, color)

    if observation.quality is not None and not observation.quality.accepted:
        quality = observation.quality
        _draw_text(
            display,
            f"Quality rejected | area:{quality.polygon_area_px:.0f} ratio:{quality.aspect_ratio:.2f} sharp:{quality.sharpness:.0f}",
            3,
            (0, 0, 255),
        )
    elif observation.trigger.locked:
        _draw_text(display, "Captured. Move or hide cube for next face", 3, (0, 255, 0))
    else:
        _draw_text(
            display,
            f"Stability: {observation.trigger.stable_frames} frames",
            3,
            (255, 255, 255),
        )
    return display


def draw_pending_preview(pending: PendingCapture) -> np.ndarray:
    """사용자가 방향을 확인할 수 있도록 원근 보정 면과 샘플 영역을 보여준다."""
    preview = draw_sampling_grid(pending.samples.warped_face, pending.samples.cell_regions)
    preview = np.rot90(preview, -pending.rotation_quarter_turns).copy()
    _draw_text(
        preview,
        f"{FACE_NAMES[pending.face_name]} rotation: {pending.rotation_quarter_turns * 90} deg CW",
        0,
        (0, 255, 0),
    )
    _draw_text(preview, pending.notice, 1, (255, 255, 255))
    return preview


def save_viewer_state(
    destination: Path,
    session: CubeSession,
    result: CubeStateResult,
) -> Path:
    """다음 Three.js 뷰어가 읽을 간단한 54칸 상태 JSON을 저장한다."""
    payload = {
        "face_order": list(session.face_order),
        "labels_by_face": {
            face: list(labels) for face, labels in result.labels_by_face.items()
        },
        "center_colors_bgr": {
            face: session.captures[face].center_bgr.tolist() for face in session.face_order
        },
        "validation": {
            "color_counts": dict(result.validation.color_counts),
            "count_errors": dict(result.validation.count_errors),
            "ambiguous_stickers": [
                {
                    "face": sticker.face_name,
                    "index": sticker.sticker_index,
                    "label": sticker.label,
                    "margin": round(sticker.margin, 3),
                }
                for sticker in result.validation.ambiguous_stickers
            ],
            "close_center_pairs": [list(pair) for pair in result.validation.close_center_pairs],
            "is_valid": result.validation.is_valid,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def complete_session(
    session: CubeSession,
    output_directory: Path,
) -> CubeStateResult:
    """6면 분류·검증을 실행하고 Three.js용 상태 파일을 생성한다."""
    result = session.classify()
    print(f"색상별 개수: {dict(result.validation.color_counts)}")
    if result.validation.count_errors:
        print(f"개수 오류: {dict(result.validation.count_errors)}")
    if result.validation.ambiguous_stickers:
        print(f"애매한 칸 수: {len(result.validation.ambiguous_stickers)}")
    if result.validation.close_center_pairs:
        print(f"유사한 중심색: {result.validation.close_center_pairs}")
    if not result.validation.is_valid:
        failed_faces = ", ".join(result.validation.count_errors) or "재캡처할 면"
        print(f"검증 실패: cube_state.json은 저장하지 않습니다. {failed_faces} 면을 재캡처하세요.")
        return result

    state_path = save_viewer_state(output_directory / "cube_state.json", session, result)
    print(f"6면 캡처 완료. 3D 뷰어용 상태 저장: {state_path}")
    return result


def launch_3d_viewer(port: int, state_path: Path) -> None:
    """별도 로컬 서버를 백그라운드로 시작하고 기본 브라우저에 뷰어를 연다."""
    viewer_server = Path(__file__).with_name("viewer_server.py")
    if not viewer_server.is_file():
        print(f"3D 뷰어 실행 실패: {viewer_server} 파일이 없습니다.")
        return

    command = [sys.executable, str(viewer_server), "--port", str(port)]
    popen_kwargs: dict[str, object] = {"cwd": str(viewer_server.parent)}
    if sys.platform == "win32":
        # 스캔 종료 후 서버용 콘솔 창이 따로 나타나지 않게 한다.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        project_directory = viewer_server.parent.resolve()
        try:
            relative_state_path = state_path.resolve().relative_to(project_directory).as_posix()
        except ValueError:
            print(
                "3D 뷰어 실행 실패: --output 폴더는 프로젝트 폴더 안에 있어야 합니다. "
                f"현재 경로: {state_path.resolve()}"
            )
            return

        subprocess.Popen(command, **popen_kwargs)
        encoded_state_path = quote(relative_state_path, safe="/")
        url = f"http://127.0.0.1:{port}/viewer.html?state={encoded_state_path}"
        webbrowser.open_new_tab(url)
        print(f"3D 뷰어를 엽니다: {url}")
    except OSError as error:
        print(f"3D 뷰어 실행 실패: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="큐브 6면 자동 캡처 프로토타입")
    # parser.add_argument("--source", default="0", help="카메라 번호 또는 IP 스트림 URL")   # 노트북 내장카메라 및 데스크톱 웹캠
    parser.add_argument("--source", default="1", help="카메라 번호 또는 IP 스트림 URL")     # 노트북 USB 웹캠
    parser.add_argument("--model", default=None, help="cube-detector.pt 파일 경로")
    parser.add_argument("--output", default="captures", help="캡처와 상태 JSON 저장 폴더")
    parser.add_argument("--width", type=int, default=1280, help="카메라 요청 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 요청 높이")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="최소 YOLO 신뢰도")
    parser.add_argument("--min-area", type=float, default=12_000, help="최소 OBB 면적(px²)")
    parser.add_argument("--min-aspect", type=float, default=0.55, help="최소 OBB 종횡비")
    # 최소 화질 선명도 초기값: 80 -> 40
    parser.add_argument("--min-sharpness", type=float, default=40, help="최소 Laplacian variance")
    parser.add_argument(
        "--confirm-capture",
        action="store_true",
        help="자동 저장 대신 면 방향을 사람이 확인·보정하는 창을 표시",
    )
    parser.add_argument("--viewer-port", type=int, default=8000, help="3D 뷰어 로컬 서버 포트")
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="정지 판정이 통과하면 자동으로 캡처합니다. 기본값은 Space 키 수동 촬영입니다.",
    )
    args = parser.parse_args()

    output_directory = Path(args.output)
    quality_gate = CaptureQualityGate(
        min_confidence=args.min_confidence,
        min_area_px=args.min_area,
        min_aspect_ratio=args.min_aspect,
        min_laplacian_variance=args.min_sharpness,
    )
    tracker = FaceTracker(
        model_path=args.model,
        confidence_threshold=args.min_confidence,
        quality_gate=quality_gate,
    )
    session = CubeSession()
    archive = CaptureArchive(output_directory)
    camera = OpenCVCameraSource(
        parse_camera_address(args.source), width=args.width, height=args.height
    )
    pending: Optional[PendingCapture] = None
    completed_result: Optional[CubeStateResult] = None
    previous_quality_reasons: tuple[str, ...] = ()

    def commit_capture(capture: FaceCapture, *, allow_duplicate: bool = False) -> bool:
        """캡처를 저장한다. 중복 의심 면은 명시적 승인 없이는 저장하지 않는다."""
        nonlocal completed_result
        duplicate_warnings = session.duplicate_center_warnings(capture)
        if duplicate_warnings and not allow_duplicate:
            print("중복 면 의심(저장하지 않음):", "; ".join(duplicate_warnings))
            return False

        session.add_capture(capture)
        metadata_path = archive.save(capture)
        print(f"{capture.face_name} 면 저장: {metadata_path}")
        next_face = session.next_face_name
        if next_face is not None:
            print(f"다음 촬영 안내 [{FACE_NAMES[next_face]}]: {FACE_SCAN_GUIDANCE_KO[next_face]}")
        if session.is_complete:
            completed_result = complete_session(session, output_directory)
        return True

    camera.open()
    try:
        while True:
            packet = camera.read()
            if packet is None:
                # 재연결이 끝날 때까지 직전 창 이벤트도 계속 처리한다.
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
                continue

            # 상하좌우가 반전되어 표시되어 뒤집었음
            frame = packet.image
            # frame = cv2.flip(packet.image, -1)
            observation: Optional[FrameObservation] = None
            if pending is None and completed_result is None:
                observation = tracker.process(
                    frame,
                    update_trigger=args.auto_capture,
                )
                current_quality_reasons = (
                    observation.quality.reasons
                    if observation.quality is not None and not observation.quality.accepted
                    else ()
                )
                if current_quality_reasons != previous_quality_reasons:
                    if current_quality_reasons:
                        print("캡처 품질 대기:", ", ".join(current_quality_reasons))
                    previous_quality_reasons = current_quality_reasons
                if args.auto_capture and observation.trigger.should_capture:
                    samples = extract_face_colors(frame, observation.detection.corners)
                    candidate = PendingCapture(
                        face_name=session.next_face_name or "F",
                        original_frame=frame.copy(),
                        detection=observation.detection,
                        samples=samples,
                    )
                    if args.confirm_capture:
                        pending = candidate
                        cv2.namedWindow("Capture confirmation", cv2.WINDOW_NORMAL)
                    else:
                        # 자동 모드는 OBB가 준 꼭짓점 순서와 기본 회전값 0을 사용한다.
                        commit_capture(candidate.build_face_capture())
                display = draw_live_view(
                    frame,
                    observation,
                    session,
                    manual_capture=not args.auto_capture,
                )
            else:
                display = draw_live_view(
                    frame,
                    None,
                    session,
                    manual_capture=not args.auto_capture,
                )
                if pending is not None:
                    cv2.imshow("Capture confirmation", draw_pending_preview(pending))
                elif completed_result is not None:
                    if completed_result.validation.is_valid:
                        _draw_text(display, "Scan complete - cube_state.json saved", 1, (0, 255, 0))
                        _draw_text(display, "Press Q to exit and launch 3D viewer", 2, (255, 255, 255))
                    else:
                        failed_faces = ", ".join(completed_result.validation.count_errors)
                        _draw_text(display, "Validation failed - 3D state was NOT saved", 1, (0, 0, 255))
                        _draw_text(display, f"Press {failed_faces} to recapture that face", 2, (0, 165, 255))
                        _draw_text(display, "Press Q to exit", 3, (255, 255, 255))

            cv2.imshow("Cube scan", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                if completed_result is not None and completed_result.validation.is_valid:
                    launch_3d_viewer(
                        args.viewer_port,
                        output_directory / "cube_state.json",
                    )
                break

            # 검증 실패 시 해당 면을 삭제하고 재촬영 모드로 돌아간다.
            if completed_result is not None:
                if not completed_result.validation.is_valid:
                    requested_face = chr(key).upper() if key else ""
                    if requested_face in FACE_ORDER:
                        session.remove_capture(requested_face)
                        completed_result = None
                        tracker.reset()
                        previous_quality_reasons = ()
                        print(f"{requested_face} 면 이전 캡처를 제거했습니다. 재캡처를 시작합니다.")
                continue

            if pending is None:
                if key == ord(" ") and not args.auto_capture:
                    if observation is None or observation.detection is None:
                        print("수동 캡처 실패: 현재 프레임에서 큐브 면을 찾지 못했습니다.")
                        continue
                    if observation.quality is None or not observation.quality.accepted:
                        reasons = observation.quality.reasons if observation.quality else ()
                        print("수동 캡처 거절:", ", ".join(reasons) or "품질 정보를 얻지 못했습니다.")
                        continue

                    samples = extract_face_colors(frame, observation.detection.corners)
                    candidate = PendingCapture(
                        face_name=session.next_face_name or "F",
                        original_frame=frame.copy(),
                        detection=observation.detection,
                        samples=samples,
                    )
                    if args.confirm_capture:
                        pending = candidate
                        cv2.namedWindow("Capture confirmation", cv2.WINDOW_NORMAL)
                    else:
                        commit_capture(candidate.build_face_capture())
                continue
            if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                pending.rotation_quarter_turns = int(chr(key))
                pending.duplicate_override_requested = False
                pending.notice = "Use 0-3 to rotate, A to accept, R to recapture"
            elif key in (ord("r"), ord("R")):
                # 다시 캡처하려면 같은 면을 약간 움직인 뒤 정지시킨다.
                pending = None
                tracker.reset()
                cv2.destroyWindow("Capture confirmation")
            elif key in (ord("a"), ord("A"), 13):
                capture = pending.build_face_capture()
                duplicate_warnings = session.duplicate_center_warnings(capture)
                if duplicate_warnings and not pending.duplicate_override_requested:
                    pending.duplicate_override_requested = True
                    pending.notice = "Possible duplicate. A again: force save | R: recapture"
                    print("중복 면 의심:", "; ".join(duplicate_warnings))
                    continue

                if commit_capture(capture, allow_duplicate=True):
                    pending = None
                    cv2.destroyWindow("Capture confirmation")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
