"""cube_state.json을 읽는 Three.js 뷰어용 로컬 HTTP 서버."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="큐브 3D 뷰어 로컬 서버")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 번호")
    args = parser.parse_args()

    # viewer.html과 captures 폴더를 함께 제공하려면 프로젝트 루트를 서버 루트로 사용한다.
    project_root = Path(__file__).resolve().parent
    handler = lambda *handler_args, **handler_kwargs: SimpleHTTPRequestHandler(
        *handler_args, directory=str(project_root), **handler_kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"3D 뷰어 서버: http://127.0.0.1:{args.port}/viewer.html")
    print("종료하려면 Ctrl+C를 누르세요.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
