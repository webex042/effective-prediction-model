"""``python -m epm_chatbot`` — start the standalone EPM Chat server."""

import argparse


def main() -> None:
    from epm_chatbot import require_runtime

    require_runtime()
    from epm_chatbot.backend.server import serve
    from epm_chatbot.config import Settings

    p = argparse.ArgumentParser(description="EPM Chat — standalone chatbot server")
    p.add_argument("--host", help="bind address (default from EPM_CHAT_HOST or 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default from EPM_CHAT_PORT or 8765)")
    args = p.parse_args()
    s = Settings.from_env()
    if args.host:
        s.host = args.host
    if args.port:
        s.port = args.port
    serve(s)


if __name__ == "__main__":
    main()
