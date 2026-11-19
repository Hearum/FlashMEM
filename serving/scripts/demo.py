"""Launch a local one-GPU server, send one request, and stop the server."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

from smoke import generate, wait_for_server


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_DIR"), help="Model ID or local checkpoint")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: pick an available port)")
    parser.add_argument("--wait", type=float, default=300, help="Startup timeout in seconds")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.wait <= 0:
        parser.error("Specify a port from 0 through 65535 and a positive startup timeout.")

    # Check an explicit port before starting, or get a free port for this run.
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", args.port))
            port = listener.getsockname()[1]
    except OSError as error:
        parser.exit(1, f"Could not reserve a loopback port: {error}\n")

    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PORT": str(port), "PYTHON": sys.executable}
    if args.model:
        local_model = Path(args.model).expanduser()
        env["MODEL_DIR"] = str(local_model.resolve()) if local_model.is_dir() else args.model
    server = subprocess.Popen(
        ["bash", "scripts/serve.sh", "demo"],
        cwd=root, env=env, stdout=sys.stderr, start_new_session=True,
    )
    try:
        endpoint = f"http://127.0.0.1:{port}"
        wait_for_server(endpoint, args.wait, server=server)
        print(json.dumps(generate(endpoint, timeout=60)))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Demo failed: {error}") from error
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
