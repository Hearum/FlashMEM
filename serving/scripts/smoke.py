"""Check an SGLang server using a single synthetic request."""

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def wait_for_server(endpoint, wait, server=None):
    deadline = time.monotonic() + wait
    while True:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"Server exited with status {server.returncode}.")
        try:
            timeout = min(5, max(0.1, deadline - time.monotonic())) if wait else 5
            with urlopen(f"{endpoint}/health", timeout=timeout) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError) as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Server is not ready at {endpoint}. Check the server terminal.") from error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(f"Server is not ready at {endpoint}. Check the server terminal.")
        time.sleep(min(remaining, 1))


def generate(endpoint, timeout):
    payload = {
        "text": "The next even number after 2, 4, and 6 is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
    }
    request = Request(
        f"{endpoint}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result.get("text"), str) or "meta_info" not in result:
        raise RuntimeError("Unexpected response from /generate")
    return {"ok": True, "text": result["text"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30014")
    parser.add_argument("--timeout", type=float, default=60, help="Generation request timeout in seconds")
    parser.add_argument("--wait", type=float, default=0, help="Seconds to wait for the server to start")
    args = parser.parse_args()
    endpoint = args.url.rstrip("/")
    wait_for_server(endpoint, args.wait)
    print(json.dumps(generate(endpoint, args.timeout)))


if __name__ == "__main__":
    main()
