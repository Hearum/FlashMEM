import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent

FAKE_SERVER = """
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import signal

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--tp-size", type=int, required=True)
parser.add_argument("--model-path", required=True)
args, _ = parser.parse_known_args()
assert args.tp_size == 1
if args.model_path == "exit-early":
    raise SystemExit(7)

def stop(*_args):
    Path("server-stopped").touch()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)

class Handler(BaseHTTPRequestHandler):
    attempts = 0

    def log_message(self, *_args):
        pass

    def do_GET(self):
        assert self.path == "/health"
        Handler.attempts += 1
        self.send_response(503 if Handler.attempts == 1 else 200)
        self.end_headers()

    def do_POST(self):
        assert self.path == "/generate"
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["text"]
        assert request["sampling_params"]["max_new_tokens"] == 4
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"text":" 8","meta_info":{"completion_tokens":4}}')

HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""


class QuickStartTests(unittest.TestCase):
    def setUp(self):
        scratch = self.enterContext(tempfile.TemporaryDirectory())
        self.project = Path(scratch)
        shutil.copytree(ROOT / "scripts", self.project / "scripts")
        package = self.project / "sglang/python/sglang"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "launch_server.py").write_text(FAKE_SERVER)

    def demo(self, model):
        return subprocess.run(
            [sys.executable, "scripts/demo.py", "--model", model, "--wait", "15"],
            cwd=self.project, capture_output=True, text=True, timeout=8,
        )

    def test_demo_waits_for_readiness_and_stops_its_server(self):
        result = self.demo("test-model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"ok": True, "text": " 8"})
        self.assertTrue((self.project / "server-stopped").is_file())

    def test_demo_reports_early_exit(self):
        result = self.demo("exit-early")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exited with status 7", result.stderr)

    def test_example_can_be_validated_without_model_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", "replay/build_full_workload.py", "examples/conversations.json", "--validate-only"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        self.assertEqual(result.stdout.strip(), "Validated 2 conversations.")

    def test_validation_reports_the_invalid_message(self):
        invalid = self.project / "invalid.json"
        invalid.write_text('[{"messages":[{"role":"user","content":null}]}]')
        result = subprocess.run(
            [sys.executable, "-S", "replay/build_full_workload.py", str(invalid), "--validate-only"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Conversation 1, message 1 needs string content.", result.stderr)
