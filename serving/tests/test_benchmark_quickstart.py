from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    import run_benchmark


class BenchmarkQuickStartTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model = self.scratch / "Kimi-K2.6"
        self.draft = self.scratch / "draft model"
        for checkpoint in (self.model, self.draft):
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
        self.output = self.scratch / "run"
        self.argv = [
            "run_benchmark.py", "--data", str(ROOT / "examples/conversations.json"),
            "--model", str(self.model), "--draft-model", str(self.draft),
            "--output", str(self.output), "--dram-cache", str(self.scratch / "dram"),
        ]
        self.enterContext(patch.object(run_benchmark.os, "umask"))
        self.phases = self.enterContext(patch.object(run_benchmark, "run_phase"))
        self.servers = self.enterContext(patch.object(run_benchmark, "run_server"))
        self.enterContext(patch("builtins.print"))

    def test_prepare_only_runs_before_any_server(self):
        with patch.object(sys, "argv", [*self.argv, "--prepare-only"]):
            run_benchmark.main()

        self.assertEqual([call.args[0] for call in self.phases.call_args_list],
                         ["Build workload", "Precompute draft keys"])
        self.servers.assert_not_called()

    def test_comparison_uses_the_same_workload_and_schedule(self):
        with patch.object(sys, "argv", [
            *self.argv, "--baseline-python", sys.executable,
            "--storage", "dram", "ssd", "--rates", "0.15", "0.30",
        ]):
            run_benchmark.main()

        pic, baseline = [call.args for call in self.servers.call_args_list]
        self.assertEqual((pic[0], baseline[0]), ("pic", "baseline"))
        self.assertEqual(pic[3]["BENCH_BASE"], baseline[3]["BENCH_BASE"])
        self.assertEqual([command[1] for _, command, _ in pic[4]], ["populate", "dram", "ssd"])
        official = baseline[4][0][1]
        for _, command, _ in pic[4][1:]:
            self.assertEqual(command[2:11], official[2:11])
            self.assertIn(self.output / "input/workload.json", command)
            self.assertIn(0.15, command)
            self.assertIn(0.30, command)
        self.assertEqual(self.phases.call_args.args[0], "Summarize trials")
