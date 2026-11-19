"""Build and replay a conversation workload on a local eight-GPU server."""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory

from demo import stop_server
from smoke import wait_for_server


ROOT = Path(__file__).resolve().parent.parent


def run_phase(label, args, cwd, env):
    print(f"\n== {label} ==", flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, check=True)


def run_server(launcher, python, directory, env, phases, timeout):
    log_path = directory / f"server-{launcher}.log"
    print(f"\n== Starting {launcher} server. Log: {log_path} ==", flush=True)
    with log_path.open("w") as log:
        server = subprocess.Popen(
            ["bash", ROOT / "scripts/serve.sh", launcher], cwd=directory,
            env={**env, "PYTHON": str(python)}, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(env["BENCH_BASE"], timeout, server=server)
            for label, command, overrides in phases:
                run_phase(label, command, directory, {**env, **overrides})
        finally:
            stop_server(server)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="JSON array of conversations")
    source.add_argument("--workload", type=Path, help="Previously built workload.json")
    parser.add_argument("--keys", type=Path, help="Precomputed draft key directory for --workload")
    parser.add_argument("--model", type=Path, required=True, help="Local Kimi-K2.6 checkpoint directory")
    parser.add_argument("--draft-model", type=Path, required=True, help="Local Qwen2.5-0.5B checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--storage", choices=("dram", "ssd"), nargs="+", default=["dram"])
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument("--rate", type=float, help="Offered requests per second for one rate")
    rates.add_argument("--rates", type=float, nargs="+", help="Run every listed offered rate")
    parser.add_argument("--max-input-tokens", type=int, default=32000)
    parser.add_argument("--limit", type=int, help="Replay the first N scheduled requests after shuffling")
    parser.add_argument("--gpu-numa", type=int, default=0, help="CPU socket index for the GPU selector")
    parser.add_argument("--selector-gpu", help="GPU for the draft selector (default: last visible GPU)")
    parser.add_argument("--decode-interval", type=int, default=1, help="Decode interval for the modified engine")
    parser.add_argument("--tpot-coalescing", type=int, nargs=2, default=[32, 32],
                        metavar=("MIN_PREFILLS", "DECODE_STEPS"))
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: pick an available port)")
    parser.add_argument("--wait", type=float, default=2400, help="Startup timeout per server in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Workers for preparing draft keys and document KV")
    parser.add_argument("--prepare-only", action="store_true", help="Build inputs and draft keys without starting a server")
    parser.add_argument("--document-cache", type=Path, help="Persistent document KV directory on SSD")
    parser.add_argument("--dram-cache", type=Path, help="A tmpfs directory for the DRAM run")
    parser.add_argument("--reuse-document-cache", action="store_true", help="Use a prepared cache and skip population")
    parser.add_argument("--baseline-python", type=Path, help="Python executable with official SGLang v0.5.23 installed")
    args = parser.parse_args()
    args.rates = args.rates or [0.15 if args.rate is None else args.rate]
    if (any(not math.isfinite(rate) or rate <= 0 for rate in args.rates)
            or not math.isfinite(args.wait) or args.wait <= 0 or args.max_input_tokens < 1 or args.workers < 1
            or args.gpu_numa < 0 or not 0 <= args.port <= 65535
            or args.decode_interval < 0 or any(not 0 <= value <= 32 for value in args.tpot_coalescing)
            or (args.limit is not None and args.limit < 2)):
        parser.error("Check the rate, timeouts, input cap, worker count, placement, and scheduler parameters.")
    if len(args.rates) != len(set(args.rates)) or len(args.storage) != len(set(args.storage)):
        parser.error("Specify each rate and storage tier once.")
    for name in ("data", "workload"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"{path} must be a file.")
    for path in (args.model, args.draft_model):
        if not (path / "config.json").is_file():
            parser.error(f"{path} must be a local checkpoint directory with config.json.")
    if args.model.name != "Kimi-K2.6":
        parser.error("Name the target checkpoint directory Kimi-K2.6 for replicated MLA storage.")
    if args.keys and (not args.workload or not any(args.keys.glob("keys-*.pt"))):
        parser.error("--keys requires --workload and a directory containing keys-*.pt.")
    if args.reuse_document_cache and not (args.document_cache and args.document_cache.is_dir()):
        parser.error("--reuse-document-cache requires an existing --document-cache directory.")
    if args.baseline_python and not (args.baseline_python.is_file() and os.access(args.baseline_python, os.X_OK)):
        parser.error("--baseline-python must point to an executable.")
    return args


def main():
    args = parse_args()
    os.umask(0o077)
    directory = args.output.absolute()
    documents = (args.document_cache or directory / "documents").absolute()
    try:
        directory.mkdir(parents=True, exist_ok=False)
        visible_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        selector_gpu = args.selector_gpu or visible_gpus[-1].strip()
        env = {
            **os.environ, "MODEL_DIR": str(args.model.absolute()),
            "DRAFT_MODEL_DIR": str(args.draft_model.absolute()), "TP_SIZE": "8",
            "SGLANG_FUSIONRAG_CACHE_DIR": str(documents),
            "SGLANG_FUSIONRAG_REPLICATED_MLA": "1",
        }
        workload = args.workload.absolute() if args.workload else directory / "input/workload.json"
        keys = args.keys.absolute() if args.keys else directory / "draft-keys"
        if args.data:
            run_phase("Build workload", [
                ROOT / "replay/build_full_workload.py", args.data.absolute(), workload.parent,
                "--materialize-all", "--max-input-tokens", args.max_input_tokens,
            ], directory, env)
        if not args.keys:
            run_phase("Precompute draft keys", [
                ROOT / "replay/precompute_keys.py", workload, keys, "--workers", args.workers,
            ], directory, env)
        if args.prepare_only:
            print(f"Prepared workload: {workload}\nDraft keys: {keys}", flush=True)
            return
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", args.port))
            port = str(listener.getsockname()[1])
        env.update(PORT=port, BENCH_BASE=f"http://127.0.0.1:{port}")
        replay = ROOT / "replay/serve_long_replay.py"
        trials = directory / "trials"
        common = ["--workload", workload, "--chunk-size", env.get("CHUNK_SIZE", "8192")]
        trial = [*common, "--rates", *args.rates, "--output", trials]
        if args.limit:
            trial += ["--limit", args.limit]
        phases = []
        if not args.reuse_document_cache:
            phases.append(("Populate document KV", [
                replay, "populate", *common, "--workers", args.workers,
                "--output", directory / "population",
            ], {}))
        for tier in args.storage:
            phases.append((f"Replay {tier}", [
                replay, tier, *trial, "--keys", keys, "--selector", "gpu",
                "--gpu-numa", args.gpu_numa, "--decode-interval", args.decode_interval,
                "--tpot-coalescing", *args.tpot_coalescing,
            ], {"CUDA_VISIBLE_DEVICES": selector_gpu}))
        dram_cache = (nullcontext(args.dram_cache.absolute()) if args.dram_cache
                      else TemporaryDirectory(dir="/dev/shm", prefix="pic-replay-"))
        with dram_cache as dram:
            run_server("pic", sys.executable, directory, {
                **env, "SGLANG_FUSIONRAG_DRAM_CACHE_DIR": str(dram),
            }, phases, args.wait)
        if args.baseline_python:
            run_server("baseline", args.baseline_python.absolute(), directory, {
                **env, "BENCH_OFFICIAL_PREFILL_GRAPH": "off", "BENCH_OFFICIAL_DEFAULT_SCHEDULER": "1",
            }, [("Replay official", [replay, "official", *trial], {})], args.wait)
        run_phase("Summarize trials", [ROOT / "replay/report_long_replay.py", trials], directory, env)
        print(f"Results: {trials / 'analysis.json'}", flush=True)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        raise SystemExit(f"Benchmark failed: {error}\nRun directory: {directory}") from error


if __name__ == "__main__":
    main()
