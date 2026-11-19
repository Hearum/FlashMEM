"""Paired LoCoMo storage-tier replay and TPOT scheduling sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import mmap
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path("results")
BASE = os.getenv("BENCH_BASE", "http://127.0.0.1:30014")
PAGE = mmap.PAGESIZE
LIBC = ctypes.CDLL(None, use_errno=True)


def api(path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def state():
    return api("/server_info")["internal_states"][0]


def set_control(values):
    for _ in range(12):
        result = api("/set_internal_state", {"server_args": values})
        if result and all(result):
            return
        time.sleep(1)
    raise RuntimeError(f"Scheduler refused to apply {values}")


def policy(decode_interval=0, prefill_limit=None, max_batch=None):
    return {
        "fusionrag_decode_interval": decode_interval,
        "prefill_max_requests": prefill_limit,
        "pp_max_micro_batch_size": max_batch,
    }


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def page_sample(path, mode):
    with path.open("rb", buffering=0) as stream:
        if mode == "cold":
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            pages = (len(mapping) + PAGE - 1) // PAGE
            residency = (ctypes.c_ubyte * pages)()
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            if LIBC.mincore(ctypes.c_void_p(ptr), ctypes.c_size_t(len(mapping)), residency):
                raise OSError(ctypes.get_errno(), str(path))
            return sum(bytes(residency)), pages


def prime_file(path):
    with path.open("rb", buffering=0) as stream:
        buffer = bytearray(4 * 1024**2)
        while stream.readinto(buffer):
            pass


def prepare_file_pages(files, mode):
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda path: page_sample(path, mode), files))
        if mode == "file_warm":
            for _ in range(5):
                misses = [path for path, (resident, pages) in zip(files, results) if resident < pages]
                if sum(pages - resident for resident, pages in results) <= .001 * sum(pages for _, pages in results):
                    break
                list(executor.map(prime_file, misses))
                results = list(executor.map(lambda path: page_sample(path, "check"), files))
    resident = sum(count for count, _ in results)
    total = sum(count for _, count in results)
    fraction = resident / total
    if (mode == "file_warm" and fraction < 0.999) or (mode == "cold" and fraction > 0.005):
        raise RuntimeError(f"{mode}: unsuitable page-cache residency {resident}/{total}")
    return {
        "mode": mode,
        "files": len(files),
        "resident_pages": resident,
        "pages": total,
        "fraction": fraction,
        "preparation_seconds": time.monotonic() - started,
    }


def summarize_restore(samples):
    latencies = sorted(sample[2] for sample in samples)
    return {
        "restore_calls": len(samples),
        "restored_blocks": sum(sample[0] for sample in samples),
        "restored_tokens": sum(sample[1] for sample in samples),
        "rank0_p50_ms": statistics.median(latencies) if latencies else None,
        "rank0_p95_ms": latencies[min(len(latencies)-1, int(0.95*len(latencies)))] if latencies else None,
    }


def replay(name, arm, count, rate, probes):
    output = ROOT / name
    before = state()
    subprocess.run(
        [sys.executable, "-u", "bench_prefix.py", "load", "--arms", arm, "--limit", str(count),
         "--rate", str(rate), "--arrival", "poisson", "--tokens", "32", "--probe-requests",
         str(probes), "--out", str(output)],
        check=True,
    )
    after = state()
    result = json.loads((output / "summary.json").read_text())[arm]
    before_cache = before["fusionrag_storage_bench"]
    after_cache = after["fusionrag_storage_bench"]
    before_samples = len(before_cache["restore_samples"])
    result["storage"] = {
        "resident_tokens_before": before_cache["resident_tokens"],
        "resident_tokens_after": after_cache["resident_tokens"],
        **summarize_restore(after_cache["restore_samples"][before_samples:]),
    }
    result["scheduler"] = {key: after[key] for key in (
        "prefill_max_requests", "pp_max_micro_batch_size", "fusionrag_decode_interval"
    )}
    save(output / "with-storage.json", result)
    if result["ok"] != count:
        raise RuntimeError(f"{name}: only {result['ok']} of {count} requests completed")
    return result


def pick_policy(pilot):
    default = pilot["default"]["result"]
    eligible = [
        row for row in pilot.values()
        if row["result"]["req_s"] >= 0.95 * default["req_s"]
        and row["result"]["ttft_ms"]["p95"] <= 1.2 * default["ttft_ms"]["p95"]
    ]
    return min(eligible, key=lambda row: (
        row["result"]["tpot_ms"]["p50"],
        row["result"]["tpot_ms"]["p95"],
        -row["result"]["req_s"],
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pilot-count", type=int, default=240)
    parser.add_argument("--limit", type=int, default=1540)
    parser.add_argument("--rate", type=float, default=6)
    args = parser.parse_args()

    before = state()
    resident = before["fusionrag_storage_bench"]
    assert resident["indexed_blocks"] == 4314 and resident["resident_tokens"] == 373925, resident
    default_cap = before["pp_max_micro_batch_size"]
    configs = {
        "default": policy(max_batch=default_cap),
        "decode1": policy(decode_interval=1, max_batch=default_cap),
        "prefill4": policy(prefill_limit=4, max_batch=default_cap),
        "prefill2": policy(prefill_limit=2, max_batch=default_cap),
        "batch24": policy(max_batch=min(24, default_cap)),
        "batch16": policy(max_batch=min(16, default_cap)),
        "decode1-prefill4": policy(decode_interval=1, prefill_limit=4, max_batch=default_cap),
    }
    pilot = {}
    for name, config in configs.items():
        set_control(config)
        result = replay(f"pilot-{name}", "online", args.pilot_count, args.rate, probes=4)
        assert result["storage"]["restore_calls"] == 0
        pilot[name] = {"name": name, "policy": config, "result": result}
        save(ROOT / "pilot.json", pilot)
    chosen = pick_policy(pilot)
    save(ROOT / "chosen-policy.json", chosen)
    print(f"Selected {chosen['name']}", flush=True)
    set_control(chosen["policy"])

    full = {}
    for mode, arm in (("dram_hot", "online"), ("file_warm", "online"), ("ssd_cold", "online"), ("dense", "dense")):
        if mode in ("file_warm", "ssd_cold"):
            set_control({"fusionrag_reset_host": True})
            empty = state()["fusionrag_storage_bench"]
            assert empty["resident_tokens"] == 0 and empty["indexed_blocks"] == 4314, empty
            files = sorted(args.cache_dir.glob("v1/*/tp8-rank*/*/kv.pt"))
            assert len(files) == 4314 * 8, len(files)
            page_state = prepare_file_pages(files, "cold" if mode == "ssd_cold" else "file_warm")
            save(ROOT / f"{mode}-page-state.json", page_state)
        full[mode] = replay(f"full-{mode}", arm, args.limit, args.rate, probes=0)
        restore = full[mode]["storage"]
        if mode in ("file_warm", "ssd_cold"):
            assert restore["restored_tokens"] == 373925 and restore["restored_blocks"] == 4314, restore
        else:
            assert restore["restore_calls"] == 0, restore
        save(ROOT / "full.json", full)

    if chosen["name"] != "default":
        set_control(configs["default"])
        for mode, arm in (("dram_hot_default", "online"), ("dense_default", "dense")):
            full[mode] = replay(f"full-{mode}", arm, args.limit, args.rate, probes=0)
            assert full[mode]["storage"]["restore_calls"] == 0
            save(ROOT / "full.json", full)


if __name__ == "__main__":
    main()
