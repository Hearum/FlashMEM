"""Compare online CPU/GPU selectors under the same open-loop arrivals."""

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import random
import statistics
from struct import pack
import time

from selector_workers import init_torch, pick_cpus, placement


_model = None


def initialize(path, keydir, cpus, device, max_logits, cache_plan=None):
    global _model
    torch = init_torch(len(cpus), cpus)
    from long_selector import LongSelector

    if device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    raw = json.loads(Path(path).read_text())
    admitted = (set(json.loads(Path(cache_plan).read_text())["blocks"])
                if cache_plan is not None else raw["blocks"].keys())
    bank = {key: {name: block[name] for name in ("draft_offsets", "target_offsets")}
            for key, block in raw["blocks"].items() if key in admitted}
    del raw
    model = LongSelector(bank, device, max_logits)
    for shard in sorted(Path(keydir).glob("keys-*.pt")):
        items = torch.load(shard, mmap=True, map_location="cpu", weights_only=True)
        model.keys.update((key, value) for key, value in items.items() if key in bank)
    if model.keys.keys() != bank.keys():
        raise RuntimeError("The draft-key index is incomplete")
    _model = model


def prefault_keys():
    # Offsets are BF16 words. Sample one word per 4 KiB page.
    checksum = sum(key.view(-1)[::2048].float().sum().item() for key in _model.keys.values())
    if not math.isfinite(checksum):
        raise ValueError("Non-finite document draft key")


def ping(warm, touch_keys=False):
    if touch_keys:
        # The replay changes the location of target-model KV only. Fault the
        # draft keys into DRAM before both SSD and DRAM timing windows.
        prefault_keys()
    for row in warm:
        work(row)
    return dict(pid=os.getpid(), cpus=sorted(os.sched_getaffinity(0)),
                device=str(_model.device), max_logits=_model.max_logits)


def work(row):
    started = time.perf_counter()
    if row["blocks"]:
        mask, detail = _model.select(row, 0.30, profile=_model.device.type == "cuda")
    else:
        mask, detail = [], dict(select_ms=0.0, selected_summary=0)
    return started, time.perf_counter(), digest_mask(mask), detail


def digest_mask(mask):
    return sha256(pack(f"<{len(mask)}I", *mask)).hexdigest()


def payload(row):
    return {name: row[name] for name in ("qid", "blocks", "draft_query", "head_len")}


def percentiles(values):
    ordered = sorted(values)

    def quantile(q):
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (index - low) * (ordered[high] - ordered[low])

    return dict(mean=statistics.fmean(ordered), p50=quantile(0.5), p95=quantile(0.95),
                p99=quantile(0.99), max=ordered[-1])


async def replay(pools, order, normalized_gaps, rate, output):
    loop = asyncio.get_running_loop()
    worker_queue = asyncio.Queue()
    for pool in pools:
        worker_queue.put_nowait(pool)
    offsets = [offset / rate for offset in normalized_gaps]
    start = time.perf_counter() + 0.25

    async def submit(row, offset):
        deadline = start + offset
        await asyncio.sleep(max(0, deadline - time.perf_counter()))
        entry = time.perf_counter()
        pool = await worker_queue.get()
        try:
            begun, done, digest, detail = await asyncio.wrap_future(pool.submit(work, payload(row)))
            returned = time.perf_counter()
        finally:
            worker_queue.put_nowait(pool)
        return dict(qid=row["qid"], offered_offset=offset, arrival_lag_ms=(entry - deadline) * 1000,
                    selector_queue_ms=(begun - entry) * 1000, time_to_mask_ms=(done - deadline) * 1000,
                    client_return_ms=(returned - done) * 1000, completed_offset=done - start,
                    input_tokens=len(row["input_ids"]), history_tokens=row["summary_len"],
                    draft_query_tokens=len(row["draft_query"]), mask_hash=digest, **detail)

    with (output / "requests.jsonl").open("w", buffering=1) as records:
        rows = []
        for pending in asyncio.as_completed([submit(row, offset) for row, offset in zip(order, offsets)]):
            row = await pending
            rows.append(row)
            records.write(json.dumps(row, separators=(",", ":")) + "\n")
        end = max(row["completed_offset"] for row in rows)
    stats = dict(requests=len(rows), offered_req_s=rate,
                 realized_req_s=(len(rows) - 1) / offsets[-1], throughput_req_s=len(rows) / end,
                 offered_window_s=offsets[-1], completion_window_s=end)
    for key in ("select_ms", "selector_queue_ms", "time_to_mask_ms", "query_ms", "keys_rope_ms",
                "draft_key_pack_ms", "draft_key_h2d_ms", "score_ms", "topk_map_ms", "client_return_ms"):
        stats[key] = percentiles(row.get(key, 0.0) for row in rows)
    stats["effective_history_recompute"] = sum(row["selected_summary"] for row in rows) / sum(
        row["history_tokens"] for row in rows)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=Path)
    parser.add_argument("keys", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--max-logits", type=int)
    parser.add_argument("--gpu-numa", type=int, choices=(0, 1))
    parser.add_argument("--cache-plan", type=Path)
    parser.add_argument("--exclude-cpus", type=int, nargs="+", default=[])
    args = parser.parse_args()
    raw = json.loads(args.workload.read_text())
    rows = raw["rows"]
    if args.cache_plan is not None:
        admitted = set(json.loads(args.cache_plan.read_text())["blocks"])
        for row in rows:
            length = next((i for i, key in enumerate(row["blocks"]) if key not in admitted),
                          len(row["blocks"]))
            row["blocks"] = row["blocks"][:length]
            row["summary_len"] = sum(len(raw["blocks"][key]["target_ids"]) for key in row["blocks"])
    del raw
    order = list(rows)
    random.Random(args.seed).shuffle(order)
    gaps = [0.0]
    random_gaps = random.Random(args.seed + 1)
    for _ in order[1:]:
        gaps.append(gaps[-1] + random_gaps.expovariate(1.0))

    sockets, _ = pick_cpus(45, sample_s=2.0)
    excluded = set(args.exclude_cpus)
    if excluded:
        siblings = set()
        for cpu in excluded:
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            for entry in path.read_text().strip().split(","):
                lo, _, hi = entry.partition("-")
                siblings.update(range(int(lo), int(hi or lo) + 1))
        sockets = [[cpu for cpu in socket if cpu not in siblings] for socket in sockets]
    if args.device == "cuda" and args.gpu_numa is not None:
        sockets = [sockets[args.gpu_numa]]
    cpus = placement(sockets, args.workers, args.threads)
    warm = [payload(row) for row in order if 1000 < row["summary_len"] < 6000][:2]
    pools = [ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=initialize,
                                initargs=(args.workload, args.keys, seat, args.device, args.max_logits,
                                          args.cache_plan))
             for seat in cpus]
    try:
        pending = [pool.submit(ping, warm, args.cache_plan is not None) for pool in pools]
        observations = [job.result() for job in pending]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "placement.json").write_text(json.dumps(observations, indent=2) + "\n")
        for rate in args.rates:
            output = args.output / f"r{rate:g}"
            output.mkdir()
            (output / "arrivals.json").write_text(json.dumps(
                [(row["qid"], gap / rate) for row, gap in zip(order, gaps)]) + "\n")
            before = Path("/proc/pressure/cpu").read_text()
            print(f"Begin {args.device} {rate:g} req/s, {len(rows)} arrivals", flush=True)
            stats = asyncio.run(replay(pools, order, gaps, rate, output))
            stats.update(cpu_pressure_before=before, cpu_pressure_after=Path("/proc/pressure/cpu").read_text())
            (output / "trial.json").write_text(json.dumps(stats, indent=2) + "\n")
            print(json.dumps(stats), flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
