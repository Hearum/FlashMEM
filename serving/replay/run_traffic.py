"""Replay identical chat prompts with official Radix or FusionRAG history KV."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid

import aiohttp

from bench_prefix import BASE, monitor, selector_call, SelectorPool, stream, summary
from storage_replay_prefix import api, prepare_file_pages, save, set_control, state, summarize_restore


ROOT = Path("results")
OUTPUT_TOKENS = 32
SEED = 20261114
WORKERS = 2
TP = 8
PREFILL_POLICY = {
    "tpot_coalescing": [32, 32],
    "fusionrag_decode_interval": 0,
    "prefill_max_requests": None,
    "pp_max_micro_batch_size": 128,
    "fusionrag_storage_read_mode": "mapped",
}
SELECTOR_FIELDS = ("query_ms", "keys_rope_ms", "score_ms", "topk_map_ms", "select_ms", "selector_queue_ms")


def make_body(data, row, mode, tag, mask=None, tokens=OUTPUT_TOKENS):
    body = {
        "rid": f"{tag}-{row['qid']}", "extra_key": tag, "stream": True,
        "input_ids": row["input_ids"],
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True},
    }
    if mode != "official":
        blocks = [data["blocks"][key] for key in row["blocks"]]
        parts = [block["text"] for block in blocks]
        part_ids = [block["target_ids"] for block in blocks]
        head = data["head_ids"]
        prefix_end = len(head) + row["summary_len"]
        body["fusionrag_params"] = {
            "save_cache": False, "load_preprocess_cache": False, "load_raw_cache": bool(blocks),
            "rope": True, "use_fusion_rag": True, "cache_radix_prefix": True,
            "prefix_cache_ids": head, "prefix_prompt": "".join(parts), "prefix_prompt_list": parts,
            "prefix_prompt_ids_list": part_ids, "prefix_prompt_ids": row["input_ids"][len(head):prefix_end],
            "prompt_ids_list": [head, *part_ids, row["input_ids"][prefix_end:]],
            "recompute_idx": [pos for pos in mask if pos >= len(head)],
        }
    return body


def prime(data, mode, tag):
    head = data["head_ids"]
    body = {"input_ids": head, "extra_key": tag,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}}
    if mode != "official":
        body["fusionrag_params"] = {"cache_radix_prefix": True, "prefix_cache_ids": head}
    result = api("/generate", body)["meta_info"]
    assert result["prompt_tokens"] == len(head) and result["cached_tokens"] == 0


def check_response(meta, row, mode, recomputed):
    assert meta["prompt_tokens"] == len(row["input_ids"]), "Prompt token count changed"
    cached = meta["cached_tokens"]
    if mode != "official":
        hits = meta["cached_tokens_details"]
        head, history = row["head_len"], row["summary_len"]
        assert cached == head + history, f"Cache hit {cached} expected {head + history}"
        assert hits["device"] == head and hits["host"] == history, f"Unexpected cache accounting {hits}"
        if history:
            assert abs(meta["recomputation_rate"] - recomputed / history) < 1e-6, "Recompute count changed"
    return cached


def order_and_times(rows, rate):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    rng = random.Random(SEED + 1)
    offsets = [0.0]
    for _ in rows[1:]:
        offsets.append(offsets[-1] + rng.expovariate(1) / rate)
    return rows, offsets


async def run_one(data, session, row, mode, tag, scheduled, selector, executor):
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    entry = time.perf_counter()
    rec = dict(qid=row["qid"], arrival=scheduled, dispatch_lag_ms=(entry - scheduled) * 1000)
    try:
        mask = None
        if mode != "official":
            queued = time.perf_counter()
            if row["blocks"]:
                if executor is None:
                    started, finished, mask, stages = await selector.select(row)
                else:
                    started, finished, mask, stages = await asyncio.get_running_loop().run_in_executor(
                        executor, selector_call, selector, row
                    )
                rec.update(stages, selector_queue_ms=(started - queued) * 1000,
                           selector_start_clock=started, selector_finish_clock=finished)
            else:
                mask = []
                rec.update({field: 0.0 for field in SELECTOR_FIELDS})
                rec["selected_summary"] = 0
        body = make_body(data, row, mode, tag, mask)
        sent, first, last, end, count, meta, _ = await stream(session, body)
        recomputed = rec.get("selected_summary", 0)
        cached = check_response(meta, row, mode, recomputed)
        assert count == OUTPUT_TOKENS, f"Expected {OUTPUT_TOKENS} output tokens, got {count}"
        rec.update(ok=True, rid=body["rid"], end=end, http_sent=sent, first_token_clock=first,
                   last_token_clock=last, completion_tokens=count, token_events=meta["token_events"],
                   ttft_ms=(first - scheduled) * 1000, engine_ttft_ms=(first - sent) * 1000,
                   tpot_ms=(last - first) * 1000 / (count - 1), e2e_ms=(end - scheduled) * 1000,
                   prompt_tokens=meta["prompt_tokens"], history_tokens=row["summary_len"],
                   cached_tokens=cached, prefill_tokens=meta["prompt_tokens"] - cached + recomputed,
                   recomputed_tokens=recomputed, retractions=meta.get("total_retractions", 0))
    except Exception as exc:
        # Do not copy request/response text into the result archive.
        rec.update(ok=False, end=time.perf_counter(), error=type(exc).__name__)
        if isinstance(exc, AssertionError):
            rec["check"] = str(exc)
    return rec


async def replay(data, mode, rate, tag, out, selector, executor, limit):
    order, offsets = order_and_times(data["rows"], rate)
    order, offsets = order[:limit], offsets[:limit]
    with (out / "arrivals.json").open("w") as target:
        json.dump([(row["qid"], offset) for row, offset in zip(order, offsets)], target)
    timeout = aiohttp.ClientTimeout(total=480)
    if mode != "official" and executor is None:
        selector.reset_queue()
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        stop_sampling = asyncio.Event()
        sampling = asyncio.create_task(monitor(session, out / "monitor.jsonl", stop_sampling))
        start = time.perf_counter() + .25
        pending = [asyncio.create_task(run_one(data, session, row, mode, tag, start + offset, selector, executor))
                   for row, offset in zip(order, offsets)]
        try:
            with (out / "requests.jsonl").open("w", buffering=1) as target:
                by_id = {}
                for job in asyncio.as_completed(pending):
                    row = await job
                    by_id[row["qid"]] = row
                    target.write(json.dumps(row, separators=(",", ":")) + "\n")
        finally:
            stop_sampling.set()
            await sampling
    rows = [by_id[row["qid"]] for row in order]
    result = summary(rows, start)
    ok = [row for row in rows if row["ok"]]
    result.update(mode=mode, offered_rate=rate, actual_offered_rate=(len(rows) - 1) / offsets[-1],
                  total_prompt_tokens=sum(r["prompt_tokens"] for r in ok),
                  total_cached_tokens=sum(r["cached_tokens"] for r in ok),
                  total_history_tokens=sum(r["history_tokens"] for r in ok),
                  total_recomputed_tokens=sum(r["recomputed_tokens"] for r in ok),
                  total_retractions=sum(r["retractions"] for r in ok))
    return result


def config(mode):
    info = api("/server_info")
    assert info["tp_size"] == TP and info["attention_backend"] == "triton"
    assert not info["disable_cuda_graph"] and not info["disable_radix_cache"]
    assert not info.get("disable_decode_cuda_graph", False)
    assert info["enable_hierarchical_cache"] == (mode != "official")
    if mode == "official":
        assert info["version"].split("+", 1)[0] == "0.5.23", info["version"]
    return {key: info.get(key) for key in (
        "version", "tp_size", "attention_backend", "disable_radix_cache", "disable_cuda_graph",
        "disable_overlap_schedule", "enable_hierarchical_cache", "max_total_num_tokens", "cuda_graph_max_bs",
        "page_size", "stream_interval", "schedule_policy", "memory_usage",
    )}


def wait_ready(max_seconds=1500):
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{BASE}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError("Timed out waiting for /health")


def host_state():
    paths = ("/proc/loadavg", "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory")
    return {path: Path(path).read_text().strip() for path in paths if Path(path).exists()}


def warm(data, workers=8):
    async def populate():
        head, head_ids = data["head"], data["head_ids"]
        timeout = aiohttp.ClientTimeout(total=480)
        sem = asyncio.Semaphore(workers)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def save_block(text, ids, include_head):
                prefix, prefix_ids = (head, head_ids) if include_head else ("", [])
                body = dict(text=prefix + text, input_ids=prefix_ids + ids,
                            sampling_params={"temperature": 0, "max_new_tokens": 0},
                            fusionrag_params={"save_cache": True, "save_preprocess_cache": False,
                                              "prefix_prompt": prefix,
                                              "prefix_prompt_list": [prefix] if prefix else [],
                                              "prefix_prompt_ids": prefix_ids,
                                              "prefix_prompt_ids_list": [prefix_ids] if prefix_ids else [],
                                              "prompt_ids_list": [prefix_ids, ids] if prefix else [ids]})
                async with sem, session.post(f"{BASE}/generate", json=body) as response:
                    response.raise_for_status()
                    result = await response.json()
                    assert result["meta_info"]["prompt_tokens"] == len(prefix_ids) + len(ids)

            await save_block(head, head_ids, False)
            for i, job in enumerate(asyncio.as_completed(
                    [save_block(b["text"], b["target_ids"], True) for b in data["blocks"].values()]), 1):
                await job
                if i % 100 == 0:
                    print(f"Persisted {i}/{len(data['blocks'])} document blocks", flush=True)

    set_control({**PREFILL_POLICY, "fusionrag_document_prefetch": False, "fusionrag_document_staging": False})
    before = time.monotonic()
    asyncio.run(populate())
    stats = state()["fusionrag_storage_bench"]
    expected = sum(len(block["target_ids"]) for block in data["blocks"].values())
    assert stats["indexed_blocks"] == len(data["blocks"]) + 1, stats["indexed_blocks"]
    assert stats["resident_tokens"] == expected + len(data["head_ids"]), stats["resident_tokens"]
    files = disk_files()
    assert len(files) == stats["indexed_blocks"] * TP
    result = {"blocks": stats["indexed_blocks"], "tokens": stats["resident_tokens"], "files": len(files),
              "file_bytes": sum(path.stat().st_size for path in files), "seconds": time.monotonic() - before}
    save(ROOT / "populate.json", result)
    print("KV population", result, flush=True)


def disk_files():
    return sorted(Path(os.environ.get("SGLANG_FUSIONRAG_CACHE_DIR", "ssd")).glob("v1/*/tp8-rank*/*/kv.pt"))


def restore_dram(data):
    expected = sum(len(block["target_ids"]) for block in data["blocks"].values())
    before = state()["fusionrag_storage_bench"]
    assert before["indexed_blocks"] == len(data["blocks"]) + 1
    if before["resident_tokens"] in (expected, expected + len(data["head_ids"])):
        return

    async def walk():
        sem = asyncio.Semaphore(8)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=480)) as session:
            async def restore(row):
                body = make_body(data, row, "dram", tag, [], tokens=1)
                async with sem:
                    _, _, _, _, n, meta, _ = await stream(session, body)
                    assert n == 1
                    check_response(meta, row, "dram", 0)

            for i, job in enumerate(asyncio.as_completed([restore(row) for row in data["rows"]]), 1):
                await job
                if i % 100 == 0:
                    print(f"Restored DRAM through {i}/{len(data['rows'])} requests", flush=True)

    started = time.monotonic()
    tag = "restore-" + uuid.uuid4().hex[:8]
    prime(data, "dram", tag)
    set_control({**PREFILL_POLICY, "tpot_coalescing": [0, 0],
                 "fusionrag_document_prefetch": True, "fusionrag_document_staging": True})
    asyncio.run(walk())
    after = state()["fusionrag_storage_bench"]
    assert after["resident_tokens"] == expected
    save(ROOT / "dram-materialization.json", {"resident_before": before["resident_tokens"],
                                              "resident_after": after["resident_tokens"],
                                              "seconds": time.monotonic() - started})


def set_tier(mode, policy=None):
    cold = mode == "ssd"
    set_control({**(policy or PREFILL_POLICY), "fusionrag_document_prefetch": cold, "fusionrag_document_staging": cold})
    if cold:
        set_control({"fusionrag_reset_host": True})
    cache = state()["fusionrag_storage_bench"]
    assert cache["document_prefetch_enabled"] is cold and cache["document_staging_enabled"] is cold
    if cold:
        assert cache["resident_tokens"] == 0
    else:
        assert cache["resident_tokens"] > 0
    return cache


def collect_trace(out, name):
    trace = Path(f"trace-{name}.csv")
    trace.rename(out / "scheduler.csv")
    if name.startswith("profile-"):
        for rank in range(TP):
            for suffix in ("", "-copies"):
                path = Path(f"profile-{name}-rank{rank}{suffix}.jsonl")
                path.rename(out / f"rank{rank}{suffix}.jsonl")


def run_arm(data, mode, rate, limit, selector, executor, profile, label=None, coalescing=(32, 32),
            prefetch_limits=None):
    suffix = f"-{label}" if label else ""
    name = f"{'profile-' if profile else ''}{mode}-r{rate:g}{suffix}"
    out = ROOT / name
    if (out / "trial.json").exists():
        result = json.loads((out / "trial.json").read_text())
        if result["ok"] == limit:
            return result
    if out.exists():
        out.rename(out.with_name(f"{name}.partial-{time.time_ns()}"))
    out.mkdir()
    tag = f"{name}-{uuid.uuid4().hex[:8]}"
    if mode == "official":
        with urlopen(Request(f"{BASE}/flush_cache?timeout=30", data=b"{}", method="POST"), timeout=40) as response:
            assert response.status == 200
    prime(data, mode, tag)
    if mode != "official":
        cache_before = set_tier(mode, {**PREFILL_POLICY, "tpot_coalescing": list(coalescing)})
        assert cache_before["indexed_blocks"] == len(data["blocks"]) + 1
        expected_tokens = sum(len(block["target_ids"]) for block in data["blocks"].values())
        if mode == "dram":
            assert cache_before["resident_tokens"] in (expected_tokens, expected_tokens + len(data["head_ids"]))
        if mode == "ssd":
            if prefetch_limits is not None:
                set_control({"fusionrag_document_prefetch_limits": prefetch_limits})
                assert state()["fusionrag_storage_bench"]["document_prefetch_limits"] == prefetch_limits
            page_state = prepare_file_pages(disk_files(), "cold")
            assert page_state["fraction"] < .0001, page_state
            save(out / "initial-page-state.json", page_state)
        set_control({"tpot_trace_label": name})
    print(f"Starting {name}: {limit} requests", flush=True)
    started = time.time()
    host_before = host_state()
    try:
        result = asyncio.run(replay(data, mode, rate, tag, out, selector, executor, limit))
    finally:
        if mode != "official":
            set_control({"tpot_trace_label": ""})
            collect_trace(out, name)
    result.update(started_epoch=started, ended_epoch=time.time(), profile_enabled=profile,
                  host_before=host_before, host_after=host_state())
    if mode != "official":
        after = state()["fusionrag_storage_bench"]
        restore = summarize_restore(after["restore_samples"][len(cache_before["restore_samples"]):])
        result.update(coalescing=list(coalescing),
                      selector_workers=selector.count if executor is None else WORKERS,
                      resident_tokens_before=cache_before["resident_tokens"],
                      resident_tokens_after=after["resident_tokens"], storage=restore,
                      prefetch={key: after["document_prefetch"][key] - cache_before["document_prefetch"][key]
                                for key in after["document_prefetch"]})
        if executor is None:
            result["selector_config"] = selector.configuration
        if prefetch_limits is not None:
            result["prefetch_limits"] = prefetch_limits
        if result["ok"] == len(data["rows"]):
            if mode == "ssd":
                assert (restore["restored_blocks"], restore["restored_tokens"]) == (len(data["blocks"]), expected_tokens)
            else:
                assert restore["restore_calls"] == 0, restore
    save(out / "trial.json", result)
    print(f"Finished {name} {result['ok']}/{limit} at {result['req_s']:.3f} req/s", flush=True)
    if result["ok"] != limit:
        raise RuntimeError(f"{name}: incomplete, see requests.jsonl")
    return result


def load_selector(data):
    import torch
    from selector import Selector

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    before = time.monotonic()
    selector = [Selector(data["blocks"]) for _ in range(WORKERS)]
    all_keys = list(data["blocks"])
    assignments = [dict((key, data["blocks"][key]) for key in all_keys[i::WORKERS]) for i in range(WORKERS)]

    def prepare(worker, blocks):
        worker.bank = blocks
        worker.prepare()
        worker.bank = data["blocks"]
        return worker.keys

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(prepare, worker, assigned) for worker, assigned in zip(selector, assignments)]
        keys = {}
        for future in futures:
            keys.update(future.result())
    for worker in selector:
        worker.keys = keys
    save(ROOT / "draft-key-precompute.json", {"seconds": time.monotonic() - before,
                                              "blocks": len(keys), "workers": WORKERS})
    print(f"Prepared {len(keys)} CPU draft key blocks in {time.monotonic()-before:.1f} seconds", flush=True)
    return SelectorPool(selector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("warm", "restore", "official", "ours", "dram", "ssd"))
    parser.add_argument("--rates", type=float, nargs="+", default=(2.0, 4.0))
    parser.add_argument("--profile-rate", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--micro", action="store_true")
    args = parser.parse_args()
    assert BASE.endswith(":30014"), BASE
    assert 2 <= args.limit <= 500 and all(rate > 0 for rate in args.rates)
    data = json.loads(Path("workload.json").read_text())
    if args.mode == "warm":
        wait_ready()
        warm(data)
        return
    if args.mode == "restore":
        wait_ready()
        restore_dram(data)
        return
    selector = None if args.mode == "official" else load_selector(data)
    wait_ready()
    save(ROOT / f"server-config-{args.mode}.json", config(args.mode))
    tiers = ("dram", "ssd") if args.mode == "ours" else (args.mode,)
    with ThreadPoolExecutor(WORKERS) as executor:
        for mode in tiers:
            for rate in args.rates:
                run_arm(data, mode, rate, args.limit, selector, executor,
                        profile=mode != "official" and rate == args.profile_rate)
    if args.micro:
        assert selector is not None
        from traffic_micro import run_micro

        worker = selector.available.get()
        run_micro(data, worker, ROOT / "profile-serial-micro")


if __name__ == "__main__":
    main()
