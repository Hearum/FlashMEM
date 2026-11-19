"""Replay capped industrial RAG requests with online selection."""

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import shutil
import time
from urllib.request import Request, urlopen
import uuid

import aiohttp

from bench_prefix import BASE, monitor, stream, summary
from run_traffic import disk_files as rank_disk_files, host_state, make_body, order_and_times, prime, wait_ready
from selector_open_loop import digest_mask, initialize, payload, ping
from selector_workers import pick_cpus, placement
from storage_replay_prefix import api, prepare_file_pages, save, set_control, state, summarize_restore

TP = 8
OUTPUT = 32


def cache_key(text):
    return sha256(f"is_preprocess_cache=False text_without_prefix={text}".encode()).hexdigest()


def disk_files():
    if os.environ.get("SGLANG_FUSIONRAG_REPLICATED_MLA") == "1":
        root = Path(os.environ["SGLANG_FUSIONRAG_CACHE_DIR"])
        return sorted(path for path in root.glob("v1/*/tp8-replicated-mla/*/kv.pt")
                      if not path.parent.name.startswith(".") and (path.parent / "metadata.json").is_file())
    return rank_disk_files()


def prime_shared_dram(files):
    root = Path(os.environ["SGLANG_FUSIONRAG_DRAM_CACHE_DIR"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    needed = sum(file.stat().st_size for file in files)
    pending = []
    for source in files:
        relative = source.relative_to(source.parents[2])
        target = root / relative
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            pending.append((source, target))
    free = shutil.disk_usage(root).free
    assert free >= sum(source.stat().st_size for source, _ in pending), (free, needed)

    def copy_one(pair):
        source, target = pair
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".copy")
        shutil.copyfile(source, temp)
        temp.replace(target)

    with ThreadPoolExecutor(max_workers=12) as workers:
        for _ in workers.map(copy_one, pending):
            pass
    assert all((root / source.relative_to(source.parents[2])).is_file() for source in files)
    return dict(files=len(files), bytes=needed, new_files=len(pending), dramsource="tmpfs")


def prepare_workload(path, cache_plan, rows_file=None):
    data = json.loads(path.read_text())
    if rows_file is not None:
        identifiers = set(json.loads(rows_file.read_text()))
        data["rows"] = [row for row in data["rows"] if row["qid"] in identifiers]
        assert len(data["rows"]) == len(identifiers)
        used = {key for row in data["rows"] for key in row["blocks"]}
        data["blocks"] = {key: data["blocks"][key] for key in sorted(used)}
    if cache_plan is None:
        return data
    admitted = set(json.loads(cache_plan.read_text())["blocks"])
    bank = data["blocks"]
    for row in data["rows"]:
        original = row["blocks"]
        length = next((i for i, key in enumerate(original) if key not in admitted), len(original))
        row["full_history_len"] = row["summary_len"]
        row["blocks"] = original[:length]
        row["summary_len"] = sum(len(bank[key]["target_ids"]) for key in row["blocks"])
    used = {key for row in data["rows"] for key in row["blocks"]}
    data["blocks"] = {key: bank[key] for key in sorted(used)}
    assert used == admitted and all(row["input_ids"] for row in data["rows"])
    return data


class Selectors:
    def __init__(self, args, data):
        device = "cpu" if args.selector == "cpu" else "cuda"
        sockets, _ = pick_cpus(45, sample_s=2)
        if device == "cuda":
            sockets = [sockets[args.gpu_numa]]
        cores = placement(sockets, args.workers, args.threads)
        key_filter = args.cache_plan
        if getattr(args, "rows", None) is not None:
            key_filter = Path(args.output) / "selector-keys.json"
            key_filter.parent.mkdir(parents=True, exist_ok=True)
            save(key_filter, {"blocks": sorted(data["blocks"])})
        self.pools = [ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=initialize,
                                         initargs=(args.workload, args.keys, seat, device, None, key_filter))
                      for seat in cores]
        defaults = [payload(row) for row in data["rows"] if 200 < row["summary_len"] < 5000][:3]
        warm_keys = os.environ.get("BENCH_PREFAULT_DRAFT_KEYS", "1") == "1"
        self.placements = [job.result() for job in [pool.submit(ping, defaults, warm_keys) for pool in self.pools]]

    def close(self):
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def queue(self):
        free = asyncio.Queue()
        for pool in self.pools:
            free.put_nowait(pool)
        return free


def select_mask(row):
    import selector_open_loop

    start = time.perf_counter()
    model = selector_open_loop._model
    mask, detail = model.select(row, 0.30, profile=model.device.type == "cuda")
    return start, time.perf_counter(), mask, detail


def populate(data, requested, subset=False):
    cache = state()["fusionrag_storage_bench"]
    shared = cache.get("replicated_document_kv", False)
    expected = sum(len(b["target_ids"]) for b in data["blocks"].values())
    if not subset and cache["indexed_blocks"] == len(data["blocks"]) + 1:
        if not shared and cache["resident_tokens"] != expected + len(data["head_ids"]):
            restore_dram(data, requested)
        print(f"Existing document cache has {cache['indexed_blocks']} blocks", flush=True)
        return
    assert shared or cache["indexed_blocks"] == 0, "Use an isolated cache root for an incomplete population"
    set_control({"fusionrag_document_prefetch": False, "fusionrag_document_staging": False,
                 "tpot_coalescing": [0, 0], "fusionrag_decode_interval": 0})
    started = time.perf_counter()
    existing = {path.parent.name for path in disk_files()} if shared else set()

    def already_saved(text):
        return cache_key(text) in existing

    async def write():
        sem = asyncio.Semaphore(requested)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            async def save_block(text, ids, prefix, prefix_ids):
                body = dict(text=prefix + text, input_ids=prefix_ids + ids,
                            sampling_params=dict(temperature=0, max_new_tokens=0),
                            fusionrag_params=dict(save_cache=True, save_raw_cache=True,
                                                  save_preprocess_cache=False, prefix_prompt=prefix,
                                                  prefix_prompt_list=[prefix] if prefix else [],
                                                  prefix_prompt_ids=prefix_ids,
                                                  prefix_prompt_ids_list=[prefix_ids] if prefix else [],
                                                  prompt_ids_list=[prefix_ids, ids] if prefix_ids else [ids]))
                async with sem, session.post(f"{BASE}/generate", json=body) as response:
                    response.raise_for_status()
                    result = await response.json()
                    assert result["meta_info"]["prompt_tokens"] == len(prefix_ids) + len(ids)

            head, head_ids = data["head"], data["head_ids"]
            if not already_saved(head):
                await save_block(head, head_ids, "", [])
            pending = [b for b in data["blocks"].values() if not already_saved(b["text"])]
            for start in range(0, len(pending), 128):
                jobs = [asyncio.create_task(save_block(b["text"], b["target_ids"], head, head_ids))
                        for b in pending[start:start + 128]]
                try:
                    await asyncio.gather(*jobs)
                finally:
                    for job in jobs:
                        job.cancel()
                    await asyncio.gather(*jobs, return_exceptions=True)
                print(f"Persisted {min(start + 128, len(pending))}/{len(pending)} document blocks", flush=True)

    asyncio.run(write())
    cache = state()["fusionrag_storage_bench"]
    if not subset:
        assert cache["indexed_blocks"] == len(data["blocks"]) + 1
    if not shared:
        assert cache["resident_tokens"] == expected + len(data["head_ids"])
    files = disk_files()
    assert len(files) == (1 if shared else TP) * cache["indexed_blocks"], (len(files), cache["indexed_blocks"])
    if subset:
        required = {cache_key(block["text"]) for block in data["blocks"].values()}
        required.add(cache_key(data["head"]))
        files = [path for path in files if path.parent.name in required]
        assert {path.parent.name for path in files} == required
    result = dict(blocks=len(files) if shared else len(files) // TP,
                  indexed_blocks=cache["indexed_blocks"], tokens=expected + len(data["head_ids"]),
                  pinned_staging_tokens=cache["resident_tokens"], bytes=sum(file.stat().st_size for file in files),
                  seconds=time.perf_counter() - started)
    return result


def restore_dram(data, workers):
    """Load the bounded cache in requests without changing the measured inputs."""
    set_control(dict(tpot_coalescing=[0, 0], fusionrag_decode_interval=0,
                     fusionrag_storage_read_mode="mapped", fusionrag_document_prefetch=True,
                     fusionrag_document_staging=True))
    tag = "warm-dram-" + uuid.uuid4().hex[:8]
    prime(data, "dram", tag)

    async def restore():
        sem = asyncio.Semaphore(workers)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            async def one(key, block):
                head = data["head_ids"]
                ids = block["target_ids"]
                row = dict(qid=key, blocks=[key], input_ids=head + ids + head[:1], summary_len=len(ids))
                body = make_body(data, row, "dram", tag, [], tokens=1)
                async with sem:
                    _, _, _, _, count, meta, _ = await stream(session, body)
                    assert count == 1 and meta["cached_tokens"] == len(head) + len(ids)
            await asyncio.gather(*(one(key, block) for key, block in data["blocks"].items()))

    asyncio.run(restore())


def runtime_options(args, data):
    config = api("/server_info")
    internal = config.get("internal_states", [{}])[0]
    pool = internal.get("memory_usage", config.get("memory_usage", {}))
    tokens = pool.get("token_capacity", config.get("max_total_num_tokens"))
    length = max(len(row["input_ids"]) for row in data["rows"])
    assert tokens and tokens > length + OUTPUT + 32, (tokens, length)
    assert config["tp_size"] == TP
    if args.mode == "official":
        assert str(config.get("version", "")).split("+", 1)[0] == "0.5.23", config.get("version")
        expected_backend = os.environ.get("BENCH_EXPECTED_ATTN_BACKEND", "fa3")
        assert config["attention_backend"] == expected_backend, config["attention_backend"]
        assert not config["disable_decode_cuda_graph"]
        graph = os.environ.get("BENCH_OFFICIAL_PREFILL_GRAPH", "default")
        if graph in {"on", "off"}:
            assert config["disable_prefill_cuda_graph"] == (graph == "off")
        if os.environ.get("BENCH_OFFICIAL_DEFAULT_SCHEDULER") == "1":
            assert not config["disable_overlap_schedule"] and not config["enable_mixed_chunk"]
    else:
        assert config["attention_backend"] == "triton"
        decode_backend = config.get("decode_attention_backend") or config["attention_backend"]
        assert decode_backend == os.environ.get("PIC_DECODE_BACKEND", "triton"), decode_backend
    assert not config["disable_cuda_graph"] and not config["disable_radix_cache"]
    assert not config.get("allow_auto_truncate", False)
    assert config["enable_hierarchical_cache"] == (args.mode != "official")
    assert config["chunked_prefill_size"] == args.chunk_size
    return {key: config.get(key) for key in (
        "version", "tp_size", "attention_backend", "prefill_attention_backend", "decode_attention_backend",
        "hicache_io_backend", "triton_attention_num_kv_splits", "disable_radix_cache", "disable_cuda_graph",
        "disable_prefill_cuda_graph", "disable_decode_cuda_graph", "disable_overlap_schedule",
        "enable_mixed_chunk", "schedule_policy", "enable_hierarchical_cache", "chunked_prefill_size", "page_size",
        "context_length", "max_running_requests", "max_prefill_tokens", "kv_cache_dtype",
        "dtype", "quantization", "cpu_offload_gb", "mem_fraction_static", "cuda_graph_max_bs",
        "cuda_graph_max_bs_decode", "hicache_size", "hicache_mem_layout", "fusionrag_cache_size",
    )} | {"actual_kv_token_slots": tokens, "max_input_tokens": length}


async def run_request(data, row, args, session, free, tag, scheduled):
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    entered = time.perf_counter()
    record = dict(qid=row["qid"], arrival=scheduled, arrival_lag_ms=(entered - scheduled) * 1000)
    mask = []
    try:
        if free is not None and row["blocks"]:
            pool = await free.get()
            try:
                started, finished, mask, stages = await asyncio.wrap_future(pool.submit(select_mask, payload(row)))
            finally:
                free.put_nowait(pool)
            record.update(stages, selector_queue_ms=(started - entered) * 1000,
                          selector_start_clock=started, selector_finish_clock=finished,
                          mask_sha256=digest_mask(mask))
        else:
            record.update(selected_summary=0, select_ms=0.0, selector_queue_ms=0.0)
        body = make_body(data, row, args.mode, tag, mask)
        sent, first, last, end, count, meta, output = await stream(session, body)
        assert count == OUTPUT
        assert meta["prompt_tokens"] == len(row["input_ids"])
        cached = meta["cached_tokens"]
        selected = record["selected_summary"]
        if args.mode != "official":
            assert cached == row["head_len"] + row["summary_len"], (cached, row["summary_len"])
            hits = meta.get("cached_tokens_details") or {}
            if os.environ.get("SGLANG_FUSIONRAG_REPLICATED_MLA") == "1" and row["summary_len"]:
                assert hits.get("device") == row["head_len"] and hits.get("host") == row["summary_len"], hits
            if row["summary_len"]:
                assert abs(meta["recomputation_rate"] - selected / row["summary_len"]) < 1e-6
        record.update(ok=True, rid=body["rid"], end=end, http_sent=sent, first_token_clock=first,
                      last_token_clock=last, completion_tokens=count, ttft_ms=(first - scheduled) * 1000,
                      tpot_ms=(last - first) * 1000 / (count - 1), e2e_ms=(end - scheduled) * 1000,
                      engine_ttft_ms=(first - sent) * 1000,
                      output_sha256=sha256(output.encode()).hexdigest(),
                      first_output_id=meta.get("first_output_id"),
                      token_events=meta["token_events"],
                      prompt_tokens=meta["prompt_tokens"], cached_tokens=cached, head_tokens=row["head_len"],
                      history_tokens=row.get("full_history_len", row["summary_len"]),
                      requested_document_blocks=len(row["blocks"]),
                      recomputed_tokens=selected, prefill_tokens=meta["prompt_tokens"] - cached + selected,
                      retractions=meta.get("total_retractions", 0))
        if args.mode == "official":
            native_fields = ("first_token_latency", "request_received_ts", "api_server_dispatch_finish_ts",
                             "queue_time", "forward_entry_time", "prefill_finished_time",
                             "prefill_launch_delay", "prefill_wait_latency", "prefill_forward_latency")
            record["official_timing"] = {name: meta[name] for name in native_fields if name in meta}
    except Exception as exc:
        record.update(ok=False, end=time.perf_counter(), error=type(exc).__name__)
        if isinstance(exc, AssertionError):
            record["check"] = str(exc)
    return record


async def replay(data, args, selector, rate, tag, out):
    if args.arrival_seed is None:
        ordered, offsets = order_and_times(data["rows"], rate)
    else:
        ordered = list(data["rows"])
        random.Random(args.arrival_seed).shuffle(ordered)
        rng = random.Random(args.arrival_seed)
        offsets = [0.0]
        for _ in ordered[1:]:
            offsets.append(offsets[-1] + rng.expovariate(rate))
    ordered, offsets = ordered[:args.limit], offsets[:args.limit]
    save(out / "arrivals.json", [(row["qid"], offset) for row, offset in zip(ordered, offsets)])
    free = selector.queue() if selector else None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10_800),
                                    connector=aiohttp.TCPConnector(limit=0)) as session:
        stop = asyncio.Event()
        monitor_task = asyncio.create_task(monitor(session, out / "monitor.jsonl", stop))
        start = time.perf_counter() + 0.25
        requests = [run_request(data, row, args, session, free, tag, start + offset)
                    for row, offset in zip(ordered, offsets)]
        records = []
        try:
            with (out / "requests.jsonl").open("w", buffering=1) as target:
                for task in asyncio.as_completed(requests):
                    result = await task
                    records.append(result)
                    target.write(json.dumps(result, separators=(",", ":")) + "\n")
        finally:
            stop.set()
            await monitor_task
    stats = summary(records, start)
    stats.update(offered_rate=rate, actual_offered_rate=(len(ordered) - 1) / offsets[-1],
                 wall_seconds=max(row["end"] for row in records) - start)
    return stats


def run_trial(data, args, selector, rate):
    out = args.output / f"{args.mode}-{args.selector}-r{rate:g}"
    if (out / "trial.json").exists():
        old = json.loads((out / "trial.json").read_text())
        if old["ok"] == args.limit:
            return
    if out.exists():
        out.rename(out.with_name(out.name + f"-partial-{time.time_ns()}"))
    out.mkdir(parents=True)
    key = out.name + "-" + uuid.uuid4().hex[:8]
    trace = "profile-" + out.name.replace(".", "p")
    before = None
    files = None
    if args.mode == "official":
        with urlopen(Request(f"{BASE}/flush_cache?timeout=30", data=b"{}", method="POST"), timeout=40) as response:
            assert response.status == 200
    else:
        shared = state()["fusionrag_storage_bench"].get("replicated_document_kv", False)
        async_stage = shared or args.mode == "ssd"
        if shared:
            set_control({"fusionrag_reset_host": True})
            files = disk_files()
            required = {cache_key(block["text"]) for block in data["blocks"].values()}
            required.add(cache_key(data["head"]))
            files = [path for path in files if path.parent.name in required]
            assert {path.parent.name for path in files} == required
            if args.mode == "dram":
                save(out / "primed-dram-store.json", prime_shared_dram(files))
            set_control({"fusionrag_document_source": args.mode})
        set_control(dict(tpot_coalescing=args.tpot_coalescing, fusionrag_decode_interval=args.decode_interval,
                         prefill_max_requests=None, fusionrag_storage_read_mode="mapped",
                         fusionrag_document_prefetch=async_stage,
                         fusionrag_document_staging=async_stage))
        if shared or args.mode == "ssd":
            set_control({"fusionrag_document_prefetch_limits": [4, 512, 131072]})
        if args.mode == "ssd":
            if not shared:
                set_control({"fusionrag_reset_host": True})
            page_state = prepare_file_pages(files if files is not None else disk_files(), "cold")
            assert page_state["fraction"] < 0.0001
            save(out / "initial-page-state.json", page_state)
        before = state()["fusionrag_storage_bench"]
        resident = sum(len(block["target_ids"]) for block in data["blocks"].values())
        if shared:
            assert before["resident_tokens"] == 0 and before["document_source"] == args.mode
            assert before["indexed_blocks"] >= len(data["blocks"]) + 1
        elif args.mode == "dram":
            assert before["resident_tokens"] in (resident, resident + len(data["head_ids"]))
        else:
            assert before["resident_tokens"] == 0
    prime(data, args.mode, key)
    if args.mode != "official":
        set_control({"tpot_trace_label": trace})
    environment = host_state()
    try:
        result = asyncio.run(replay(data, args, selector, rate, key, out))
    finally:
        if args.mode != "official":
            set_control({"tpot_trace_label": ""})
            Path(f"trace-{trace}.csv").rename(out / "scheduler.csv")
            for rank in range(TP):
                for suffix in ("", "-copies"):
                    Path(f"profile-{trace}-rank{rank}{suffix}.jsonl").rename(out / f"rank{rank}{suffix}.jsonl")
    result.update(host_before=environment, host_after=host_state(), chunk_size=args.chunk_size,
                  decode_interval=args.decode_interval if args.mode != "official" else None,
                  tpot_coalescing=args.tpot_coalescing if args.mode != "official" else None)
    if selector:
        result["selector_config"] = selector.placements
        after = state()["fusionrag_storage_bench"]
        result["document_source"] = after.get("document_source", args.mode)
        result["document_resident_tokens_after"] = after["resident_tokens"]
        result["prefetch"] = {name: n - before["document_prefetch"][name]
                              for name, n in after["document_prefetch"].items()}
        result["storage"] = summarize_restore(after["restore_samples"][len(before["restore_samples"]):])
    save(out / "trial.json", result)
    print(out.name, result["ok"], "of", args.limit, f"{result['req_s']:.3f} req/s", flush=True)
    assert result["ok"] == args.limit, f"Inspect errors in {out / 'requests.jsonl'}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("populate", "dram", "ssd", "official"))
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--rows", type=Path)
    p.add_argument("--cache-plan", type=Path)
    p.add_argument("--keys", type=Path)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--selector", choices=("cpu", "gpu", "none"), default="none")
    p.add_argument("--rates", type=float, nargs="+", default=[.5, 1])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-numa", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--decode-interval", type=int, default=1)
    p.add_argument("--tpot-coalescing", type=int, nargs=2, default=[0, 0], metavar=("MIN_PREFILLS", "DECODE_STEPS"))
    p.add_argument("--limit", type=int)
    p.add_argument("--arrival-seed", type=int, help="Use a shared seed for request order and Poisson gaps")
    args = p.parse_args()
    assert len(args.rates) > 0 and min(args.rates) > 0
    assert all(0 <= value <= 32 for value in args.tpot_coalescing)
    if args.mode in ("dram", "ssd"):
        assert args.selector != "none" and args.keys
    else:
        assert args.selector == "none"
    wait_ready()
    data = prepare_workload(args.workload, args.cache_plan, args.rows)
    if args.limit is None:
        args.limit = len(data["rows"])
    assert 2 <= args.limit <= len(data["rows"])
    config = runtime_options(args, data)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / f"runtime-{args.mode}-{args.selector}.json", config)
    if args.mode == "populate":
        populated = populate(data, args.workers, subset=args.rows is not None)
        if populated is not None:
            save(args.output / "population.json", populated)
            print(json.dumps(populated), flush=True)
        return
    selector = Selectors(args, data) if args.selector != "none" else None
    try:
        for rate in args.rates:
            run_trial(data, args, selector, rate)
    finally:
        if selector:
            selector.close()

if __name__ == "__main__":
    main()
