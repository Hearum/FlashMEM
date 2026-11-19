"""Paired LoCoMo replay with streamed token times and an online selector."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import SimpleQueue
import random
import statistics
import subprocess
import time
from urllib.parse import urlsplit
import uuid

import aiohttp

BASE = os.getenv("BENCH_BASE", "http://127.0.0.1:30014")
PREFILL_BASE = os.getenv("BENCH_PREFILL")
RATE = .3


def body_for(row, arm, blocks, run, tokens, mask=None, natural=False):
    body = dict(
        rid=f"{run}-{row['qid']}",
        extra_key=run if arm == "radix" else f"{run}-{row['qid']}",
        stream=True,
        sampling_params=dict(temperature=0, max_new_tokens=tokens, ignore_eos=not natural),
    )
    if arm in {"dense", "radix"}:
        body["input_ids"] = row["input_ids"]
    else:
        prefix = [row["head"], *(blocks[key]["text"] for key in row["blocks"])]
        parts = [*prefix, row["tail"]]
        if arm == "segmented":
            prefix = []
            mask = []
        if arm == "reuse100":
            mask = list(range(row["head_len"] + row["summary_len"]))
        body.update(
            text="".join(parts),
            fusionrag_params=dict(
                save_cache=False, load_preprocess_cache=False, load_raw_cache=bool(prefix),
                rope=True, use_fusion_rag=True, prefix_prompt="".join(prefix),
                prompt_list=parts, prefix_prompt_list=prefix, recompute_idx=mask,
            ),
        )
        if arm in {"fixed_ids", "online_ids"}:
            prefix_ids = [row["input_ids"][:row["head_len"]]]
            prefix_ids.extend(blocks[key]["target_ids"] for key in row["blocks"])
            prefix_count = row["head_len"] + row["summary_len"]
            body.pop("text")
            body["input_ids"] = row["input_ids"]
            body["fusionrag_params"].update(
                prompt_ids_list=[*prefix_ids, row["input_ids"][prefix_count:]],
                prefix_prompt_ids_list=prefix_ids,
                prefix_prompt_ids=row["input_ids"][:prefix_count],
                prefix_cache_ids=[],
            )
        if arm == "online":
            # Reuse the shared instruction head through Radix and load the
            # remaining blocks through the document cache.
            body["extra_key"] = run
            body["fusionrag_params"].update(
                cache_radix_prefix=True,
                prefix_cache_prompt=row["head"],
                prefix_prompt="".join(prefix[1:]),
                prefix_prompt_list=prefix[1:],
                recompute_idx=[index for index in mask if index >= row["head_len"]],
            )
    return body


async def warm(session, rows, blocks):
    jobs = {}
    for row in rows:
        for key in row["blocks"]:
            jobs[(row["head"], blocks[key]["text"])] = None
    print("prewarm document/head pairs", len(jobs), flush=True)
    sem = asyncio.Semaphore(16)

    async def save(head, document):
        parts = [head] if document is None else [head, document]
        prefix = parts[:-1]
        body = dict(
            text="".join(parts), sampling_params=dict(temperature=0, max_new_tokens=0),
            fusionrag_params=dict(save_cache=True, save_raw_cache=True,
                                  save_preprocess_cache=False, prefix_prompt="".join(prefix),
                                  prompt_list=parts, prefix_prompt_list=prefix),
        )
        if PREFILL_BASE:
            body.update(bootstrap_host=os.environ["BENCH_BOOTSTRAP_HOST"],
                        bootstrap_port=int(os.environ["BENCH_BOOTSTRAP_PORT"]),
                        bootstrap_room=uuid.uuid4().int >> 65)
        async with sem, session.post(f"{PREFILL_BASE or BASE}/generate", json=body) as response:
            response.raise_for_status()
            result = await response.json()
            assert "meta_info" in result, result

    start = time.perf_counter()
    await asyncio.gather(*(save(head, None) for head in {head for head, _ in jobs}))
    await asyncio.gather(*(save(head, doc) for head, doc in jobs))
    print("prewarm finished seconds", round(time.perf_counter() - start, 3), flush=True)


async def stream(session, body):
    sent = time.perf_counter()
    first = last = None
    count = 0
    token_events = []
    first_id = None
    meta = {}
    text = ""
    async with session.post(f"{BASE}/generate", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} {(await response.text())[:300]}")
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            meta = item["meta_info"]
            received = time.perf_counter()
            current = meta["completion_tokens"]
            if current > count:
                token_events.append((current, received))
                if first is None:
                    first = received
                    ids = item.get("output_ids", [])
                    first_id = ids[0] if ids else None
                last, count = received, current
            text = item.get("text", text)
    end = time.perf_counter()
    assert first is not None and meta.get("finish_reason"), meta
    return sent, first, last, end, count, dict(meta, first_output_id=first_id, token_events=token_events), text



async def stream_pd(session, body, input_ids):
    bootstrap = dict(bootstrap_host=urlsplit(PREFILL_BASE).hostname, bootstrap_port=31431,
                     bootstrap_room=uuid.uuid4().int >> 65)
    prefill = dict(body, **bootstrap, stream=False)
    decode = {k: body[k] for k in ("rid", "extra_key", "sampling_params", "stream")}
    decode.update(bootstrap, input_ids=input_ids)

    async def send_prefill():
        async with session.post(f"{PREFILL_BASE}/generate", json=prefill) as response:
            if response.status != 200:
                raise RuntimeError(f"prefill HTTP {response.status} {(await response.text())[:240]}")
            data = await response.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data["meta_info"]

    sent = time.perf_counter()
    ptask = asyncio.create_task(send_prefill())
    dtask = asyncio.create_task(stream(session, decode))
    try:
        meta_p, (_, first, last, end, count, meta_d, text) = await asyncio.gather(ptask, dtask)
    finally:
        for task in (ptask, dtask):
            if not task.done():
                task.cancel()
        await asyncio.gather(ptask, dtask, return_exceptions=True)
    return sent, first, last, end, count, dict(meta_p, decode_meta=meta_d), text


class SelectorPool:
    def __init__(self, workers):
        self.available = SimpleQueue()
        for worker in workers:
            self.available.put(worker)

    def select(self, row, rate):
        worker = self.available.get()
        try:
            return worker.select(row, rate)
        finally:
            self.available.put(worker)


def selector_call(selector, row):
    start = time.perf_counter()
    mask, stages = selector.select(row, RATE)
    return start, time.perf_counter(), mask, stages


async def run_one(session, row, arm, blocks, run, tokens, masks, selector, executor,
                  scheduled, natural=False):
    entry = time.perf_counter()
    clock_offset = time.time() - entry
    rec = dict(qid=row["qid"], episode=row["episode"], arm=arm, arrival=scheduled,
               dispatch_lag_ms=(entry - scheduled) * 1e3)
    try:
        mask = masks[row["qid"]]["mask"] if arm in {"fixed", "fixed_ids", "online", "online_ids"} else None
        if arm in {"online", "online_ids"}:
            queued = time.perf_counter()
            started, finished, online_mask, stages = await asyncio.get_running_loop().run_in_executor(
                executor, selector_call, selector, row
            )
            assert online_mask == mask, "Online and precomputed masks differ"
            rec.update(stages, selector_queue_ms=(started - queued) * 1e3,
                       selector_start_clock=started, selector_finish_clock=finished,
                       selector_start_epoch=started + clock_offset,
                       selector_finish_epoch=finished + clock_offset)
        body = body_for(row, arm, blocks, run, tokens, mask, natural)
        if PREFILL_BASE:
            sent, first, last, end, count, meta, text = await stream_pd(session, body, row["input_ids"])
        else:
            sent, first, last, end, count, meta, text = await stream(session, body)
        expected = row["head_len"] + row["summary_len"]
        assert meta["prompt_tokens"] == len(row["input_ids"]), ("tokenization", meta)
        cached = meta["cached_tokens"]
        if arm == "online":
            assert cached == expected, ("loaded prefix length", cached, expected)
            hits = meta["cached_tokens_details"]
            head = row["head_len"]
            summary_len = row["summary_len"]
            assert hits["device"] == head and hits["host"] == summary_len, hits
            recomputed = len(body["fusionrag_params"]["recompute_idx"])
            assert recomputed == rec["selected_summary"], rec
            assert abs(meta["recomputation_rate"] - recomputed / summary_len) < 1e-6, meta
            rec["system_prefix_hit_tokens"] = hits["device"]
        elif arm in {"fixed", "fixed_ids", "online_ids", "reuse100"}:
            assert cached == expected, ("cache hit", cached, expected)
            recomputed = expected if arm == "reuse100" else len(mask)
            assert abs(meta["recomputation_rate"] - recomputed / cached) < 1e-6, meta
        else:
            if arm in {"dense", "segmented"}:
                assert cached == 0, ("unexpected radix hit", cached)
            recomputed = 0
        if not natural:
            assert count == tokens, ("output length", count, tokens)
        rec.update(
            ok=True, end=end, http_sent=sent, first_token_clock=first, last_token_clock=last, completion_tokens=count,
            ttft_ms=(first - scheduled) * 1e3, engine_ttft_ms=(first - sent) * 1e3,
            tpot_ms=(last - first) * 1e3 / (count - 1) if count > 1 else None,
            e2e_ms=(end - scheduled) * 1e3, prompt_tokens=meta["prompt_tokens"],
            cached_tokens=cached, prefill_tokens=meta["prompt_tokens"] - cached + recomputed,
            recomputed_tokens=recomputed, meta=meta,
        )
        if natural:
            rec["answer"] = text
    except Exception as exc:
        rec.update(ok=False, end=time.perf_counter(), error=f"{type(exc).__name__}: {exc}"[:500])
    return rec


async def monitor(session, path, stop=None):
    wanted = ("sglang:token_usage", "sglang:num_running_reqs", "sglang:num_queue_reqs",
              "sglang:num_used_tokens", "sglang:load_back", "sglang:evicted_tokens",
              "sglang:fusionrag_load_back", "sglang:fusionrag_evicted_tokens")
    with path.open("w", buffering=1) as out:
        while stop is None or not stop.is_set():
            t = time.perf_counter()
            try:
                async with session.get(f"{BASE}/metrics", timeout=2) as r:
                    payload = await r.text()
                gpu = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                gpu_out, _ = await gpu.communicate()
                out.write(json.dumps(dict(
                    time=t, gpu=gpu_out.decode().strip().splitlines(),
                    metrics=[line for line in payload.splitlines() if line.startswith(wanted)],
                )) + "\n")
            except Exception as exc:
                out.write(json.dumps(dict(time=t, error=repr(exc))) + "\n")
            if stop is None:
                await asyncio.sleep(1)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass


def summary(rows, start):
    good = [r for r in rows if r["ok"]]
    wall = max(r["end"] for r in rows) - start
    result = dict(n=len(rows), ok=len(good), duration_s=wall, req_s=len(good) / wall,
                  output_tok_s=sum(r["completion_tokens"] for r in good) / wall)
    for field in ("ttft_ms", "engine_ttft_ms", "tpot_ms", "e2e_ms", "select_ms",
                  "selector_queue_ms", "prefill_tokens"):
        x = sorted(r[field] for r in good if r.get(field) is not None)
        if x:
            result[field] = dict(mean=statistics.fmean(x), **{
                f"p{q}": x[min(len(x) - 1, int(q / 100 * len(x)))] for q in (50, 95, 99)
            })
    result["goodput_ttft_2s_tpot_150ms"] = sum(
        r["ttft_ms"] <= 2000 and (r["tpot_ms"] is None or r["tpot_ms"] <= 150) for r in good
    ) / wall
    result["goodput_ttft_1s_tpot_100ms"] = sum(
        r["ttft_ms"] <= 1000 and (r["tpot_ms"] is None or r["tpot_ms"] <= 100) for r in good
    ) / wall
    return result


async def main(args, data, masks, selector, executor):
    rng = random.Random(20261111)
    rows = list(data["rows"])
    rng.shuffle(rows)
    rows = rows[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=360)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        if args.warm or args.mode == "warm":
            await warm(session, rows if args.warm_subset else data["rows"], data["blocks"])
        if args.mode == "warm":
            return
        if args.mode == "micro":
            arms = args.micro_arms
            recs = []
            with (out / "requests.jsonl").open("w", buffering=1) as fd:
                for i, row in enumerate(rows):
                    order = list(arms)
                    rng.shuffle(order)
                    for arm in order:
                        rec = await run_one(session, row, arm, data["blocks"], f"{args.out}-{i}-{arm}",
                                            args.tokens, masks, selector, executor,
                                            time.perf_counter(), args.natural)
                        recs.append(rec)
                        fd.write(json.dumps(rec) + "\n")
                        if not rec["ok"]:
                            raise RuntimeError(rec["error"])
            stats = {arm: summary([r for r in recs if r["arm"] == arm], recs[0]["arrival"])
                     for arm in arms}
        else:
            stats = {}
            if args.arrival == "poisson":
                arrivals = [0.0]
                arrival_rng = random.Random(args.seed)
                for _ in rows[1:]:
                    arrivals.append(arrivals[-1] + arrival_rng.expovariate(args.rate))
            else:
                arrivals = [i / args.rate for i in range(len(rows))]
            for index, arm in enumerate(args.arms):
                tag = args.namespace or f"{args.out}-{index}-{arm}"
                for i, row in enumerate(rows[:args.probe_requests]):
                    probe = await run_one(session, row, arm, data["blocks"], f"{tag}-warm-{i}",
                                          args.tokens, masks, selector, executor,
                                          time.perf_counter(), args.natural)
                    if not probe["ok"]:
                        raise RuntimeError(probe["error"])
                watcher = asyncio.create_task(monitor(session, out / f"{arm}.resources.jsonl"))
                start = time.perf_counter() + .1
                tasks = []
                for offset, row in zip(arrivals, rows):
                    scheduled = start + offset
                    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                    tasks.append(asyncio.create_task(run_one(
                        session, row, arm, data["blocks"], tag, args.tokens, masks,
                        selector, executor, scheduled, args.natural,
                    )))
                recs = await asyncio.gather(*tasks)
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                with (out / f"{arm}.jsonl").open("w") as fd:
                    for rec in recs:
                        fd.write(json.dumps(rec) + "\n")
                stats[arm] = summary(recs, start)
                stats[arm].update(offered_rps=args.rate, realized_arrival_rps=(len(rows)-1)/arrivals[-1]
                                  if len(rows) > 1 else None, arrival=args.arrival)
                print(arm, json.dumps(stats[arm]), flush=True)
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("warm", "micro", "load"))
    p.add_argument("--out", required=True)
    p.add_argument("--namespace")
    p.add_argument("--warm", action="store_true")
    p.add_argument("--warm-subset", action="store_true")
    p.add_argument("--limit", type=int, default=1540)
    p.add_argument("--rate", type=float, default=6)
    p.add_argument("--arrival", choices=("periodic", "poisson"), default="periodic")
    p.add_argument("--probe-requests", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--arms", nargs="+", default=["dense", "online"])
    p.add_argument("--micro-arms", nargs="+",
                   default=["dense", "segmented", "reuse100", "fixed", "online"])
    p.add_argument("--natural", action="store_true")
    p.add_argument("--selector-device", default="cpu")
    p.add_argument("--selector-workers", type=int, default=1)
    p.add_argument("--mask-file", default="mask-micro.jsonl")
    args = p.parse_args()
    if args.selector_workers < 1 or (args.selector_workers > 1 and args.selector_device != "cpu"):
        p.error("selector workers must be positive and multiple workers require CPU")
    data = json.loads(Path("workload.json").read_text())
    masks = {}
    if args.mode != "warm":
        masks = {r["qid"]: r for line in Path(args.mask_file).open() if (r := json.loads(line))}
    selector = None
    selected_arms = args.micro_arms if args.mode == "micro" else args.arms
    if args.mode != "warm" and {"online", "online_ids"}.intersection(selected_arms):
        import torch
        from selector import Selector
        torch.set_num_threads(int(os.getenv("SELECTOR_THREADS", "4")))
        torch.set_num_interop_threads(1)
        if args.selector_device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(args.selector_device)
            if free < 3 * 2**30:
                raise RuntimeError(f"Selector requires at least 3 GiB free, have {free / 2**30:.2f}")
            torch.cuda.set_per_process_memory_fraction(1.55 * 2**30 / total, args.selector_device)
        keys = torch.load("selector-keys.pt", weights_only=True, map_location="cpu")
        workers = [Selector(data["blocks"], device=args.selector_device)
                   for _ in range(args.selector_workers)]
        for worker in workers:
            worker.keys = keys
        selector = SelectorPool(workers)
    with ThreadPoolExecutor(max_workers=args.selector_workers) as executor:
        asyncio.run(main(args, data, masks, selector, executor))
