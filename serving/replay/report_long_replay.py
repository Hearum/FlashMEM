"""Report serving metrics and additive request-level TTFT for chunked prefill."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


def dist(values):
    values = list(values)
    return dict(mean=statistics.fmean(values), **{
        f"p{p}": quantile(values, p / 100) for p in (50, 95, 99)})


def span(a, b):
    return (b - a) * 1000


def gpu_spans(batch):
    gpu = batch["gpu_ms"]
    start, end = gpu["model_start"], gpu["model_end"]
    copy_start = gpu.get("h2d_start", start)
    copy_end = gpu.get("h2d_end", copy_start)
    return dict(model_stream_ms=end - start, rope_stream_ms=gpu["rope_end"] - gpu["rope_start"],
                h2d_stream_ms=copy_end - copy_start,
                h2d_overlapping_model_ms=max(0, min(end, copy_end) - max(start, copy_start)),
                ssd_scheduler_restore_ms=sum(span(a, b) for a, b, _, _ in batch["ssd"]))


def read_json(path):
    return json.loads(path.read_text())


def request_path(row, batches):
    first, last = batches[0], batches[-1]
    queue = row["selector_queue_ms"]
    execution = row["select_ms"]
    prep = sum(span(b["start"], b["selected"]) for b in batches)
    forward = sum(span(b["selected"], b["ended"]) for b in batches)
    gaps = sum(span(a["ended"], b["start"]) for a, b in zip(batches, batches[1:]))
    queue_to_batch = span(row["http_sent"], first["start"])
    client = span(row["arrival"], row["http_sent"]) - queue - execution
    delivery = span(last["ended"], row["first_token_clock"])
    # Socket delivery and the scheduler's post-send bookkeeping can overlap.
    # Assign the part of the final batch seen by the client, ending at TTFT.
    forward += min(0.0, delivery)
    delivery = max(0.0, delivery)
    path = dict(selector_queue_ms=queue, selector_execution_ms=execution,
                dispatch_and_wait_ms=client + queue_to_batch + gaps,
                scheduling_and_kv_preparation_ms=prep, gpu_batch_and_first_token_ms=forward + delivery)
    assert abs(sum(path.values()) - row["ttft_ms"]) < 0.01
    computed = sum(b["request_computed_tokens"][b["rids"].index(row["rid"])] for b in batches)
    assert computed == row["prefill_tokens"], (row["qid"], computed, row["prefill_tokens"])
    path["ttft_ms"] = row["ttft_ms"]
    return path


def official_request_path(row):
    timing = row["official_timing"]
    received = timing["request_received_ts"]
    dispatch = timing["api_server_dispatch_finish_ts"]
    forward = timing["forward_entry_time"]
    first = timing["first_token_latency"] * 1000
    intake_ms = span(received, dispatch)
    dispatch_to_forward_ms = span(dispatch, forward)
    result = dict(client_and_first_token_delivery_ms=row["ttft_ms"] - first,
                  api_intake_and_dispatch_ms=intake_ms,
                  dispatch_to_first_forward_ms=dispatch_to_forward_ms,
                  first_forward_to_first_token_ms=first - intake_ms - dispatch_to_forward_ms)
    assert abs(sum(result.values()) - row["ttft_ms"]) < 0.01
    assert min(result.values()) >= -1.0, (row["qid"], result)
    return dict(result, ttft_ms=row["ttft_ms"])


def monitor_peaks(root):
    metrics = defaultdict(float)
    path = root / "monitor.jsonl"
    if not path.exists():
        return dict(metrics)
    for row in read(path):
        for index, gpu in enumerate(row.get("gpu", [])):
            memory = int(gpu.split(",", 1)[0])
            name = f"gpu_{index}_memory_mib"
            metrics[name] = max(metrics[name], memory)
        for entry in row.get("metrics", []):
            label, _, rest = entry.partition("{")
            if not rest or label not in {"sglang:token_usage", "sglang:num_used_tokens",
                                         "sglang:num_running_reqs", "sglang:num_queue_reqs"}:
                continue
            value = float(rest.rsplit("}", 1)[-1].strip())
            metrics[label] = max(metrics[label], value)
    return dict(metrics)


def pressure_during_trial(stats):
    """Use cumulative PSI counters to summarize the measured wall-clock window."""
    window_us = stats["wall_seconds"] * 1_000_000
    before = stats.get("host_before", {})
    after = stats.get("host_after", {})
    pressure = {}
    for resource in ("cpu", "io", "memory"):
        path = f"/proc/pressure/{resource}"
        if path not in before or path not in after:
            continue
        start = {line.split()[0]: int(line.rsplit("total=", 1)[1]) for line in before[path].splitlines()}
        finish = {line.split()[0]: int(line.rsplit("total=", 1)[1]) for line in after[path].splitlines()}
        pressure[resource] = {name: (count - start[name]) / window_us for name, count in finish.items()}
    return pressure


def scheduler_intervals(path, rows):
    steps = defaultdict(list)
    intervals = []
    with path.open(newline="") as stream:
        for event in csv.DictReader(stream):
            start, selected, end = (float(event[key]) for key in ("start", "selected", "end"))
            steps[event["mode"]].append((span(start, selected), span(selected, end)))
            intervals.append((start, end, event["mode"]))
    summary = {mode: dict(steps=len(items), scheduling_ms=dist(item[0] for item in items),
                          execution_ms=dist(item[1] for item in items))
               for mode, items in steps.items()}
    starts = [start for start, _, _ in intervals]

    def occupied_windows(row):
        first, last = row["first_token_clock"], row["last_token_clock"]
        lo = max(0, bisect_left(starts, first) - 1)
        hi = bisect_left(starts, last)
        mode_ms = defaultdict(float)
        for start, end, mode in intervals[lo:hi]:
            mode_ms[mode] += max(0.0, span(max(start, first), min(end, last)))
        tokens = row["completion_tokens"] - 1
        return dict(prefill_ms_per_token=mode_ms["prefill"] / tokens,
                    decode_ms_per_token=mode_ms["decode"] / tokens,
                    outside_batch_ms_per_token=(span(first, last) - sum(mode_ms.values())) / tokens)

    all_windows = [occupied_windows(row) for row in rows]
    top = sorted(zip(rows, all_windows), key=lambda pair: pair[0]["tpot_ms"], reverse=True)
    tail_windows = [measurement for _, measurement in top[:max(1, len(rows) // 20)]]
    tpot = {label: {key: dist(item[key] for item in group) for key in group[0]}
            for label, group in (("all_requests", all_windows), ("slowest_5pct", tail_windows))}
    return summary, tpot


def trial(root):
    rows = [row for row in read(root / "requests.jsonl") if row["ok"]]
    arrivals = read_json(root / "arrivals.json")
    assert sorted(row["qid"] for row in rows) == sorted(qid for qid, _ in arrivals)
    assert all(row["completion_tokens"] == 32 for row in rows)
    trial_stats = read_json(root / "trial.json")
    metrics = dict(requests=len(rows), offered_req_s=trial_stats["offered_rate"],
                   actual_offered_req_s=trial_stats["actual_offered_rate"],
                   wall_seconds=trial_stats["wall_seconds"],
                   throughput_req_s=trial_stats["req_s"],
                   output_tok_s=sum(row["completion_tokens"] for row in rows) / trial_stats["wall_seconds"])
    # Preserve the percentile convention used in the earlier serving tables.
    for tag in ("ttft_ms", "tpot_ms", "e2e_ms"):
        ordered = sorted(row[tag] for row in rows)
        metrics[tag] = dict(mean=statistics.fmean(ordered), **{
            f"p{p}": ordered[min(len(ordered) - 1, len(ordered) * p // 100)]
            for p in (50, 95, 99)})
    metrics["computed_prefill_tokens"] = sum(row["prefill_tokens"] for row in rows)
    metrics["input_tokens"] = sum(row["prompt_tokens"] for row in rows)
    metrics["cached_tokens"] = sum(row["cached_tokens"] for row in rows)
    metrics["recomputed_tokens"] = sum(row["recomputed_tokens"] for row in rows)
    metrics["retractions"] = sum(row["retractions"] for row in rows)
    metrics["input_tokens_dist"] = dist(row["prompt_tokens"] for row in rows)
    metrics["prefill_tokens_dist"] = dist(row["prefill_tokens"] for row in rows)
    metrics["cached_tokens_dist"] = dist(row["cached_tokens"] for row in rows)
    metrics["monitor_peaks"] = monitor_peaks(root)
    metrics["host_pressure_fraction"] = pressure_during_trial(trial_stats)
    if rows and all(row.get("official_timing") for row in rows):
        paths = [official_request_path(row) for row in rows]
        metrics["official_ttft_profile"] = {key: dist(path[key] for path in paths) for key in paths[0]}
    for key in ("host_before", "host_after", "prefetch", "storage", "selector_config", "document_source",
                "document_resident_tokens_after", "goodput_ttft_2s_tpot_150ms",
                "goodput_ttft_1s_tpot_100ms"):
        if key in trial_stats:
            metrics[key] = trial_stats[key]
    if metrics.get("document_source") in {"dram", "ssd"}:
        with_history = [row for row in rows if row["history_tokens"]]
        metrics["requests_with_history"] = len(with_history)
        metrics["full_history_hit_requests"] = sum(
            row["cached_tokens"] == row["history_tokens"] + row["head_tokens"] for row in with_history)
        metrics["document_token_references"] = sum(row["history_tokens"] for row in rows)
        metrics["document_chunk_references"] = sum(row.get("requested_document_blocks", 0) for row in rows)
    if (root / "rank0.jsonl").exists():
        ranks = [read(root / f"rank{rank}.jsonl") for rank in range(8)]
        assert all(len(rank) == len(ranks[0]) for rank in ranks)
        first_tokens = {row["rid"]: row["first_token_clock"] for row in rows}
        by_rid = defaultdict(list)
        post_first_token_batches = set()
        post_first_token_compute = 0
        for batch in ranks[0]:
            for rid, computed_tokens in zip(batch["rids"], batch["request_computed_tokens"]):
                # A decode-stage retraction can trigger another prefill under
                # the same request ID after its first token has been delivered.
                if batch["start"] < first_tokens[rid]:
                    by_rid[rid].append(batch)
                else:
                    post_first_token_batches.add(rid)
                    post_first_token_compute += computed_tokens
        assert post_first_token_batches <= {row["rid"] for row in rows if row["retractions"]}
        metrics["requests_with_post_first_token_prefill"] = len(post_first_token_batches)
        metrics["prefill_tokens_after_first_token"] = post_first_token_compute
        paths = [request_path(row, by_rid[row["rid"]]) for row in rows]
        metrics["ttft_profile"] = {key: dist(path[key] for path in paths) for key in paths[0]}
        metrics["ttft_wait_breakdown"] = dict(
            client_to_first_batch_ms=dist(span(row["http_sent"], by_rid[row["rid"]][0]["start"])
                                         for row in rows),
            between_chunks_ms=dist(sum(span(a["ended"], b["start"]) for a, b in
                                       zip(batches, batches[1:])) for batches in by_rid.values()),
        )
        metrics["selector_instrumentation"] = {
            key: dist(row.get(key, 0.0) for row in rows) for key in (
                "query_ms", "score_ms", "draft_key_pack_ms", "draft_key_h2d_ms",
                "keys_rope_ms", "topk_map_ms",
            )
        }
        metrics["prefill_batches"] = len(ranks[0])
        metrics["prefill_chunks_per_request"] = dist(len(items) for items in by_rid.values())
        metrics["scheduler_steps"], metrics["tpot_time_allocation"] = scheduler_intervals(root / "scheduler.csv", rows)
        metrics["largest_prefill_compute_batch"] = max(
            sum(batch["request_computed_tokens"]) for batch in ranks[0])
        assert metrics["largest_prefill_compute_batch"] <= trial_stats.get("chunk_size", 8192)
        micros = []
        request_model_ms = defaultdict(float)
        for rank_group in zip(*ranks):
            assert all(batch["rids"] == rank_group[0]["rids"] for batch in rank_group)
            intervals = [gpu_spans(batch) for batch in rank_group]
            h2d = max(intervals, key=lambda x: x["h2d_stream_ms"])
            micros.append(dict(model_stream_ms=max(x["model_stream_ms"] for x in intervals),
                               rope_stream_ms=max(x["rope_stream_ms"] for x in intervals),
                               ssd_scheduler_restore_ms=max(x["ssd_scheduler_restore_ms"] for x in intervals),
                               h2d_stream_ms=h2d["h2d_stream_ms"],
                               h2d_overlapping_model_ms=h2d["h2d_overlapping_model_ms"]))
            for rid in rank_group[0]["rids"]:
                if rank_group[0]["start"] < first_tokens[rid]:
                    request_model_ms[rid] += micros[-1]["model_stream_ms"]
        metrics["prefill_forward_ms_per_request"] = dist(request_model_ms.values())
        metrics["per_batch_storage_and_gpu"] = {
            key: dist(micro[key] for micro in micros) for key in micros[0]}
        transfers = [micro for micro in micros if micro["h2d_stream_ms"] > 0]
        metrics["document_h2d_batches"] = len(transfers)
        if transfers:
            metrics["per_batch_with_document_h2d"] = {
                key: dist(micro[key] for micro in transfers) for key in
                ("h2d_stream_ms", "h2d_overlapping_model_ms")}
        copies = read(root / "rank0-copies.jsonl")
        background = [span(record[0], record[2]) for record in copies if record[4]]
        if background:
            metrics["background_to_pinned_ms_per_rank0_block"] = dist(background)
    cold = root / "initial-page-state.json"
    if cold.exists():
        metrics["ssd_initial_page_residency"] = read_json(cold)
    return metrics


def format_row(name, m):
    ttft = " / ".join(f'{m["ttft_ms"][f"p{p}"] / 1000:.2f}' for p in (50, 95, 99))
    tpot = " / ".join(f'{m["tpot_ms"][f"p{p}"]:.2f}' for p in (50, 95, 99))
    e2e = " / ".join(f'{m["e2e_ms"][f"p{p}"] / 1000:.2f}' for p in (50, 95, 99))
    return (f'| {m["offered_req_s"]:.2f} | {name} | {m["throughput_req_s"]:.3f} | '
            f'{m["output_tok_s"]:.2f} | {ttft} | {tpot} | {e2e} |')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    args = p.parse_args()
    inputs = [path.parent for path in args.results.glob("*/trial.json") if "-partial-" not in path.parent.name]
    assert inputs
    results = {root.name: trial(root) for root in inputs}
    paired_inputs = {}
    paired_ours = {}
    for root in inputs:
        key = (results[root.name]["offered_req_s"], results[root.name]["requests"])
        arrivals = read_json(root / "arrivals.json")
        rows = [row for row in read(root / "requests.jsonl") if row["ok"]]
        prompts = {row["qid"]: row["prompt_tokens"] for row in rows}
        matched = (arrivals, prompts)
        assert paired_inputs.setdefault(key, matched) == matched, root
        if not root.name.startswith("official-"):
            masks = {row["qid"]: (row.get("mask_sha256"), row["cached_tokens"], row["recomputed_tokens"])
                     for row in rows}
            paired_ours[root.name] = masks
    for name, masks in paired_ours.items():
        if name.startswith("dram-"):
            peer = name.replace("dram-", "ssd-", 1)
            if peer in paired_ours:
                different = sum(value != paired_ours[peer][qid] for qid, value in masks.items())
                for mode in (name, peer):
                    results[mode]["matched_cache_and_masks"] = len(masks) - different
    output = args.results / "analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output)
    print("| Input (req/s) | Method | Throughput (req/s) | Output (tok/s) | TTFT P50/P95/P99 (s) | TPOT P50/P95/P99 (ms) | E2E P50/P95/P99 (s) |")
    print("| ---: | --- | ---: | ---: | --- | --- | --- |")
    for name, data in sorted(results.items(), key=lambda x: (x[1]["offered_req_s"], x[0])):
        print(format_row(name, data))


if __name__ == "__main__":
    main()
