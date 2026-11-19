"""Bench-only CPU timestamps and nonblocking CUDA events for the prefill path."""

import json
from pathlib import Path

import torch


class TTFTProbe:
    def __init__(self):
        self.output = None
        self.rank = None
        self.batch = None
        self.copies = None

    def set_label(self, label, rank):
        if self.output is not None:
            self.output.close()
            path = Path(f"profile-{self.label}-rank{self.rank}-copies.jsonl")
            with path.open("w") as stream:
                for record in self.copies:
                    stream.write(json.dumps(record) + "\n")
        self.output = None
        self.copies = None
        if label.startswith("profile-"):
            self.label = label
            self.rank = rank
            self.copies = []
            self.output = Path(f"profile-{label}-rank{rank}.jsonl").open(
                "w", buffering=1024 * 1024
            )
        self.batch = None

    def begin(self, start):
        self.batch = {"start": start, "ssd": [], "events": {}}

    def restore(self, started, ended, tokens, blocks):
        if self.batch is not None:
            self.batch["ssd"].append([started, ended, tokens, blocks])

    def mark(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.batch["events"][name] = event

    def finish(self, batch, selected, ended):
        record = self.batch
        self.batch = None
        events = record.pop("events")
        assert {"origin", "model_start", "rope_start", "rope_end", "model_end"} <= events.keys(), events
        origin = events["origin"]
        # The normal result path already copies next_token_ids to CPU.
        assert all(event.query() for event in events.values()), events
        record.update(
            rids=[req.rid for req in batch.reqs],
            queued=[req.time_stats.wait_queue_entry_time for req in batch.reqs],
            extend_tokens=batch.extend_num_tokens,
            request_computed_tokens=[len(req.all_compute_idx) for req in batch.reqs],
            selected=selected,
            ended=ended,
            gpu_ms={name: origin.elapsed_time(event) for name, event in events.items() if name != "origin"},
        )
        self.output.write(json.dumps(record, separators=(",", ":")) + "\n")


probe = TTFTProbe()
