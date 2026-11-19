"""Long-context implementation of the existing document-attention selector."""

from bisect import bisect_left
import math
import time

import torch

from selector import Selector


def map_positions(selected, blocks, head_len):
    """Preserve the existing character overlap rule with bounded range lookups."""
    result = []
    draft_start = 0
    target_start = head_len
    for block in blocks:
        source, target = block["draft_offsets"], block["target_offsets"]
        lo = bisect_left(selected, draft_start)
        hi = bisect_left(selected, draft_start + len(source), lo)
        spans = [source[index - draft_start] for index in selected[lo:hi]]
        j = 0
        for i, (start, end) in enumerate(target):
            while j < len(spans) and spans[j][1] <= start:
                j += 1
            if j < len(spans) and start < end and spans[j][0] < end:
                result.append(target_start + i)
        target_start += len(target)
        draft_start += len(source)
    return result


def attention_importance(query, keys, max_logits):
    """Sum softmax probabilities by key with full, row-wise normalization."""
    layers, groups, n, dim = keys.shape
    score = torch.zeros(n, device=keys.device, dtype=torch.float32)
    scale = 1 / math.sqrt(dim)
    # Each tile contains every document position. Tiling the independent query
    # rows preserves softmax normalization and bounds the temporary memory.
    rows = max(1, max_logits // (groups * n))
    for q_layer, k_layer in zip(query, keys):
        transposed = k_layer.transpose(-1, -2)
        for q_tile in q_layer.split(rows, dim=-2):
            logits = torch.matmul(q_tile, transposed).float()
            logits.mul_(scale)
            score.add_(torch.softmax(logits, dim=-1).sum(dim=(0, 1)))
    return score / (layers * groups * query.shape[-2])


def rotary_in_place(values, cos, sin):
    for layer in values:
        first, second = layer.chunk(2, dim=-1)
        rotated_first = first * cos - second * sin
        second.mul_(cos).add_(first * sin)
        first.copy_(rotated_first)


class LongSelector(Selector):
    def __init__(self, bank, device="cpu", max_logits=None):
        super().__init__(bank, device=device)
        self.max_logits = max_logits or (2**22 if self.device.type == "cpu" else 2**24)

    def key_tensor(self, blocks):
        layers, groups, _, dim = blocks[0].shape
        n = sum(block.shape[-2] for block in blocks)
        if self.device.type != "cuda":
            return torch.cat(blocks, dim=-2), None
        elements = layers * groups * n * dim
        if self.key_staging is None or self.key_staging.numel() < elements:
            self.key_staging = torch.empty(elements, dtype=blocks[0].dtype, pin_memory=True)
        packed = self.key_staging[:elements].view(layers, groups, n, dim)
        torch.cat(blocks, dim=-2, out=packed)
        return packed, self.device

    def rotary_factors(self, n, m, dtype):
        inv = self.model.model.rotary_emb.inv_freq
        positions = torch.arange(n + m, dtype=torch.float32, device=self.device)
        phase = positions[:, None] * inv.float()[None, :]
        return phase.cos().to(dtype), phase.sin().to(dtype)

    @torch.inference_mode()
    def select(self, row, rate, profile=False, score_layers=None, reference=False):
        if reference:
            return super().select(row, rate, profile, score_layers, reference=True)
        gpu = self.device.type == "cuda"
        stamp = time.perf_counter
        events = {}

        def mark(name):
            if gpu and profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[name] = event
            return stamp()

        started = mark("query_start")
        query = self.project(row["draft_query"], "query")
        query_end = mark("query_end")
        blocks = [self.keys[key] for key in row["blocks"]]
        keys, copy_to = self.key_tensor(blocks)
        packed = mark("copy_start")
        if copy_to is not None:
            keys = keys.to(copy_to, non_blocking=True)
        copied = mark("copy_end")
        layers, groups, n, dim = keys.shape
        m = query.shape[-2]
        c, s = self.rotary_factors(n, m, keys.dtype)
        rotary_in_place(query, c[n:], s[n:])
        rotary_in_place(keys, c[:n], s[:n])
        query = query.reshape(layers, groups, (self.heads // groups) * m, dim)
        key_end = mark("score_start")
        scores = attention_importance(query, keys, self.max_logits)
        mark("score_end")
        score_end = stamp()
        indices = scores.topk(int(n * rate), sorted=False).indices.tolist()
        indices.sort()
        mask = map_positions(indices, [self.bank[key] for key in row["blocks"]], row["head_len"])
        ended = stamp()
        detail = dict(select_ms=(ended - started) * 1000, selected_summary=len(mask),
                      query_ms=(query_end - started) * 1000, keys_rope_ms=(key_end - query_end) * 1000,
                      score_ms=(score_end - key_end) * 1000, topk_map_ms=(ended - score_end) * 1000)
        if gpu and profile:
            detail.update(query_ms=events["query_start"].elapsed_time(events["query_end"]),
                          keys_rope_ms=events["copy_end"].elapsed_time(events["score_start"]),
                          score_ms=events["score_start"].elapsed_time(events["score_end"]),
                          draft_key_pack_ms=(packed - query_end) * 1000,
                          draft_key_h2d_ms=events["copy_start"].elapsed_time(events["copy_end"]))
        elif not gpu:
            detail["draft_key_pack_ms"] = (packed - query_end) * 1000
        return list(range(row["head_len"])) + mask, detail
