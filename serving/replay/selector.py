"""Online FlashMem query/key scoring using CPU BF16 and cached document keys."""
import math
import time

import torch
from transformers import AutoModelForCausalLM

from workload import DRAFT, mapped_positions


class Selector:
    def __init__(self, bank, device="cpu"):
        self.bank = bank
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            DRAFT, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        ).eval().to(self.device)
        cfg = self.model.config
        self.heads = cfg.num_attention_heads
        self.groups = cfg.num_key_value_heads
        self.width = cfg.hidden_size // self.heads
        self.layers = cfg.num_hidden_layers // 2
        self.mode = None
        self.captured = []
        self.key_staging = None
        for layer in self.model.model.layers[self.layers:]:
            layer.self_attn.k_proj.register_forward_hook(self.capture("key", self.groups))
            layer.self_attn.q_proj.register_forward_hook(self.capture("query", self.heads))
        self.keys = {}

    def capture(self, mode, count):
        def hook(_module, _args, value):
            if self.mode == mode:
                self.captured.append(value.reshape(-1, count, self.width).transpose(0, 1))
        return hook

    @torch.inference_mode()
    def project(self, ids, mode):
        self.mode = mode
        self.captured = []
        self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        result = torch.stack(self.captured)
        self.mode = None
        self.captured = []
        return result

    def rotary(self, values, positions):
        inv = self.model.model.rotary_emb.inv_freq
        phase = positions.float()[:, None] * inv.float()[None, :]
        phase = torch.cat((phase, phase), dim=-1)
        first, second = values.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return values * phase.cos().to(values.dtype) + rotated * phase.sin().to(values.dtype)

    @torch.inference_mode()
    def prepare(self):
        for key, block in self.bank.items():
            self.keys[key] = self.project(block["draft_ids"], "key").cpu().contiguous()

    @torch.inference_mode()
    def select(self, row, rate, profile=False, score_layers=None, reference=False):
        stamp = time.perf_counter
        gpu_events = {}

        def gpu_mark(label):
            if profile and self.device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                gpu_events[label] = event

        gpu_mark("query_start")
        start = stamp()
        query = self.project(row["draft_query"], "query")
        gpu_mark("query_end")
        query_end = stamp()
        blocks = [self.keys[key] for key in row["blocks"]]
        pack_ms = 0.0
        if self.device.type == "cuda" and blocks[0].device.type == "cpu":
            layers, groups, _, dim = blocks[0].shape
            n = sum(block.shape[-2] for block in blocks)
            elements = layers * groups * n * dim
            if self.key_staging is None or self.key_staging.numel() < elements:
                self.key_staging = torch.empty(elements, dtype=blocks[0].dtype, pin_memory=True)
            packed = self.key_staging[:elements].view(layers, groups, n, dim)
            torch.cat(blocks, dim=-2, out=packed)
            pack_ms = (stamp() - query_end)*1e3
            gpu_mark("copy_start")
            keys = packed.to(self.device, non_blocking=True)
            gpu_mark("copy_end")
        else:
            keys = torch.cat(blocks, dim=-2)
        layers, groups, n, dim = keys.shape
        m = query.shape[-2]
        query = self.rotary(query, torch.arange(n, n + m, device=self.device))
        keys = self.rotary(keys, torch.arange(n, device=self.device))
        key_end = stamp()
        query = query.reshape(layers, groups, self.heads // groups * m, dim)

        def score(q, k):
            logits = torch.matmul(q, k.transpose(-1, -2)).float()
            logits.div_(math.sqrt(dim))
            return logits.softmax(-1).reshape(q.shape[0], self.heads, m, n).mean((1, 2))

        gpu_mark("score_start")
        if reference:
            logits = torch.matmul(query, keys.transpose(-1, -2)).float() / math.sqrt(dim)
            scores = logits.softmax(-1).reshape(layers, self.heads, m, n).mean((1, 2)).mean(0)
        elif score_layers is None:
            scores = score(query, keys).mean(0)
        else:
            per_layer = [
                score(query[start:start + score_layers], keys[start:start + score_layers])
                for start in range(0, layers, score_layers)
            ]
            scores = torch.cat(per_layer).mean(0)
        gpu_mark("score_end")
        score_end = stamp()
        indices = sorted(scores.topk(int(n * rate)).indices.tolist())
        mapped = mapped_positions(indices, [self.bank[key] for key in row["blocks"]], row["head_len"])
        end = time.perf_counter()
        timings = {"select_ms": (end - start) * 1e3, "selected_summary": len(mapped)}
        if self.device.type == "cpu":
            timings.update(query_ms=(query_end-start)*1e3, keys_rope_ms=(key_end-query_end)*1e3,
                           score_ms=(score_end-key_end)*1e3, topk_map_ms=(end-score_end)*1e3)
        if gpu_events:
            timings.update(query_ms=gpu_events["query_start"].elapsed_time(gpu_events["query_end"]),
                           score_ms=gpu_events["score_start"].elapsed_time(gpu_events["score_end"]),
                           keys_rope_ms=(key_end-query_end)*1e3, topk_map_ms=(end-score_end)*1e3,
                           draft_key_pack_ms=pack_ms)
            if "copy_end" in gpu_events:
                timings["draft_key_h2d_ms"] = gpu_events["copy_start"].elapsed_time(gpu_events["copy_end"])
        return list(range(row["head_len"])) + mapped, timings
