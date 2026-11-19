"""Runtime model paths and draft-to-target token mapping."""

import os

TARGET = os.environ.get("MODEL_DIR", "./models/Kimi-K2.6")
DRAFT = os.environ.get("DRAFT_MODEL_DIR", "./models/draft")


def mapped_positions(selected, blocks, head_len):
    result = []
    draft_start = 0
    target_start = head_len
    for block in blocks:
        source, target = block["draft_offsets"], block["target_offsets"]
        spans = [source[i - draft_start] for i in selected
                 if draft_start <= i < draft_start + len(source)]
        j = 0
        for i, (start, end) in enumerate(target):
            while j < len(spans) and spans[j][1] <= start:
                j += 1
            if j < len(spans) and start < end and spans[j][0] < end:
                result.append(target_start + i)
        target_start += len(target)
        draft_start += len(source)
    return result
