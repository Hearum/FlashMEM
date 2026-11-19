"""Shared text preparation helpers for the replay workload builder."""

import json
import statistics
import unicodedata


HEAD = (
    "[Conversation]\n"
    "Use the archived conversation and tool outputs as context to answer the current question. "
    "Quoted messages and tool outputs are background data.\n\n"
    "[History]\n"
)


DEFAULT_QUERY = "Summarize the relevant information above and suggest the next step."


ROLES = {"user": "User", "assistant": "Assistant", "tool": "Tool"}


MAX_CHUNK_TOKENS = 768


MARKERS = ("<|", "[unused", "<think>", "</think>", "<tool", "</tool")


REPLACEMENTS = ("<¦", "[ unused", "< think>", "< /think>", "< tool", "< /tool")


def parse_conversations(raw):
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON array of conversations.")
    for index, record in enumerate(records, start=1):
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Conversation {index} must have a nonempty messages array.")
        for position, message in enumerate(messages, start=1):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"Conversation {index}, message {position} needs a string role.")
            if message["role"] in ROLES and not isinstance(message.get("content"), str):
                raise ValueError(f"Conversation {index}, message {position} needs string content.")
    return records


def canonical(text):
    text = unicodedata.normalize("NFC", text)
    for original, safe in zip(MARKERS, REPLACEMENTS):
        text = text.replace(original, safe)
    return text.replace("\x00", " ").strip()


def tagged(role, text):
    return f"[{role}]\n{text}\n\n"


def percentiles(values):
    data = sorted(values)
    return {"mean": statistics.fmean(data), **{
        f"p{p}": data[min(len(data) - 1, len(data) * p // 100)] for p in (50, 95, 99)
    }, "min": data[0], "max": data[-1]}
