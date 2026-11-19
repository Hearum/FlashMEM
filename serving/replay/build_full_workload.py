"""Build a text-history replay with optional trimming to an input token cap."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_workload import DEFAULT_QUERY, HEAD, MAX_CHUNK_TOKENS, ROLES, canonical, parse_conversations, percentiles, tagged
from workload import DRAFT, TARGET


def split_payload(encode, model, role, content):
    rendered = tagged(role, content)
    if len(encode(rendered)) <= MAX_CHUNK_TOKENS:
        return [rendered]

    raw = content.encode()
    token_ends = [0]
    for token in encode(content):
        token_ends.append(token_ends[-1] + len(model.decode_single_token_bytes(token)))
    assert token_ends[-1] == len(raw)
    chunks = []
    reconstructed = []
    start = 0
    while start < len(token_ends) - 1:
        end = min(start + MAX_CHUNK_TOKENS - 18, len(token_ends) - 1)
        while True:
            boundary = token_ends[end]
            if boundary < len(raw) and raw[boundary] & 0xC0 == 0x80:
                end -= 1
                continue
            piece = raw[token_ends[start]:boundary].decode()
            rendered = tagged(role, piece)
            if len(encode(rendered)) <= MAX_CHUNK_TOKENS:
                break
            end -= 1
        assert end > start
        chunks.append(rendered)
        reconstructed.append(piece)
        start = end
    assert "".join(reconstructed) == content
    return chunks


def history_chunks(encode, model, messages):
    turns = []
    turn = []
    for role, content in messages:
        if role == "user" and turn:
            turns.append(turn)
            turn = []
        turn.extend(split_payload(encode, model, ROLES[role], content))
    if turn:
        turns.append(turn)
    chunks = []
    rendered_pieces = []
    for i in range(0, len(turns), 2):
        block = ""
        for piece in (p for t in turns[i:i + 2] for p in t):
            candidate = block + piece
            if block and len(encode(candidate)) > MAX_CHUNK_TOKENS:
                chunks.append(block)
                candidate = piece
            block = candidate
            rendered_pieces.append(piece)
        if block:
            chunks.append(block)
    assert "".join(chunks) == "".join(rendered_pieces)
    return chunks


def offsets(model, text, ids):
    char_ends = [0]
    for char in text:
        char_ends.append(char_ends[-1] + len(char.encode()))
    start = 0
    boundaries = []
    pieces = [model.decode_single_token_bytes(token) for token in ids]
    assert b"".join(pieces) == text.encode()
    for piece in pieces:
        end = start + len(piece)
        boundaries.append((bisect_right(char_ends, start) - 1, bisect_left(char_ends, end)))
        start = end
    return boundaries


def row_indices(index_path):
    if index_path is None:
        return set()
    return set(json.loads(index_path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="Directory for generated workload files")
    parser.add_argument("--validate-only", action="store_true", help="Check the input schema without loading tokenizers")
    parser.add_argument("--rows", type=Path, help="JSON array of source indices for workload.json")
    parser.add_argument("--materialize-all", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, help="Keep the full question and trim the oldest history blocks")
    args = parser.parse_args()
    try:
        raw = args.source.read_bytes()
        source = parse_conversations(raw)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.validate_only:
        print(f"Validated {len(source)} conversations.")
        return
    if args.output is None:
        parser.error("Specify an output directory when building a workload.")
    if args.max_input_tokens is not None and args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive.")

    from transformers import AutoTokenizer

    target = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True, local_files_only=True)
    draft = AutoTokenizer.from_pretrained(DRAFT, local_files_only=True, use_fast=True)
    encode = target.model.encode_ordinary
    selected_rows = set(range(len(source))) if args.materialize_all else row_indices(args.rows)
    assert selected_rows <= set(range(len(source)))
    head_ids = encode(HEAD)
    rows, stats_rows = [], []
    bank, bank_lengths = {}, {}
    references = Counter()
    flags = Counter()
    for index, record in enumerate(source):
        messages = [(m["role"], canonical(m["content"])) for m in record["messages"]
                    if m["role"] in ROLES]
        if index < 8:
            for _, text in messages:
                assert encode(text) == target.encode(text, add_special_tokens=False)
        body_tokens = sum(len(encode(text)) for _, text in messages)
        body_bytes = sum(len(text.encode()) for _, text in messages)
        messages = [(role, content) for role, content in messages if content]
        last_user = next((i for i in range(len(messages) - 1, -1, -1)
                          if messages[i][0] == "user"), None)
        question = DEFAULT_QUERY if last_user is None else messages.pop(last_user)[1]
        flags["synthetic_question"] += last_user is None
        assert (sum(len(text.encode()) for _, text in messages)
                + (len(question.encode()) if last_user is not None else 0)) == body_bytes
        blocks = history_chunks(encode, target.model, messages)
        tail_ids = encode("[Question]\n" + question + "\n\n[Answer]\n")
        query_ids = draft.encode(question, add_special_tokens=False, verbose=False)
        target_tokens = [encode(text) for text in blocks]
        before = len(head_ids) + sum(map(len, target_tokens)) + len(tail_ids)
        drop = 0
        if args.max_input_tokens is not None:
            if len(head_ids) + len(tail_ids) > args.max_input_tokens:
                raise ValueError(f"Request {index}: full question exceeds the model-input cap")
            remaining = before
            while remaining > args.max_input_tokens:
                remaining -= len(target_tokens[drop])
                drop += 1
        flags["history_trimmed"] += bool(drop)
        flags["history_tokens_removed"] += sum(map(len, target_tokens[:drop]))
        lengths = []
        keys = []
        for text, ids in zip(blocks[drop:], target_tokens[drop:]):
            key = hashlib.sha256(text.encode()).hexdigest()
            assert 0 < len(ids) <= MAX_CHUNK_TOKENS
            keys.append(key)
            lengths.append(len(ids))
            bank_lengths[key] = len(ids)
            if index in selected_rows and key not in bank:
                d = draft(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
                bank[key] = dict(text=text, target_ids=ids, target_offsets=offsets(target.model, text, ids),
                                 draft_ids=d.input_ids, draft_offsets=d.offset_mapping)
        references.update(keys)
        summary_len = sum(lengths)
        input_len = len(head_ids) + summary_len + len(tail_ids)
        qid = f"traffic-{index:03d}"
        stats_rows.append(dict(source_index=index, qid=qid, source_body_tokens=body_tokens, uncapped_input_tokens=before,
                               input_tokens=input_len, body_bytes=body_bytes,
                               history_tokens=summary_len, draft_query_tokens=len(query_ids),
                               question_tokens=len(encode(question)), blocks=len(keys), history_blocks_removed=drop,
                               history_block_ids=keys, original_last_role=record["messages"][-1]["role"]))
        if index in selected_rows:
            input_ids = head_ids + [token for key in keys for token in bank[key]["target_ids"]] + tail_ids
            assert len(input_ids) == input_len
            rows.append(dict(qid=qid, source_index=index, blocks=keys, head_len=len(head_ids),
                             draft_query=query_ids, input_ids=input_ids, summary_len=summary_len,
                             question_tokens=len(encode(question)), tail_len=len(tail_ids),
                             original_message_count=len(record["messages"]),
                             original_last_role=record["messages"][-1]["role"]))

    unique_tokens = sum(bank_lengths.values())
    stats = dict(requests=len(source), restored_body_bytes=sum(r["body_bytes"] for r in stats_rows),
                 unique_blocks=len(bank_lengths), block_references=references.total(),
                 unique_document_tokens=unique_tokens,
                 history_token_references=sum(r["history_tokens"] for r in stats_rows),
                 repeated_document_tokens=sum((count - 1) * bank_lengths[k]
                                              for k, count in references.items()),
                 shared_token_references=sum(count * bank_lengths[k] for k, count in references.items()
                                             if count > 1),
                 source_last_role=dict(Counter(r["original_last_role"] for r in stats_rows)),
                 source_sha256=hashlib.sha256(raw).hexdigest(),
                 source_scope="NFC-normalized text content from user, assistant, tool",
                 source_arrivals_available=False, max_chunk_tokens=MAX_CHUNK_TOKENS,
                 input_token_cap=args.max_input_tokens, history_trimmed_requests=flags["history_trimmed"],
                 history_tokens_removed=flags["history_tokens_removed"],
                 synthetic_questions=flags["synthetic_question"], output_tokens=32,
                 materialized=len(rows))
    for field in ("source_body_tokens", "input_tokens", "history_tokens", "question_tokens", "draft_query_tokens", "blocks"):
        stats[field] = percentiles(r[field] for r in stats_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "workload-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output / "workload-manifest.json").write_text(json.dumps(
        {"rows": stats_rows, "block_lengths": bank_lengths}, separators=(",", ":")) + "\n")
    if rows:
        with (args.output / "workload.json").open("w") as stream:
            json.dump(dict(head=HEAD, head_ids=head_ids, rows=rows, blocks=bank),
                      stream, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
