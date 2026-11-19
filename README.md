# FlashMEM

**Lightweight memory for long-horizon language agents.**

FlashMEM builds compact memory entries that point back to the original conversation. At query time, it retrieves relevant entries and uses their source links to recover details from short excerpts of the conversation. On the serving side, it reuses key–value (KV) states for recurring context to reduce repeated prefill.

The [serving engine](serving/README.md) includes the modified SGLang runtime, a one-GPU server demo, and a benchmark for document KV reuse. The agent-side memory implementation is in preparation.

## How it works

1. **Build source-linked memory.** Process each new conversation segment once and append self-contained entries with references to their source turns.
2. **Retrieve memory and evidence.** Retrieve relevant entries, then search the regions they reference for query-specific evidence. Send both the entries and the selected excerpts to the model in a single answer-generation call.
3. **Reuse KV states across requests.** Reuse recurring context during construction and retain the KV states of compact memory across queries. When cached chunks move to new positions or appear under different prefixes, a lightweight selector chooses which token positions the serving model recomputes.

## Evaluation

The study evaluates answer quality and serving efficiency on LoCoMo, LongMemEval-S, and HaluMem-Medium. See [Evaluation](docs/evaluation.md) for benchmark settings and reported results.
