# Evaluation

FlashMEM is evaluated on long-term conversational question answering with Qwen3-8B, GLM-4.5-Air, and Kimi-K2.6. The main comparison measures token F1, semantic accuracy, and token usage during memory construction and answering. Separate serving experiments measure prefill, latency, and persistent key–value (KV) cache capacity.

## Benchmarks

| Benchmark | Questions | Evaluation setting |
| --- | ---: | --- |
| [LoCoMo](https://github.com/snap-research/LoCoMo) | 1,540 | Non-adversarial questions across ten multi-session conversations |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | 500 | Long-history questions covering information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention |
| [HaluMem-Medium](https://arxiv.org/abs/2511.03506) | 3,467 | Memory question answering over each user's conversation history up to the time of the query |

## Answer quality

The table reports the Qwen3-8B results for FlashMEM with KV reuse disabled. The best baseline is selected by token F1 on each benchmark. Semantic accuracy is judged by GLM-5.3.

| Benchmark | Best baseline F1 (%) | FlashMEM F1 (%) | FlashMEM accuracy (%) |
| --- | ---: | ---: | ---: |
| LoCoMo | 42.78 | 50.22 | 56.10 |
| LongMemEval-S | 33.36 | 44.41 | 61.40 |
| HaluMem-Medium | 23.87 | 38.75 | 74.94 |

## Serving efficiency

- **Time to first token.** On LoCoMo with Qwen3-8B, KV reuse delivered a 2.71× speedup in median time to first token compared with the same system with reuse disabled.
- **Persistent KV storage.** With Qwen3-8B, estimated persistent KV capacity was 86.9%, 89.6%, and 93.7% lower than full-history caching on LoCoMo, LongMemEval-S, and HaluMem-Medium, respectively. The estimates include the cached keys of the lightweight selector.
