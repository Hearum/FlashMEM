# Launch the FlashMEM Inference Engine

The bundled runtime extends SGLang with document KV reuse. Run the commands
below from `serving/` in the FlashMEM repository.

## Install

Use x86-64 Linux, Python 3.12, a CUDA toolchain, and an NVIDIA GPU with BF16
support. Install the included runtime in a fresh virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./sglang/python
```

## Run inference on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/demo.py
```

The demo launches a loopback server, sends a request, prints JSON with
`"ok": true` and generated `text`, and stops the server. The default model
downloads on first use. To use a local checkpoint, pass `--model PATH`.
[Default model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

To keep the server running, start it in one terminal.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh demo
```

In a second terminal, activate the same environment and send a request.

```bash
source .venv/bin/activate
python scripts/smoke.py --wait 300
```

## Replay document KV reuse on 8 H20 GPUs

The replay uses a single host with eight H20 GPUs, TP=8, a local Kimi-K2.6
W4A16 checkpoint, and a local Qwen2.5-0.5B-Instruct draft checkpoint. The target
checkpoint directory must be named `Kimi-K2.6`. The public checkpoints can be
downloaded as follows. Allow sufficient disk space for the target weights.
[Target checkpoint](https://huggingface.co/moonshotai/Kimi-K2.6) · [Draft checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

```bash
hf download moonshotai/Kimi-K2.6 --local-dir models/Kimi-K2.6
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/draft
```

Run the bundled example with a new output directory.

```bash
python scripts/run_benchmark.py \
  --data examples/conversations.json \
  --model models/Kimi-K2.6 \
  --draft-model models/draft \
  --output results/example
```

The runner builds the workload, prepares draft keys, starts the server,
populates the cache, and runs one DRAM trial at 0.15 req/s. The two synthetic
conversations exercise the replay pipeline. [Example input](examples/conversations.json)

To replay your own data, supply a JSON array of at least two conversations,
each with a nonempty `messages` array and string message content. The runner
uses the last user message as the query. Validate the input with
`python replay/build_full_workload.py data/conversations.json --validate-only`.

| Option | Effect |
| --- | --- |
| `--storage dram ssd --rates 0.15 0.30` | Run both cache tiers at both offered rates. |
| `--document-cache PATH` | Put the persistent document cache on a local SSD. |
| `--dram-cache PATH` | Use a tmpfs directory with space for the DRAM copy. The default is under `/dev/shm`. |
| `--baseline-python PATH` | Add a baseline run using the Python executable from a separate environment with official SGLang v0.5.23. |

The benchmark generates 32 tokens per request with EOS stopping disabled.
Client TTFT is measured from scheduled arrival to the first received token.
Find the metrics in `trials/analysis.json`, the runtime settings in
`trials/runtime-*.json`, and server startup messages in `server-pic.log`, all
relative to `--output`. Run `python scripts/run_benchmark.py --help` for the
full options.

The CPU checks run with `python -m unittest discover -s tests -v`.

The runtime is based on SGLang revision `d2b3c7f`.
[Apache 2.0 license](LICENSE) · [Upstream source](https://github.com/sgl-project/sglang/commit/d2b3c7fb14d97d744b8df198ef45c767237720f6)
