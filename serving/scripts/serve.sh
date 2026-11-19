#!/usr/bin/env bash
set -euo pipefail

mode=${1:-demo}
case "$mode" in
  demo|pic|baseline) ;;
  -h|--help)
    echo 'Usage: bash scripts/serve.sh {demo|pic|baseline}'
    echo 'demo uses a small public model on one GPU. pic and baseline require MODEL_DIR.'
    echo 'Benchmark settings: PORT, TP_SIZE, CONTEXT_LENGTH, KV_POOL_TOKENS, CHUNK_SIZE.'
    exit 0
    ;;
  *) echo 'Choose demo, pic, or baseline.' >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON:-python3}
port=${PORT:-30014}

if [[ "$mode" != baseline ]]; then
  export PYTHONPATH="$root/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ "$mode" == demo ]]; then
  # The small demo model uses MHA. The persistent document store requires MLA.
  unset SGLANG_FUSIONRAG_CACHE_DIR SGLANG_FUSIONRAG_DRAM_CACHE_DIR
  export SGLANG_FUSIONRAG_REPLICATED_MLA=0 FUSIONRAG_STORAGE_BENCH=0
  exec "$python" -m sglang.launch_server \
    --host 127.0.0.1 --port "$port" --model-path "${MODEL_DIR:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --tp-size 1 --dtype bfloat16 --context-length 2048 --max-total-tokens 4096 \
    --max-running-requests 4 --mem-fraction-static 0.70 \
    --attention-backend triton --chunked-prefill-size 512 --max-prefill-tokens 512 \
    --enable-hierarchical-cache --hicache-ratio 2 --hicache-mem-layout layer_first \
    --hicache-io-backend kernel --hicache-write-policy write_back --page-size 1 \
    --disable-hicache-numa-detect --disable-overlap-schedule --disable-cuda-graph
fi

: "${MODEL_DIR:?Set MODEL_DIR to a compatible target checkpoint}"
tp_size=${TP_SIZE:-8}
chunk=${CHUNK_SIZE:-8192}

args=(
  --host 127.0.0.1 --port "$port" --model-path "$MODEL_DIR"
  --trust-remote-code --tp-size "$tp_size" --mem-fraction-static 0.85
  --context-length "${CONTEXT_LENGTH:-34816}" --max-running-requests "${MAX_RUNNING_REQUESTS:-24}"
  --max-total-tokens "${KV_POOL_TOKENS:-110000}" --chunked-prefill-size "$chunk"
  --page-size 1 --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" --schedule-policy fcfs
  --enable-metrics --enable-p2p-check --watchdog-timeout 3000
)

if [[ "$mode" == pic ]]; then
  export SGLANG_FUSIONRAG_CACHE_DIR="${SGLANG_FUSIONRAG_CACHE_DIR:-$root/cache/documents}"
  export SGLANG_FUSIONRAG_DRAM_CACHE_DIR="${SGLANG_FUSIONRAG_DRAM_CACHE_DIR:-/dev/shm/pic-serving-${UID}}"
  export SGLANG_FUSIONRAG_REPLICATED_MLA="${SGLANG_FUSIONRAG_REPLICATED_MLA:-1}"
  export FUSIONRAG_STORAGE_BENCH=1
  args+=(
    --attention-backend triton --decode-attention-backend "${PIC_DECODE_BACKEND:-triton}"
    --cuda-graph-max-bs "${MAX_RUNNING_REQUESTS:-24}"
    --triton-attention-num-kv-splits "${TRITON_KV_SPLITS:-64}"
    --disable-shared-experts-fusion --disable-overlap-schedule --disable-custom-all-reduce
    --enable-hierarchical-cache --hicache-size "${HICACHE_GB:-12}"
    --fusionrag-cache-size "${DOCUMENT_CACHE_GB:-26}"
    --hicache-mem-layout layer_first --hicache-io-backend kernel --hicache-write-policy write_back
    --max-prefill-tokens "$chunk"
  )
  if [[ "$SGLANG_FUSIONRAG_REPLICATED_MLA" == 1 && "$tp_size" != 8 ]]; then
    echo 'Replicated MLA storage expects TP_SIZE=8. Set SGLANG_FUSIONRAG_REPLICATED_MLA=0 for a different setup.' >&2
    exit 2
  fi
  model_path=${MODEL_DIR%/}
  if [[ "$SGLANG_FUSIONRAG_REPLICATED_MLA" == 1 && "${model_path##*/}" != Kimi-K2.6 ]]; then
    echo 'Replicated MLA storage expects the checkpoint directory to be named Kimi-K2.6.' >&2
    exit 2
  fi
  mkdir -p -- "$SGLANG_FUSIONRAG_CACHE_DIR" "$SGLANG_FUSIONRAG_DRAM_CACHE_DIR"
else
  args+=(--disable-prefill-cuda-graph --cuda-graph-max-bs-decode "${MAX_RUNNING_REQUESTS:-24}")
fi

exec "$python" -m sglang.launch_server "${args[@]}"
