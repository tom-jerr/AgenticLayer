#!/usr/bin/env bash
set -euo pipefail
: "${RUN:?set RUN to a private experiment directory}"
: "${MODEL:?set MODEL to the pinned snapshot}"
: "${PYTHON:?set PYTHON to the isolated interpreter}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
export MAX_JOBS="${MAX_JOBS:-3}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export SGLANG_CACHE_DIR="$RUN/cache/sglang"
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_DECODE_STATE_FREE_RADIX=0 SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT=0
export SGLANG_PLUGINS=__no_plugins__
extra=()
hicache=(--enable-hierarchical-cache --hicache-size "${AGENTIC_HOST_GB:-16}" --hicache-write-policy write_back
         --hicache-mem-layout page_first --hicache-io-backend kernel)
case "${ENGINE_ARM:-native-radix}" in
  native-radix) ;;
  reference-cold) extra+=(--disable-radix-cache) ;;
  native-hicache) extra+=("${hicache[@]}") ;;
  extended|extended-hicache)
    export SGLANG_PLUGINS=agentic_layer AGENTIC_ENGINE_SOCKET="$RUN/engine.sock"
    export AGENTIC_TRITON_COPY="${AGENTIC_TRITON_COPY:-0}"
    extra+=(--radix-cache-backend agentic_layer)
    if [[ "$ENGINE_ARM" == extended-hicache ]]; then extra+=("${hicache[@]}"); fi
    ;;
  *) printf 'Unknown ENGINE_ARM: %s\n' "$ENGINE_ARM" >&2; exit 2 ;;
esac
if [[ -n "${AGENTIC_FULL_TOKENS:-}" ]]; then extra+=(--max-total-tokens "$AGENTIC_FULL_TOKENS"); fi
if [[ -n "${AGENTIC_MAMBA_SLOTS:-}" ]]; then extra+=(--max-mamba-cache-size "$AGENTIC_MAMBA_SLOTS"); fi
decode_graph="${AGENTIC_DECODE_GRAPH:-disabled}"
case "$decode_graph" in
  disabled) ;;
  full) extra+=(--cuda-graph-max-bs-decode 8 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8) ;;
  *) printf 'Unknown AGENTIC_DECODE_GRAPH: %s\n' "$decode_graph" >&2; exit 2 ;;
esac
case "${AGENTIC_DETERMINISTIC:-0}" in
  0) ;;
  1) extra+=(--enable-deterministic-inference) ;;
  *) printf 'Invalid AGENTIC_DETERMINISTIC: %s\n' "$AGENTIC_DETERMINISTIC" >&2; exit 2 ;;
esac
# Fixed revision deducts six engine slots from the input ceiling. The public
# budget is still 32768; 32767 prompt + 1 output is verified on every baseline.
exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name qwen35-9b-nvfp4 \
  --host 127.0.0.1 --port 30080 --tp-size 1 \
  --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
  --dtype bfloat16 --kv-cache-dtype bf16 --attention-backend triton \
  --linear-attn-backend triton --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton --page-size 1 \
  --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule \
  --mamba-ssm-dtype float32 --hybrid-cache-checkpoints-level "${AGENTIC_CHECKPOINT_LEVEL:-2}" \
  --context-length 32774 --chunked-prefill-size 2048 --mem-fraction-static 0.80 \
  --max-running-requests 8 --radix-eviction-policy lru \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-metrics \
  --cuda-graph-backend-decode "$decode_graph" --cuda-graph-backend-prefill disabled \
  "${extra[@]}"
