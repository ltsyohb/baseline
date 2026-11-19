#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/env.sh"
export PATH="$AGEMEM_HOME/inference-env/bin:$PATH"
mkdir -p "$AGEMEM_HOME/logs"
vllm serve "$MODEL_PATH" --served-model-name "$QWEN_MODEL" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm 2>&1 | tee "$AGEMEM_HOME/logs/vllm.log"
