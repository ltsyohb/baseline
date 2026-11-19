#!/usr/bin/env bash
export AGEMEM_HOME="${AGEMEM_HOME:-/root/autodl-tmp/agemem-norl}"
export HF_HOME="$AGEMEM_HOME/cache/huggingface"
export MODEL_PATH="$AGEMEM_HOME/models/Qwen2.5-7B-Instruct"
export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8000/v1}"
export QWEN_MODEL="qwen2.5-7b-instruct"
export AGEMEM_STATE_DIR="${AGEMEM_STATE_DIR:-$AGEMEM_HOME/state/default}"
