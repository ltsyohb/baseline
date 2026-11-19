#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/env.sh"
if [ -f /root/.config/agemem/dashscope.env ]; then
  source /root/.config/agemem/dashscope.env
fi
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  read -rsp 'DashScope key (embedding only): ' DASHSCOPE_API_KEY
  echo
  export DASHSCOPE_API_KEY
fi
"$AGEMEM_HOME/agent-env/bin/python" "$KIT/app.py" "$@"
