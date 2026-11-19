#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/env.sh"
BASE_PYTHON="${BASE_PYTHON:-/root/miniconda3/bin/python}"
"$BASE_PYTHON" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 12), 'Use the tested Python 3.12 image.'
assert torch.__version__.split('+')[0] == '2.8.0', 'Use the tested PyTorch 2.8.0 image.'
assert torch.cuda.is_available(), 'GPU unavailable.'
print('Base runtime:', sys.version.split()[0], torch.__version__)
PY
mkdir -p "$AGEMEM_HOME/logs"
if [ ! -x "$AGEMEM_HOME/inference-env/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$AGEMEM_HOME/inference-env"
fi
"$AGEMEM_HOME/inference-env/bin/python" -m pip install 'vllm==0.10.2' 'transformers==4.56.2' 'huggingface_hub==0.34.4'
"$AGEMEM_HOME/inference-env/bin/python" -m pip check
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
"$AGEMEM_HOME/inference-env/bin/python" "$KIT/download_model.py"
"$AGEMEM_HOME/inference-env/bin/python" -m pip freeze > "$AGEMEM_HOME/logs/inference-pip-freeze.txt"
if [ ! -x "$AGEMEM_HOME/agent-env/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$AGEMEM_HOME/agent-env"
fi
"$AGEMEM_HOME/agent-env/bin/python" -m pip install -r "$KIT/requirements-agent.txt"
"$AGEMEM_HOME/agent-env/bin/python" -m pip check
"$AGEMEM_HOME/agent-env/bin/python" -m pip freeze > "$AGEMEM_HOME/logs/agent-pip-freeze.txt"
"$AGEMEM_HOME/agent-env/bin/python" "$KIT/test_offline.py"
"$AGEMEM_HOME/agent-env/bin/python" "$KIT/test_experiments.py"
echo 'Setup complete. Run bash start.sh, bash acceptance.sh, then bash chat.sh.'
