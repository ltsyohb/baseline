#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/env.sh"
source /root/.config/agemem/dashscope.env
exec "$AGEMEM_HOME/agent-env/bin/python" "$KIT/acceptance.py"
