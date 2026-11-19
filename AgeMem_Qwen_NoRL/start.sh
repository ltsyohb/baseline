#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT/env.sh"
exec "$AGEMEM_HOME/agent-env/bin/python" "$KIT/service.py" start
