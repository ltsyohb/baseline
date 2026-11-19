"""Idempotent local Qwen startup and health check."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ['AGEMEM_HOME'])
HOME.mkdir(parents=True, exist_ok=True)


def ready():
    try:
        with urllib.request.urlopen(os.environ['QWEN_BASE_URL'] + '/models', timeout=3) as response:
            models = json.load(response)['data']
        return any(x['id'] == os.environ['QWEN_MODEL'] for x in models)
    except Exception:
        return False


if sys.argv[1] == 'status':
    result = ready()
    print('Qwen: READY' if result else 'Qwen: NOT READY')
    print('Endpoint:', os.environ['QWEN_BASE_URL'])
    print('Default memory:', os.environ['AGEMEM_STATE_DIR'])
    print('Credential file:', 'present' if Path('/root/.config/agemem/dashscope.env').is_file() else 'missing')
    raise SystemExit(0 if result else 1)

with (HOME / 'service.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if ready():
        print('Qwen is already ready; reused the existing service.')
        raise SystemExit(0)
    pidfile = HOME / 'service.pid'
    if pidfile.exists():
        pid = int(pidfile.read_text())
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and str(ROOT / 'serve.sh').encode() in cmdline.read_bytes():
            raise SystemExit('Server is still starting. Inspect logs/vllm-launch.log.')
    (HOME / 'logs').mkdir(exist_ok=True)
    with (HOME / 'logs/vllm-launch.log').open('a') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'serve.sh')], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(process.pid))
    for i in range(180):
        if ready():
            print('Qwen is ready. Run bash chat.sh.')
            break
        if process.poll() is not None:
            raise SystemExit('Server exited. See logs/vllm-launch.log.')
        if i % 10 == 0:
            print('Loading Qwen; progress in logs/vllm-launch.log.', flush=True)
        time.sleep(2)
    else:
        raise SystemExit('Server is not ready after 6 minutes. Inspect logs before restarting.')
