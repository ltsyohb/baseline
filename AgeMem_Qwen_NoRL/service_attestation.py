"""Create and verify the runtime evidence required by the frozen baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

WEIGHTS = dict(zip([f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)], [
    'a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7',
    'f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185',
    '8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5',
    '1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd']))


def process_evidence(pid):
    proc = Path('/proc') / str(pid)
    argv = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'argv': argv, 'start_ticks': stat[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def check_process(proc, model_path, runtime):
    argv = proc['argv']
    if 'serve' not in argv or not any('vllm' in x for x in argv):
        raise ValueError('PID must be the vLLM serve process')
    if str(Path(argv[argv.index('serve') + 1]).resolve()) != str(Path(model_path).resolve()):
        raise ValueError('Live process loaded a different model path')
    required = {'--served-model-name': runtime['model'], '--host': '127.0.0.1', '--port': '8000',
                '--max-model-len': '8192', '--tool-call-parser': 'hermes', '--generation-config': 'vllm'}
    for flag, expected in required.items():
        if flag not in argv or argv[argv.index(flag)+1] != expected:
            raise ValueError('Service launch argument mismatch: ' + flag)


def check_listener(pid):
    """Bind the advertised localhost endpoint to the attested server process."""
    inodes = set()
    for name in ('tcp', 'tcp6'):
        path = Path('/proc/net') / name
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1].split(':')[-1].upper() == '1F40' and fields[3] == '0A':
                inodes.add(fields[9])
    owned = set()
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            owned.add(target[8:-1])
    if not inodes.intersection(owned):
        raise ValueError('Attested vLLM PID does not own the port 8000 listener')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(root: Path):
    files = [root / name for name in ('app.py', 'baseline_v2.py', 'experiments.py', 'service_attestation.py')]
    files += sorted((root / 'vendor/AgeMem_code_agentscope').glob('*.py'))
    return {str(p.relative_to(root)): sha256(p) for p in files if p.exists()}


def attest(args):
    root = Path(args.root).resolve()
    client = OpenAI(api_key='local', base_url=args.base_url, timeout=15, max_retries=0)
    models = client.models.list()
    ids = [m.id for m in models.data]
    if args.model not in ids:
        raise SystemExit(f'model {args.model!r} not advertised: {ids!r}')
    weights = []
    for path in sorted(Path(args.model_path).glob('*.safetensors')):
        weights.append({'path': str(path), 'sha256': sha256(path), 'size': path.stat().st_size})
    observed = {Path(w['path']).name: w['sha256'] for w in weights}
    if observed != WEIGHTS:
        raise ValueError('Actual four weight shards differ from pinned Qwen revision')
    proc = process_evidence(args.pid)
    check_process(proc, args.model_path, {'model': args.model})
    check_listener(args.pid)
    inference_python = Path(os.getenv('AGEMEM_HOME', '/root/autodl-tmp/agemem-norl')) / 'inference-env/bin/python'
    inference_version = subprocess.run([str(inference_python), '-c',
        'import importlib.metadata; print(importlib.metadata.version("vllm"))'],
        capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    evidence = {
        'schema_version': 1, 'model': args.model, 'base_url': args.base_url,
        'advertised_model_ids': ids, 'model_path': str(Path(args.model_path).resolve()),
        'weight_files': weights, 'revision': args.revision,
        'process': proc, 'created_at': datetime.now(timezone.utc).isoformat(),
        'versions': {**{k: importlib.metadata.version(k) for k in ('openai', 'agentscope')},
                     'inference_env_vllm': inference_version},
        'source_hashes': source_hashes(root), 'python': platform.python_version(),
        'platform': platform.platform(), 'cwd': str(Path.cwd()),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                   capture_output=True, text=True).stdout.strip() or None,
        'env': {'QWEN_MODEL': os.getenv('QWEN_MODEL'), 'QWEN_BASE_URL': os.getenv('QWEN_BASE_URL'),
                'MODEL_PATH': os.getenv('MODEL_PATH')},
    }
    Path(args.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': args.output, 'model': args.model, 'weights': len(weights)}, ensure_ascii=False))


def verify_live(evidence, runtime):
    if evidence.get('model') != runtime['model'] or evidence.get('base_url') != runtime['base_url']:
        raise ValueError('Service evidence does not match frozen model or base URL')
    if evidence.get('revision') != runtime['declared_revision']:
        raise ValueError('Service evidence revision differs from frozen revision')
    if not evidence.get('weight_files'):
        raise ValueError('No verified weight files in service evidence')
    if runtime['model'] not in evidence.get('advertised_model_ids', []):
        raise ValueError('Frozen model absent from service advertisement')
    proc = process_evidence(evidence['process']['pid'])
    if proc != evidence['process']:
        raise ValueError('Service process changed since attestation')
    check_process(proc, evidence['model_path'], runtime)
    check_listener(proc['pid'])
    if {Path(w['path']).name: sha256(Path(w['path'])) for w in evidence['weight_files']} != WEIGHTS:
        raise ValueError('Live weights differ from pinned hashes')
    if source_hashes(Path(__file__).resolve().parent) != evidence['source_hashes']:
        raise ValueError('Source changed after service attestation')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--base-url', default=os.getenv('QWEN_BASE_URL', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--model', default=os.getenv('QWEN_MODEL', 'qwen2.5-7b-instruct'))
    p.add_argument('--model-path', default=os.getenv('MODEL_PATH', 'models/Qwen2.5-7B-Instruct'))
    p.add_argument('--revision', default='a09a35458c702b33eeacc393d103063234e8bc28')
    p.add_argument('--output', required=True)
    p.add_argument('--pid', type=int, required=True, help='Local Linux vLLM serve PID; verified via /proc')
    args = p.parse_args()
    attest(args)


if __name__ == '__main__':
    main()
