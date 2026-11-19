"""Download a fixed revision and verify the four weight shards."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
HASHES = [
    'a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7',
    'f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185',
    '8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5',
    '1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd',
]
root = Path(os.environ['MODEL_PATH'])
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision=REVISION,
                  local_dir=str(root), max_workers=4)
rows = []
for index, expected in enumerate(HASHES, 1):
    path = root / f'model-{index:05d}-of-00004.safetensors'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    rows.append({'file': path.name, 'sha256': digest.hexdigest(),
                 'expected': expected, 'passed': digest.hexdigest() == expected})
report = {'revision': REVISION, 'files': rows, 'passed': all(x['passed'] for x in rows)}
log = Path(os.environ['AGEMEM_HOME']) / 'logs/model-verification.json'
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text(json.dumps(report, indent=2), encoding='utf-8')
if not report['passed']:
    raise SystemExit('Weight verification failed: ' + str(log))
print('All four weight shards verified.')
