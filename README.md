# baseline

AgeMem / Qwen memory-poisoning baseline.

This repository contains the executable baseline implementation, the host-side
provenance and evidence-preserving adaptations, the frozen-pipeline code, and
offline tests. Dataset files, model weights, service attestations, raw runs and
generated result tables are intentionally excluded. The code expects those
artifacts to be supplied in a separate local workspace.

## Mechanism

The baseline evaluates four declared direct-store adaptations under a shared
AgeMem host:

- `AP_frozen_transfer`: a frozen AgentPoison DPR transfer payload;
- `L1_template`: a single template record;
- `L2_codebook`: a two-record codebook construction;
- `L3_conditional`: a trigger-conditioned template record.

Each case is paired across `clean`, `benign`, `poisoned_benign`,
`clean_triggered` and `poisoned_triggered`. The `L2_codebook` method also has
two fragment controls. Natural retrieval and a separate host diagnostic track
are distinguished in the code and must not be pooled.

## Components and dependencies

| Component | Location |
|---|---|
| AgeMem/Qwen adapter and logging | `AgeMem_Qwen_NoRL/app.py` |
| Baseline v2 runner | `AgeMem_Qwen_NoRL/baseline_v2.py` |
| Provenance v3 host | `AgeMem_Qwen_NoRL/provenance_v3.py` |
| Evidence-preserving v4 host | `AgeMem_Qwen_NoRL/host_v4.py` |
| Frozen joint pipeline | `joint_baseline_20261110/joint_pipeline_v4.py` |
| Full-matrix runner | `joint_baseline_20261110/03_code/run_full_matrix_20261116.py` |
| Operational summaries | `joint_baseline_20261110/05_results/` |
| Vendored AgeMem AgentScope layer | `AgeMem_Qwen_NoRL/vendor/AgeMem_code_agentscope/` |

The tested reference environment uses Python 3.12, PyTorch 2.8.0,
AgentScope 1.0.8, OpenAI SDK 1.109.1, vLLM 0.10.2 and
`text-embedding-v4` with 256 dimensions. The local Qwen endpoint is expected
at `http://127.0.0.1:8000/v1` and the served model id is
`qwen2.5-7b-instruct`.

Install the agent environment with:

```bash
cd AgeMem_Qwen_NoRL
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-agent.txt
python -m pip install tokenizers
```

The model server and the embedding credential are external dependencies. Keep
credentials in the process environment; do not put them in this repository.

## Data contract

The runners do not download or fabricate benchmark data. Put the separately
managed files in the paths described by
`joint_baseline_20261110/02_data/README.md` and create a frozen plan in the
local workspace. The expected external inputs include the StrategyQA and
MemPoison case files, the tokenizer, the frozen transfer trigger, training
demos, and a clean AgeMem snapshot. Their hashes are recorded in the local
manifest before a run.

Generated queues belong under `joint_baseline_20261110/02_data/`; raw runs
belong under `joint_baseline_20261110/04_runs/`; derived tables belong under
`joint_baseline_20261110/05_results/`. These directories are ignored by Git.

## Flow

1. Prepare the external dataset, tokenizer and prior pilot files described below.
2. Freeze and hash the experiment plan using `joint_pipeline_v4.py freeze`.
3. Run the offline preflight and unit tests.
4. Generate a fresh service attestation for the running local Qwen server.
5. Run the selected queue with append-only JSONL checkpoints.
6. Build a blind semantic-review queue before calculating ASR.
7. Generate summaries from the raw records; never edit raw records by hand.

## Commands

From the repository root:

```bash
python -m compileall AgeMem_Qwen_NoRL joint_baseline_20261110
python AgeMem_Qwen_NoRL/test_offline.py
python AgeMem_Qwen_NoRL/test_experiments.py
```

After the external plan and evidence files are in place, run the tests that
exercise frozen plans and saved model traces:

```bash
python -m unittest discover -s AgeMem_Qwen_NoRL -p 'test_*.py' -v
python joint_baseline_20261110/joint_pipeline_v4.py freeze
python joint_baseline_20261110/joint_pipeline_v4.py preflight
```

After the external plan and service evidence exist:

```bash
python joint_baseline_20261110/joint_pipeline_v4.py joint \
  --service-evidence <SERVICE_EVIDENCE_JSON> \
  --output joint_baseline_20261110/04_runs/<RUN_ID>
```

The full-matrix runner is restartable. Reuse the same output directory and
the same frozen queue after an interruption. It records errors and unresolved
semantic labels instead of replacing them with successful samples.

## Verification boundaries

Offline tests verify schemas, provenance transitions, snapshot isolation,
formatter/tool boundaries, frozen-file hashes, paired-job coverage and
checkpoint behavior. They do not prove that the external model server is
available or that a semantic attack succeeded. `semantic_label` remains
`U` until an independent review file is applied. Literal target strings,
retrieval calls and Top-k exposure are operational diagnostics, not semantic
ASR.

The currently used local subset is an exploratory completed-case subset. It is
not a random sample of the full matrix and must not be reported as a full
benchmark result.

## Threat model

The declared permissions and limitations are recorded in
`joint_baseline_20261110/00_idea/威胁模型.md`. The code must not be run with
permissions broader than that document without creating a new versioned plan.
