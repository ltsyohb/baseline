# External data contract

This directory is intentionally empty in the GitHub repository. Keep the
following files in the private experiment workspace and do not commit them:

- `full_matrix_20261116_v2/tasks.json`
- `full_matrix_20261116_v2/jobs.jsonl`
- `full_matrix_20261116_v2/MANIFEST.json`

The full-matrix builder expects the following external paths relative to the
repository root. These inputs are intentionally not included in this GitHub
repository:

- `AgeMem_Qwen_NoRL/data/strategyqa_dev.json`
- `AgeMem_Qwen_NoRL/data/MemPoison_full1227.json`
- `AgeMem_Qwen_NoRL/data/agentpoison_train_demos_2.json`
- `AgentPoison-preparation/model-small-files/Qwen2.5-7B-Instruct/tokenizer.json`
- `joint_baseline_20261108/agentpoison/frozen_transfer_trigger.json`
- `joint_baseline_20261108/agentpoison/backup_20261108_184406/files/agemem-norl/formal_runs/agentpoison-surrogate-v1/evaluation/clean_snapshot.json`
- `joint_baseline_20261108/mempoison/strategyqa_pilot_plan.json`
- `joint_baseline_20261110/data/provenance_v3_dev_01.json`

The builder also records SHA-256 hashes in the generated manifest. A queue is
valid only when those hashes still match.

The repository ships no benchmark examples, task text, attack payload,
generated output, model weight or service credential.
