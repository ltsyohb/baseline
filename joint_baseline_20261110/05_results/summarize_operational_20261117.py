"""Report observed execution and paired coverage from an archived matrix run.

These are operational diagnostics, not semantic attack success metrics.
"""

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
MATRIX = BASE / "02_data/full_matrix_20261116_v2"
DEFAULT_ARCHIVE = BASE / "04_runs/remote_checkpoints/full_matrix_20261117_0600_partial.tar.gz"
RAW_MEMBER = "joint_baseline_20261110/04_runs/full_matrix_20261117_mempoison_pilot/raw.jsonl"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads((MATRIX / "MANIFEST.json").read_text(encoding="utf-8"))
    jobs_file = MATRIX / "jobs.jsonl"
    if sha256(jobs_file) != manifest["files_sha256"]["jobs.jsonl"]:
        raise ValueError("Frozen jobs differ from manifest")
    jobs = [json.loads(line) for line in jobs_file.open(encoding="utf-8") if line.strip()]
    if len(jobs) != manifest["job_count"]:
        raise ValueError("Frozen job count differs from manifest")
    expected = {job["key"]: (index, job) for index, job in enumerate(jobs)}
    if len(expected) != len(jobs):
        raise ValueError("Duplicate frozen job keys")

    with tarfile.open(args.archive, "r:gz") as archive:
        archive_names = set(archive.getnames())
        source = archive.extractfile(RAW_MEMBER)
        if source is None:
            raise ValueError("Main raw.jsonl missing from archive")
        recorded = {}
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key in recorded or key not in expected:
                raise ValueError(f"Unknown or duplicate raw key: {key}")
            index, job = expected[key]
            if row["job_index"] != index or any(
                    row[field] != job[field]
                    for field in ("dataset", "case_id", "method", "condition")):
                raise ValueError(f"Raw metadata differs from frozen job: {key}")
            recorded[key] = row

    # A compressed tar is expensive to seek repeatedly. Stream it once and
    # select each valid job's answer trace by its immutable key digest.
    trace_status = {}
    trace_priority = {}
    run_prefix = RAW_MEMBER.rsplit("/", 1)[0]
    valid_digest_to_key = {
        hashlib.sha256(key.encode("utf-8")).hexdigest(): key
        for key, row in recorded.items()
        if row.get("error") is None and row.get("protocol_valid", False)
    }
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            name = member.name
            if not member.isfile() or not name.endswith("/answer_events.jsonl"):
                continue
            if not name.startswith(run_prefix + "/"):
                continue
            parts = name[len(run_prefix) + 1:].split("/")
            if len(parts) == 4 and parts[0] == "jobs":
                priority, digest = 0, parts[2]
            elif (len(parts) == 6 and parts[0] == "shards"
                  and parts[2] == "jobs"):
                priority, digest = 1, parts[4]
            else:
                continue
            key = valid_digest_to_key.get(digest)
            if key is None or trace_priority.get(key, 2) <= priority:
                continue
            searches = []
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for line in stream:
                if line.strip():
                    event = json.loads(line)
                    if event.get("kind") == "vector_search_compact":
                        searches.append(event)
            expected_calls = recorded[key].get("retrieval_calls", 0)
            if len(searches) != expected_calls:
                status = "search_call_mismatch"
            else:
                external = any(
                    result.get("item", {}).get("metadata", {})
                    .get("_host_provenance_v3", {}).get("source") == "EXTERNAL_NOTE"
                    for search in searches for result in search.get("results", []))
                status = "external_in_topk" if external else "no_external_in_topk"
            trace_status[key] = status
            trace_priority[key] = priority
    for key in valid_digest_to_key.values():
        trace_status.setdefault(key, "missing_trace")

    counts = defaultdict(Counter)
    pair_conditions = defaultdict(set)
    pair_valid_conditions = defaultdict(set)
    condition_sets = defaultdict(set)
    for key, (index, job) in expected.items():
        cell = (job["dataset"], job["method"], job["condition"])
        method_case = (job["dataset"], job["method"], job["case_id"])
        condition_sets[(job["dataset"], job["method"])].add(job["condition"])
        counts[cell]["planned"] += 1
        row = recorded.get(key)
        if row is None:
            counts[cell]["missing"] += 1
            continue
        pair_conditions[method_case].add(job["condition"])
        counts[cell]["recorded"] += 1
        if row.get("error") is not None or not row.get("protocol_valid", False):
            counts[cell]["protocol_or_run_error"] += 1
            continue
        pair_valid_conditions[method_case].add(job["condition"])
        counts[cell]["protocol_valid"] += 1
        if row.get("retrieval_calls", 0) > 0:
            counts[cell]["retrieval_invoked"] += 1
        status = trace_status[key]
        counts[cell][status] += 1

    cells = []
    for cell in sorted(counts):
        c = counts[cell]
        if c["planned"] != c["missing"] + c["protocol_or_run_error"] + c["protocol_valid"]:
            raise ValueError(f"Cell denominator mismatch: {cell}")
        cells.append({
            "dataset": cell[0], "method": cell[1], "condition": cell[2],
            "planned": c["planned"], "recorded": c["recorded"],
            "protocol_valid": c["protocol_valid"],
            "protocol_or_run_error": c["protocol_or_run_error"],
            "missing": c["missing"],
            "retrieval_invoked": c["retrieval_invoked"],
            "external_in_topk": c["external_in_topk"],
            "missing_trace": c["missing_trace"],
            "search_call_mismatch": c["search_call_mismatch"],
            "retrieval_rate_among_valid": c["retrieval_invoked"] / c["protocol_valid"]
            if c["protocol_valid"] else "",
        })

    methods = []
    for dataset, method in sorted(condition_sets):
        conditions = condition_sets[(dataset, method)]
        cases = {job["case_id"] for job in jobs
                 if job["dataset"] == dataset and job["method"] == method}
        methods.append({
            "dataset": dataset, "method": method, "planned_cases": len(cases),
            "all_conditions_recorded": sum(
                pair_conditions[(dataset, method, case_id)] == conditions for case_id in cases),
            "all_conditions_protocol_valid": sum(
                pair_valid_conditions[(dataset, method, case_id)] == conditions for case_id in cases),
            "conditions_per_case": len(conditions),
        })

    args.output.mkdir(parents=True)
    for name, rows in (("cells.csv", cells), ("paired_coverage.csv", methods)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "archive_sha256": sha256(args.archive),
        "matrix_version": manifest["version"],
        "planned_jobs": len(jobs), "recorded_jobs": len(recorded),
        "protocol_valid_jobs": sum(c["protocol_valid"] for c in counts.values()),
        "protocol_or_run_error_jobs": sum(c["protocol_or_run_error"] for c in counts.values()),
        "missing_jobs": len(jobs) - len(recorded),
        "external_in_topk_jobs": sum(c["external_in_topk"] for c in counts.values()),
        "missing_trace_jobs": sum(c["missing_trace"] for c in counts.values()),
        "search_call_mismatch_jobs": sum(c["search_call_mismatch"] for c in counts.values()),
        "all_conditions_protocol_valid_cases_by_method": {
            row["dataset"] + "/" + row["method"]: row["all_conditions_protocol_valid"]
            for row in methods},
        "interpretation": "Execution, retrieval, external-memory Top-k and paired coverage diagnostics only; semantic ASR remains pending.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
