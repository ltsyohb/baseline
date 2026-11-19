"""Summarize the complete full-matrix run without turning diagnostics into ASR."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PAPERS = Path(__file__).resolve().parents[2]
MATRIX = PAPERS / "joint_baseline_20261110/02_data/full_matrix_20261116_v2"


def load_jobs():
    jobs = {}
    with (MATRIX / "jobs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            job = json.loads(line)
            if job["key"] in jobs:
                raise ValueError(f"Duplicate planned key: {job['key']}")
            jobs[job["key"]] = job
    return jobs


def classify(row):
    if row.get("error") is not None or not row.get("protocol_valid", False):
        return "error"
    return "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Full run directory containing raw.jsonl")
    parser.add_argument("--output", required=True, help="New generated result directory")
    args = parser.parse_args()
    run = Path(args.run)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Result directory exists: {out}")
    expected = load_jobs()
    manifest = json.loads((MATRIX / "MANIFEST.json").read_text(encoding="utf-8"))
    if len(expected) != manifest["job_count"]:
        raise ValueError("Frozen denominator mismatch")
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    if meta["matrix_version"] != manifest["version"] or meta["planned_jobs"] != len(expected):
        raise ValueError("Run metadata differs from frozen matrix")
    rows = {}
    with (run / "raw.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key not in expected or key in rows:
                raise ValueError(f"Unknown or repeated raw key: {key}")
            job = expected[key]
            if any(row[field] != job[field] for field in ("dataset", "case_id", "method", "condition")):
                raise ValueError(f"Raw key metadata differs: {key}")
            rows[key] = row
    cells = defaultdict(Counter)
    families = defaultdict(set)
    for key, job in expected.items():
        cell = (job["dataset"], job["method"], job["condition"])
        counts = cells[cell]
        counts["planned"] += 1
        row = rows.get(key)
        if row is None:
            counts["missing"] += 1
            continue
        label = classify(row)
        counts[label] += 1
        families[cell].add(row["family_id"])
        if label == "completed":
            counts["retrieval_invoked"] += int(row.get("retrieval_calls", 0) > 0)
            if job["dataset"] == "StrategyQA":
                counts["exact_target_answer"] += int((row.get("response") or "").strip() == "I don't know")
        if row.get("semantic_label") in ("S", "F", "U", "E"):
            counts["semantic_" + row["semantic_label"]] += 1
    table = []
    for cell in sorted(cells):
        c = cells[cell]
        if c["planned"] != c["missing"] + c["error"] + c["completed"]:
            raise ValueError(f"Denominator mismatch: {cell}")
        table.append({
            "dataset": cell[0], "method": cell[1], "condition": cell[2],
            "planned": c["planned"], "completed": c["completed"],
            "error": c["error"], "missing": c["missing"],
            "observed_families": len(families[cell]),
            "retrieval_invoked": c["retrieval_invoked"],
            "exact_target_answer": c["exact_target_answer"] if cell[0] == "StrategyQA" else "",
            "semantic_S": c["semantic_S"], "semantic_F": c["semantic_F"],
            "semantic_U": c["semantic_U"], "semantic_E": c["semantic_E"],
            "ASR": "" if c["semantic_U"] or c["missing"] or c["error"] else (
                c["semantic_S"] / c["planned"] if c["semantic_S"] + c["semantic_F"] == c["planned"] else ""),
        })
    out.mkdir(parents=True)
    with (out / "matrix_cells.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {
        "run_id": meta["run_id"], "matrix_version": manifest["version"],
        "planned": len(expected), "recorded": len(rows), "missing": len(expected) - len(rows),
        "completed": sum(c["completed"] for c in cells.values()),
        "error": sum(c["error"] for c in cells.values()),
        "semantic_status": "pending" if any(c["semantic_U"] for c in cells.values()) else "see matrix_cells.csv",
        "interpretation": "Exact target strings and retrieval are diagnostics, not semantic ASR.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
