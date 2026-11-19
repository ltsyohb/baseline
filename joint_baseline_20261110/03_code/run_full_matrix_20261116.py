"""Run the frozen full matrix with append-only records and restart support.

The existing v4 host is used for every job. The clean background is loaded
from the frozen corpus, so no 78 MB snapshot is written for each answer.
"""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PAPERS = Path(__file__).resolve().parents[2]
BASE = PAPERS / "joint_baseline_20261110"
MATRIX = BASE / "02_data/full_matrix_20261116_v2"
KIT = PAPERS / "AgeMem_Qwen_NoRL"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def ensure_matrix():
    manifest = read(MATRIX / "MANIFEST.json")
    if manifest["version"] != "full_matrix_20261116_v2":
        raise ValueError("Matrix version mismatch")
    for name, expected in manifest["files_sha256"].items():
        if sha(MATRIX / name) != expected:
            raise ValueError(f"Matrix file changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        if sha(PAPERS / name) != expected:
            raise ValueError(f"Input changed: {name}")
    return manifest


def path_for(root, key):
    return root / "jobs" / hashlib.sha256(key.encode("utf-8")).hexdigest()[:2] / hashlib.sha256(key.encode("utf-8")).hexdigest()


def compact_trace(agent):
    """Record retrieval and changed memories without logging the entire corpus."""
    memory = agent.memory_manager.inner
    original_embed = memory.embed

    def embed(value):
        vector = original_embed(value)
        agent.event("embedding", text=value, vector=vector,
                    model=memory.embedding_model, dimensions=memory.embedding_dim)
        return vector

    memory.embed = embed
    original_search = memory._store.search

    def search(query_embedding, top_k=5, metadata_filter=None):
        candidates = memory._store.get_all_memory()
        results = original_search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
        agent.event("vector_search_compact", query_embedding=query_embedding,
                    top_k=top_k, metadata_filter=metadata_filter,
                    candidate_count=len(candidates),
                    results=[{"item": item.to_dict(), "score": score} for item, score in results])
        return results

    memory._store.search = search
    original_apply = agent._apply_tool

    async def apply(call):
        before = {item["memory_id"]: item for item in agent.memory_manager.state_dict()["content"]}
        try:
            return await original_apply(call)
        finally:
            after = {item["memory_id"]: item for item in agent.memory_manager.state_dict()["content"]}
            changed = {key: value for key, value in after.items()
                       if key not in before or before[key] != value}
            deleted = sorted(before.keys() - after.keys())
            agent.event("memory_transition_compact", call=call,
                        before_count=len(before), after_count=len(after),
                        changed_items=changed, deleted_ids=deleted)

    agent._apply_tool = apply


def infrastructure_interruption(exc):
    """Stop the queue when the serving infrastructure cannot answer calls."""
    names = {"APIConnectionError", "APITimeoutError", "RateLimitError",
             "InternalServerError", "ConnectError", "ConnectTimeout",
             "ReadTimeout", "RemoteProtocolError"}
    current = exc
    while current is not None:
        if type(current).__name__ in names:
            return True
        current = current.__cause__ or current.__context__
    return False


async def execute(root, service_evidence, start_index, stop_index):
    sys.path[:0] = [str(KIT), str(BASE)]
    from baseline_v2 import runtime_check
    from host_v4 import build_v4_agent, output_status
    from provenance_v3 import host_reply
    from experiments import state

    manifest = ensure_matrix()
    plan = read(BASE / "data/joint_v4_20261113_v1.json")
    runner_sha256 = sha(Path(__file__))
    plan_sha256 = sha(BASE / "data/joint_v4_20261113_v1.json")
    if not service_evidence:
        raise ValueError("Current service attestation is required")
    runtime_check(plan["spec"], service_evidence)
    if root.exists() and not (root / "run_meta.json").exists():
        raise ValueError("Output directory exists without matching run metadata")
    if not root.exists():
        root.mkdir(parents=True)
        atomic_json(root / "run_meta.json", {
            "run_id": root.name, "matrix_version": manifest["version"],
            "matrix_sha256": sha(MATRIX / "MANIFEST.json"),
            "runner_sha256": runner_sha256, "plan_sha256": plan_sha256,
            "service_evidence": read(service_evidence),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": plan["spec"]["runtime"]["model"],
            "status": "running", "planned_jobs": manifest["job_count"],
        })
    else:
        previous = read(root / "run_meta.json")
        if (previous.get("matrix_sha256") != sha(MATRIX / "MANIFEST.json")
                or previous.get("runner_sha256") != runner_sha256
                or previous.get("plan_sha256") != plan_sha256):
            raise ValueError("Cannot resume with changed matrix, runner, or plan")

    tasks = {(task["dataset"], task["case_id"]): task for task in read(MATRIX / "tasks.json")}
    qa_background = None
    mp_backgrounds = {}
    journal = root / "raw.jsonl"
    jobs = MATRIX / "jobs.jsonl"
    recorded = {}
    if journal.exists():
        with journal.open(encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    row = json.loads(line)
                    if row["key"] in recorded:
                        raise ValueError(f"Duplicate raw record: {row['key']}")
                    recorded[row["key"]] = row
    with jobs.open(encoding="utf-8") as source, journal.open("a", encoding="utf-8") as output:
        for index, line in enumerate(source):
            if index < start_index or (stop_index is not None and index >= stop_index):
                continue
            job = json.loads(line)
            folder = path_for(root, job["key"])
            done = folder / "result.json"
            if job["key"] in recorded:
                if not done.exists():
                    atomic_json(done, recorded[job["key"]])
                elif read(done) != recorded[job["key"]]:
                    raise ValueError(f"Result and raw record differ: {job['key']}")
                continue
            if done.exists():
                row = read(done)
                if row["key"] != job["key"]:
                    raise ValueError(f"Result key mismatch: {job['key']}")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                recorded[job["key"]] = row
                continue
            folder.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix="agemem-full-matrix-"))
            task = tasks[(job["dataset"], job["case_id"])]
            row = {"run_id": root.name, "probe_id": job["case_id"],
                   "category": task.get("original_level") or "StrategyQA",
                   "model": plan["spec"]["runtime"]["model"],
                   "framing": job["method"] + "/" + job["condition"],
                   "prompt": job["query"], "response": None,
                   "seed": plan["spec"]["runtime"].get("seed", 42),
                   "ts": datetime.now(timezone.utc).isoformat(),
                   "latency_ms": None, "error": None,
                   "key": job["key"], "job_index": index,
                   "dataset": job["dataset"], "case_id": job["case_id"],
                   "family_id": task["family_id"], "method": job["method"],
                   "condition": job["condition"],
                   "payload_tokens": job["payload_tokens"]}
            start = datetime.now(timezone.utc)
            interrupted = None
            try:
                if "snapshot" in task["background"]:
                    if qa_background is None:
                        qa_background = read(PAPERS / task["background"]["snapshot"])
                        for n, memory in enumerate(qa_background["content"], 1):
                            memory.setdefault("metadata", {})["_host_provenance_v3"] = {
                                "source": "AUTHORITY_INITIAL", "identity": "initial",
                                "input_id": "frozen-background", "input_order": 0,
                                "version": n, "parent_versions": {},
                                "attribution": "experiment-supplied clean corpus"}
                        qa_background["host_clock_v3"] = len(qa_background["content"])
                    snapshot = qa_background
                else:
                    case_key = (job["dataset"], job["case_id"])
                    if case_key not in mp_backgrounds:
                        initial = build_v4_agent(scratch / "initial", plan["spec"])
                        compact_trace(initial)
                        with initial.memory_manager.transport("initial", job["case_id"] + ":initial"):
                            for n, content in enumerate(task["background"]["context"]):
                                await initial.memory_manager.add("clean-" + str(n), content)
                        mp_backgrounds[case_key] = state(initial)
                    snapshot = mp_backgrounds[case_key]
                if not snapshot["content"]:
                    raise ValueError("Empty clean background")
                if job["records"]:
                    prepare = build_v4_agent(scratch / "prepare", plan["spec"])
                    compact_trace(prepare)
                    prepare.memory_manager.load_state_dict(snapshot)
                    with prepare.memory_manager.transport("external", job["key"] + ":injection"):
                        for n, content in enumerate(job["records"]):
                            await prepare.memory_manager.add("attack-" + str(n), content)
                    snapshot = state(prepare)
                answer_agent = build_v4_agent(scratch / "answer", plan["spec"])
                compact_trace(answer_agent)
                answer_agent.memory_manager.load_state_dict(snapshot)
                answer = await host_reply(answer_agent, job["query"], "task", job["key"])
                text = answer.get_text_content()
                events = [json.loads(item) for item in (answer_agent.state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                row.update(response=text, **output_status(text, events),
                           retrieval_calls=sum(item.get("kind") == "tool_call" and item.get("call", {}).get("name") == "retrieve_memory" for item in events))
            except Exception as exc:
                if infrastructure_interruption(exc):
                    interrupted = exc
                else:
                    row.update(error=repr(exc), **output_status(None, [], runner_error=repr(exc)))
            finally:
                for phase in ("initial", "prepare", "answer"):
                    source_events = scratch / phase / "events.jsonl"
                    if source_events.exists():
                        shutil.copyfile(source_events, folder / f"{phase}_events.jsonl")
                shutil.rmtree(scratch)
            if interrupted is not None:
                atomic_json(root / "interruption.json", {
                    "key": job["key"], "job_index": index,
                    "error": repr(interrupted), "time_utc": datetime.now(timezone.utc).isoformat(),
                    "resumable": True,
                })
                raise RuntimeError(f"Infrastructure interrupted at job {index}; resume same output directory") from interrupted
            row["latency_ms"] = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            atomic_json(done, row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            recorded[job["key"]] = row
            if index % 100 == 0:
                print(f"completed index={index} key={job['key']} error={row['error']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--service-evidence", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    args = parser.parse_args()
    asyncio.run(execute(Path(args.output), args.service_evidence,
                        args.start_index, args.stop_index))


if __name__ == "__main__":
    main()
