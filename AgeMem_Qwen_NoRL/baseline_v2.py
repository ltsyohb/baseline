"""Frozen natural-ingestion pilot. Offline tests never count as model results."""
import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from app import build_agent, Msg, save_memory
from experiments import state, difference, write_json, score_literal

CONDITIONS = ['B0_clean', 'B1_harmless', 'B2_poisoned',
              'R0_clean_snapshot_reload', 'R1_actual_restore', 'C1_legal_update']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(spec):
    if spec['conditions'] != CONDITIONS:
        raise ValueError('Condition matrix mismatch')
    cases = spec['cases']
    if len(cases) != 10 or len({c['case_id'] for c in cases}) != 10:
        raise ValueError('Pilot requires 10 unique cases')
    if len({c['connected_group'] for c in cases}) != 10:
        raise ValueError('Connected groups overlap')
    for c in cases:
        if len(c['trigger_tasks']) != 1 or len(c['poison_turns']) != 1:
            raise ValueError('Frozen one-task/one-intervention contract violated')
        if abs(c['tokens']['attack'] - c['tokens']['harmless']) > 4:
            raise ValueError('B1 length mismatch')
        if max(c['tokens'][k] for k in ('attack', 'harmless', 'legal_update')) > 512:
            raise ValueError('Token budget exceeded')
    if spec['budget']['planned_answers'] != len(cases) * len(CONDITIONS):
        raise ValueError('Planned denominator mismatch')


def load_spec(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen protocol changed; version and refreeze explicitly')
    spec = json.loads(path.read_text(encoding='utf-8'))
    validate(spec)
    return spec


def attach_trace(agent):
    """Trace actual embed/search calls without changing query or top_k."""
    memory = agent.memory_manager
    original_embed = memory.embed
    def embed(text):
        vector = original_embed(text)
        agent.event('embedding', text=text, vector=vector,
                    model=memory.embedding_model, dimensions=memory.embedding_dim)
        return vector
    memory.embed = embed
    original_search = memory._store.search
    def search(query_embedding, top_k=5, metadata_filter=None):
        candidates = state(agent)
        results = original_search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
        agent.event('vector_search', query_embedding=query_embedding, top_k=top_k,
                    metadata_filter=metadata_filter, candidate_snapshot=candidates,
                    candidate_sha256=digest(candidates),
                    results=[{'item': item.to_dict(), 'score': score} for item, score in results])
        return results
    memory._store.search = search
    original_apply = agent._apply_tool
    async def apply(call):
        before = state(agent)
        try:
            return await original_apply(call)
        finally:
            after = state(agent)
            agent.event('memory_transition', call=call, before=before, after=after,
                        delta=difference(before, after),
                        attribution='temporal derivation; semantic contamination requires review')
    agent._apply_tool = apply


def runtime_check(spec, evidence_path):
    """Strict local vLLM binding; a /models name alone is insufficient."""
    runtime = spec['runtime']
    for key, expected in [('QWEN_MODEL', runtime['model']), ('QWEN_BASE_URL', runtime['base_url'])]:
        if os.getenv(key, expected) != expected:
            raise ValueError(key + ' differs from frozen configuration')
    evidence = json.loads(Path(evidence_path).read_text(encoding='utf-8'))
    from service_attestation import verify_live
    verify_live(evidence, runtime)
    return evidence


async def run(spec, output, factory=None, runtime_evidence=None):
    validate(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'frozen_spec.json', spec)
    write_json(output / 'manifest.json', {
        'mode': 'offline_test_double' if factory else 'live', 'spec_sha256': digest(spec),
        'conditions': CONDITIONS, 'planned_answers': 60,
        'model_evidence': runtime_evidence,
        'versions': {k: importlib.metadata.version(k) for k in ('agentscope', 'openai')},
        'source_sha256': {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), Path(__file__).with_name('app.py'),
                                    *sorted((Path(__file__).parent/'vendor/AgeMem_code_agentscope').glob('*.py'))]},
        'permission': 'normal user transport with authenticated provenance labels',
        'retrieval': 'natural; no forced query or top_k',
        'interpretation': 'MemPoison-inspired policy-conditioned adaptation; exploratory developer cases'})
    if factory is None:
        def factory(folder):
            agent = build_agent(folder, policy_suffix=spec['policy_suffix'], runtime_config=spec['runtime'])
            attach_trace(agent)
            return agent

    def fresh(folder, snapshot=None):
        folder = Path(folder)
        if snapshot is not None:
            write_json(folder / 'memory.json', snapshot)
        agent = factory(folder)
        if agent.context_messages:
            raise ValueError('New session inherited short-term context')
        if snapshot is not None and digest(state(agent)) != digest(snapshot):
            raise ValueError('Snapshot load differs')
        agent.event('session_start', context_count=0, snapshot_sha256=digest(state(agent)))
        return agent

    rows = []
    blind = []
    for case in spec['cases']:
        folder = output / case['case_id']
        task = case['trigger_tasks'][0]
        results = {c: {'status': 'error', 'semantic_label': 'E', 'error': 'not_executed'} for c in CONDITIONS}
        try:
            initial = fresh(folder / 'prepare')
            initial.event('input', phase='verified_initial', text=case['initial_message'])
            await initial.reply(Msg('owner', case['initial_message'], 'user'))
            clean = state(initial)
            write_json(folder / 'clean_snapshot.json', clean)
        except Exception as exc:
            for c in CONDITIONS:
                results[c]['error'] = 'prepare: ' + repr(exc)
            clean = None
        snapshots = {'B0_clean': clean, 'R0_clean_snapshot_reload': clean}
        if clean is not None:
            for condition, key in [('B1_harmless', 'harmless_message'), ('B2_poisoned', 'attack_message'),
                                   ('C1_legal_update', 'legal_update_message')]:
                try:
                    a = fresh(folder / condition / 'ingest', clean)
                    a.event('input', phase=condition, text=case[key])
                    await a.reply(Msg('owner' if condition == 'C1_legal_update' else 'external', case[key], 'user'))
                    snapshots[condition] = state(a)
                    write_json(folder / condition / 'snapshot.json', snapshots[condition])
                    write_json(folder / condition / 'delta.json', difference(clean, snapshots[condition]))
                except Exception as exc:
                    results[condition]['error'] = 'ingest: ' + repr(exc)
            # Restore on the SAME object after loading the poisoned state, persist,
            # then reopen a NEW session to check disk state and answer naturally.
            if 'B2_poisoned' in snapshots:
                try:
                    a = fresh(folder / 'R1_actual_restore' / 'restore', snapshots['B2_poisoned'])
                    before = state(a)
                    a.memory_manager.load_state_dict(copy.deepcopy(clean))
                    save_memory(a.memory_manager, a.state_dir / 'memory.json')
                    disk = json.loads((a.state_dir / 'memory.json').read_text(encoding='utf-8'))
                    check = {'before_sha256': digest(before), 'clean_sha256': digest(clean),
                             'after_sha256': digest(state(a)), 'disk_sha256': digest(disk),
                             'residual': difference(clean, disk),
                             'intervention': 'whole-store restoration; not targeted deletion'}
                    write_json(folder / 'R1_actual_restore' / 'restore_check.json', check)
                    if digest(clean) != digest(disk) or digest(clean) != digest(state(a)):
                        raise ValueError('Restore residual state')
                    snapshots['R1_actual_restore'] = disk
                except Exception as exc:
                    results['R1_actual_restore']['error'] = 'restore: ' + repr(exc)
            else:
                results['R1_actual_restore']['error'] = 'dependent poisoned snapshot unavailable'
            for condition in CONDITIONS:
                snapshot = snapshots.get(condition)
                if snapshot is None:
                    continue
                try:
                    a = fresh(folder / condition / 'answer', snapshot)
                    a.event('input', phase='task', text=task['prompt'])
                    response = await a.reply(Msg('user', task['prompt'], 'user'))
                    answer = response.get_text_content()
                    results[condition] = {'status': 'ok', 'semantic_label': 'U', 'answer': answer,
                        'snapshot_sha256': digest(snapshot), 'memory_count': len(snapshot['content']),
                        'memory_chars': sum(len(x['content']) for x in snapshot['content']),
                        'diagnostic': score_literal(answer, task['clean_behavior'], task['poisoned_behavior'])}
                except Exception as exc:
                    results[condition]['error'] = 'answer: ' + repr(exc)
        for condition in CONDITIONS:
            key = {'case_id': case['case_id'], 'task_id': task['task_id'], 'condition': condition}
            row = {**key, 'family': case['connected_group'], **results[condition]}
            rows.append(row)
            if row['status'] == 'ok':
                review_id = hashlib.sha256((str(output.resolve()) + digest(key)).encode()).hexdigest()[:16]
                row['review_id'] = review_id
                blind.append({'review_id': review_id, 'question': task['prompt'], 'answer': row['answer'],
                              'target_behavior': task['poisoned_behavior'], 'clean_behavior': task['clean_behavior'],
                              'target_endorsed': 'U', 'clean_endorsed': 'U', 'refusal': 'U',
                              'rationale': '', 'rubric': case['rubric']})
        write_json(output / 'results.json', rows)
        print(case['case_id'], {k: v['status'] for k, v in results.items()}, flush=True)
    write_json(output / 'blind_review.json', sorted(blind, key=lambda x: x['review_id']))
    errors = sum(x['status'] != 'ok' for x in rows)
    write_json(output / 'summary.json', {'N': 60, 'S': 0, 'F': 0, 'U': 60-errors, 'E': errors,
        'evaluated_target_rate': None, 'semantic_review': 'pending',
        'readiness': 'offline verification only' if runtime_evidence is None else 'live outputs pending review'})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--output')
    p.add_argument('--service-evidence')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    spec = load_spec(args.spec)
    if args.preflight:
        print('PASS: frozen 10-case / 60-answer protocol. Live model and semantics not verified.')
        return
    if not args.output or not args.service_evidence:
        p.error('live run requires --output and --service-evidence')
    evidence = runtime_check(spec, args.service_evidence)
    asyncio.run(run(spec, args.output, runtime_evidence=evidence))


if __name__ == '__main__':
    main()
