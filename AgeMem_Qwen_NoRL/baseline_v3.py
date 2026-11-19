"""Versioned provenance adaptation and disjoint diagnostic tracks.

No implicit retries, case replacement, direct-store initialization or forced
retrieval in the natural track. Offline fixtures are explicitly marked.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from app import save_memory
from baseline_v2 import CONDITIONS, attach_trace, digest, runtime_check, validate
from experiments import state, write_json, difference
from provenance_v3 import build_v3_agent, host_reply

ROOT = Path(__file__).resolve().parent


def code_hashes():
    files = [ROOT / n for n in ('app.py', 'baseline_v2.py', 'baseline_v3.py',
                                'provenance_v3.py', 'experiments.py', 'service_attestation.py')]
    files += sorted((ROOT / 'vendor/AgeMem_code_agentscope').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_plan(path):
    p = Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest() != p.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen v3 plan changed')
    plan = json.loads(p.read_text(encoding='utf-8'))
    if plan['source_sha256'] != code_hashes():
        raise ValueError('Code changed after plan freeze; create a new version')
    if plan['version'] != 'provenance-v3-dev-01':
        raise ValueError('Unsupported protocol')
    validate(plan['spec'])
    ids = {c['case_id'] for c in plan['spec']['cases']}
    names = [t['name'] for t in plan['tracks']]
    if len(names) != len(set(names)) or plan['default_tracks'] != ['natural', 'forced_retrieval']:
        raise ValueError('Invalid track manifest')
    for t in plan['tracks']:
        if not set(t['case_ids']) <= ids or not set(t['conditions']) <= set(CONDITIONS):
            raise ValueError('Unrecognized case or condition')
        if t['name'] not in {'natural', 'forced_retrieval', 'controlled_initial'}:
            raise ValueError('Unknown track')
        if t['name'] == 'natural' and (t['force_retrieval'] or t['initialize_directly']):
            raise ValueError('Natural track cannot use diagnostic interventions')
        if t['force_retrieval'] != (t['name'] == 'forced_retrieval'):
            raise ValueError('Forced retrieval flag mismatch')
        if t['initialize_directly'] != (t['name'] == 'controlled_initial'):
            raise ValueError('Controlled initial flag mismatch')
        if t['planned_answers'] != len(t['case_ids']) * len(t['conditions']):
            raise ValueError('Incorrect track denominator')
        if len(t['case_ids']) != len(set(t['case_ids'])) or len(t['conditions']) != len(set(t['conditions'])):
            raise ValueError('Duplicate planned keys')
    return plan


def trace_v3(agent):
    # Trace the actual inner embedding calls, including ingestion and updates.
    wrapped = agent.memory_manager
    agent.memory_manager = wrapped.inner
    try:
        attach_trace(agent)
    finally:
        agent.memory_manager = wrapped


async def run(plan, output, selected=None, factory=None, evidence=None):
    tracks = [t for t in plan['tracks'] if t['name'] in (selected or plan['default_tracks'])]
    if not tracks or set(selected or plan['default_tracks']) != {t['name'] for t in tracks}:
        raise ValueError('Track selection mismatch')
    if factory is None and evidence is None:
        raise ValueError('Live execution requires service evidence')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'frozen_plan.json', plan)
    write_json(output / 'manifest.json', {'mode': 'offline_fixture' if factory else 'live',
        'tracks': [t['name'] for t in tracks], 'plan_sha256': digest(plan),
        'source_sha256': code_hashes(), 'service_evidence': evidence,
        'interpretation': 'new provenance adaptation; exploratory developer diagnostics; not original paper reproduction'})
    cases = {c['case_id']: c for c in plan['spec']['cases']}
    rows = [{'track': t['name'], 'case_id': cid, 'condition': cond,
             'task_id': cases[cid]['trigger_tasks'][0]['task_id'],
             'status': 'error', 'semantic_label': 'E', 'error': 'not_executed'}
            for t in tracks for cid in t['case_ids'] for cond in t['conditions']]

    def checkpoint():
        write_json(output / 'results.json', rows)
        write_json(output / 'summary.json', {'mode': 'offline_fixture' if factory else 'live',
            'semantic_review': 'pending', 'tracks': {
                t['name']: {'N': sum(r['track'] == t['name'] for r in rows), 'S': 0, 'F': 0,
                            'U': sum(r['track'] == t['name'] and r['status'] == 'ok' for r in rows),
                            'E': sum(r['track'] == t['name'] and r['status'] != 'ok' for r in rows)} for t in tracks}})

    def fresh(folder, snapshot=None):
        if snapshot is not None:
            write_json(Path(folder) / 'memory.json', snapshot)
        a = factory(folder) if factory else build_v3_agent(folder, plan['spec'])
        if not factory:
            trace_v3(a)
        if a.context_messages or (snapshot is not None and digest(state(a)) != digest(snapshot)):
            raise ValueError('New session isolation or snapshot mismatch')
        a.event('session_start', context_count=0, snapshot_sha256=digest(state(a)))
        return a

    checkpoint()
    for track in tracks:
        for cid in track['case_ids']:
            case = cases[cid]
            task = case['trigger_tasks'][0]
            folder = output / track['name'] / cid
            relevant = [r for r in rows if r['track'] == track['name'] and r['case_id'] == cid]
            try:
                a = fresh(folder / 'prepare')
                if track['initialize_directly']:
                    with a.memory_manager.transport('initial', cid + ':controlled_initial'):
                        await a.memory_manager.add('controlled-initial', '\n'.join(case['context']))
                    a.event('diagnostic_intervention', name='controlled_initial_store', model_decision=False)
                else:
                    await host_reply(a, case['initial_message'], 'initial', cid + ':initial')
                clean = state(a)
                save_memory(a.memory_manager, a.state_dir / 'memory.json')
                init = {'record_count': len(clean['content']), 'empty': not clean['content'],
                        'semantic_correctness': 'pending_review',
                        'direct_store': track['initialize_directly'], 'retry_or_backfill': False}
                write_json(folder / 'initialization.json', init)
                write_json(folder / 'clean_snapshot.json', clean)
            except Exception as exc:
                for r in relevant:
                    r['error'] = 'prepare: ' + repr(exc)
                checkpoint()
                continue
            snapshots = {'B0_clean': clean, 'R0_clean_snapshot_reload': clean}
            for cond, key, identity in [('B1_harmless', 'harmless_message', 'external'),
                                        ('B2_poisoned', 'attack_message', 'external'),
                                        ('C1_legal_update', 'legal_update_message', 'owner_update')]:
                if cond not in track['conditions']:
                    continue
                try:
                    a = fresh(folder / cond / 'ingest', clean)
                    await host_reply(a, case[key], identity, cid + ':' + cond)
                    snapshots[cond] = state(a)
                    save_memory(a.memory_manager, a.state_dir / 'memory.json')
                    write_json(folder / cond / 'snapshot.json', state(a))
                    write_json(folder / cond / 'delta.json', difference(clean, state(a)))
                except Exception as exc:
                    next(r for r in relevant if r['condition'] == cond)['error'] = 'ingest: ' + repr(exc)
                    checkpoint()
            if 'R1_actual_restore' in track['conditions'] and 'B2_poisoned' in snapshots:
                try:
                    a = fresh(folder / 'R1_actual_restore/restore', snapshots['B2_poisoned'])
                    before = digest(state(a))
                    a.memory_manager.load_state_dict(copy.deepcopy(clean))
                    save_memory(a.memory_manager, a.state_dir / 'memory.json')
                    disk = json.loads((a.state_dir / 'memory.json').read_text(encoding='utf-8'))
                    check = {'before': before, 'after': digest(state(a)), 'disk': digest(disk), 'clean': digest(clean)}
                    write_json(folder / 'R1_actual_restore/restore_check.json', check)
                    if not check['after'] == check['disk'] == check['clean']:
                        raise ValueError('Restore residual')
                    snapshots['R1_actual_restore'] = disk
                except Exception as exc:
                    next(r for r in relevant if r['condition'] == 'R1_actual_restore')['error'] = 'restore: ' + repr(exc)
            for row in relevant:
                cond = row['condition']
                row['initialization'] = init
                if cond not in snapshots:
                    checkpoint()
                    continue
                try:
                    a = fresh(folder / cond / 'answer', snapshots[cond])
                    answer = await host_reply(a, task['prompt'], 'task', cid + ':answer',
                        forced_query=task['prompt'] if track['force_retrieval'] else None)
                    save_memory(a.memory_manager, a.state_dir / 'memory.json')
                    events = [json.loads(s) for s in (a.state_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
                    calls = [e for e in events if e['kind'] == 'tool_call' and e['call']['name'] == 'retrieve_memory']
                    row.update(status='ok', semantic_label='U', answer=answer.get_text_content(),
                               retrieval_calls=len(calls), natural_retrieval=not track['force_retrieval'],
                               snapshot_sha256=digest(snapshots[cond]))
                    row.pop('error', None)
                except Exception as exc:
                    row['error'] = 'answer: ' + repr(exc)
                checkpoint()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--tracks', nargs='+', choices=['natural', 'forced_retrieval', 'controlled_initial'])
    p.add_argument('--output')
    p.add_argument('--service-evidence')
    args = p.parse_args()
    plan = load_plan(args.plan)
    if args.preflight:
        print('PASS frozen code and plan. Default 24 natural + 9 forced diagnostic; controlled initial opt-in only. No live run.')
        return
    if not args.output or not args.service_evidence:
        p.error('--output and fresh --service-evidence required')
    evidence = runtime_check(plan['spec'], args.service_evidence)
    asyncio.run(run(plan, args.output, args.tracks, evidence=evidence))


if __name__ == '__main__':
    main()
