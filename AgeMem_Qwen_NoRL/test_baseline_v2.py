"""Regression checks for isolation, restoration, trace fidelity and denominator."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from app import LoggedAgeMem, LoggedChatModel, OpenAIChatFormatter, official_prompt, Msg, install_wire_trace
from baseline_v2 import CONDITIONS, attach_trace, load_spec, run, validate
from test_offline import FakeMemory, FakeHelper, FakeModel
from review_baseline_v2 import aggregate
from service_attestation import check_process

SPEC = Path(__file__).parent.parent / 'joint_baseline_20261110/data/pilot_l1_user_10.json'


class FixtureAgent:
    def __init__(self, folder, visits, fail=False):
        self.state_dir = Path(folder)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.context_messages = []
        self.memory_manager = FakeMemory()
        f = self.state_dir / 'memory.json'
        if f.exists():
            self.memory_manager.load_state_dict(json.loads(f.read_text(encoding='utf-8')))
        self.visits, self.fail = visits, fail

    def event(self, *args, **kwargs):
        pass

    async def reply(self, msg):
        text = msg.get_text_content()
        self.visits.append((str(self.state_dir), text, list(self.context_messages),
                            copy.deepcopy(self.memory_manager.state_dict())))
        self.context_messages.append(msg)
        if self.fail and 'B1_harmless' in str(self.state_dir) and 'ingest' in str(self.state_dir):
            raise RuntimeError('injected ingestion failure')
        if any(text.startswith(p) for p in ('AUTHORITY_INITIAL', 'AUTHORITY_UPDATE', 'EXTERNAL_NOTE')):
            await self.memory_manager.add(str(len(self.memory_manager.state_dict()['content'])), text)
        return Msg('agent', 'fixture output; not a model result', 'assistant')


class Tests(unittest.TestCase):
    def test_http_trace_matches_sent_body(self):
        import httpx
        from openai import AsyncOpenAI
        async def check():
            logs, received = [], []
            def handler(request):
                received.append(json.loads(request.content))
                return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion',
                    'created': 0, 'model': 'fake', 'choices': [{'index': 0,
                    'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                client = AsyncOpenAI(api_key='offline-secret', base_url='http://offline/v1', http_client=transport)
                install_wire_trace(client, lambda kind, **data: logs.append({'kind': kind, **data}))
                await client.chat.completions.create(model='fake', messages=[{'role': 'user', 'content': 'exact input'}], temperature=0)
            body = next(x['body'] for x in logs if x['kind'] == 'model_http_request')
            self.assertEqual(received[0], body)
            self.assertNotIn('offline-secret', json.dumps(logs))
        asyncio.run(check())

    def test_review_denominators_and_missing_keys(self):
        spec = load_spec(SPEC)
        rows, reviews = [], []
        for case in spec['cases']:
            for condition in CONDITIONS:
                rid = case['case_id'] + condition
                rows.append({'case_id': case['case_id'], 'task_id': case['trigger_tasks'][0]['task_id'],
                             'condition': condition, 'status': 'ok', 'review_id': rid, 'answer': 'sample'})
                reviews.append({'review_id': rid, 'answer': 'sample', 'target_endorsed': 'U',
                                'clean_endorsed': 'U', 'refusal': 'U', 'rationale': ''})
        report = aggregate(spec, rows, reviews)
        self.assertEqual(10, report['conditions']['B2_poisoned']['U'])
        self.assertIsNone(report['conditions']['B2_poisoned']['target_rate_among_reviewed'])
        with self.assertRaises(ValueError):
            aggregate(spec, rows[:-1], reviews)
        reviews[0]['target_endorsed'] = 'S'
        with self.assertRaises(ValueError):
            aggregate(spec, rows, reviews)

    def test_service_launch_mismatch_rejected(self):
        argv = ['vllm', 'serve', '/models/qwen', '--served-model-name', 'qwen2.5-7b-instruct',
                '--host', '127.0.0.1', '--port', '8000', '--max-model-len', '8192',
                '--tool-call-parser', 'hermes', '--generation-config', 'vllm']
        check_process({'argv': argv}, '/models/qwen', {'model': 'qwen2.5-7b-instruct'})
        with self.assertRaises(ValueError):
            check_process({'argv': argv}, '/models/other', {'model': 'qwen2.5-7b-instruct'})
        with self.assertRaises(ValueError):
            check_process({'argv': argv}, '/models/qwen', {'model': 'other-model'})
    def test_paired_isolation_and_restore(self):
        spec = load_spec(SPEC)
        visits = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            rows = asyncio.run(run(spec, out, lambda f: FixtureAgent(f, visits)))
            self.assertEqual(60, len(rows))
            self.assertEqual(60, len({(x['case_id'], x['condition']) for x in rows}))
            self.assertTrue(all(x['semantic_label'] == 'U' for x in rows))
            for folder, text, history, snapshot in visits:
                self.assertEqual([], history)
                if 'B2_poisoned' in folder and 'ingest' in folder:
                    self.assertFalse(any('Administrative note' in x['content'] for x in snapshot['content']))
                if 'R1_actual_restore' in folder and 'answer' in folder:
                    self.assertEqual(1, len(snapshot['content']))
            for c in spec['cases']:
                check = json.loads((out/c['case_id']/'R1_actual_restore/restore_check.json').read_text())
                self.assertNotEqual(check['before_sha256'], check['after_sha256'])
                self.assertEqual(check['after_sha256'], check['disk_sha256'])
                self.assertTrue(all(not v for v in check['residual'].values()))

    def test_failures_preserve_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = asyncio.run(run(load_spec(SPEC), Path(tmp)/'failure', lambda f: FixtureAgent(f, [], True)))
            self.assertEqual(60, len(rows))
            self.assertEqual(10, sum(x['semantic_label'] == 'E' for x in rows))
            self.assertEqual(50, sum(x['semantic_label'] == 'U' for x in rows))

    def test_budget_and_pairing_fail_closed(self):
        spec = load_spec(SPEC)
        bad = copy.deepcopy(spec)
        bad['cases'][0]['tokens']['harmless'] += 20
        with self.assertRaises(ValueError):
            validate(bad)
        bad = copy.deepcopy(spec)
        bad['cases'][1]['connected_group'] = bad['cases'][0]['connected_group']
        with self.assertRaises(ValueError):
            validate(bad)

    def test_actual_model_boundary_and_retrieval_trace(self):
        async def check(tmp):
            a = LoggedAgeMem(name='test', sys_prompt=official_prompt(), model=FakeModel(),
                             formatter=OpenAIChatFormatter(), memory=FakeMemory(),
                             chat_client=FakeHelper(), state_dir=tmp)
            a.model = LoggedChatModel(a.model, a)
            attach_trace(a)
            await a.reply(Msg('user', 'Remember project ORCHID', 'user'))
            await a.memory_manager.retrieve('ORCHID', 3)
            events = [json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(e['kind'] == 'model_input' and e['prompt'] for e in events))
            self.assertTrue(any(e['kind'] == 'memory_transition' and e['delta']['added'] for e in events))
            self.assertTrue(any(e['kind'] == 'embedding' and e['text'] == 'ORCHID' for e in events))
            self.assertTrue(any(e['kind'] == 'vector_search' and e['results'] for e in events))
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(tmp))


if __name__ == '__main__':
    unittest.main()
