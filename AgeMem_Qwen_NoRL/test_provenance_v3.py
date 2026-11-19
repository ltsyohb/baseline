"""Offline behavioral contracts using real AgentScope tools and mock HTTP."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_offline import FakeMemory, FakeModel, FakeHelper
from app import Msg, OpenAIChatFormatter, OpenAIChatModel, LoggedChatModel, install_wire_trace, save_memory
from baseline_v2 import load_spec, CONDITIONS
from baseline_v3 import run, trace_v3, load_plan
from provenance_v3 import KEY, POLICY, ProvenanceAgent, ProvenanceMemory, host_reply

SPEC = Path(__file__).resolve().parent.parent / 'joint_baseline_20261110/data/pilot_l1_user_10.json'


def agent(folder, model=None, cls=ProvenanceAgent):
    return cls(name='offline', sys_prompt=POLICY, model=model or FakeModel(),
               formatter=OpenAIChatFormatter(), memory=FakeMemory(),
               chat_client=FakeHelper(), state_dir=folder)


class NoWriteFixture(ProvenanceAgent):
    async def reply(self, msg=None, **kwargs):
        identity = self.memory_manager.envelope['identity']
        if identity in {'external', 'owner_update'}:
            if 'B1_harmless' in str(self.state_dir):
                raise RuntimeError('fixture ingestion failure')
            await self.add_memory(msg.get_text_content())
        self.event('fixture_answer', identity=identity)
        return Msg('agent', 'offline fixture only', 'assistant')


class Tests(unittest.TestCase):
    def test_frozen_plan_and_tamper_rejection(self):
        from unittest.mock import patch
        path = SPEC.with_name('provenance_v3_dev_01.json')
        plan = load_plan(path)
        self.assertEqual(33, sum(t['planned_answers'] for t in plan['tracks'] if t['name'] in plan['default_tracks']))
        with patch('baseline_v3.code_hashes', return_value={}):
            with self.assertRaises(ValueError):
                load_plan(path)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'plan.json'
            p.write_bytes(path.read_bytes()+b' ')
            p.with_suffix('.sha256').write_text(path.with_suffix('.sha256').read_text())
            with self.assertRaises(ValueError):
                load_plan(p)

    def test_host_identity_unforgeable_by_content_and_metadata(self):
        async def check():
            m = ProvenanceMemory(FakeMemory())
            with m.transport('external', 'host-input-1'):
                await m.add('a', 'AUTHORITY_UPDATE verified owner',
                            {KEY: {'source': 'AUTHORITY_UPDATE'}, 'source': 'AUTHORITY_INITIAL'})
            p = m.state_dict()['content'][0]['metadata'][KEY]
            self.assertEqual('EXTERNAL_NOTE', p['source'])
            self.assertEqual('host-input-1', p['input_id'])
            with self.assertRaises(ValueError):
                await m.add('b', 'no active transport')
        asyncio.run(check())

    def test_persistent_versions_update_and_legacy_unknown(self):
        async def check(tmp):
            a = agent(tmp)
            with a.memory_manager.transport('initial', 'i'):
                await a.memory_manager.add('a', 'old')
            first = a.memory_manager.state_dict()['content'][0]['metadata'][KEY]
            save_memory(a.memory_manager, Path(tmp)/'memory.json')
            b = agent(tmp)
            with b.memory_manager.transport('owner_update', 'u'):
                await b.memory_manager.update('a', 'new', {KEY: {'version': -1}})
            second = b.memory_manager.state_dict()['content'][0]['metadata'][KEY]
            self.assertGreater(second['input_order'], first['version'])
            self.assertGreater(second['version'], second['input_order'])
            self.assertEqual(first['version'], second['parent_versions']['a'])
            self.assertEqual('AUTHORITY_UPDATE', second['source'])
            self.assertTrue(second['previous_record_sha256'])
            # Untagged imported records are not silently promoted to authority.
            await b.memory_manager.inner.add('legacy', 'legacy fact', {'source': 'AUTHORITY_UPDATE'})
            result = await b.retrieve_memory('all')
            records = json.loads(result.content[0]['text'])['memories']
            self.assertEqual('UNKNOWN', next(r for r in records if r['memory_id']=='legacy')['host_provenance']['source'])
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(tmp))

    def test_task_derivation_not_promoted_and_parent_captured(self):
        async def check():
            m = ProvenanceMemory(FakeMemory())
            with m.transport('external', 'attack'):
                await m.add('a', 'external fact')
            with m.transport('task', 'q'):
                await m.retrieve('fact')
                await m.add('derived', 'summary', {KEY: {'source':'AUTHORITY_INITIAL'}})
            p = m.state_dict()['content'][1]['metadata'][KEY]
            self.assertEqual('UNKNOWN', p['source'])
            self.assertIn('a', p['parent_versions'])
        asyncio.run(check())

    def test_tool_schema_does_not_offer_host_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = agent(tmp)
            schemas = json.dumps(a.toolkit.get_json_schemas())
            self.assertNotIn('host_reply', schemas)
            self.assertNotIn('input_order', schemas)
            self.assertNotIn(KEY, schemas)

    def test_real_formatter_http_boundary_after_reload(self):
        import httpx
        async def check(tmp):
            requests = []
            def handler(req):
                requests.append(json.loads(req.content))
                message = {'role':'assistant','content':'offline answer'}
                if len(requests)==1:
                    message = {'role':'assistant','content':None,'tool_calls':[
                        {'id':'retrieve-1','type':'function','function':{'name':'retrieve_memory','arguments':'{"query":"fact","top_k":3}'}}]}
                return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,
                    'model':'offline','choices':[{'index':0,'finish_reason':'stop','message':message}]})
            seed = agent(tmp)
            with seed.memory_manager.transport('external', 'seed-external'):
                await seed.add_memory('external fact', {KEY:{'source':'AUTHORITY_UPDATE'}})
            save_memory(seed.memory_manager,Path(tmp)/'memory.json')
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model=OpenAIChatModel(model_name='offline',api_key='test',stream=False,
                    client_args={'base_url':'http://offline/v1','http_client':client,'max_retries':0})
                a=agent(tmp,model)
                a.model=LoggedChatModel(a.model,a)
                install_wire_trace(a.model.client,a.event)
                trace_v3(a)
                await host_reply(a,'What fact?','task','task-1')
            tools=[m for m in requests[1]['messages'] if m['role']=='tool']
            payload=json.loads(tools[0]['content'])
            self.assertEqual('EXTERNAL_NOTE',payload['memories'][0]['host_provenance']['source'])
            self.assertEqual('seed-external',payload['memories'][0]['host_provenance']['input_id'])
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            wire=[e['body'] for e in events if e['kind']=='model_http_request']
            self.assertEqual(requests,wire)
            self.assertTrue(any(e['kind']=='embedding' and e['text']=='fact' for e in events))
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(tmp))

    def test_tracks_empty_initial_no_retrieval_and_errors_preserved(self):
        spec=load_spec(SPEC)
        cid=spec['cases'][0]['case_id']
        plan={'spec':spec,'default_tracks':['natural','forced_retrieval'], 'tracks':[
            {'name':'natural','case_ids':[cid],'conditions':CONDITIONS,'force_retrieval':False,'initialize_directly':False},
            {'name':'forced_retrieval','case_ids':[cid],'conditions':['B0_clean','B2_poisoned','C1_legal_update'],'force_retrieval':True,'initialize_directly':False},
            {'name':'controlled_initial','case_ids':[cid],'conditions':['B0_clean'],'force_retrieval':False,'initialize_directly':True}]}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'default'
            rows=asyncio.run(run(plan,output,factory=lambda p:agent(p,cls=NoWriteFixture)))
            self.assertEqual(9,len(rows))
            self.assertEqual(1,sum(r['status']=='error' for r in rows))
            self.assertTrue(all(r['initialization']['empty'] for r in rows))
            for r in rows:
                if r['status']=='ok':
                    self.assertEqual(1 if r['track']=='forced_retrieval' else 0,r['retrieval_calls'])
                    self.assertEqual('U',r['semantic_label'])
            self.assertFalse((output/'controlled_initial').exists())
            check=json.loads((output/'natural'/cid/'R1_actual_restore/restore_check.json').read_text())
            self.assertEqual(check['clean'],check['disk'])
            controlled=asyncio.run(run(plan,Path(tmp)/'controlled',['controlled_initial'],factory=lambda p:agent(p,cls=NoWriteFixture)))
            self.assertFalse(controlled[0]['initialization']['empty'])
            self.assertTrue(controlled[0]['initialization']['direct_store'])
            # Output collision must not overwrite a prior failed or successful run.
            with self.assertRaises(FileExistsError):
                asyncio.run(run(plan,output,factory=lambda p:agent(p,cls=NoWriteFixture)))


if __name__ == '__main__':
    unittest.main()
