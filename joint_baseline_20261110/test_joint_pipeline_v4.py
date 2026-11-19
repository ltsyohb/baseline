import asyncio
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joint_pipeline_v4 as pipeline
sys.path.insert(0,str(pipeline.KIT))
from host_v4 import EvidencePreservingAgent
from provenance_v3 import POLICY
from app import Msg, OpenAIChatFormatter
from test_offline import FakeMemory, FakeModel, FakeHelper


class FixtureAgent(EvidencePreservingAgent):
    async def reply(self,msg=None,**kwargs):
        if '/benign/' in self.state_dir.as_posix():
            raise RuntimeError('fixture failure retained')
        self.event('fixture_only',is_model_result=False)
        return Msg('fixture','<tool_call>{}</tool_call>' if '/fragment_A/' in self.state_dir.as_posix() else 'fixture answer','assistant')


def factory(folder,variant):
    # Upstream eagerly constructs its default client even when a helper is
    # supplied. Offline fixtures never need an SSL client or network transport.
    with patch('AgeMem_code_agentscope.agent.chat_client',return_value=FakeHelper()), patch('AgeMem_code_agentscope.memory.OpenAI'):
        return FixtureAgent(name='fixture',sys_prompt=POLICY,model=FakeModel(),
            formatter=OpenAIChatFormatter(),memory=FakeMemory(),chat_client=FakeHelper(),state_dir=folder)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=pipeline.load_plan()

    def test_complete_matrix_same_queries_and_budget(self):
        p=self.plan
        self.assertEqual(8,len({(j['dataset'],j['method']) for j in p['jobs']}))
        self.assertEqual(132,len(p['jobs']))
        self.assertEqual(68,len(p['replay_jobs'])*len(p['replay_variants']))
        pipeline.validate(p)
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(pipeline.TOKENIZER))
        for j in p['jobs']:
            self.assertEqual(j['payload_tokens'],sum(len(tokenizer.encode(t,add_special_tokens=False).ids) for t in j['records']))

    def test_missing_pair_and_dirty_control_rejected(self):
        p=copy.deepcopy(self.plan);p['jobs'].pop()
        with self.assertRaises(ValueError):pipeline.validate(p)
        p=copy.deepcopy(self.plan);p['jobs'][0]['records']=['unexpected']
        with self.assertRaises(ValueError):pipeline.validate(p)

    def test_evaluation_answers_not_used_by_attack_constructor(self):
        t=copy.deepcopy(self.plan['tasks'][0])
        before=pipeline.attack_templates(t['target'],'frozen suffix',['training question'])
        t['clean_behavior']='FLIPPED EVALUATION LABEL'
        after=pipeline.attack_templates(t['target'],'frozen suffix',['training question'])
        self.assertEqual(before,after)
        self.assertNotIn(t['clean_behavior'],json.dumps(after))

    def test_full_runner_retains_errors_no_overwrite_and_pairing(self):
        p=copy.deepcopy(self.plan)
        t=next(t for t in p['tasks'] if t['dataset']=='MemPoison-Bench')
        p['tasks']=[t]
        p['jobs']=[j for j in p['jobs'] if j['dataset']==t['dataset'] and j['case_id']==t['case_id']]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            rows=asyncio.run(pipeline.execute(p,out,'joint',factory=factory))
            self.assertEqual(22,len(rows))
            self.assertEqual(4,sum(r['status']=='error' for r in rows))
            self.assertEqual(5,sum(r['semantic_label']=='E' for r in rows))
            self.assertEqual(17,sum(r['semantic_label']=='U' for r in rows))
            clean=[r['input_snapshot_sha256'] for r in rows if r['condition']=='clean']
            self.assertEqual(1,len(set(clean)))
            self.assertEqual('offline_fixture',pipeline.read(out/'run_meta.json')['mode'])
            with self.assertRaises(FileExistsError):
                asyncio.run(pipeline.execute(p,out,'joint',factory=factory))

    def test_same_snapshot_replay_keeps_v3_v4_separate(self):
        p=copy.deepcopy(self.plan);p['replay_jobs']=p['replay_jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            rows=asyncio.run(pipeline.execute(p,Path(tmp)/'replay','replay',factory=factory))
            self.assertEqual({'v3','v4'},{r['variant'] for r in rows})
            self.assertEqual(rows[0]['input_snapshot_sha256'],rows[1]['input_snapshot_sha256'])


if __name__=='__main__':
    unittest.main()
