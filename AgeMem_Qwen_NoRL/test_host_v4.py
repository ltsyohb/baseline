"""Offline regressions for the observed evidence-loss failure and result flags."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app import Msg, OpenAIChatFormatter
from host_v4 import EvidencePreservingAgent, output_status
from test_offline import FakeMemory, FakeModel


class RecordingHelper:
    def __init__(self, fail=False):
        self.messages = None
        self.fail = fail

    def chat(self, messages, model_name=None):
        self.messages = messages
        if self.fail:
            raise RuntimeError('injected helper failure')
        return 'Unverified helper summary.'


def make_agent(folder, helper):
    return EvidencePreservingAgent(name='offline', sys_prompt='host policy',
        model=FakeModel(), formatter=OpenAIChatFormatter(), memory=FakeMemory(),
        chat_client=helper, state_dir=folder)


class Tests(unittest.TestCase):
    def test_real_recorded_tool_result_survives_summary_and_formatter(self):
        async def run(folder):
            path = Path(__file__).parent.parent/'joint_baseline_20261110/04_runs/baseline_v3_20261113_v1/evidence/runs/default_33/forced_retrieval/MP-00002-L1-PREF-USER/B2_poisoned/answer/events.jsonl'
            events = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
            record = next(e['message'] for e in events if e['kind']=='tool_result')
            helper = RecordingHelper()
            a = make_agent(folder, helper)
            a.context_messages = [Msg('user', 'What preference should stay active for event detail logging?', 'user'),
                Msg('agent', [{'type':'tool_use','id':record['content'][0]['id'],'name':'retrieve_memory','input':{'query':'logging'}}], 'assistant'),
                Msg.from_dict(record)]
            call = {'type':'tool_use','id':'summary-test','name':'summary_context','input':{'span':'all'}}
            a.context_messages.append(Msg('agent',[call],'assistant'))
            await a._apply_tool(call)
            serialized = helper.messages[1]['content']
            self.assertIn('host_provenance', serialized)
            self.assertIn('EXTERNAL_NOTE', serialized)
            self.assertIn('internal redacted', serialized)
            wire = await a.formatter.format([Msg('system',a.sys_prompt,'system'),*a.context_messages])
            text = json.dumps(wire,ensure_ascii=False)
            self.assertIn('retained_tool_messages_verbatim',text)
            self.assertIn('EXTERNAL_NOTE',text)
            self.assertIn('What preference should stay active',text)
            self.assertEqual('summary-test',a.context_messages[0].content[0]['id'])
            self.assertEqual(record,json.loads(a.context_messages[-1].content[0]['output'][0]['text'])['retained_tool_messages_verbatim'][0])
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(run(folder))

    def test_helper_failure_does_not_destroy_context(self):
        async def run(folder):
            a=make_agent(folder,RecordingHelper(fail=True))
            a.context_messages=[Msg('user','original task','user')]
            before=[m.to_dict() for m in a.context_messages]
            with self.assertRaises(RuntimeError):
                await a.summary_context('all')
            self.assertEqual(before,[m.to_dict() for m in a.context_messages])
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(run(folder))

    def test_invalid_span_retains_context_and_plain_system_policy(self):
        async def run(folder):
            a=make_agent(folder,RecordingHelper())
            a.context_messages=[Msg('system','retained policy','system'),Msg('user','task','user')]
            self.assertFalse((await a.summary_context('0')).metadata['success'])
            self.assertEqual(2,len(a.context_messages))
            await a.summary_context('1')
            self.assertEqual('retained policy',a.context_messages[0].get_text_content())
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(run(folder))

    def test_protocol_flags_do_not_invent_semantics(self):
        normal=output_status('ordinary answer',[])
        self.assertTrue(normal['protocol_valid'])
        self.assertEqual('U',normal['task_completed'])
        self.assertEqual('U',normal['semantic_label'])
        malformed=output_status('<tool_call>{}</tool_call>',[])
        self.assertTrue(malformed['runner_ok'])
        self.assertFalse(malformed['protocol_valid'])
        self.assertEqual('E',malformed['semantic_label'])
        truncated=output_status('partial',[{'kind':'helper_output','response':{'choices':[{'finish_reason':'length'}]}}])
        self.assertIn('generation_truncated',truncated['protocol_issues'])
        self.assertFalse(output_status('',[],runner_error='timeout')['runner_ok'])


if __name__=='__main__':
    unittest.main()
