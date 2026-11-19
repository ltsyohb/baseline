"""Contract tests: real upstream tools, deterministic offline substitutes."""
import asyncio
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('DASHSCOPE_API_KEY', 'offline-test-not-a-real-key')
from app import LoggedAgeMem, AgentScopeLongtermMemory, Msg, OpenAIChatFormatter, official_prompt, save_memory
from agentscope.model import ChatModelBase, ChatResponse


class FakeMemory(AgentScopeLongtermMemory):
    def embed(self, content):
        return [1.0, 0.5]  # No semantic-quality claim; tests store mechanics.


class FakeHelper:
    def chat(self, messages, model_name=None):
        return '0.0' if 'similarity' in messages[0]['content'].lower() else 'Test summary'


class FakeModel(ChatModelBase):
    def __init__(self):
        super().__init__('offline-fake', False)
        self.count = 0

    async def __call__(self, messages, tools=None, **kwargs):
        self.count += 1
        if self.count == 1:
            return ChatResponse(content=[{'type': 'tool_use', 'id': 'test-add', 'name': 'add_memory', 'input': {'content': 'project ORCHID'}}])
        return ChatResponse(content=[{'type': 'text', 'text': 'Saved.'}])


async def run():
    with tempfile.TemporaryDirectory() as folder:
        agent = LoggedAgeMem(name='test', sys_prompt=official_prompt(),
                             model=FakeModel(), formatter=OpenAIChatFormatter(),
                             memory=FakeMemory(), chat_client=FakeHelper(), state_dir=folder)
        names = {s['function']['name'] for s in agent.toolkit.get_json_schemas()}
        assert names == {'summary_context', 'filter_context', 'retrieve_memory', 'add_memory', 'update_memory', 'delete_memory', 'generate_response'}
        reply = await agent.reply(Msg('user', 'Remember project ORCHID', 'user'))
        assert reply.get_text_content() == 'Saved.'
        items = await agent.memory_manager.retrieve('project')
        assert len(items) == 1
        mid = items[0].memory_id
        assert (await agent.update_memory(mid, 'project LILY')).metadata['success']
        save_memory(agent.memory_manager, Path(folder) / 'memory.json')
        restored = FakeMemory()
        restored.load_state_dict(json.loads((Path(folder) / 'memory.json').read_text()))
        assert (await restored.get_memory())[0].content == 'project LILY'
        agent.context_messages.clear()
        assert await agent.memory_manager.size() == 1
        agent.context_messages.append(Msg('user', 'test fact', 'user'))
        assert (await agent.summary_context('all')).metadata['success']
        assert (await agent.filter_context('unrelated')).metadata['success']
        assert not (await agent.delete_memory(mid, confirmation=False)).metadata['success']
        assert (await agent.delete_memory(mid, confirmation=True)).metadata['success']
        assert await agent.memory_manager.size() == 0
        events = [json.loads(x) for x in (Path(folder) / 'events.jsonl').read_text().splitlines()]
        assert any(x['kind'] == 'tool_call' for x in events)
        print('PASS: upstream six-tool registration, tool loop, CRUD, persistence, context reset and event logging (offline substitutes).')


if __name__ == '__main__':
    asyncio.run(run())
