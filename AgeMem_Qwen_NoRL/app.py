"""Local Qwen + upstream AgeMem tools. No RL or adapter weights."""
import argparse
import asyncio
import ast
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from openai import OpenAI
from AgeMem_code_agentscope.agent import AgeMem
from AgeMem_code_agentscope.memory import AgentScopeLongtermMemory

UPSTREAM_COMMIT = '98f563f907d67b2f2436e3ae7b7ceff32e482814'
MODEL_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def official_prompt():
    tree = ast.parse((ROOT / 'vendor/AgeMem_code_agentscope/main.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sys_prompt' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('Upstream system prompt missing')


def save_memory(memory, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(memory.state_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


class LocalHelper:
    def __init__(self, base_url, model):
        self.client = OpenAI(api_key='local', base_url=base_url, timeout=120, max_retries=0)
        self.model = model
        self.event = lambda *args, **kwargs: None

    def chat(self, messages, model_name=None):
        # Upstream tools hard-code qwen-max. This adapter explicitly routes
        # auxiliary summaries and relevance scores to the local Qwen instead.
        self.event('helper_input', model=self.model, messages=messages,
                   temperature=0, max_tokens=1024, seed=42)
        result = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=1024, seed=42)
        self.event('helper_output', response=result.model_dump())
        return result.choices[0].message.content or ''


class LoggedAgeMem(AgeMem):
    def __init__(self, *args, state_dir, **kwargs):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        super().__init__(*args, **kwargs)
        # The official class also exposes execute_python_code. The baseline
        # deliberately exposes only the six memory tools and response tool.
        self.toolkit.remove_tool_function('execute_python_code')

    def event(self, kind, **data):
        record = dict(time=datetime.now(timezone.utc).isoformat(), run_id=self.run_id,
                      kind=kind, **data)
        with (self.state_dir / 'events.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    async def _apply_tool(self, tool_call):
        self.event('tool_call', call=tool_call)
        try:
            result = await super()._apply_tool(tool_call)
            self.event('tool_result', message=self.context_messages[-1].to_dict())
            return result
        except Exception as exc:
            self.event('tool_error', error=repr(exc))
            raise
        finally:
            save_memory(self.memory_manager, self.state_dir / 'memory.json')


class LoggedChatModel:
    """Keep the exact formatted prompt at the model boundary.

    AgentScope's formatter output is the authoritative model input. Logging
    only tool calls loses the distinction between a retrieved item and text
    that was actually sent to the model, so this small delegating wrapper
    records both for every iteration.
    """
    def __init__(self, model, owner):
        self._model = model
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._model, name)

    async def __call__(self, prompt, **kwargs):
        self._owner.event('model_input', prompt=prompt, tool_schema=kwargs.get('tools', []))
        result = await self._model(prompt, **kwargs)
        self._owner.event('model_output', response=getattr(result, 'content', result))
        return result


def install_wire_trace(client, event):
    """Attach async HTTP hooks without persisting authorization headers."""
    async def log_request(request):
        if request.url.path.endswith('/chat/completions'):
            event('model_http_request', body=json.loads(request.content))
    async def log_response(response):
        if response.request.url.path.endswith('/chat/completions'):
            await response.aread()
            event('model_http_response', status=response.status_code, body=response.json())
    client._client.event_hooks['request'].append(log_request)
    client._client.event_hooks['response'].append(log_response)


def build_agent(state_dir, *, policy_suffix='', runtime_config=None, agent_class=LoggedAgeMem):
    if not os.getenv('DASHSCOPE_API_KEY'):
        raise RuntimeError('Set DASHSCOPE_API_KEY for official text-embedding-v4 retrieval.')
    base_url = os.getenv('QWEN_BASE_URL', 'http://127.0.0.1:8000/v1')
    name = os.getenv('QWEN_MODEL', 'qwen2.5-7b-instruct')
    if runtime_config:
        expected_name = runtime_config['model']
        expected_url = runtime_config['base_url']
        if name != expected_name or base_url != expected_url:
            raise ValueError('Runtime environment differs from frozen model/base_url')
    helper = LocalHelper(base_url, name)
    model_inventory = helper.client.models.list()  # Fail before starting if absent.
    if runtime_config and name not in [x.id for x in model_inventory.data]:
        raise ValueError('Frozen model ID not advertised by server')
    memory = AgentScopeLongtermMemory(embedding_model='text-embedding-v4', embedding_dim=256)
    state_file = Path(state_dir) / 'memory.json'
    if state_file.exists():
        memory.load_state_dict(json.loads(state_file.read_text(encoding='utf-8')))
    agent = agent_class(
        name='AgeMem-Qwen-noRL', sys_prompt=official_prompt() + policy_suffix,
        model=OpenAIChatModel(model_name=name, api_key='local', stream=False,
                              client_args={'base_url': base_url, 'timeout': 120, 'max_retries': 0},
                              generate_kwargs={'temperature': 0, 'max_tokens': 1024, 'seed': 42}),
        formatter=OpenAIChatFormatter(), memory=memory, chat_client=helper,
        max_iters=10, max_context_tokens=8192, state_dir=state_dir)
    agent.model = LoggedChatModel(agent.model, agent)
    helper.event = agent.event
    # Persist the actual JSON HTTP request after SDK formatting, not just the
    # messages handed to the wrapper. Authorization headers are never saved.
    install_wire_trace(agent.model.client, agent.event)
    try:
        advertised_models = [getattr(x, 'id', str(x)) for x in model_inventory.data]
    except Exception:
        advertised_models = [str(model_inventory)]
    agent.event('run_start', model=name, base_model='Qwen/Qwen2.5-7B-Instruct',
                model_revision=MODEL_REVISION, upstream_commit=UPSTREAM_COMMIT,
                qwen_base_url=base_url, advertised_models=advertised_models,
                model_env=os.getenv('QWEN_MODEL', ''),
                condition='official-agentscope-tools-local-qwen-noRL',
                embedding_model='text-embedding-v4', embedding_dim=256,
                helper_model=name, temperature=0, max_tokens=1024, seed=42,
                tools=agent.toolkit.get_json_schemas())
    return agent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', default=os.getenv('AGEMEM_STATE_DIR', str(ROOT / 'state/default')))
    parser.add_argument('--smoke', action='store_true', help='Live tool checks; requires Qwen and embedding API')
    args = parser.parse_args()
    if args.smoke:
        args.state_dir = str(Path(args.state_dir) / ('smoke-' + uuid.uuid4().hex[:8]))
    agent = build_agent(args.state_dir)
    if args.smoke:
        result = await agent.add_memory('The test project codename is ORCHID-742.')
        mid = result.metadata['memory_id']
        found = await agent.memory_manager.retrieve('What is the test project codename?', 3)
        assert any(x.memory_id == mid for x in found), 'Retrieval did not return the stored record'
        assert (await agent.update_memory(mid, 'The test project codename is ORCHID-743.')).metadata['success']
        save_memory(agent.memory_manager, agent.state_dir / 'memory.json')
        restored = AgentScopeLongtermMemory()
        restored.load_state_dict(json.loads((agent.state_dir / 'memory.json').read_text(encoding='utf-8')))
        assert (await restored.get_memory())[0].content.endswith('ORCHID-743.')
        answer = await agent.reply(Msg('user', 'Use retrieve_memory to find the test project codename, then answer.', 'user'))
        agent.event('smoke_answer', message=answer.to_dict())
        assert (await agent.delete_memory(mid, confirmation=True)).metadata['success']
        save_memory(agent.memory_manager, agent.state_dir / 'memory.json')
        print('Live CRUD, persistence and model-loop checks passed. Inspect events.jsonl to assess model tool use.')
        return
    print('Qwen + official AgeMem tools (no RL). /new clears conversation; /memory shows saved memories; /quit exits.')
    while True:
        try:
            text = input('user> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == '/quit':
            break
        if text == '/new':
            agent.context_messages.clear()
            agent.event('new_session')
            print('Conversation cleared; long-term memory retained.')
            continue
        if text == '/memory':
            for item in await agent.memory_manager.get_memory():
                print(item.memory_id, item.content)
            continue
        if not text:
            continue
        agent.event('user', text=text)
        try:
            reply = await agent.reply(Msg('user', text, 'user'))
            agent.event('answer', message=reply.to_dict())
            print('agent>', reply.get_text_content())
        except Exception as exc:
            agent.event('error', error=repr(exc))
            print('Request failed:', exc)
            print('Use /new if context exceeds 8192 tokens; memory is retained.')
        finally:
            save_memory(agent.memory_manager, agent.state_dir / 'memory.json')


if __name__ == '__main__':
    asyncio.run(main())
