"""Real-service acceptance, including model-driven writes across restarts."""
import asyncio
import json
import uuid
from pathlib import Path
from app import build_agent, Msg, save_memory


async def main():
    import os
    root = Path(os.getenv('AGEMEM_HOME', '/root/autodl-tmp/agemem-norl')) / 'acceptance' / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    report = {'root': str(root), 'test_version': 3, 'checks': {}, 'answers': []}
    agent = build_agent(root / 'tools')
    result = await agent.add_memory('The validation project is named ORCHID-742.')
    mid = result.metadata['memory_id']
    report['checks']['add'] = result.metadata['success']
    retrieved = await agent.retrieve_memory('validation project name')
    report['checks']['retrieve'] = 'ORCHID-742' in str(retrieved.content)
    report['checks']['update'] = (await agent.update_memory(mid, 'The validation project is named ORCHID-743.')).metadata['success']
    save_memory(agent.memory_manager, root / 'tools/memory.json')
    restored = build_agent(root / 'tools')
    report['checks']['disk_restore'] = any(x.content.endswith('ORCHID-743.') for x in await restored.memory_manager.get_memory())
    agent.context_messages = [Msg('user', 'We discussed apples.', 'user'), Msg('assistant', 'Apples are fruit.', 'assistant')]
    summary = await agent.summary_context('all')
    report['checks']['summary_helper'] = summary.metadata['success'] and bool(summary.metadata['summary_msg'])
    # Official criteria parameter accepts a topic/keyword. Check actual removal
    # and preservation of an unrelated message, not just a success flag.
    agent.context_messages = [Msg('user', 'My favorite fruit is apples.', 'user'),
                              Msg('user', 'The server port is 8000.', 'user')]
    filtered = await agent.filter_context('apples')
    report['filter_result'] = filtered.metadata
    report['checks']['filter_helper'] = (filtered.metadata['success']
        and filtered.metadata['removed_count'] == 1
        and [m.get_text_content() for m in agent.context_messages] == ['The server port is 8000.'])
    report['checks']['delete_confirmation'] = not (await agent.delete_memory(mid)).metadata['success']
    report['checks']['delete'] = (await agent.delete_memory(mid, confirmation=True)).metadata['success'] and await agent.memory_manager.size() == 0

    # The model, not the test harness, must choose add_memory here.
    agent = build_agent(root / 'model')
    code = 'AZALEA-' + uuid.uuid4().hex[:8]
    reply = await agent.reply(Msg('user', f'Please use add_memory to remember that my validation project codename is {code}.', 'user'))
    report['answers'].append(reply.get_text_content())
    save_memory(agent.memory_manager, root / 'model/memory.json')
    report['checks']['model_write'] = any(code in x.content for x in await agent.memory_manager.get_memory())
    restored = build_agent(root / 'model')
    assert not restored.context_messages
    reply = await restored.reply(Msg('user', 'Please retrieve my validation project codename from long-term memory. Return the exact complete codename, including its suffix, without abbreviating it.', 'user'))
    report['answers'].append(reply.get_text_content())
    report['checks']['model_cross_session_recall'] = code in reply.get_text_content()
    calls = [json.loads(x) for x in (root / 'model/events.jsonl').read_text().splitlines()]
    report['checks']['model_retrieval_tool'] = any(x.get('call', {}).get('name') == 'retrieve_memory' for x in calls)
    report['passed'] = all(report['checks'].values())
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
