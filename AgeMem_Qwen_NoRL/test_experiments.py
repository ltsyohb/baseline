"""Check injection roles and snapshot isolation using upstream formatter."""
import asyncio
import tempfile
from pathlib import Path
from test_offline import FakeMemory, FakeHelper, FakeModel
from app import LoggedAgeMem, OpenAIChatFormatter, official_prompt
from experiments import ingest, state, difference


async def main():
    with tempfile.TemporaryDirectory() as folder:
        for channel in ['user_input','tool_return','cross_agent']:
            a = LoggedAgeMem(name='test',sys_prompt=official_prompt(),model=FakeModel(),
                             formatter=OpenAIChatFormatter(),memory=FakeMemory(),
                             chat_client=FakeHelper(),state_dir=Path(folder)/channel)
            before=state(a)
            await ingest(a,channel,{'content':'A synthetic test observation'})
            after=state(a)
            assert len(difference(before,after)['added']) == 1
            a.memory_manager.load_state_dict(before)
            assert await a.memory_manager.size() == 0
            assert len(after['content']) == 1
    print('PASS: three injection channels format successfully; clean/poison snapshot isolation.')


if __name__=='__main__':
    asyncio.run(main())
