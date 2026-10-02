import pytest

from butler_bridge import heads


@pytest.mark.parametrize('result', [
    heads.ProcResult(1, stderr='authentication failed'),
    heads.ProcResult(1, stderr='No conversation found with session ID: other'),
    heads.ProcResult(1, stdout='partial output',
                     stderr='No conversation found with session ID: lost'),
    heads.ProcResult(-1, timed_out=True,
                     stderr='No conversation found with session ID: lost'),
    heads.ProcResult(0, stderr='No conversation found with session ID: lost'),
])
def test_only_exact_pre_execution_rejection_is_retried(result):
    assert not heads.missing_claude_session(result, 'lost')


@pytest.mark.asyncio
async def test_failed_recovery_preserves_session_and_history(config, state, monkeypatch):
    state.activate('claude', 'lost')
    state.append_history('owner', 'keep me')
    before = state.sessions_path.read_bytes(), state.history_path.read_bytes()
    calls = []

    async def run(cmd, **kwargs):
        calls.append(cmd)
        return heads.ProcResult(1, stderr='No conversation found with session ID: lost')

    monkeypatch.setattr(heads, 'run_process', run)
    result = await heads.run_head('claude', 'question', config, state)
    assert not result.text
    assert len(calls) == 2
    assert (state.sessions_path.read_bytes(), state.history_path.read_bytes()) == before
