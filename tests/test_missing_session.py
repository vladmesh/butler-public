import json
from dataclasses import replace

import pytest

from butler_bridge import heads


@pytest.mark.asyncio
@pytest.mark.parametrize('rotation_due', [False, True])
async def test_missing_claude_session_retries_with_history_without_reset(
    config, state, monkeypatch, rotation_due
):
    state.activate('claude', 'lost-session')
    if rotation_due:
        config = replace(config, session_max_turns=1)
        state.note_service_failure()
        state.note_service_failure()
    state.append_history('owner', 'remember this')
    state.digest_path.write_text('durable digest')
    original_history = state.history_path.read_bytes()
    calls = []

    async def run(cmd, **kwargs):
        calls.append(cmd)
        if '--resume' in cmd:
            return heads.ProcResult(
                1, stderr='No conversation found with session ID: lost-session\n'
            )
        assert '--resume' not in cmd
        assert 'remember this' in cmd[-1]
        assert 'durable digest' in cmd[-1]
        assert 'current question' in cmd[-1]
        return heads.ProcResult(
            0, stdout=json.dumps({'result': 'answer', 'session_id': 'new-session'})
        )

    monkeypatch.setattr(heads, 'run_process', run)
    result = await heads.run_head('claude', 'current question', config, state)
    assert result.text == 'answer'
    assert len(calls) == (3 if rotation_due else 2)
    assert state.history_path.read_bytes() == original_history
    assert state.read_dialog().resume_id('claude') == 'new-session'
