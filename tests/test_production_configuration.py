import asyncio
import json
import os

from witness.judge import JudgeConfig
from witness.subnet import production


def test_evaluation_worker_receives_frozen_provider_and_no_wallet_credentials(tmp_path,monkeypatch):
    config={'env':'/private/selected.env','judge_cache':'/private/cache','budget_path':'/private/daily.sqlite3',
            'calibration':'/private/calibration.json'}
    monkeypatch.setenv('BT_PW_fixture','wallet-test-only')
    monkeypatch.setenv('OPENAI_API_KEY','provider-test-only')
    monkeypatch.setenv('CREDENTIALS_DIRECTORY','/private/systemd')
    captured=[]
    async def process(command,**kwargs):
        captured.append((command,kwargs));return b'{}'
    monkeypatch.setattr(production,'run_process',process)
    _,evaluate=production.callbacks(config,tmp_path,JudgeConfig())
    asyncio.run(evaluate({'reference':{}},{'response':{}}))
    command,kwargs=captured[0]
    assert command[command.index('--provider')+1]=='saygm'
    assert command[command.index('--model')+1]=='gpt-5.6-luna'
    assert command[command.index('--effort')+1]=='low'
    assert kwargs['timeout']==300
    assert set(kwargs['env']) <= {'PATH','LANG','PYTHONPATH'}
    assert 'wallet-test-only' not in json.dumps(kwargs)
