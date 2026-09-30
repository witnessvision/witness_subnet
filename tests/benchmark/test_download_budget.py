"""Acquisition quotas survive restart and never create model scores."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from witness.benchmark import download_budget as module
from witness.benchmark.download_budget import DownloadBudget, DownloadBudgetExceeded

MODEL = 'a' * 64


def test_active_time_accumulates_across_restart_but_waiting_time_does_not(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    with DownloadBudget(tmp_path, MODEL) as first:
        assert first.remaining_s() == 900
        now[0] += 45
    now[0] += 10000  # Queued or offline time is not a download.
    with DownloadBudget(tmp_path, MODEL) as restarted:
        assert restarted.used == 45
        now[0] += 15
    path = tmp_path / (MODEL + '.json')
    assert json.loads(path.read_text()) == {'charged_s': 60., 'active': False}
    assert path.stat().st_mode & 0o777 == 0o600


def test_crash_keeps_reserved_time_and_cannot_reset_quota(tmp_path):
    script = ('from pathlib import Path; import os; '
              'from witness.benchmark.download_budget import DownloadBudget; '
              f'b=DownloadBudget(Path({str(tmp_path)!r}), {MODEL!r}); b.__enter__(); os._exit(0)')
    subprocess.run([sys.executable, '-c', script], check=True, timeout=10)
    record = json.loads((tmp_path / (MODEL + '.json')).read_text())
    assert record == {'charged_s': 900., 'active': True}
    with pytest.raises(DownloadBudgetExceeded):
        with DownloadBudget(tmp_path, MODEL):
            pytest.fail('crash reset the acquisition quota')


def test_exhausted_quota_survives_restart_and_does_not_touch_existing_partial(tmp_path):
    path = tmp_path / (MODEL + '.json')
    path.write_text(json.dumps({'charged_s': 5400., 'active': False}))
    partial = tmp_path / 'existing.part'
    partial.write_bytes(b'preserved prefix')
    for _ in range(2):
        with pytest.raises(DownloadBudgetExceeded):
            with DownloadBudget(tmp_path, MODEL):
                pytest.fail('exhausted model was allowed to download')
    assert partial.read_bytes() == b'preserved prefix'
    assert json.loads(path.read_text())['charged_s'] == 5400


def test_concurrent_download_cannot_spend_same_budget_twice(tmp_path):
    with DownloadBudget(tmp_path, MODEL):
        with pytest.raises(BlockingIOError):
            with DownloadBudget(tmp_path, MODEL):
                pytest.fail('second writer obtained budget')


@pytest.mark.parametrize('record', ['invalid json', '[]', '{"charged_s": -1}', '{"charged_s": "0"}'])
def test_corrupt_budget_is_infrastructure_failure_not_invalid_model(tmp_path, record):
    (tmp_path / (MODEL + '.json')).write_text(record)
    with pytest.raises(OSError, match='invalid_download_budget_record'):
        with DownloadBudget(tmp_path, MODEL):
            pytest.fail('invalid quota was accepted')


def test_only_fresh_readmission_renews_shared_prefetch_and_foreground_quota(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    budget = DownloadBudget(tmp_path, MODEL)
    budget.renew(200)
    with budget:
        now[0] += 900
    for block in (199, 200):
        DownloadBudget(tmp_path, MODEL).renew(block)
        with pytest.raises(DownloadBudgetExceeded):
            with DownloadBudget(tmp_path, MODEL):
                pytest.fail('replay granted a new quota')
    DownloadBudget(tmp_path, MODEL).renew(201)
    with DownloadBudget(tmp_path, MODEL) as resumed:
        assert resumed.used == 0
        now[0] += 300
    DownloadBudget(tmp_path, MODEL).renew(201)  # Foreground cannot reset prefetch spending.
    with DownloadBudget(tmp_path, MODEL) as foreground:
        assert foreground.used == 300
        assert foreground.remaining_s() == 600


def test_readmission_cannot_reset_a_live_acquisition(tmp_path):
    with DownloadBudget(tmp_path, MODEL):
        with pytest.raises(BlockingIOError):
            DownloadBudget(tmp_path, MODEL).renew(200)
