from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sqlite3

import pytest

from witness.budget import BudgetUnavailable, DailyBudget


def test_atomic_reservations_shared_by_provider_role_and_restart(tmp_path):
    path = tmp_path/'budget.sqlite3'
    budget = DailyBudget(path)
    def reserve(i):
        try:
            return DailyBudget(path).reserve(role='validator', provider=['saygm','openai'][i%2],
                                            input_hash=str(i), upper_usd=1.)
        except BudgetUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = [v for v in pool.map(reserve, range(30)) if v]
    assert len(claims) == 9
    assert budget.totals()['validator']['reserved_usd'] == 9
    budget.settle(claims[0], .5)
    assert DailyBudget(path).reserve(role='validator', provider='openai', input_hash='new-key', upper_usd=.5)
    with pytest.raises(BudgetUnavailable):
        budget.reserve(role='validator', provider='saygm', input_hash='x', upper_usd=.000001)
    miner = budget.reserve(role='miner', provider='saygm', input_hash='m', upper_usd=1.)
    with pytest.raises(BudgetUnavailable):
        budget.reserve(role='miner', provider='openai', input_hash='m2', upper_usd=.001)
    budget.settle(miner, .25)
    budget.settle(miner, .25)
    with pytest.raises(BudgetUnavailable, match='settlement_changed'):
        budget.settle(miner, .1)


def test_utc_rollover_retains_unsettled_calls_on_reservation_day(tmp_path):
    clock = [datetime(2026,9,13,23,59,59,tzinfo=timezone.utc).timestamp()]
    budget = DailyBudget(tmp_path/'budget.sqlite3', clock=lambda: clock[0])
    claim = budget.reserve(role='validator', provider='saygm', input_hash='x', upper_usd=9.)
    clock[0] += 2
    budget.reserve(role='validator', provider='openai', input_hash='y', upper_usd=9.)
    budget.settle(claim, 1.)
    assert budget.totals('2026-09-13')['validator']['settled_usd'] == 1
    assert budget.totals('2026-09-14')['validator']['reserved_usd'] == 9


def test_invalid_cost_and_overrun_do_not_erase_debt(tmp_path):
    budget = DailyBudget(tmp_path/'budget.sqlite3')
    for value in (float('nan'), float('inf'), -1):
        with pytest.raises(ValueError):
            budget.reserve(role='miner', provider='saygm', input_hash='x', upper_usd=value)
    claim = budget.reserve(role='miner', provider='saygm', input_hash='x', upper_usd=.01)
    with pytest.raises(BudgetUnavailable, match='exceeded'):
        budget.settle(claim, 1.1)
    with pytest.raises(BudgetUnavailable):
        budget.reserve(role='miner', provider='saygm', input_hash='y', upper_usd=.001)
    assert budget.totals()['miner']['settled_usd'] == 1.1
