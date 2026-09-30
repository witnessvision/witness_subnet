"""Persistent limits on active acquisition time, independent of scoring."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from witness.storage import write_private

MODEL_DOWNLOAD_BUDGET_S = 15 * 60
MODEL_DOWNLOAD_TURN_S = 15 * 60


class DownloadBudgetExceeded(TimeoutError):
    """Acquisition limit reached; this is not a model score."""


class DownloadBudget:
    def __init__(self, root: Path, model: str):
        if not re.fullmatch('[0-9a-f]{64}', model):
            raise ValueError('invalid_cache_key')
        self.root, self.model = Path(root), model
        self.path = self.root / (model + '.json')
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.root / (self.model + '.lock')
        if any(p.is_symlink() for p in (self.path, lock_path, self.root, *self.root.parents)):
            raise OSError('unsafe_download_budget_path')
        self.lock = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'r+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                record = json.loads(self.path.read_text()) if self.path.exists() else {}
                self.used = record.get('charged_s', 0.)
                self.readmission_block = record.get('readmission_block')
                if (type(self.used) not in (int, float) or not math.isfinite(self.used) or self.used < 0):
                    raise ValueError('invalid_charge')
            except (ValueError, AttributeError) as error:
                raise OSError('invalid_download_budget_record') from error
            if self.used >= MODEL_DOWNLOAD_BUDGET_S:
                raise DownloadBudgetExceeded('model_download_time_budget_exhausted')
            self.reserved = min(MODEL_DOWNLOAD_TURN_S, MODEL_DOWNLOAD_BUDGET_S - self.used)
            # Charge before starting. A crash keeps the reservation; restarting
            # cannot reset the quota. Normal exits refund unused active time.
            write_private(self.path, self._record(self.used + self.reserved, True))
            self.started = time.monotonic()
            return self
        except BaseException:
            self.lock.close()
            raise

    def _record(self, charged, active):
        record = {'charged_s': charged, 'active': active}
        if self.readmission_block is not None:
            record['readmission_block'] = self.readmission_block
        return record

    def renew(self, block):
        """A fresh finalized readmission grants one new quota, never a restart."""
        if type(block) is not int or block <= 0:
            raise ValueError('invalid_readmission_block')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.root / (self.model + '.lock')
        if any(p.is_symlink() for p in (self.path, lock_path, self.root, *self.root.parents)):
            raise OSError('unsafe_download_budget_path')
        with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = json.loads(self.path.read_text()) if self.path.exists() else {}
            if block > record.get('readmission_block', 0):
                write_private(self.path, {'charged_s': 0., 'active': False, 'readmission_block': block})

    def remaining_s(self):
        return max(0., self.reserved - (time.monotonic() - self.started))

    def exhausted(self):
        return self.used + time.monotonic() - self.started >= MODEL_DOWNLOAD_BUDGET_S

    def __exit__(self, *exc):
        try:
            spent = min(self.reserved, max(0., time.monotonic() - self.started))
            write_private(self.path, self._record(self.used + spent, False))
        finally:
            self.lock.close()
