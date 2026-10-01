"""Measured evaluator timings. Diagnostic data has no role in consensus."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import math
import time
import threading

from witness.storage import write_private


class Performance:
    def __init__(self, root, entry, window):
        self.root, self.entry, self.window = Path(root), entry, window
        self.started, self.started_unix = time.monotonic(), time.time()
        self.phases, self.clips = {}, []
        self.models = []
        self.lock = threading.Lock()
        self.phase_intervals, self.phase_work = {}, {}

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            finished = time.monotonic()
            from .event_scoring import intervals, length
            with self.lock:
                self.phase_work[name] = self.phase_work.get(name,0.)+finished-started
                self.phase_intervals[name] = intervals(self.phase_intervals.get(name,[])+[(started,finished)])
                self.phases[name] = length(self.phase_intervals[name])

    def finish(self, status, error=None):
        elapsed = time.monotonic()-self.started
        path = self.root/'performance'/str(self.window['id'])/(self.entry['model_id']+'.json')
        previous = json.loads(path.read_text()) if path.exists() else {}
        count = previous.get('attempts', 0)+1
        row = {'status': status, 'error_type': type(error).__name__ if error else None,
               'started_unix': self.started_unix, 'finished_unix': time.time(), 'active_s': elapsed,
               'phases_s': self.phases, 'phase_work_s': self.phase_work, 'models': self.models, 'clips': self.clips}
        cumulative = dict(previous.get('cumulative_phases_s', {}))
        for key,value in self.phases.items(): cumulative[key] = cumulative.get(key,0.)+value
        data = {'schema_version':'witness-performance-1','window_id':self.window['id'],
                'uid':self.entry['uid'],'model_id':self.entry['model_id'],'attempts':count,
                'first_started_unix':previous.get('first_started_unix',self.started_unix),
                'active_s':previous.get('active_s',0.)+elapsed,'cumulative_phases_s':cumulative,
                'latest':row,'status':status}
        data['wall_s'] = row['finished_unix']-data['first_started_unix']
        prefetch=path.with_suffix('.prefetch.json')
        if prefetch.exists():
            data['prefetch']=json.loads(prefetch.read_text())
            data['wall_from_prefetch_start_s']=row['finished_unix']-data['prefetch']['started_unix']
        samples={}
        for sample in previous.get('clip_samples',[])+self.clips:
            samples.setdefault((sample['model_id'],sample['task_id']),sample)
        data['clip_samples']=list(samples.values())[-2000:]
        measured=sorted(r['elapsed_s'] for r in data['clip_samples'])
        if measured:
            data['clip_latency_s']={'count':len(measured),'p50':measured[math.ceil(len(measured)*.5)-1],
                                   'p95':measured[math.ceil(len(measured)*.95)-1],'max':measured[-1]}
        write_private(path,data)
        write_private(self.root/'performance-latest.json',data)
        return data
