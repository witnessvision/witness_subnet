"""A GPU worker independent of the finalized-chain loop. No local coronations."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import threading
import time

from witness.events import content_hash
from witness.storage import write_private
from .contract import InfrastructureError, Policy
from .duel import cases, grade
from .protocol import BASELINE, CONTROLS_OK, EARLY_STOP, REJECTED, Result, decide, quantize
from .reward import EVAL, eval_score, judge_clip, video_scores
from .stopping import futility, upper_units
from .round import controls
from .submission import parse_submission, verify_directory
from .triggers import Triggers


class ModelRejected(ValueError):
    """A definitive error in miner-supplied bytes, never a judge/provider failure."""


class Evaluator:
    def __init__(self, *, root, hotkey, ledger, policy, judge, batch_factory, download, runner, gpu=None):
        self.root, self.hotkey, self.ledger = Path(root), hotkey, ledger
        self.policy, self.judge, self.batch_factory = policy, judge, batch_factory
        self.download, self.runner, self.gpu = download, runner, gpu
        self.triggers = Triggers(self.root / 'triggers.sqlite3')
        self.lock, self.stopping, self.context = threading.RLock(), threading.Event(), None
        self.thread = None
        self.guard = None
        self.deadline = None
        self.attempt_epoch = None
        path = self.root / 'outbox.json'
        self.outbox = json.loads(path.read_text()) if path.exists() else []

    @classmethod
    def from_config(cls, config, root, hotkey, ledger, keypair):
        from witness.providers import ApiText
        from .gpu import make_gpu
        from .judge import CodexJudge
        from .media import PREPROCESSOR_ID
        from .p2p import ModelClient
        from .pool import window_batch
        from .runner import PodRunner, runtime_identity
        from .submission import ARCHITECTURES
        enabled = config.get('enabled_architectures', [])
        if (not isinstance(enabled, list) or not enabled
                or any(not isinstance(a, str) or a not in ARCHITECTURES for a in enabled)):
            raise ValueError('explicit_GPU_qualified_enabled_architectures_required')
        if config.get('policy') or config.get('label_model', 'gpt-6-luna') != 'gpt-6-luna' or config.get(
                'judge_model', 'gpt-5.6-terra') != 'gpt-5.6-terra':
            raise ValueError('mainnet_v2_scoring_policy_is_versioned_not_per_validator')
        gpu = make_gpu(config['gpu'], root)
        source_urls = json.loads(Path(config['source_urls']).read_text()) if config.get('source_urls') else None
        if source_urls is not None and not isinstance(source_urls, dict):
            raise ValueError('source_urls_requires_mapping')
        provider = config.get('provider', 'openai')
        labeler = ApiText('gpt-6-luna', root / 'label-cache', provider=provider, effort='low', max_tokens=16000,
                          budget_path=str(root / 'budget.sqlite3'), daily_limit_usd=config.get('api_daily_limit_usd'))
        api = ApiText('gpt-5.6-terra', root / 'judge-cache', provider=provider, effort='low', max_tokens=16000,
                      budget_path=str(root / 'budget.sqlite3'), daily_limit_usd=config.get('api_daily_limit_usd'))
        base = dict(runtime_hash=content_hash(runtime_identity(config['gpu'])), preprocessing_hash=PREPROCESSOR_ID,
                    reference_kind='machine', min_annotators=1, confirmation_size=2, screen_size=1,
                    quality_floor=0., judge_id='pending')
        judge = CodexJudge(Policy(**base), model=api.model, effort=api.effort, api=api)
        policy = Policy(**{**base, 'judge_id': judge.identity})

        def download(entry, snapshot, cancelled):
            submission = parse_submission(entry['value'])
            cache = root / 'models'
            directory = cache / entry['model_id']
            manifest_file = directory / 'witness-manifest.json'
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                if manifest.get('arch') not in enabled:
                    raise InfrastructureError('architecture_not_enabled_on_this_evaluator')
                try:
                    info = verify_directory(directory, manifest)
                except ValueError as error:
                    raise InfrastructureError('local_model_cache_corrupted') from error
                if info['model_id'] != submission.model_id:
                    raise InfrastructureError('cached_model_identity_mismatch')
                return {**info, 'path': str(directory), 'manifest': manifest}
            address = snapshot.get('endpoints', {}).get(entry['hotkey'])
            if address is None:
                raise InfrastructureError('miner_endpoint_unavailable')
            try:
                client = ModelClient(keypair, entry['hotkey'], address['host'], address['port'], submission)
                client.cancelled = cancelled
                client.remaining_s = lambda: (max(0., evaluator.deadline - time.monotonic())
                                               if evaluator.deadline else 30.)
                directory, info = client.download(cache, cancelled=cancelled, accepted_architectures=enabled,
                                                  cache_key=entry['model_id'])
            except NotImplementedError as error:
                raise InfrastructureError('architecture_not_enabled_on_this_evaluator') from error
            except (ValueError, RecursionError) as error:
                raise ModelRejected('invalid_model_package') from error
            return {**info, 'path': str(directory)}

        evaluator = cls(root=root, hotkey=hotkey, ledger=ledger, policy=policy, judge=judge, gpu=gpu,
                   batch_factory=lambda window: window_batch(root / 'pool-v2', window['id'], hotkey,
                       gpu=gpu, api=labeler, policy=policy, source_urls=source_urls), download=download,
                   runner=lambda submissions, job: PodRunner(gpu, submissions, policy, job))
        api.request_timeout_s = labeler.request_timeout_s = lambda: (
            min(30., evaluator.deadline - time.monotonic()) if evaluator.deadline else 30.)
        return evaluator

    def progress(self, stage, window, model=None, completed=0):
        write_private(self.root / 'progress.json', {'stage': stage, 'window_id': window['id'],
                      'model_id': model, 'completed_clips': completed, 'total_clips': 10,
                      'updated_unix': time.time()})

    def update(self, snapshot, *, start_worker=True):
        """Called by the chain thread; never runs inference or paid labeling."""
        submissions = self.ledger.submissions()
        self.triggers.observe(submissions, {r['hotkey'] for r in submissions},
                              coldkeys={r['hotkey']: r['coldkey'] for r in submissions},
                              uids={r['hotkey']: r['uid'] for r in submissions})
        for hotkey, usage in self.ledger.usage(self.hotkey).items():
            self.triggers.reconcile(hotkey, usage['result'], window=usage['window'],
                                    rejected=bool(usage['result']['flags'] & REJECTED))
        with self.lock:
            self.context = (self.ledger.active, snapshot)
            for item in list(self.outbox):
                used = self.ledger.usage(self.hotkey)
                acknowledged = any(r['hotkey'] == self.hotkey and r['value'] == item['value'] for r in
                                   self.ledger.history(item['start_block'], snapshot['block'] + 1))
                expired = self.context[0] is None or self.context[0]['id'] != item['window']
                if acknowledged or expired:
                    self.outbox.remove(item)
                    if expired and not acknowledged and item.get('trigger_id'):
                        self.triggers.defer(item['trigger_id'], 'window_closed_before_publication')
            self._save_outbox()
        if self.thread is None and start_worker:
            self.thread = threading.Thread(target=self._loop, name='witness-evaluator', daemon=True)
            self.thread.start()
            if self.gpu:
                self.guard = threading.Thread(target=self._guard, name='witness-gpu-budget', daemon=True)
                self.guard.start()

    def _guard(self):
        while not self.stopping.wait(15):
            try:
                self.gpu.enforce_budget()
            except Exception as error:
                write_private(self.root / 'gpu-guard-error.json', {'type': type(error).__name__, 'unix': time.time()})

    def _save_outbox(self):
        write_private(self.root / 'outbox.json', self.outbox)

    def flush_one(self, store, snapshot):
        with self.lock:
            if not self.outbox:
                return
            item = dict(self.outbox[0])
        if not store.writable:
            write_private(self.root / 'intended-commitments.json', self.outbox)
            return
        if item['window'] != (self.ledger.active or {}).get('id'):
            return
        # Publishing does not consume the hotkey: finalized history acknowledges it.
        receipt = store.publish(self.hotkey, item['value'], snapshot['block'])
        write_private(self.root / 'last-commitment-send.json', {'value': item['value'], 'receipt': receipt})

    def _loop(self):
        while not self.stopping.is_set():
            with self.lock:
                context = self.context
            try:
                if context and context[0]:
                    self.run_once(*context)
            except Exception as error:
                # Redacted, typed error only; providers and model paths may contain private data.
                write_private(self.root / 'worker-error.json', {'type': type(error).__name__,
                                                               'unix': time.time()})
            self.stopping.wait(1)

    def _cancelled(self, window):
        with self.lock:
            return (self.stopping.is_set() or self.context is None or self.context[0] is None
                    or self.context[0]['id'] != window['id']
                    or (self.deadline is not None and time.monotonic() >= self.deadline)
                    or (self.attempt_epoch is not None and self.context[1]['epoch_index'] != self.attempt_epoch))

    def _batch(self, window):
        root = self.root / 'windows' / str(window['id'])
        path = root / 'batch.json'
        rows = json.loads(path.read_text()) if path.exists() else self.batch_factory(window)
        if len(rows) != EVAL.videos * EVAL.clips_per_video or len({r['video'] for r in rows}) != EVAL.videos:
            raise InfrastructureError('incorrect_window_batch')
        videos = [row['video'] for row in rows[::EVAL.clips_per_video]]
        if len(set(videos)) != EVAL.videos or any(
                [r['index'] for r in rows[i:i + EVAL.clips_per_video]] != list(range(EVAL.clips_per_video))
                or len({r['video'] for r in rows[i:i + EVAL.clips_per_video]}) != 1
                for i in range(0, len(rows), EVAL.clips_per_video)):
            raise InfrastructureError('window_batch_not_grouped_by_random_video_order')
        selected = cases(rows, self.policy, str(window['id']))
        check = root / 'controls.json'
        if not check.exists():
            measured = []
            for case, references in selected:
                if self._cancelled(window):
                    raise InterruptedError('window_closed')
                for name, response in controls(case.task.duration).items():
                    score = judge_clip(references, response, judge=self.judge, policy=self.policy, root=root / 'judge')[0]
                    measured.append({'video': case.source_group, 'control': name, 'quality': score['quality']})
            if any(sum(r['quality'] for r in measured if r['control'] == name) / len(selected) > .05
                   for name in ('empty', 'generic')):
                raise InfrastructureError('generic_controls_failed')
            write_private(check, {'pass': True, 'measurements': measured})
        write_private(path, rows)
        return rows, selected

    def _evaluate(self, entry, window, snapshot, rows, selected, *, baseline=None, resume=False):
        model_id = entry['model_id']
        root = self.root / 'windows' / str(window['id'])
        path = root / 'models' / model_id / 'score.json'
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and (not cached.get('early_stop') or not resume):
            return cached
        if self._cancelled(window):
            raise InterruptedError('window_closed')
        self.progress('downloading', window, model_id)
        item = self.download(entry, snapshot, lambda: self._cancelled(window))
        content_path = self.root / 'content-owners.json'
        owners = json.loads(content_path.read_text()) if content_path.exists() else {}
        owner = owners.get(item['content_id'])
        if owner is not None and owner != model_id:
            raise ModelRejected('duplicate_model_content')
        owners[item['content_id']] = model_id
        write_private(content_path, owners)
        attempts = root / 'models' / model_id / 'attempt.json'
        attempt = json.loads(attempts.read_text())['count'] + 1 if attempts.exists() else 1
        write_private(attempts, {'count': attempt})
        run = self.runner({model_id: item}, f'w{window["id"]}-{model_id[:16]}-{attempt}')
        run.cancelled = lambda: self._cancelled(window)
        run.remaining_s = lambda: max(0., self.deadline - time.monotonic()) if self.deadline else 900.
        hardware_file = root / 'hardware.json'
        progress_path = path.with_name('progress.json')
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        grades = list(progress.get('grades', cached['grades'] if cached else []))
        resume = resume or progress.get('continue_to_full', False)
        if (len(grades) % EVAL.clips_per_video or len(grades) > len(rows)
                or [r['id'] for r in grades] != [c.task.clip_sha256 for c, _ in selected[:len(grades)]]):
            raise InfrastructureError('cached_progress_binding_failed')
        stopped = None
        first = len(grades)
        boundaries = [3 * EVAL.clips_per_video, len(rows)] if baseline and not resume and first < 6 else [len(rows)]
        for last in boundaries:
            if last <= first:
                continue
            if self._cancelled(window):
                raise InterruptedError('evaluation_budget_or_epoch_closed')
            batch = selected[first:last]
            self.progress('evaluating', window, model_id, first)
            executions = run([model_id], [c.task for c, _ in batch], [Path(c.media_path) for c, _ in batch])[model_id]
            hardware = getattr(run, 'hardware_id', None)
            if hardware_file.exists() and json.loads(hardware_file.read_text())['id'] != hardware:
                raise InfrastructureError('hardware_changed_wait_for_next_window')
            if not hardware_file.exists():
                write_private(hardware_file, {'id': hardware})
            for (case, references), row in zip(batch, rows[first:last]):
                if self._cancelled(window):
                    raise InterruptedError('evaluation_budget_or_epoch_closed')
                execution = executions.get(case.task.id)
                if execution is None:
                    raise InfrastructureError('missing_execution')
                value = grade(case, references, execution, model_id, policy=self.policy, judge=self.judge,
                              root=root / 'judge')
                grades.append({**value, 'id': case.task.clip_sha256, 'duration': case.task.duration,
                               'start': row['start'], 'file': row['file']})
                self.progress('judging', window, model_id, len(grades))
                if len(grades) % EVAL.clips_per_video == 0:
                    write_private(progress_path, {'grades': grades, 'continue_to_full': resume})
            first = last
            if baseline and not resume:
                stopped = futility(list(video_scores(grades).values()), baseline['reward'])
                if stopped:
                    break
        videos = video_scores(grades)
        result = {'model_id': model_id, 'grades': grades, 'per_video': videos, **eval_score(videos)}
        if stopped:
            result.update(early_stop=stopped, quality=stopped['quality_upper'], reward=stopped['reward_upper'])
        write_private(path, result)
        return result

    def _report(self, entry, window, scored, baseline):
        videos = []
        for video, summary in scored['per_video'].items():
            clips = []
            for row in scored['grades']:
                if row['video'] != video:
                    continue
                other = next((x for x in (baseline or {}).get('grades', []) if x['id'] == row['id']), None)
                def metrics(value):
                    return {'response': value.get('response'), 'quality': value['quality'],
                            'latency_s': value.get('latency_s'), 'time_score': value.get('speed'),
                            'reward': value['reward']}
                clips.append({'id': row['id'], 'start': row['start'], 'duration': row['duration'],
                              'video_url': f'/api/media/{self.hotkey}/{window["id"]}/{row["id"]}.mp4',
                              **metrics(row), 'king': metrics(other) if other else None})
            other = (baseline or {}).get('per_video', {}).get(video, {})
            videos.append({'id': video, 'quality': summary['quality'], 'reward': summary['reward'],
                           'king_quality': other.get('quality'), 'king_reward': other.get('reward'), 'clips': clips})
        partial = scored.get('early_stop')
        return {'schema_version': 'witness-evaluation-2', 'available': True, 'validator': self.hotkey,
                'evaluation_status': 'early_stop' if partial else 'complete', 'early_stop': partial,
                'window_id': window['id'], 'hotkey': entry['hotkey'], 'model_id': entry['model_id'],
                'opponent': window['king'], 'videos': videos, 'policy_hash': self.ledger.policy,
                'total': {'quality': None if partial else scored['quality'], 'reward': None if partial else scored['reward'],
                          'quality_upper': scored['quality'] if partial else None,
                          'reward_upper': scored['reward'] if partial else None,
                          'king_quality': (baseline or {}).get('quality'), 'king_reward': (baseline or {}).get('reward')}}

    def _enqueue(self, entry, window, scored, baseline, *, flags, trigger_id=None):
        report = self._report(entry, window, scored, baseline)
        evidence = content_hash(report)
        encode = upper_units if flags & EARLY_STOP else quantize
        result = Result(window['id'], entry['uid'], entry['model_id'], self.ledger.policy[:24], evidence,
                        encode(scored['quality']), encode(scored['reward']),
                        quantize((baseline or {}).get('quality', 0.)), quantize((baseline or {}).get('reward', 0.)), flags)
        write_private(self.root / 'reports' / (evidence + '.json'), report)
        with self.lock:
            if not any(item['value'] == result.commitment for item in self.outbox):
                self.outbox.append({'window': window['id'], 'start_block': window['start_block'],
                                    'value': result.commitment, 'trigger_id': trigger_id, 'report_hash': evidence})
                self._save_outbox()

    def run_once(self, window, snapshot):
        self.deadline = None
        self.attempt_epoch = snapshot['epoch_index']
        with self.lock:
            waiting = {r['trigger_id'] for r in self.outbox if r.get('trigger_id')}
        published, partials = set(), set()
        for row in self.ledger.history(window['start_block'], self.ledger.cursor + 1):
            if row['hotkey'] != self.hotkey:
                continue
            try:
                record = Result.parse(row['value'])
            except ValueError:
                continue
            if record.window == window['id'] and record.policy == self.ledger.policy[:24]:
                (partials if record.flags & EARLY_STOP else published).add(record.model_id)
        comparison = decide(self.ledger._voting_history(window, self.ledger.cursor + 1),
                            window=window['id'], policy=self.ledger.policy, king=window['king'],
                            candidates=window['candidates'], snapshot=snapshot)
        pending = [r for r in self.triggers.pending(10000, before_block=window['start_block'],
                    current_block=snapshot['block']) if r['model_id'] in window['candidates']
                   and window['candidates'][r['model_id']]['hotkey'] == r['hotkey'] and r['id'] not in waiting
                   and r['model_id'] not in published]
        pending = [r for r in pending if r['model_id'] not in partials
                   or r['model_id'] in comparison['inconclusive']]
        if not pending or self._cancelled(window):
            self.progress('waiting', window)
            if self.gpu:
                self.gpu.stop_if_idle()
            return
        entry = {**pending[0], 'uid': window['candidates'][pending[0]['model_id']]['uid']}
        self.triggers.start(entry['id'], window['id'])
        try:
            self.deadline = time.monotonic() + 900.
            if self.gpu:
                self.gpu.cancelled = lambda: self._cancelled(window)
                self.gpu.remaining_s = lambda: max(0., self.deadline - time.monotonic())
            with self.gpu.lease() if self.gpu else nullcontext():
                self._attempt(entry, window, snapshot, partials)
        except Exception as error:
            self.progress('deferred', window, entry['model_id'])
            self.triggers.defer(entry['id'], type(error).__name__, snapshot['block'] + min(300, 10 * 2**min(entry['attempts'], 5)))
            raise
        finally:
            if self.gpu:
                self.gpu.cancelled = lambda: False
                self.gpu.remaining_s = lambda: float('inf')

    def _attempt(self, entry, window, snapshot, partials):
        baseline = None
        self.progress('preparing', window, entry['model_id'])
        rows, selected = self._batch(window)
        if window['king']:
            king = {**window['king'], 'uid': snapshot['uids'][window['king']['hotkey']]}
            baseline = self._evaluate(king, window, snapshot, rows, selected)
            history = self.ledger.history(window['start_block'], snapshot['block'] + 1)
            if not any(r['hotkey'] == self.hotkey and r['value'].startswith('wr2|') and
                       (lambda v: v.flags & BASELINE and v.model_id == king['model_id'])(Result.parse(r['value']))
                       for r in history):
                self._enqueue(king, window, baseline, None, flags=CONTROLS_OK | BASELINE)
        try:
            scored = self._evaluate(entry, window, snapshot, rows, selected, baseline=baseline,
                                    resume=entry['model_id'] in partials)
            flags = CONTROLS_OK | (EARLY_STOP if scored.get('early_stop') else 0)
        except ModelRejected as error:
            scored = {'quality': 0., 'reward': 0., 'per_video': {}, 'grades': [], 'reason': str(error)[:120]}
            flags = CONTROLS_OK | REJECTED
        if self._cancelled(window):
            raise InterruptedError('window_closed')
        self._enqueue(entry, window, scored, baseline, flags=flags, trigger_id=entry['id'])

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=45)
