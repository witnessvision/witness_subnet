"""Five direct MP4 requests per observed epoch; no weight-writing capability.

The caller supplies private acquisition/evaluation and a read-only chain view.
Only send_clip constructs a miner request, from the anonymous clip and spec.
"""
from __future__ import annotations
import asyncio
import fcntl
import hashlib
from pathlib import Path
import time
import httpx

from witness.events import EventsTaskSpec
from witness.mp4 import send_clip
from witness.score_v5_1_0 import latency_reward
from witness.storage import write_private
from witness.subnet.events_state import EventsState


class MP4Validator:
    def __init__(self, chain, *, root: Path, target_hotkey: str, url: str, signing_key,
                 jobs_factory, evaluate, start_after_epoch: int = -1):
        if not target_hotkey:
            raise ValueError('one_target_hotkey_required')
        self.chain,self.root,self.target_hotkey=chain,root,target_hotkey
        self.url,self.key,self.jobs_factory,self.evaluate=url,signing_key,jobs_factory,evaluate
        self.state=EventsState(root/'scheduler',netuid=chain.netuid,target_hotkey=target_hotkey,
                               start_after_epoch=start_after_epoch)
        # Endpoint and policy changes require a new history root.
        import json
        identity={'target_hotkey':target_hotkey,'url':url,'reward_version':'5.1.0',
                  'transport_version':'5.1','signing_public_key':signing_key.public_key().public_bytes_raw().hex()}
        path=root/'identity.json'
        if path.exists() and json.loads(path.read_text())!=identity:
            self.state.close()
            raise ValueError('mp4_validator_identity_changed')
        write_private(path,identity)

    def target(self):
        targets=[e for e in self.chain.miner_endpoints() if e.hotkey==self.target_hotkey]
        if len(targets)!=1:
            raise ValueError('registered_target_required')
        return targets[0]

    async def dispatch(self, task):
        job=task['payload'];began=time.monotonic()
        result={'status':'failed','dispatch_attempted':False,'evaluation':None,
                'miner_elapsed_s':None,'evaluator_elapsed_s':None,'observation_calls':0,'feedback_calls':0}
        try:
            self.target()
            clip=Path(job['clip'])
            if hashlib.sha256(clip.read_bytes()).hexdigest()!=job['clip_sha256']:
                raise ValueError('prepared_clip_hash_changed')
            def dispatched(request):
                result.update(dispatch_attempted=True,task=request.model_dump())
                write_private(self.root/'requests'/(task['id']+'.dispatch.json'),result)
            received=await send_clip(self.url,clip,EventsTaskSpec.model_validate(job['spec']),self.key,
                                      on_dispatch=dispatched)
            result.update(status='valid',miner_elapsed_s=received['elapsed_s'],received=received)
        except Exception as error:
            result.update(error_type=type(error).__name__,miner_elapsed_s=time.monotonic()-began)
            if isinstance(error,TimeoutError):result['status']='expired'
            elif isinstance(error,httpx.HTTPStatusError):
                result['status']='expired' if error.response.status_code==504 else 'rejected'
        if result['status']=='valid':
            started=time.monotonic()
            try:
                async with asyncio.timeout(300):
                    result['evaluation']=await self.evaluate(job,result['received'])
            except Exception as error:
                result['evaluator_error']=type(error).__name__
            result['evaluator_elapsed_s']=time.monotonic()-started
        try:
            quality=result['evaluation']['score']['f1'] if result['evaluation'] else 0.
            result['reward']=latency_reward(quality,result['miner_elapsed_s'],response_received=result['status']=='valid')
        except (KeyError,TypeError,ValueError):
            result.update(evaluation=None,evaluator_error='invalid_quality_score')
            result['reward']=latency_reward(0.,result['miner_elapsed_s'],response_received=result['status']=='valid')
        write_private(self.root/'requests'/(task['id']+'.json'),result)
        self.state.finish_task(task['id'],result)

    async def step(self):
        active=self.state.active()
        if active is None:
            observed=self.chain.epoch_state()
            if not self.state.eligible(int(observed['epoch_index'])):return None
            self.target()
            try:
                jobs=await self.jobs_factory(int(observed['epoch_index']))
                if len(jobs)!=5 or len({job['original'] for job in jobs})!=5:
                    raise ValueError('round_requires_five_distinct_originals')
            except Exception as error:
                consumed=int(self.chain.epoch_state()['epoch_index'])
                self.state.skip_epoch(consumed)
                write_private(self.root/'preparation-failures'/(str(observed['epoch_index'])+'.json'),
                    {'epoch':observed['epoch_index'],'consumed_through_epoch':consumed,
                     'planned':5,'dispatch_attempted':0,'error_type':type(error).__name__})
                raise
            # Acquisition can cross an epoch. Claim only the currently observed
            # epoch when clips are ready, never a backlog of missed rounds.
            epoch=int(self.chain.epoch_state()['epoch_index'])
            round_id=self.state.begin(epoch,jobs)
            if round_id is None:return None
            active=self.state.active()
        for task in self.state.tasks(active['id']):
            if self.state.claim(task['id']):await self.dispatch(task)
        finish_epoch=int(self.chain.epoch_state()['epoch_index'])
        self.state.finish_round(active['id'],finish_epoch)
        tasks=self.state.tasks(active['id']);results=[t['result'] for t in tasks]
        for task,result in zip(tasks,results):
            if result['status']=='interrupted_unknown':
                result['dispatch_attempted']=(self.root/'requests'/(task['id']+'.dispatch.json')).exists()
        report={'round_id':active['id'],'epoch':active['epoch'],'finish_epoch':finish_epoch,
            'skipped_epochs':list(range(active['epoch']+1,finish_epoch+1)),
            'planned':5,'dispatch_attempted':sum(bool(r.get('dispatch_attempted')) for r in results),
            'valid':sum(r['status']=='valid' for r in results),
            'scored':sum(bool(r.get('evaluation')) for r in results),
            'observation_calls':sum(r.get('observation_calls',0) for r in results),
            'feedback_calls':sum(r.get('feedback_calls',0) for r in results),
            'reward':sum(r.get('reward',{}).get('reward',0.) for r in results)/5,
            'statuses':{s:sum(r['status']==s for r in results)
                        for s in ('valid','failed','rejected','expired','interrupted_unknown')},
            'weights_enabled':False}
        import numpy as np
        for metric in ('f1','precision','recall'):
            report[metric]=sum(r['evaluation']['score'].get(metric,0.) if r.get('evaluation') else 0.
                               for r in results)/5
        for key in ('miner_elapsed_s','evaluator_elapsed_s'):
            samples=[r[key] for r in results if r.get(key) is not None]
            report[key]={'count':len(samples),'mean':float(np.mean(samples)) if samples else None,
                         'p50':float(np.percentile(samples,50)) if samples else None,
                         'p95':float(np.percentile(samples,95)) if samples else None,
                         'max':max(samples) if samples else None}
        write_private(self.root/'rounds'/(active['id']+'.json'),report)
        return report

    async def run(self, *, poll_interval_s=12.):
        if poll_interval_s<=0:raise ValueError('positive_poll_interval_required')
        with (self.root/'.owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state.recover()
            while True:
                try:
                    await self.step()
                except Exception as error:
                    write_private(self.root/'last-error.json',{'error_type':type(error).__name__,'unix':time.time()})
                await asyncio.sleep(poll_interval_s)

    def close(self):
        self.state.close()
