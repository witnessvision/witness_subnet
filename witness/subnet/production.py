"""Public 5.2 validator CLI. Source manifests and evaluation state stay private."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import uuid

from witness.events import content_hash
from witness.judge import JudgeConfig
from witness.sources.catalog import prepare_catalog, jobs_from
from witness.storage import write_private
from witness.subnet.processes import run_process
from witness.subnet.production_chain import ProductionChain
from witness.subnet.production_validator import ProductionValidator


def callbacks(config, root, judge_config):
    async def jobs(epoch):
        catalog_path = Path(config['catalog'])
        catalog = json.loads(catalog_path.read_text())
        if content_hash(catalog) != config['catalog_hash'] or len(catalog['originals']) != config['catalog_count']:
            raise ValueError('catalogue_identity_changed')
        state = await prepare_catalog(argparse.Namespace(catalog=catalog_path,
            output=root/'preparation'/str(epoch), count=5, seed=None,
            max_attempts=config.get('max_source_attempts', 60),
            max_seconds=config.get('max_preparation_s', 1800), cache_bytes=512*1024**2,
            mirror_root=Path(config['mirror_root']), require_available_catalog=True))
        result = jobs_from(state)
        if len(result) != 5:
            raise ValueError('five_clips_unavailable')
        return result

    async def evaluate(job, received):
        path = root/'evaluation-inputs'/(uuid.uuid4().hex+'.json')
        write_private(path, {'reference': job['reference'], 'response': received['response']})
        command = [sys.executable, '-m', 'witness.judge', '--input', str(path), '--env', config['env'],
            '--cache', config['judge_cache'], '--budget', config['budget_path'],
            '--provider', judge_config.judge_provider, '--model', judge_config.judge_model,
            '--effort', judge_config.judge_effort, '--calibration', config['calibration']]
        env = {k:v for k,v in os.environ.items() if k in ('PATH','LANG','PYTHONPATH')}
        return json.loads(await run_process(command, timeout=300, max_output=4*1024**2, env=env))
    return jobs, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    judge = JudgeConfig(**{k: config[k] for k in ('judge_provider','judge_model','judge_effort') if k in config})
    root = Path(config['root'])
    calibration = json.loads(Path(config['calibration']).read_text())
    identity = judge.identity(config['judge_cache'], budget_path=config['budget_path'])
    chain = ProductionChain(netuid=20, network=config.get('network', 'finney'),
        wallet_name=config['wallet'], wallet_hotkey=config['wallet_hotkey'], wallet_path=config['wallet_path'])
    validator = None
    try:
        if chain.validator_hotkey != config['expected_hotkey']:
            raise ValueError('wrong_validator_hotkey')
        jobs, evaluate = callbacks(config, root, judge)
        validator = ProductionValidator(chain, root=root, jobs_factory=jobs, evaluate=evaluate,
            evaluator_identity=identity, calibration=calibration, start_after_epoch=config['start_after_epoch'])
        asyncio.run(validator.run())
    finally:
        if validator:
            validator.close()
        chain.close()


if __name__ == '__main__':
    main()
