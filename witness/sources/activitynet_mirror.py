"""Bounded access to a pinned, public ActivityNet research mirror.

This is an acquisition backend for validator evaluation,
never a source of validator selections for the miner. No YouTube challenges,
cookies or proxy rotation are involved. Original media is retained unchanged.
"""
import asyncio
import json
from pathlib import Path
import re
import shutil

import httpx

from witness.storage import write_private
from witness.subnet.processes import run_process
from .activitynet import SourceCache, MAX_SOURCE_BYTES
from witness.storage import sha256_file

PIN = {
    'dataset': 'TornadoLabs/activitynet',
    'revision': '2dafb988f8854a0a801c40a697032e77de67b249',
    'metadata_sha256': 'b3849e17b6b821044c87789c2b1108009a4c50a0b08d255cd908a97255916042',
    'readme_sha256': 'a8480d9585bae575c5e77e52c10acb3a46519c08cfd6154624b98b6eb9679483',
}
URL_PREFIX = 'https://huggingface.co/buckets/TornadoLabs/staging/resolve/videos/activitynet/'


class AcquisitionPaused(RuntimeError):
    """An infrastructure error requires diagnosis, not catalogue exclusions."""


async def validate_native_media(media, probe, duration):
    await run_process(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-xerror',
                       '-i',str(media),'-threads','1','-f','null','-'],
                      timeout=max(120.,min(600.,duration)))


class MirrorSourceCache(SourceCache):
    def __init__(self, root: Path, mirror_root: Path, *, max_bytes=512*1024**2):
        super().__init__(root, max_bytes=max_bytes)
        if (sha256_file(mirror_root/'metadata.jsonl') != PIN['metadata_sha256'] or
                sha256_file(mirror_root/'README.md') != PIN['readme_sha256']):
            raise ValueError('mirror_pin_mismatch')
        self.identity = dict(PIN)
        self.sources = {}
        for line in (mirror_root/'metadata.jsonl').read_text().splitlines():
            row = json.loads(line)
            original = row['video_id']
            if (not re.fullmatch(r'[A-Za-z0-9_-]{11}', original) or original in self.sources or
                    row['video_url'] != URL_PREFIX+original+'.mp4' or
                    type(row['size_bytes']) is not int or row['size_bytes'] <= 0):
                raise ValueError('invalid_mirror_metadata')
            self.sources[original] = row

    async def get(self, row):
        original = row['original']
        source = self.sources.get(original)
        if source is None:
            raise ValueError('source_absent_from_public_mirror')
        if source['size_bytes'] > MAX_SOURCE_BYTES:
            raise ValueError('source_media_too_large')
        destination = self.root/original
        receipt = destination/'receipt.json'
        media = destination/'video.mp4'
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (record['mirror'] != self.identity or sha256_file(media) != record['media_sha256']):
                raise ValueError('cached_media_hash_mismatch')
            destination.touch()
            return media, record
        self.prune()
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(mode=0o700)
        phase='download'
        try:
            async with asyncio.timeout(120):
                async with httpx.AsyncClient(timeout=30, follow_redirects=True, trust_env=False) as client:
                    async with client.stream('GET', source['video_url']) as response:
                        if response.status_code in (401, 403, 429) or response.status_code >= 500:
                            raise AcquisitionPaused('mirror_http_'+str(response.status_code))
                        if response.status_code != 200:
                            raise ValueError('mirror_source_http_'+str(response.status_code))
                        size = 0
                        with media.open('wb') as output:
                            async for block in response.aiter_bytes():
                                size += len(block)
                                if size > MAX_SOURCE_BYTES or size > source['size_bytes']:
                                    raise ValueError('mirror_source_size_changed')
                                output.write(block)
            phase='validation'
            if size != source['size_bytes']:
                raise ValueError('mirror_source_size_changed')
            probe = json.loads(await run_process(['ffprobe','-v','error','-show_streams',
                '-show_format','-of','json',str(media)],timeout=30))
            duration = float(probe['format']['duration'])
            if abs(duration-row['duration']) > .5:
                raise ValueError('source_duration_changed')
            await validate_native_media(media,probe,duration)
            record = {'original':original, 'media_sha256':sha256_file(media), 'bytes':size,
                'native_duration':duration, 'source_url':row['source_url'],
                'mirror_url':source['video_url'], 'mirror':self.identity,
                'source_license':'academic-source-terms; original uploader rights retained',
                'channel_id':None, 'partition':'unverified_channel',
                'partition_policy':'mirror lacks channel metadata; diagnostic original-level sampling only',
                'streams':[{k:s[k] for k in ('codec_type','codec_name','sample_rate','channels','r_frame_rate')
                            if k in s} for s in probe['streams']]}
            media.chmod(0o600)
            write_private(receipt, record)
            return media, record
        except (httpx.TransportError, TimeoutError) as error:
            shutil.rmtree(destination, ignore_errors=True)
            if phase=='validation' and isinstance(error,TimeoutError):
                raise RuntimeError('native_validation_timeout') from None
            raise AcquisitionPaused('mirror_transport_interrupted') from None
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
