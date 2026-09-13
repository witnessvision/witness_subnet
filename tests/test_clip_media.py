import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from witness.tools.clip_media import crop_mp4


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg required')
def test_crop_preserves_timing_audio_and_removes_source_metadata(tmp_path):
    source,clip=tmp_path/'source.mp4',tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=2:duration=64',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=64','-metadata','title=SOURCE_CANARY',
        '-metadata:s:v:0','handler_name=SOURCE_CANARY_VIDEO','-metadata:s:a:0','handler_name=SOURCE_CANARY_AUDIO',
        '-c:v','libx264','-threads','1','-c:a','alac',str(source)],check=True,stdin=subprocess.DEVNULL)
    measured=asyncio.run(crop_mp4(source,clip,start=2.,duration=60.))
    assert measured['duration']==60 and measured['fps']==2
    assert measured['has_audio'] and measured['audio']['sample_rate']==44100
    probe=subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(clip)],stdin=subprocess.DEVNULL)
    assert b'SOURCE_CANARY' not in probe
    assert abs(float(json.loads(probe)['format']['start_time']))<.001
    def frame(path,t):
        return np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(t),'-i',str(path),
            '-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','-threads','1','-'],stdin=subprocess.DEVNULL),dtype=np.uint8)
    for t in [0,20,59]:
        assert np.abs(frame(source,t+2).astype(float)-frame(clip,t)).mean()<5
    def audio(path,t):
        return subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(t),'-i',str(path),'-t','1',
            '-vn','-f','s16le','-acodec','pcm_s16le','-'],stdin=subprocess.DEVNULL)
    assert audio(source,2)==audio(clip,0)


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg required')
def test_variable_rate_silent_video_preserves_frame_timestamps(tmp_path):
    source,clip=tmp_path/'vfr.mp4',tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=6:duration=64',
        '-vf',"select=if(lt(t\\,32)\\,not(mod(n\\,2))\\,1)",'-fps_mode','vfr',
        '-c:v','libx264','-threads','1',str(source)],check=True,stdin=subprocess.DEVNULL)
    measured=asyncio.run(crop_mp4(source,clip,start=0.,duration=60.))
    assert measured['has_audio'] is False
    def times(path):
        output=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],stdin=subprocess.DEVNULL)
        return [float(f['best_effort_timestamp_time']) for f in json.loads(output)['frames']]
    expected=[t for t in times(source) if t<60]
    assert times(clip)==pytest.approx(expected,abs=.001)
