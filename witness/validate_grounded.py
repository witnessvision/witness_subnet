"""Fail-closed media/label checks for grounded scenes; no miner/model dependency."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from witness.render import FFMPEG
from witness.scene import COLORS
from witness.score.oracle import perfect_reconstruction
from witness.score_v3_0_0 import score_reconstruction


def decode_checkpoints(video: Path, numbers: list[int], width: int, height: int):
    numbers = sorted(set(numbers))
    def expression(items):
        if len(items) == 1:
            return f'eq(n\\,{items[0]})'
        middle = len(items) // 2
        return f'({expression(items[:middle])}+{expression(items[middle:])})'
    if not numbers:
        return {}
    result = subprocess.run([str(FFMPEG), '-v', 'error', '-threads', '2', '-i', str(video),
        '-vf', f'select={expression(numbers)}', '-fps_mode', 'passthrough',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],capture_output=True,check=True)
    size = width * height * 3
    if len(result.stdout) != size * len(numbers):
        raise ValueError('decoded checkpoint count mismatch')
    return {n: np.frombuffer(result.stdout[i*size:(i+1)*size],np.uint8).reshape(height,width,3)
            for i,n in enumerate(numbers)}


def _screen(scene, number, point):
    width, height = scene['resolution']; appearance = scene['appearance']
    zoom = appearance['zoom']; drift = appearance['pan'] * np.sin(number / scene['fps'] / 8)
    return [zoom * (point[0] - width/2*(1-1/zoom) - drift),
            zoom * (point[1] - height/2*(1-1/zoom))]


def _badge_visible(image, scene, number, position, expected):
    """Verify a private known position still contains a distinguishable ID.

    This uses the renderer's fixed glyph vocabulary, not a learned perception
    model. Blank, occluded, or swapped badges must not validate an identity label.
    """
    x, y = _screen(scene, number, position)
    zoom = scene['appearance']['zoom']
    observed = Image.fromarray(image).convert('L')
    patches = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            patch = observed.transform((19, 23), Image.Transform.AFFINE,
                (zoom, 0, x-9*zoom+dx, 0, zoom, y-11*zoom+dy), Image.Resampling.BILINEAR)
            patches.append(np.array(patch, dtype=float)[2:-2, 2:-2] / 255)
    if max(p.std() for p in patches) < .2:
        return False
    errors = {}
    font = ImageFont.load_default(size=20)
    for letter in 'ABC':
        template = Image.new('L', (40, 48), 255)
        ImageDraw.Draw(template).text((13, 12), letter, font=font, fill=18)
        glyph = np.array(template.crop((11, 13, 30, 36)), dtype=float)[2:-2, 2:-2] / 255
        errors[letter] = min(np.abs(patch-glyph).mean() for patch in patches)
    # Occlusion of a border changes absolute pixel error without making the
    # letter ambiguous. Require contrast and separation from both other IDs.
    return (min(errors, key=errors.get) == expected
            and min(value for letter, value in errors.items() if letter != expected) - errors[expected] > .025)


def _check_world(scene, video):
    from witness.grounded import event_phrase, state_at, ZONES
    failures = []
    numbers = []
    for e in scene['episodes']:
        numbers.extend([e['start_frame']-1,e['start_frame']+2,
                        (e['start_frame']+e['end_frame'])//2,e['end_frame']-2,e['end_frame']+2])
    width,height=scene['resolution']
    images=decode_checkpoints(video,numbers,width,height)
    objects={o['id']:o for o in scene['objects']}
    for e in scene['episodes']:
        matched=[v for v in scene['events'] if v['id']==e['id']]
        if len(matched)!=1 or any(matched[0][f]!=e[f] for f in scene['evaluation']['event_fields']):
            failures.append('event disagrees with interaction episode')
        if e['action'] not in {'carry','push','lift','touch','pass'}:
            failures.append('unknown interaction action');continue
        if e['end_frame']-e['start_frame'] < 18:
            failures.append('interaction shorter than observable minimum')
        if e['target'] not in ZONES or abs(e['destination'][0]-ZONES[e['target']])>.01:
            failures.append('destination label disagrees with visible zone')
        if e['action'] in {'lift','touch'} and e['origin']!=e['destination']:
            failures.append('stationary action changes its ground position')
        if e['action'] in {'carry','push','pass'} and abs(e['origin'][0]-e['destination'][0])<60:
            failures.append('transport action has no visible displacement')
        clear_views={actor:0 for actor in (e['actor'],e['recipient']) if actor}
        for number in (e['start_frame']+2,(e['start_frame']+e['end_frame'])//2,e['end_frame']-2):
            positions=state_at(scene,number)['positions']
            for actor in clear_views:
                clear_views[actor]+=int(_badge_visible(images[number],scene,number,positions[actor],actor))
        # Brief occlusion is compatible with video tracking; there must still be
        # two distinguishable views of every participant during its interaction.
        if any(count<2 for count in clear_views.values()):
            failures.append('actor identity lacks two distinguishable interaction views')
        color=COLORS[objects[e['object']]['color']].lstrip('#')
        expected=np.array([int(color[i:i+2],16) for i in (0,2,4)])
        # Decode both boundaries: a lift visibly leaves the ground while a touch
        # does not; destinations are measured from final pixels, not predicted data.
        for number, position in [(e['start_frame']-1,list(e['origin'])),
                                 (e['start_frame']+2,list(e['origin'])),
                                 (e['end_frame']+2,list(e['destination']))]:
            if number==e['start_frame']+2 and e['action'] in {'carry','lift','pass'}:
                position[1]-=18
            x,y=map(round,_screen(scene,number,position))
            if not (20<=x<width-25 and 16<=y<height-16):
                failures.append(f'object outside image at {number}');continue
            crop=images[number][y-10:y+11,x-14:x+20].astype(float)
            count=np.count_nonzero(np.linalg.norm(crop-expected,axis=2)<60)
            if count<25:
                failures.append(f'object not visible at expected position frame{number}')
    for q in scene['qa']:
        if q['group']=='history':
            kind=q['id'].removeprefix('history_')
            answer=' > '.join(event_phrase(e) for e in scene['events'] if e['object']==kind)
            if answer!=q['a']:failures.append('history answer disagrees with events')
            if q.get('event_ids') != [e['id'] for e in scene['events'] if e['object']==kind]:
                failures.append('history supporting events disagree with visible sequence')
    selections=[]
    selected_ids=[]
    for d in scene['dialogue']:
        words=d['text'].split();kind=words[2]
        after=[e for e in scene['events'] if e['object']==kind and e['start']>=d['start']]
        if not after:
            failures.append('spoken request has no following interaction');continue
        event=after[0]
        if event['start']-d['start']>3 or d['end']>event['start']:
            failures.append('speech does not finish before the requested observation')
        selections.append(f"{kind} {event_phrase(event)}")
        selected_ids.append(event['id'])
    actual=next(q['a'] for q in scene['qa'] if q['id']=='spoken_records')
    if ' > '.join(selections)!=actual:failures.append('audio-grounding answer disagrees with speech/events')
    if next(q.get('event_ids') for q in scene['qa'] if q['id']=='spoken_records') != selected_ids:
        failures.append('audio-grounding support disagrees with observed instructions')
    # The decoded track must contain the actual synthesized words, not simply a
    # non-silent sound with correct metadata. AAC is lossy; compare waveform shape.
    result=subprocess.run([str(FFMPEG),'-v','error','-threads','2','-i',str(video),
        '-vn','-ac','1','-ar',str(scene['audio']['sample_rate']),'-f','s16le','pipe:1'],
        capture_output=True,check=True)
    audio=np.frombuffer(result.stdout,dtype='<i2').astype(float)
    references=scene['audio'].get('validation_pcm',[])
    if len(references)!=len(scene['dialogue']):
        return failures+['missing exact generated speech evidence']
    for d,reference in zip(scene['dialogue'],references):
        a,b=d['start_sample'],d['end_sample']
        if b>len(audio) or b-a<100:
            failures.append('invalid decoded speech interval');continue
        decompressor=zlib.decompressobj()
        pcm=decompressor.decompress(base64.b64decode(reference['zlib_base64'],validate=True),2*(b-a)+1)
        if len(pcm)!=2*(b-a) or not decompressor.eof or hashlib.sha256(pcm).hexdigest()!=reference['sha256']:
            failures.append('invalid exact generated speech evidence');continue
        waveform=np.frombuffer(pcm,dtype='<i2').astype(float)
        correlation=np.corrcoef(waveform,audio[a:b])[0,1]
        if not np.isfinite(correlation) or correlation<.95:
            failures.append('decoded speech differs from declared spoken content')
    return failures


def _check_real(scene, video):
    """Bind scored labels to reviewed intervals and actually decoded footage.

    The recorded reviewer is still the label authority; pixel agreement does not
    turn one review into independent human adjudication.
    """
    from witness.grounded_real import episode_filters
    failures = []
    evidence = scene['annotation_evidence']
    annotation_path = Path(evidence['annotation_path'])
    raw = annotation_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence['annotation_sha256']:
        return ['reviewed annotation bytes changed']
    annotation = json.loads(raw)
    if scene['composition'].get('context_policy') != 'complete_reviewed_interval_v2':
        failures.append('natural task omits the required complete reviewed context')
    if not annotation.get('reviewed') or annotation.get('reviewer') != evidence.get('reviewer'):
        failures.append('real action labels require recorded visual review')
    source = Path(annotation['source_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != annotation['source_sha256']:
        return failures + ['reviewed source video bytes changed']
    if evidence.get('video_sha256') != hashlib.sha256(video.read_bytes()).hexdigest():
        failures.append('real action evidence does not identify exact video bytes')
    if (scene['provenance']['source_id'] != annotation['source_id']
            or scene['provenance']['uploader_id'] != annotation['uploader_id']):
        failures.append('source provenance mismatch')
    for review in annotation.get('review_evidence', []):
        if hashlib.sha256(Path(review['path']).read_bytes()).hexdigest() != review['sha256']:
            failures.append('visual review evidence changed')
    if not annotation.get('review_evidence'):
        failures.append('missing recorded visual review evidence')
    entries = [entry for group in scene['composition']['groups'] for entry in group]
    events = scene['events']
    if len(entries) != len(events):
        return failures + ['action episode count disagrees with composition']
    checkpoints = []
    for event in events:
        checkpoints.extend([event['start_frame'] + 2,
                            (event['start_frame'] + event['end_frame']) // 2,
                            event['end_frame'] - 3, event['end_frame'] + 1])
    actual = decode_checkpoints(video, checkpoints, *scene['resolution'])
    frame = 0
    for entry, event in zip(entries, events):
        clip = annotation['clips'][entry['clip_index']]
        if not clip['start'] <= entry['source_start'] < entry['source_end'] <= clip['end']:
            failures.append('episode samples outside its visually reviewed interval')
        if entry['source_start'] != clip['start'] or entry['source_end'] != clip['end']:
            failures.append('episode truncates the complete reviewed action context')
        if not 120 <= entry['frames'] <= 168:
            failures.append('episode does not use the independent normalized duration range')
        if scene['composition']['options'].get(entry['action']) != clip['label']:
            failures.append('action code disagrees with reviewed visible action')
        if (event['start_frame'] != frame or event['end_frame'] != frame + entry['frames']
                or event['action'] != entry['action']):
            failures.append('scored episode differs from sampled composition')
        offsets = [2, entry['frames'] // 2, entry['frames'] - 3]
        expression = '+'.join(f'eq(n\\,{n})' for n in offsets)
        filters = episode_filters(annotation, clip, entry) + ',select=' + expression
        result = subprocess.run([str(FFMPEG), '-v', 'error', '-threads', '2',
            '-ss', str(entry['source_start']), '-t', str(entry['source_end']-entry['source_start']),
            '-i', str(source), '-an', '-vf', filters, '-frames:v', '3',
            '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            capture_output=True, check=True)
        references = np.frombuffer(result.stdout, np.uint8).reshape(3, 360, 640, 3)
        for offset, expected in zip(offsets, references):
            error = np.abs(actual[frame + offset].astype(float) - expected).mean()
            if error > 8:
                failures.append(f'encoded natural episode differs from reviewed source pixels: {error:.2f}')
        separator = actual[event['end_frame'] + 1]
        if np.abs(separator.astype(float) - 112).mean() > 2:
            failures.append('declared episode separator is not visible')
        frame += entry['frames'] + 4
    if frame != scene['duration_frames']:
        failures.append('natural composition does not cover the video duration')
    index = 0
    for group, question in zip(scene['composition']['groups'], scene['qa'], strict=True):
        if question['a'] != ' > '.join(entry['action'] for entry in group):
            failures.append('natural action history disagrees with composition')
        if question.get('event_ids') != [e['id'] for e in events[index:index+len(group)]]:
            failures.append('natural action history support disagrees with episode order')
        index += len(group)
    return failures


def _check_frame_clock(video: Path, fps: int, count: int) -> list[str]:
    """Frame ordinals and seek timestamps must describe the same video."""
    probe = json.loads(subprocess.check_output([
        str(FFMPEG.with_name('ffprobe')), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
    frames = probe.get('frames', [])
    if len(frames) != count:
        return ['decoded frame clock count mismatch']
    if any(abs(float(frame['best_effort_timestamp_time']) - i/fps) > 1e-6
           for i, frame in enumerate(frames)):
        return ['encoded frame timestamps disagree with declared frame clock']
    return []


def validate_grounded(scene: dict, video: Path) -> list[str]:
    failures=[]
    try:
        if scene.get('schema_version')!='3.0':return ['expected schema3.0']
        probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe')),'-v','error',
            '-show_streams','-show_format','-of','json',str(video)]))
        videos=[s for s in probe['streams'] if s['codec_type']=='video']
        audios=[s for s in probe['streams'] if s['codec_type']=='audio']
        if len(videos)!=1 or len(audios)!=1:return ['expected one video and audio stream']
        stream=videos[0]
        if [stream['width'],stream['height']]!=scene['resolution']:failures.append('encoded resolution mismatch')
        if stream.get('r_frame_rate')!=f"{scene['fps']}/1":failures.append('encoded frame rate mismatch')
        if int(stream.get('nb_frames',-1))!=scene['duration_frames']:failures.append('encoded frame count mismatch')
        if abs(float(probe['format']['duration'])-scene['duration'])>1/scene['fps']+.001:failures.append('encoded duration mismatch')
        failures.extend(_check_frame_clock(video, scene['fps'], scene['duration_frames']))
        if abs(score_reconstruction(scene,perfect_reconstruction(scene))['quality']-1)>1e-9:
            failures.append('private labels do not receive full oracle credit')
        if scene.get('content_kind')=='interaction_world':
            failures.extend(_check_world(scene,video))
        elif scene.get('content_kind')=='real_actions':
            failures.extend(_check_real(scene,video))
        else:
            failures.append('unknown grounded content kind')
    except (KeyError,TypeError,ValueError,OSError,StopIteration,zlib.error,subprocess.SubprocessError) as exc:
        failures.append(f'invalid grounded contract/media: {type(exc).__name__}: {str(exc)[:180]}')
    return failures
