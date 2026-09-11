"""Compose private reviewed real action intervals into fresh temporal tasks.

No labels, source names or reference media are shipped by this public module.
The annotation file must remain private and includes its own visual-review and
source-byte provenance. Source/uploader partitions belong to the evaluation run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from witness.render import FFMPEG
from witness.temporal import _rng


def load_annotation_pool(manifest: Path) -> list[tuple[Path, str]]:
    """Load one explicitly selected private partition, with immutable labels."""
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get('schema_version') != 'witness-grounded-pool-3.0':
        raise ValueError('expected a grounded annotation pool, not a raw video pool')
    entries = data.get('annotations')
    if not isinstance(entries, list) or not entries:
        raise ValueError('grounded annotation pool is empty')
    result = []
    sources = set()
    for entry in entries:
        path = (manifest.parent / entry['path']).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('grounded annotation revision changed')
        annotation = json.loads(raw)
        if annotation.get('partition') != data.get('partition'):
            raise ValueError('grounded pool mixes evaluation partitions')
        if annotation.get('reviewed') is not True or not annotation.get('reviewer'):
            raise ValueError('grounded pool contains unreviewed labels')
        identity = annotation['source_id']
        if identity in sources:
            raise ValueError('duplicate source in grounded annotation pool')
        sources.add(identity)
        result.append((path, digest))
    return result


def build_plan(annotation: dict, seed: int, tier: int, *, counterfactual=False):
    if annotation.get('reviewed') is not True or not annotation.get('reviewer'):
        raise ValueError('real-action intervals require recorded visual review')
    clips=annotation['clips'];foils=annotation['foils']
    if not 3<=len(clips)<=6 or len({c['label'] for c in clips})!=len(clips):
        raise ValueError('need three to six distinct reviewed action classes')
    descriptions=[c['label'] for c in clips]+foils
    if len(set(descriptions))!=len(descriptions) or len(descriptions)<8:
        raise ValueError('at least eight distinct, plausible action options required')
    story,timing,labels=(_rng(seed,'real:'+v) for v in ('story','timing','labels'))
    options=list(descriptions);labels.shuffle(options)
    names={description:f'K{index+1}' for index,description in enumerate(options)}
    n=7 if len(clips)<=4 else 8
    groups=[]
    for group in range(3):
        for attempt in range(2000):
            sequence=list(range(len(clips)))+[story.randrange(len(clips)) for _ in range(n-len(clips))]
            story.shuffle(sequence)
            counts=Counter(sequence)
            orders=math.factorial(n)//math.prod(math.factorial(v) for v in counts.values())
            boundary_ok=not groups or sequence[0]!=groups[-1][-1]['clip_index']
            if orders>=200 and boundary_ok and all(a!=b for a,b in zip(sequence,sequence[1:])):
                break
        else:
            raise ValueError('could not generate a sufficiently diverse action sequence')
        entries=[]
        for index in sequence:
            clip=clips[index]
            lo,hi=float(clip['start']),float(clip['end'])
            if hi-lo<1.0:raise ValueError('reviewed action interval shorter than one second')
            # Preserve the complete reviewed context. An arbitrary subwindow can
            # hide the very tool/object named by its inherited label. Normalize
            # playback to an independently sampled duration so source interval
            # length does not disclose an action's identity or order.
            frames=timing.randint(120,168)
            entries.append({'clip_index':index,'source_start':lo,'source_end':hi,
                            'frames':frames,'action':names[clip['label']]})
        groups.append(entries)
    if counterfactual:
        permutation=_rng(seed,'real:counterfactual')
        for group,entries in enumerate(groups):
            # Preserve complete multiset and public block durations, and preserve
            # the entire video's first and last clips exactly.
            left=1 if group==0 else 0;right=len(entries)-1 if group==2 else len(entries)
            original=entries[left:right]
            for attempt in range(2000):
                shifted=list(original);permutation.shuffle(shifted)
                candidate=entries[:left]+shifted+entries[right:]
                codes=[e['action'] for e in candidate]
                if (codes!=[e['action'] for e in entries]
                    and all(a!=b for a,b in zip(codes,codes[1:]))
                    and (group==0 or codes[0]!=groups[group-1][-1]['action'])):
                    entries[:]=candidate
                    break
            else:
                raise ValueError('could not construct a distinct observable counterfactual')
    return {'seed':seed,'tier':tier,'context_policy':'complete_reviewed_interval_v2',
            'options':{names[d]:d for d in options},'groups':groups}


def episode_filters(annotation, clip, entry):
    """The declared pixel transform; validators independently decode its output."""
    filters = []
    if clip.get('crop') or annotation.get('crop'):
        filters.append(clip.get('crop') or annotation['crop'])
    speed = (entry['frames'] / 24) / (entry['source_end'] - entry['source_start'])
    filters.extend([f'setpts=(PTS-STARTPTS)*{speed:.12f}', 'fps=24',
                    'scale=640:360:flags=lanczos', 'setsar=1', 'format=yuv420p',
                    'tpad=stop_mode=clone:stop_duration=0.1'])
    return ','.join(filters)


def generate(annotation_path: Path, seed: int, tier: int, output: Path, *, counterfactual=False):
    output=Path(output)
    if any((output/name).exists() for name in ('scene.json','video.mp4')):
        raise FileExistsError('grounded real generation requires a fresh output directory')
    annotation_path=Path(annotation_path).resolve()
    annotation=json.loads(annotation_path.read_text())
    source=Path(annotation['source_path']).expanduser().resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=annotation['source_sha256']:
        raise ValueError('reviewed source bytes have changed')
    plan=build_plan(annotation,seed,tier,counterfactual=counterfactual)
    output.mkdir(parents=True,exist_ok=True)
    scene={'schema_version':'3.0','renderer_version':'witness-real-actions-3.0-context-v3',
           'seed':seed,'difficulty':tier,'fps':24,'resolution':[640,360],'debug_labels':False,
           'content_kind':'real_actions','actors':[],'objects':[],'events':[],
           'dialogue':[],'shots':[],'on_screen_text':[],'audio_events':[],
           'intentional_errors':[],'qa':[],'audio':{'sample_rate':22050,'channels':1},
           'evaluation':{'event_fields':['action'],'required_qa_groups':['history'],
                         'interval_tolerance':.35},
           'provenance':{'generator':'witness.grounded_real.generate','source_id':annotation['source_id'],
                         'uploader_id':annotation['uploader_id'],'counterfactual':counterfactual},
           'composition':plan}
    frame=0;parts=[]
    options='; '.join(f'{code}: {description}' for code,description in plan['options'].items())
    with tempfile.TemporaryDirectory(prefix='witness-real-actions-') as temp:
        temporary=Path(temp)
        separator=temporary/'separator.mp4'
        subprocess.run([str(FFMPEG),'-v','error','-y','-f','lavfi',
            '-i','color=c=0x707070:s=640x360:r=24','-frames:v','4','-an',
            '-c:v','libx264','-threads','2','-preset','veryfast','-crf','18',str(separator)],
            check=True,capture_output=True)
        for group_index,group in enumerate(plan['groups']):
            start=frame
            first_event_index=len(scene['events'])
            for entry in group:
                clip=annotation['clips'][entry['clip_index']]
                source_seconds=entry['source_end']-entry['source_start']
                path=temporary/f'{len(parts):03d}.mp4'
                subprocess.run([str(FFMPEG),'-v','error','-y','-threads','2',
                    '-ss',str(entry['source_start']),'-t',str(source_seconds),'-i',str(source),
                    '-an','-vf',episode_filters(annotation,clip,entry),'-frames:v',str(entry['frames']),
                    '-c:v','libx264','-threads','2','-preset','veryfast','-crf','18',str(path)],
                    check=True,capture_output=True)
                parts.append(path)
                event={'id':f'event_{len(scene["events"])}','start_frame':frame,
                       'end_frame':frame+entry['frames'],'start':frame/24,
                       'end':(frame+entry['frames'])/24,'action':entry['action']}
                scene['events'].append(event);frame+=entry['frames']
                parts.append(separator);frame+=4
            scene['qa'].append({'id':f'history_{group_index+1}','type':'grounded_sequence','group':'history',
                'q':f'From {start/24:.6f} to {frame/24:.6f} seconds, list the observed action class of every episode, '
                    'in chronological order, including repeats. Return action codes separated by >. '
                    'Brief gray screens separate episodes; camera cuts inside an episode do not create another answer. '
                    'Also reconstruct every action episode in events with start/end seconds and action code. '
                    'The labels describe visible actions, tools and affected objects; do not infer a recipe order. '
                    'Some action options never occur. Action options: '+options,
                'a':' > '.join(e['action'] for e in group),
                'event_ids':[e['id'] for e in scene['events'][first_event_index:]]})
        scene['duration_frames']=frame;scene['duration']=frame/24
        listing=temporary/'parts.txt'
        listing.write_text(''.join(f"file '{p.as_posix()}'\n" for p in parts))
        video=output/'video.mp4'
        subprocess.run([str(FFMPEG),'-v','error','-y','-f','concat','-safe','0','-i',str(listing),
            '-f','lavfi','-i','anullsrc=r=22050:cl=mono','-map','0:v','-map','1:a',
            '-c:v','copy',
            # Concatenated container durations can drift by fractions of a
            # millisecond. Restore the declared frame clock without re-encoding
            # pixels, including the last frame requested by the observation API.
            '-bsf:v','setts=pts=round(PTS*TB*24)/(24*TB):dts=round(DTS*TB*24)/(24*TB)',
            '-c:a','aac','-t',str(frame/24),'-movflags','+faststart',str(video)],
            check=True,capture_output=True)
    scene['annotation_evidence']={'reviewed':True,'reviewer':annotation['reviewer'],
        'annotation_path':str(annotation_path),
        'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'sources':[{'source_id':annotation['source_id'],'uploader_id':annotation['uploader_id'],
                    'sha256':annotation['source_sha256']}],
        'review_evidence':annotation.get('review_evidence',[])}
    labels_path=output/'scene.json';labels_path.write_text(json.dumps(scene,indent=2)+'\n')
    return labels_path,video
