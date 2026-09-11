from copy import deepcopy
import asyncio
import json

import pytest
import numpy as np
from PIL import Image

from witness.grounded import build_scene
from witness.score.oracle import perfect_reconstruction
from witness.score_v3_0_0 import apply_fact_sharing, score_reconstruction
from witness.subnet.validator import aggregate_miner_scores, public_task_spec


@pytest.fixture(scope='module')
def scene():
    return build_scene(72836219098376291, 1)


def record(scene, pred, uid):
    report = score_reconstruction(scene, pred)
    return {"uid": uid, "scene_id": "private-scene", "responded": True,
            "score_version": "3.0.0", "score_before_duplicates": report['score'],
            "semantic_credits": report['semantic_credits'],
            "semantic_fingerprint": report['semantic_fingerprint']}


def test_perfect_and_wrong_results(scene):
    pred = perfect_reconstruction(scene)
    assert score_reconstruction(scene, pred)['quality'] == pytest.approx(1.)
    for event in pred['events']:
        event['target'] = 'unobserved destination'
    report = score_reconstruction(scene, pred)
    assert report['quality'] == 0
    assert report['score'] == 0


def test_actor_a_is_not_an_article(scene):
    pred = perfect_reconstruction(scene)
    for question in scene['qa']:
        if any(part.startswith('A ') for part in question['a'].split(' > ')):
            pred['qa'][question['id']] = ' > '.join(part[2:] if part.startswith('A ') else part
                                                   for part in question['a'].split(' > '))
    report = score_reconstruction(scene, pred)
    assert report['quality'] < 1


def test_changing_actor_colors_cannot_stand_in_for_stable_identities(scene):
    pred = perfect_reconstruction(scene)
    colors = {actor['id']: actor['color'] for actor in scene['actors']}
    for event in pred['events']:
        event['actor'] = colors[event['actor']]
        if event['recipient']:
            event['recipient'] = colors[event['recipient']]
    assert score_reconstruction(scene, pred)['score'] == 0
    pred = perfect_reconstruction(scene)
    for event in pred['events']:
        event['actor'] = 'actor ' + event['actor']
        if event['recipient']:
            event['recipient'] = 'actor ' + event['recipient']
    assert score_reconstruction(scene, pred)['quality'] == pytest.approx(1.)


def test_speech_and_visual_evidence_both_required(scene):
    pred = perfect_reconstruction(scene)
    pred['qa'].pop('spoken_records')
    assert score_reconstruction(scene, pred)['score'] == 0
    pred = perfect_reconstruction(scene)
    pred['events'] = []
    assert score_reconstruction(scene, pred)['score'] == 0


def test_partial_copies_conserve_credit(scene):
    full = perfect_reconstruction(scene)
    records = [record(scene, full, 0)]
    for uid in range(1, len(scene['events']) + 1):
        pred = deepcopy(full)
        pred['events'].pop(uid - 1)
        records.append(record(scene, pred, uid))
    apply_fact_sharing(records)
    assert sum(r['score'] for r in records) <= 1 + 1e-9
    assert all(r['score'] < .2 for r in records)


def test_exact_copies_share_and_noise_does_not_change_facts(scene):
    full = perfect_reconstruction(scene)
    variant = deepcopy(full)
    variant['events'].reverse()
    variant['ignored'] = 'arbitrary extra data'
    for e in variant['events']:
        e['start'] += 1e-7
    records = [record(scene, full, 0), record(scene, variant, 1)]
    assert records[0]['semantic_fingerprint'] == records[1]['semantic_fingerprint']
    apply_fact_sharing(records)
    assert [r['score'] for r in records] == pytest.approx([.5, .5])


def test_mass_alternatives_cannot_create_quality(scene):
    pred = perfect_reconstruction(scene)
    pred['events'] = [dict(e, action=action) for e in pred['events']
                      for action in ('carry', 'push', 'lift', 'touch', 'pass')]
    assert score_reconstruction(scene, pred)['score'] == 0


def test_correct_answers_cannot_use_missing_or_contradictory_event_support(scene):
    pred = perfect_reconstruction(scene)
    supporting_id = next(q['event_ids'][0] for q in scene['qa'] if q['id'] == 'spoken_records')
    index = next(i for i, e in enumerate(scene['events']) if e['id'] == supporting_id)
    pred['events'][index]['action'] = 'unobserved action'
    report = score_reconstruction(scene, pred)
    assert report['grounding']['question_groups']['audio_grounding'] == 0
    assert report['score'] == 0


def test_hallucinations_in_an_absent_family_still_reduce_precision(scene):
    pred = perfect_reconstruction(scene)
    assert not scene['on_screen_text']
    pred['on_screen_text'] = [{'start':i*.1,'end':i*.1+.09,'text':'This is not in the image'}
                              for i in range(100)]
    report = score_reconstruction(scene, pred)
    assert report['grounding']['claim_precision'] < .4
    assert report['score'] == 0


def test_answers_to_nonexistent_questions_are_not_free_claims(scene):
    pred = perfect_reconstruction(scene)
    pred['qa'].update({f'invented_{i}':'fabricated answer' for i in range(100)})
    assert score_reconstruction(scene, pred)['score'] == 0


def test_malformed_family_cannot_hide_many_false_claims_as_one_object(scene):
    pred = perfect_reconstruction(scene)
    pred['on_screen_text'] = {str(i):'fabricated text' for i in range(100)}
    assert score_reconstruction(scene, pred)['score'] == 0


@pytest.mark.parametrize('mutation', ['nan', 'dual_time', 'missing_target', 'missing_recipient'])
def test_malformed_or_omitted_semantics_do_not_match(scene, mutation):
    pred = perfect_reconstruction(scene)
    for e in pred['events']:
        if mutation == 'nan':
            e['start'] = float('nan')
        elif mutation == 'dual_time':
            e['start'] += 1
        else:
            e.pop(mutation.removeprefix('missing_'))
    assert score_reconstruction(scene, pred)['score'] == 0


def test_public_task_hides_actions_seeds_and_audio_schedule(scene):
    task = public_task_spec(scene)
    assert set(task) == {'duration', 'fps', 'tier', 'schema_version', 'qa'}
    assert all(set(q) == {'id', 'q'} for q in task['qa'])
    paired = build_scene(scene['seed'], scene['difficulty'], counterfactual=True)
    assert public_task_spec(paired) == task
    assert [e['action'] for e in paired['events']] != [e['action'] for e in scene['events']]


def test_event_counts_and_gaps_are_not_a_fixed_scene_calendar():
    counts = set()
    gaps = set()
    durations = set()
    for seed in range(20):
        s = build_scene(seed, 2, include_tts_timing=False)
        counts.add(len(s['events']))
        gaps.update(b['start_frame']-a['end_frame'] for a,b in zip(s['events'],s['events'][1:]))
        durations.update(e['end_frame']-e['start_frame'] for e in s['events'])
        assert s['duration'] < 120
    assert len(counts) >= 4
    assert max(gaps)-min(gaps) >= 15
    assert max(durations)-min(durations) >= 15


@pytest.mark.parametrize('value', [None, '', 'unregistered actor'])
def test_bad_private_identity_invalidates_sample_instead_of_matching(value, scene):
    bad = deepcopy(scene)
    bad['events'][0]['actor'] = value
    with pytest.raises(ValueError, match='private semantic'):
        score_reconstruction(bad, perfect_reconstruction(bad))


def test_media_identity_guard_rejects_blank_and_wrong_badges(scene):
    from witness.grounded import render_frame, state_at
    from witness.validate_grounded import _badge_visible
    event = scene['episodes'][0]
    frame = event['start_frame'] + 2
    actor = event['actor']
    position = state_at(scene, frame)['positions'][actor]
    image = np.array(render_frame(scene, frame))
    assert _badge_visible(image, scene, frame, position, actor)
    assert not _badge_visible(image, scene, frame, position, next(a for a in 'ABC' if a != actor))
    blank = np.array(Image.new('RGB', (640, 360), 'white'))
    assert not _badge_visible(blank, scene, frame, position, actor)


def test_hybrid_round_requires_both_slices_and_counts_missing_scenes():
    kinds = {'a': 'interaction_world', 'b': 'real_actions', 'c': 'real_actions'}
    rows = [{'uid': uid, 'scene_id': scene_id, 'score': score,
             'score_version': '3.0.0', 'responded': True}
            for uid, scene_id, score in [(1, 'a', 1), (2, 'a', .8), (2, 'b', .6)]]
    scores, ema, responders = aggregate_miner_scores([1, 2, 3], rows, 3, {}, .2, scene_kinds=kinds)
    assert scores == pytest.approx({1: 0, 2: .3, 3: 0})
    assert ema == scores
    assert responders == {1, 2}
    old_scores, _, _ = aggregate_miner_scores([1, 2, 3], rows, 3, {}, .2)
    assert old_scores == pytest.approx({1: 1/3, 2: 1.4/3, 3: 0})


def test_natural_composition_preserves_full_context_without_duration_label_leakage():
    from witness.grounded_real import build_plan
    annotation = {'reviewed': True, 'reviewer': 'test fixture',
        'clips': [{'start': i*20., 'end': i*20.+length, 'label': f'action {i}'}
                  for i,length in enumerate((2.,4.,8.,6.))],
        'foils': [f'foil {i}' for i in range(4)]}
    plan = build_plan(annotation, 1826517092, 1)
    changed = deepcopy(annotation)
    for clip in changed['clips']:
        clip['end'] += 3.
    other = build_plan(changed, 1826517092, 1)
    entries = [entry for group in plan['groups'] for entry in group]
    assert len(entries) == 21
    for entry in entries:
        clip = annotation['clips'][entry['clip_index']]
        assert entry['source_start'] == clip['start']
        assert entry['source_end'] == clip['end']
        assert 120 <= entry['frames'] <= 168
    assert [e['frames'] for e in entries] == [e['frames'] for g in other['groups'] for e in g]
    assert plan['context_policy'] == 'complete_reviewed_interval_v2'


def test_hybrid_round_rejects_duplicate_and_unknown_scene_rows():
    row = {'uid': 1, 'scene_id': 'a', 'score': 1., 'score_version': '3.0.0', 'responded': True}
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_miner_scores([1], [row, row], 1, {}, .2, scene_kinds={'a': 'real_actions'})
    with pytest.raises(ValueError, match='inventory'):
        aggregate_miner_scores([1], [row], 1, {}, .2, scene_kinds={'b': 'real_actions'})


def test_natural_generator_preserves_seekable_frame_clock(tmp_path):
    import hashlib
    import io
    import json
    import subprocess
    from PIL import Image
    from witness.grounded_real import generate
    from witness.render import FFMPEG
    from witness.tools.server import _extract_frame
    from witness.validate_grounded import _check_frame_clock

    source = tmp_path/'source.mp4'
    subprocess.run([str(FFMPEG), '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=s=160x90:r=24:d=3', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'ultrafast', str(source)], check=True, capture_output=True)
    annotation = {'reviewed': True, 'reviewer': 'synthetic transport fixture',
        'source_path': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_id': 'synthetic-fixture', 'uploader_id': 'synthetic-fixture',
        'clips': [{'start': i, 'end': i+1, 'label': f'fixture action {i}'} for i in range(3)],
        'foils': [f'fixture foil {i}' for i in range(5)]}
    path = tmp_path/'annotation.json'
    path.write_text(json.dumps(annotation))
    labels, video = generate(path, 1826517092, 1, tmp_path/'generated')
    scene = json.loads(labels.read_text())
    assert _check_frame_clock(video, 24, scene['duration_frames']) == []
    jpeg = _extract_frame(video, (scene['duration_frames']-1)/24, 80, 45, FFMPEG)
    assert Image.open(io.BytesIO(jpeg)).size == (80, 45)

    drifted = tmp_path/'drifted.mp4'
    subprocess.run([str(FFMPEG), '-v', 'error', '-i', str(video), '-map', '0',
        '-c', 'copy', '-bsf:v', 'setts=pts=PTS*1.00001:dts=DTS*1.00001', str(drifted)],
        check=True, capture_output=True)
    assert _check_frame_clock(drifted, 24, scene['duration_frames']) == [
        'encoded frame timestamps disagree with declared frame clock']


def test_grounded_round_scores_metered_observations_and_partial_clones(tmp_path):
    from witness.grounded import generate
    from witness.subnet.chain import InMemoryChainAdapter
    from witness.subnet.validator import ValidatorConfig, WitnessValidator
    from witness.tools.client import WitnessClient

    source = tmp_path / 'source'
    labels, _ = generate(92872109007231, 1, source)
    truth = json.loads(labels.read_text())
    chain = InMemoryChainAdapter()

    def handler(kind):
        async def respond(task):
            # This is a validator-only oracle fixture, not a miner implementation.
            assert task.task_spec == public_task_spec(truth)
            client = WitnessClient(task.tool_base_url, session_id=task.session_id)
            assert 'seed' not in client.get_meta().data
            client.get_frame(0, res='320x180')
            prediction = perfect_reconstruction(truth)
            if kind == 'partial':
                required = next(q['event_ids'] for q in truth['qa'] if q['id'] == 'spoken_records')
                index = next(i for i, e in enumerate(truth['events']) if e['id'] not in required)
                prediction['events'].pop(index)
            elif kind == 'wrong':
                for event in prediction['events']:
                    event['target'] = 'unobserved destination'
            task.reconstruction = prediction
            task.trace_summary = {'status': 'ok'}
            return task
        return respond

    for uid, kind in enumerate(('oracle', 'partial', 'wrong'), 1):
        chain.add_miner(uid, handler(kind))
    result = asyncio.run(WitnessValidator(chain, ValidatorConfig(
        round_root=tmp_path/'rounds', source_scenes=(source,), scene_count=1,
        score_version='3.0.0', allow_unlocked=True, transcript_source='none',
        tool_host='127.0.0.1', tool_port=0, set_weights_enabled=False,
    )).run_round())
    first, partial, wrong = (result['miners'][str(uid)] for uid in (1, 2, 3))
    assert .4 < first['round_score'] < .6
    assert 0 < partial['round_score'] < .6
    assert wrong['round_score'] == 0
    assert first['round_score'] + partial['round_score'] <= 1
    assert all(m['scenes'][0]['cost']['visual_tokens'] > 0 for m in (first, partial, wrong))
    assert result['aggregation_identity']['scene_reduction'] == 'minimum_content_slice_mean'
    assert result['weight_submission']['status'] == 'disabled'
    assert not chain.weight_history
    assert all(row['status'] == 'not_accepted' for row in result['feedback']['deliveries'].values())
