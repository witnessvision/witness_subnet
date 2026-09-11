"""Decoded-media observability checks for temporal scenes, separate from scoring."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import ImageFont

from witness.render import FFMPEG, world_to_screen
from witness.scene import COLORS, state_at


def validate_temporal(scene: dict, video: Path) -> list[str]:
    failures: list[str] = []
    checks: dict[int, list[dict]] = {}
    for event in scene['events']:
        if event['action'] in {'pick_up', 'drop'}:
            for number in (event['frame'] - 1, event['frame'], event['frame'] + 2):
                checks.setdefault(number, []).append(event)
    for shot in scene['shots'][1:]:
        checks.setdefault(shot['start_frame'] - 1, [])
        checks.setdefault(shot['start_frame'], [])
    for text in scene['on_screen_text']:
        checks.setdefault(text['start_frame'] + 2, [])
        font = ImageFont.load_default(size=18)
        box = font.getbbox(text['text'])
        if box[2] - box[0] + 10 > text['bbox'][2] - text['bbox'][0]:
            failures.append(f"text does not fit its visible box: {text['id']}")
    for question in scene['qa']:
        expected = [e['actor'] for e in scene['events']
                    if e['action'] == 'pick_up' and e.get('object') == question['object']]
        if expected != question['actor_sequence']:
            failures.append(f"history label disagrees with events: {question['id']}")
        orders = math.factorial(len(expected)) // math.prod(math.factorial(n) for n in Counter(expected).values())
        if orders < 200:
            failures.append(f"insufficient temporal ambiguity: {question['id']}")
    numbers = sorted(checks)
    def selection(items):
        if len(items) == 1:
            return f'eq(n\\,{items[0]})'
        midpoint = len(items) // 2
        return f'({selection(items[:midpoint])}+{selection(items[midpoint:])})'
    selector = selection(numbers)
    raw = subprocess.check_output([str(FFMPEG), '-v', 'error', '-i', str(video),
        '-vf', f'select={selector}', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    width, height = scene['resolution']
    size = width * height * 3
    if len(raw) != size * len(numbers):
        return failures + ['decoded checkpoint count does not match requested frames']
    images = {n: np.frombuffer(raw[i*size:(i+1)*size], np.uint8).reshape(height,width,3)
              for i,n in enumerate(numbers)}
    definitions = {o['id']: o for o in scene['objects']}
    for number, events in checks.items():
        state = state_at(scene, number)
        for event in events:
            obj = state['objects'][event['object']]
            x,y = world_to_screen(scene, number, obj['position'])
            if not (15 <= x < width-20 and 12 <= y < height-12):
                failures.append(f"object outside visible image at frame {number}")
                continue
            color = COLORS[definitions[event['object']]['color']].lstrip('#')
            expected_color = np.array([int(color[i:i+2],16) for i in (0,2,4)])
            region = images[number][y-9:y+10,x-12:x+14].astype(float)
            visible_pixels = np.count_nonzero(np.linalg.norm(region-expected_color,axis=2)<55)
            if visible_pixels < 15:
                failures.append(f"object color not observable at frame {number}")
    for event in scene['events']:
        if event['action'] not in {'pick_up','drop'}:
            continue
        number = event['frame']
        obj = state_at(scene, number)['objects'][event['object']]
        x,y = world_to_screen(scene, number, obj['position'])
        before = images[number-1][max(0,y-20):y+21,max(0,x-22):x+23].astype(float)
        after = images[number][max(0,y-20):y+21,max(0,x-22):x+23].astype(float)
        if np.abs(after-before).mean() < 1.5:
            failures.append(f"no visible pickup/drop change at frame {number}")
    for shot in scene['shots'][1:]:
        number = shot['start_frame']
        difference = np.abs(images[number].astype(float)-images[number-1]).mean()
        if difference < 3:
            failures.append(f"shot boundary not observable at frame {number}")
    return failures
