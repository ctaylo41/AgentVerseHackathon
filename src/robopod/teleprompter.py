"""One-at-a-time instructions and bounded, explicitly scripted mocap responses."""
from dataclasses import dataclass, asdict
import math

import numpy as np

from .control import advance


@dataclass(frozen=True)
class Instruction:
    source: str
    object_id: str
    text: str
    axis: int
    displacement: float

    @property
    def key(self):
        return (self.object_id, self.axis, int(np.sign(self.displacement)))


class Teleprompter:
    def __init__(self, cooldown=1.0):
        self.cooldown = cooldown
        self.active = None
        self.history = []
        self.ready_at = 0.0
        self.failed = set()
        self.last_direction = {}

    def issue(self, instruction, now):
        direction = self.last_direction.get((instruction.object_id, instruction.axis))
        if self.active is not None or now < self.ready_at or instruction.key in self.failed:
            return False
        # Do not oscillate back and forth during one shot.
        if direction is not None and direction != instruction.key[2]:
            return False
        self.active = instruction
        self.last_direction[(instruction.object_id, instruction.axis)] = instruction.key[2]
        self.history.append(dict(**asdict(instruction), issued_at=now, status='active'))
        return True

    def finish(self, now, observed, evidence):
        if self.active is None:
            raise ValueError('No active instruction')
        self.history[-1].update(status='completed' if observed else 'not_completed',
                                finished_at=now, evidence=evidence)
        if not observed:
            self.failed.add(self.active.key)
        self.active = None
        self.ready_at = now+self.cooldown


def creative_action(subject_id, text):
    """Small exact phrase vocabulary; never guess arbitrary natural-language actions."""
    normalized = text.lower().strip().rstrip('.!')
    phrases = {'step to your left': (0, .25), 'step slightly to your left': (0,.25),
               'step to your right': (0,-.25), 'step slightly to your right': (0,-.25),
               'step closer to the camera': (1,-.3), 'step back': (1,.3)}
    if normalized in phrases:
        axis, delta = phrases[normalized]
        return Instruction('creative', subject_id, text, axis, delta)
    if normalized in ('raise the product', 'move the product higher'):
        return Instruction('creative', 'body:product', text, 2, .15)
    return None


def correction(shot, measured, scene):
    # Camera has already been optimised. Prioritise scale, which the fixed rail cannot freely change.
    for target in shot['targets']:
        key = target['object_id']
        if key not in ('body:person', 'body:product'):
            continue
        metric = measured['targets'][key]
        actual = metric['framing']
        if actual is None:
            continue  # An unobserved subject cannot safely be positioned from stale estimates.
        if metric['scale_error'] > shot['completion']['max_scale_error']:
            closer = actual['frame_height_fraction'] < target['frame_height_fraction']
            label = scene['objects'][key]['label']
            delta = -.5 if closer else .5
            return Instruction('geometric', key, f'Move {label} {"closer to" if closer else "farther from"} the camera.', 1, delta)
    return None


def respond(sim, instruction, *, callback=None, trace=None):
    """Kinematic demo actor, not a physically simulated grasp or walking controller."""
    name = instruction.object_id.removeprefix('body:')
    if name not in ('person','product') or instruction.axis not in (0,1,2):
        raise ValueError('Unsupported scripted response')
    if not math.isfinite(instruction.displacement) or abs(instruction.displacement) > .5:
        raise ValueError('Scripted displacement must be finite and at most 0.5 m')
    mocap = int(sim.model.body(name).mocapid[0])
    if mocap < 0:
        raise ValueError('Subject is not controllable')
    start = sim.data.mocap_pos[mocap].copy()
    end = start.copy()
    end[instruction.axis] += instruction.displacement
    low = np.array([-1.2, -.9 if name == 'product' else .1, .82 if name == 'product' else 0])
    high = np.array([1.2, 2.2, 1.6 if name == 'product' else 0])
    end = np.clip(end, low, high)
    for i in range(1,51):
        t = i/50
        sim.data.mocap_pos[mocap] = start+(3*t*t-2*t*t*t)*(end-start)
        advance(sim, .02, trace)
        if callback and i % 5 == 0:
            callback(sim)
    return dict(scripted=True, applied_displacement=(end-start).tolist(),
                bounded=not np.isclose(end[instruction.axis]-start[instruction.axis], instruction.displacement))


def observe_response(before, after, instruction):
    old = next((d for d in before if d.identity.object_id == instruction.object_id), None)
    new = next((d for d in after if d.identity.object_id == instruction.object_id), None)
    if old is None or new is None:
        return False, dict(reason='Subject must be detected before and after response')
    delta = float(np.median(new.points_world[:,instruction.axis])-np.median(old.points_world[:,instruction.axis]))
    passed = delta*np.sign(instruction.displacement) >= min(.05, abs(instruction.displacement)*.25)
    return bool(passed), dict(observed_surface_displacement_m=delta,
                             criterion='At least 5 cm (or 25% of small requests) in requested direction, from fresh segmented depth')
