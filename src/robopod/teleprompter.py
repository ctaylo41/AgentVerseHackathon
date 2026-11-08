"""One-at-a-time instructions and bounded, explicitly scripted mocap responses."""
from dataclasses import dataclass, asdict
import math
import re

import numpy as np

from .control import advance


@dataclass(frozen=True)
class Instruction:
    source: str
    object_id: str
    text: str
    axis: int
    displacement: float
    action: str = 'translate'

    @property
    def key(self):
        return (self.object_id, self.axis, int(np.sign(self.displacement)), self.action)


class Teleprompter:
    def __init__(self, cooldown=1.0):
        self.cooldown = cooldown
        self.active = None
        self.history = []
        self.ready_at = 0.0
        self.failed = set()
        self.last_direction = {}

    def issue(self, instruction, now):
        direction = self.last_direction.get((instruction.object_id, instruction.axis, instruction.action))
        if self.active is not None or now + 1e-9 < self.ready_at or instruction.key in self.failed:
            return False
        # Do not oscillate back and forth during one shot.
        if direction is not None and direction != instruction.key[2]:
            return False
        self.active = instruction
        self.last_direction[(instruction.object_id, instruction.axis, instruction.action)] = instruction.key[2]
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
    """Interpret Gemma's natural-language teleprompter directions into executable subject responses."""
    if subject_id != 'body:person':
        return None  # Unfamiliar people have no configured actor.
    normalized = text.lower().strip().rstrip('.!')
    if not normalized:
        return None
    if re.search(r'\b(?:backflip|flip|juggle|sing|song|dance|jump|cartwheel|fly|swim|climb|handstand)\b', normalized):
        return None
    if re.search(r'\b(?:start close to|frame the presenter|camera tracks|camera moves|dolly|pedestal|rule of thirds)\b', normalized):
        return None
    if 'smile' in normalized and 'point' in normalized:
        return None
    performances = {
        'turn your shoulders toward the camera': 'turn_to_camera', 'turn toward the camera': 'turn_to_camera',
        'turn toward the instrument': 'turn_to_product', 'turn toward the synthesizer': 'turn_to_product',
        'look at the camera': 'look_at_camera', 'look at the camera and smile': 'look_and_smile',
        'look away from the camera': 'look_away', 'look away': 'look_away',
        'turn away from the camera': 'look_away', 'look to the side': 'look_away',
        'smile': 'smile', 'point to the controls': 'point_to_product', 'point to the product': 'point_to_product',
        'hold the product closer to your face': 'hold_near_face', 'hold the product near your face': 'hold_near_face',
    }
    if normalized in performances:
        return Instruction('creative',subject_id,text,-1,0.,performances[normalized])
    phrases = {
        'step to your left': (0, .25), 'step slightly to your left': (0, .25),
        'move to your left': (0, .25), 'move slightly to your left': (0, .25), 'take a step left': (0, .25),
        'step to your right': (0, -.25), 'step slightly to your right': (0, -.25),
        'move to your right': (0, -.25), 'move slightly to your right': (0, -.25), 'take a step right': (0, -.25),
        'step closer to the camera': (1, -.3), 'move closer to the camera': (1, -.3),
        'step forward': (1, -.3), 'step closer': (1, -.3),
        'step back': (1, .3), 'move back': (1, .3), 'step farther from the camera': (1, .3),
    }
    if normalized in phrases:
        axis, delta = phrases[normalized]
        return Instruction('creative', subject_id, text, axis, delta)
    product_phrases = {
        'raise the product': (2, .15), 'move the product higher': (2, .15),
        'lower the product': (2, -.15), 'move the product lower': (2, -.15),
        'move the product closer': (1, -.3), 'move the product closer to the camera': (1, -.3),
        'bring the product closer': (1, -.3), 'push the product back': (1, .3),
        'move the product to the left': (0, .2), 'move the product to the right': (0, -.2),
    }
    if normalized in product_phrases:
        axis, delta = product_phrases[normalized]
        return Instruction('creative', 'body:product', text, axis, delta)
    # Natural-language interpretation for unscripted Gemma teleprompter directions:
    no_touch = bool(re.search(r'\b(?:do not|don\'t|without|never|avoid|hands off)\b.*\b(?:move|touch|lift|hold)\b', normalized))
    if re.search(r'\b(?:look|gaze|glance|turn|face)\b.*\b(?:away|side|off[- ]camera|elsewhere|profile|window|plant)\b', normalized):
        return Instruction('creative', subject_id, text, -1, 0., 'look_away')
    if re.search(r'\b(?:step|move|shift|walk|lean)\b', normalized) and not re.search(r'\b(?:camera|rig|dolly|truck)\s+(?:moves|tracks|sweeps)\b', normalized):
        if not no_touch and re.search(r'\b(?:product|synth|synthesizer|instrument|keyboard)\b', normalized):
            if re.search(r'\b(?:raise|lift|higher|up)\b', normalized):
                return Instruction('creative', 'body:product', text, 2, .15)
            if re.search(r'\b(?:lower|down)\b', normalized):
                return Instruction('creative', 'body:product', text, 2, -.15)
            if re.search(r'\b(?:closer|forward|toward)\b', normalized):
                return Instruction('creative', 'body:product', text, 1, -.3)
            if re.search(r'\b(?:back|farther|away)\b', normalized):
                return Instruction('creative', 'body:product', text, 1, .3)
        if re.search(r'\b(?:left)\b', normalized):
            return Instruction('creative', subject_id, text, 0, .25)
        if re.search(r'\b(?:right)\b', normalized):
            return Instruction('creative', subject_id, text, 0, -.25)
        if re.search(r'\b(?:closer|forward|toward the camera|in)\b', normalized):
            return Instruction('creative', subject_id, text, 1, -.3)
        if re.search(r'\b(?:back|backward|farther|away)\b', normalized):
            return Instruction('creative', subject_id, text, 1, .3)
    if not no_touch and re.search(r'\b(?:hold|lift|bring)\b.*\b(?:face|up|close)\b', normalized):
        return Instruction('creative', subject_id, text, -1, 0., 'hold_near_face')
    if re.search(r'\b(?:point|gesture|show|indicate|reach|touch|controls|keys|knobs)\b', normalized) and not no_touch:
        return Instruction('creative', subject_id, text, -1, 0., 'point_to_product')
    if re.search(r'\b(?:turn|angle|orient|face)\b.*\b(?:synth|synthesizer|instrument|product|keyboard|desk|table)\b', normalized):
        return Instruction('creative', subject_id, text, -1, 0., 'turn_to_product')
    if re.search(r'\b(?:smile|smiling|grin|warmly|friendly)\b', normalized):
        if re.search(r'\b(?:look|camera|lens|viewer)\b', normalized):
            return Instruction('creative', subject_id, text, -1, 0., 'look_and_smile')
        return Instruction('creative', subject_id, text, -1, 0., 'smile')
    if re.search(r'\b(?:turn|shoulders|orient)\b.*\b(?:camera|lens|viewer|front)\b', normalized):
        return Instruction('creative', subject_id, text, -1, 0., 'turn_to_camera')
    return Instruction('creative', subject_id, text, -1, 0., 'look_at_camera')


def correction(shot, measured, scene):
    # Camera has already been optimised. Check scale, lateral composition (e.g. rule-of-thirds), vertical height and occlusion.
    criteria = shot['completion']
    explicit_prop_move = bool(re.search(
        r'(?:bring|move|push|shift|slide|raise|lower|lift) (?:the )?(?:synth|synthesizer|product|instrument)',
        shot.get('description', '').lower(),
    ))
    for target in shot['targets']:
        key = target['object_id']
        actor_key = scene['objects'].get(key, {}).get('parent_id') or key
        if actor_key not in ('body:person', 'body:product'):
            continue
        # Do not slide tabletop props during automatic geometric camera refinement unless explicitly requested.
        if scene['objects'].get(actor_key, {}).get('category') == 'instrument' and not explicit_prop_move and 'shot_type' in shot:
            continue
        metric = measured['targets'][key]
        actual = metric['framing']
        if actual is None:
            continue  # An unobserved subject cannot safely be positioned from stale estimates.
        label = scene['objects'][key]['label']
        if metric.get('scale_error', 0.0) > criteria.get('max_scale_error', 1.0):
            closer = actual['frame_height_fraction'] < target['frame_height_fraction']
            delta = -.5 if closer else .5
            return Instruction('geometric', actor_key, f'Move {label} {"closer to" if closer else "farther from"} the camera.', 1, delta)
        if 'center_x' in target and 'center_y' in target and metric.get('center_error', 0.0) > criteria.get('max_center_error', 1.0):
            dx = target['center_x'] - actual['center_x']
            dy = target['center_y'] - actual['center_y']
            if abs(dx) >= abs(dy) or actor_key != 'body:product':
                # In the camera frame (+X right), positive world X shifts target right (+center_x).
                delta = .25 if dx > 0 else -.25
                direction = 'to your left' if actor_key == 'body:person' and delta > 0 else 'to your right' if actor_key == 'body:person' else 'to the left' if delta > 0 else 'to the right'
                return Instruction('geometric', actor_key, f'Move {label} {direction} to align with the composition.', 0, delta)
            delta = -.15 if dy > 0 else .15
            return Instruction('geometric', actor_key, f'Move {label} {"lower" if delta < 0 else "higher"} in frame.', 2, delta)
        if metric.get('visible_surface_fraction', 1.0) < criteria.get('min_visible_fraction', 0.0):
            return Instruction('geometric', actor_key, f'Move {label} slightly to clear occlusion.', 0, .25 if actor_key == 'body:person' else -.25)
    return None


def respond(sim, instruction, *, callback=None, trace=None):
    """Kinematic demo actor, not a physically simulated grasp or walking controller."""
    if instruction.action != 'translate':
        from .actors import perform
        return perform(sim,instruction.action,callback,trace)
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


def observe_response(before, after, instruction, camera_position=None):
    if instruction.action != 'translate':
        from .actors import observed
        return observed(before,after,instruction.action,camera_position)
    old = next((d for d in before if d.identity.object_id == instruction.object_id), None)
    new = next((d for d in after if d.identity.object_id == instruction.object_id), None)
    if old is None or new is None:
        return False, dict(reason='Subject must be detected before and after response')
    delta = float(np.median(new.points_world[:,instruction.axis])-np.median(old.points_world[:,instruction.axis]))
    passed = delta*np.sign(instruction.displacement) >= min(.05, abs(instruction.displacement)*.25)
    return bool(passed), dict(observed_surface_displacement_m=delta,
                             criterion='At least 5 cm (or 25% of small requests) in requested direction, from fresh segmented depth')
