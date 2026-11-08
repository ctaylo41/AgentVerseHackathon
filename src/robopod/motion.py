"""Validated high-level motion intent; distances in metres, rig directions in world axes."""
import math

DIRECTIONS = {'push_in': ('dolly', 1, 'forward'), 'pull_back': ('dolly', -1, 'backward'),
              'truck_left': ('truck', -1, 'left'), 'truck_right': ('truck', 1, 'right'),
              'pedestal_up': ('height', 1, 'up'), 'pedestal_down': ('height', -1, 'down')}


def direction_for(kind):
    return DIRECTIONS[kind][2] if kind in DIRECTIONS else 'none' if kind=='static' else 'auto'


def parameters(shot):
    kind = shot.get('motion', 'static')
    return shot.get('motion_parameters', dict(amplitude_m=0. if kind == 'static' else .15,
                    direction=direction_for(kind),
                    subject_lock='none' if kind=='static' else shot['targets'][0]['object_id']))


def validate(shot, limits=None):
    from .planning import PlanValidationError
    from .control import RATES
    kind = shot.get('motion','static')
    p = parameters(shot)
    expected = direction_for(kind)
    if p['direction'] != expected:
        raise PlanValidationError(f'Motion {kind} requires direction {expected}')
    amp = p['amplitude_m']
    if not math.isfinite(amp) or (kind=='static' and amp != 0) or (kind!='static' and not .02 <= amp <= 1):
        raise PlanValidationError('Static amplitude must be zero; moving amplitude must be 0.02–1 metres')
    if p['subject_lock'] != 'none' and p['subject_lock'] not in {t['object_id'] for t in shot['targets']}:
        raise PlanValidationError('Subject lock must reference a target in this shot, or none')
    if limits is not None and kind in DIRECTIONS:
        axis = DIRECTIONS[kind][0]
        if axis not in limits or amp > limits[axis][1]-limits[axis][0]:
            raise PlanValidationError(f'Motion exceeds available {axis} travel')
        if 1.875*amp/shot['duration_seconds'] > RATES[axis]:
            raise PlanValidationError('Motion amplitude/duration exceeds actuator setpoint speed; reduce amplitude or increase duration')
    if limits is not None and kind == 'sweep':
        possible = any(n in limits and amp <= limits[n][1]-limits[n][0]
                       and 1.875*amp/shot['duration_seconds'] <= RATES[n]
                       for n in ('height','truck','dolly'))
        if not possible:
            raise PlanValidationError('Sweep amplitude/duration exceeds available travel or speed')
    return p
