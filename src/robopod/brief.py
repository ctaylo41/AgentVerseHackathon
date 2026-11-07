"""Conservative checks for explicit brief requirements; never substitutes a plan."""
import re

from .planning import PlanValidationError


def requirements(brief):
    text = brief.lower()
    close = re.search(r'close[ -]?up|up close|detail shot', text)
    reveal = re.search(r'zoom out|move out|pull back|widen|return to (?:the )?(?:presenter|reviewer)', text)
    sequence = bool(close and reveal)
    starts_close = bool(close and re.search(r'start[^.!]*(?:close[ -]?up|up close)', text))
    return dict(minimum_shots=2 if sequence and starts_close else 3 if sequence else 2 if close and 'then' in text else 1,
                close_up_required=bool(close), return_to_presenter=bool(sequence),
                starts_close=starts_close,
                sequence_guidance=('close-up of product, then wider presenter shot' if starts_close else 'presenter and product, then product close-up, then wider presenter shot') if sequence else '',
                rule_of_thirds=bool(re.search(r'rule of thirds|rule-of-thirds', text)),
                movement_requested=bool(re.search(r'movement|motion|moition|pan|tilt|move (?:the )?camera|moving camera|sweep|track|dolly|orbit|crane', text)))


def validate_brief(plan, brief, scene):
    spec = requirements(brief)
    shots = plan['shots']
    if len(shots) < spec['minimum_shots']:
        raise PlanValidationError(f"Brief requires at least {spec['minimum_shots']} separate shots; received {len(shots)}. Do not collapse the sequence into one description.")
    close_indices = [i for i,s in enumerate(shots) if s['shot_type'] in ('close_up','detail')]
    if spec['close_up_required'] and not close_indices:
        raise PlanValidationError('Brief requests a close-up: include a separate close_up/detail shot.')
    people = {k for k,v in scene['objects'].items() if v['category']=='person'}
    products = {k for k,v in scene['objects'].items() if v['category'] == 'instrument'}
    if 'synth' in brief.lower() and products and close_indices:
        if not any(any(t['object_id'] in products for t in shots[i]['targets']) for i in close_indices):
            raise PlanValidationError('The requested synthesizer close-up must target the instrument, not just the presenter.')
    if spec['return_to_presenter']:
        if not any(i > close_indices[0] and s['shot_type'] in ('wide','medium') and
                   any(t['object_id'] in people for t in s['targets']) for i,s in enumerate(shots)):
            raise PlanValidationError('After the close-up, include a wider medium/wide shot targeting the presenter. Zoom is unavailable; describe the feasible alternative.')
    if spec['starts_close'] and shots[0]['shot_type'] not in ('close_up','detail'):
        raise PlanValidationError('The brief explicitly asks to start with a close-up.')
    for shot in shots:
        for target in shot['targets']:
            half_height = target['frame_height_fraction']/2
            if not half_height <= target['center_y'] <= 1-half_height:
                raise PlanValidationError(f"Shot {shot['shot_id']}: target {target['object_id']} extends beyond the top/bottom of frame. Use a coherent center_y and frame_height_fraction.")
        if spec['rule_of_thirds']:
            primary = shot['targets'][0]
            if min(abs(primary['center_x']-1/3), abs(primary['center_x']-2/3)) > .06:
                raise PlanValidationError(f"Shot {shot['shot_id']}: rule of thirds requires the primary target center_x near 0.333 or 0.667, not {primary['center_x']}.")
    return spec
