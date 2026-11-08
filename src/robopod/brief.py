"""Conservative checks for explicit brief requirements; never substitutes a plan."""
import re

from .planning import PlanValidationError


def requirements(brief):
    text = brief.lower()
    counts = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6}
    count_match = re.search(r'\b(one|two|three|four|five|six|[1-6])(?:\s+(?:wide|medium|close[ -]?up|static|locked[ -]off|separate|distinct)){0,3}\s+shots?\b',text)
    exact_count = None
    if count_match:
        token = count_match.group(1)
        exact_count = int(token) if token.isdigit() else counts[token]
    close = re.search(
        r'close[ -]?up|up close|detail shot|as close as|get close'
        r'|close to (?:the )?(?:synth|synthesizer|product|instrument|controls|keys)',
        text,
    )
    reveal = re.search(
        r'zoom out|move out|pull(?:ed)? back|move (?:up and (?:then )?)?back|back up'
        r'|widen|return to (?:the )?(?:presenter|reviewer)',
        text,
    )
    sequence = bool(close and reveal and re.search(r'presenter|reviewer|person|\bme\b', text))
    opening = re.split(r'\bthen\b|[.!;]', text)[0]
    starts_close = bool(close and close.start() < len(opening) and
                        (re.search(r'\b(?:start|first|open)\b', opening) or
                         re.match(r'\s*(?:a )?close[ -]?up', opening)))
    static = bool(re.search(r'locked[ -]off|static (?:(?:wide|medium|close[ -]?up|detail) )?(?:camera|shot)|no camera (?:movement|motion)|keep (?:the )?camera still', text))
    negative_dutch = bool(re.search(r'\b(?:no|avoid|without|never|not)\b[^.!]*dutch', text))
    movement_text = re.sub(r'no camera (?:movement|motion)|(?:do not|don.t) move (?:the )?camera', '', text)
    moving = bool(re.search(r'movement|motion|moition|\bpan\b|tilt|push[ -]?in|pull[ -]?back|zoom out|move (?:the )?camera|moving camera|move up|move back|pull(?:ed)? back|sweep|track|truck|pedestal|dolly|orbit|crane', movement_text))
    rule_of_thirds = bool(re.search(r'(?:rule|room)[ -]of[ -]thirds', text))
    return dict(human_actions_forbidden=bool(re.search(r'no (?:human|subject|object) (?:action|movement|motion)|(?:do not|don.?t) move (?:the )?(?:presenter|subject|objects?|anything)',text)),
                centered_requested=bool(re.search(r'\bcenter(?:ed|ing| the)?\b|\bcentre(?:d| the)?\b|\bmiddle\b',text)) and not rule_of_thirds,
                exact_shots=exact_count, static_requested=static and not moving,
                dutch_requested='dutch' in text and not negative_dutch,
                minimum_shots=2 if sequence and starts_close else 3 if sequence else 2 if close and 'then' in text else 1,
                close_up_required=bool(close), return_to_presenter=bool(sequence),
                starts_close=starts_close,
                sequence_guidance='Preserve all requested subjects and shot order; return to the presenter after the close-up.' if sequence else '',
                rule_of_thirds=rule_of_thirds,
                movement_requested=moving)


def validate_brief(plan, brief, scene):
    spec = requirements(brief)
    shots = plan['shots']
    if spec['human_actions_forbidden'] and any(s.get('human_instructions') for s in shots):
        raise PlanValidationError('Brief forbids human actions')
    if spec['centered_requested'] and any(abs(t['center_x']-.5)>.03 or abs(t['center_y']-.5)>.03 for s in shots for t in s['targets']):
        raise PlanValidationError('Brief explicitly requests a centered single shot')
    if spec['exact_shots'] is not None and len(shots) != spec['exact_shots']:
        raise PlanValidationError(f"Brief requests exactly {spec['exact_shots']} shots")
    if len(shots) < spec['minimum_shots']:
        raise PlanValidationError(f"Brief requires at least {spec['minimum_shots']} separate shots; received {len(shots)}. Do not collapse the sequence into one description.")
    close_indices = [i for i,s in enumerate(shots) if s['shot_type'] in ('close_up','detail')]
    if spec['close_up_required'] and not close_indices:
        raise PlanValidationError('Brief requests a close-up: include a separate close_up/detail shot.')
    people = {k for k,v in scene['objects'].items() if v['category']=='person'}
    products = {k for k,v in scene['objects'].items() if v['category'] in ('instrument','component')}
    instrument_closeup = re.search(r'close[ -]?up (?:of |on )?(?:the |its )?(?:synth|synthesizer|instrument|controls|keys)', brief.lower())
    if instrument_closeup and products and close_indices:
        if not any(any(t['object_id'] in products for t in shots[i]['targets']) for i in close_indices):
            raise PlanValidationError('The requested synthesizer close-up must target the instrument, not just the presenter.')
    for part in ('controls','keys'):
        key=f'part:product:{part}'
        if spec['close_up_required'] and part in brief.lower() and key in scene['objects']:
            if not any(any(t['object_id']==key for t in shots[i]['targets']) for i in close_indices):
                raise PlanValidationError(f'Requested detail shot must target {key}, not only the whole product')
    if spec['return_to_presenter']:
        if not any(i > close_indices[0] and s['shot_type'] in ('wide','medium') and
                   any(t['object_id'] in people for t in s['targets']) for i,s in enumerate(shots)):
            raise PlanValidationError('After the close-up, include a wider medium/wide shot targeting the presenter. Zoom is unavailable; describe the feasible alternative.')
    if spec['starts_close'] and shots[0]['shot_type'] not in ('close_up','detail'):
        raise PlanValidationError('The brief explicitly asks to start with a close-up.')
    for shot in shots:
        if (spec['static_requested'] or not spec['movement_requested']) and shot.get('motion', 'static') != 'static':
            raise PlanValidationError('Use static motion unless camera movement is requested in the brief.')
        if not spec['dutch_requested'] and shot.get('horizon', 'level') != 'level':
            raise PlanValidationError('Use a level horizon unless the brief explicitly requests a Dutch angle.')
        for target in shot['targets']:
            half_height = target['frame_height_fraction']/2
            if not half_height <= target['center_y'] <= 1-half_height:
                raise PlanValidationError(f"Shot {shot['shot_id']}: target {target['object_id']} extends beyond the top/bottom of frame. Use a coherent center_y and frame_height_fraction.")
        if spec['rule_of_thirds']:
            primary = shot['targets'][0]
            if min(abs(primary['center_x']-1/3), abs(primary['center_x']-2/3)) > .06:
                raise PlanValidationError(f"Shot {shot['shot_id']}: rule of thirds requires the primary target center_x near 0.333 or 0.667, not {primary['center_x']}.")
    return spec
