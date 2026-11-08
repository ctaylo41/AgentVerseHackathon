"""Strict high-level shot objectives; no language-model motor commands."""
import copy
import json
import math


class PlanValidationError(ValueError):
    pass


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def text(maximum=500):
    return dict(type='string', minLength=1, maxLength=maximum)


def number(low, high):
    return dict(type='number', minimum=low, maximum=high)


def array(items, minimum=0, maximum=8):
    return dict(type='array', items=items, minItems=minimum, maxItems=maximum)


SHOT_PLAN_SCHEMA = obj({
    'schema_version': dict(type='integer', enum=[1]),
    'title': text(120),
    'shots': array(obj({
        'shot_id': text(40),
        'shot_type': dict(type='string', enum=['wide', 'medium', 'close_up', 'detail']),
        'description': text(),
        'duration_seconds': number(1, 60),
        'targets': array(obj({
            'object_id': text(100),
            'center_x': number(0, 1),
            'center_y': number(0, 1),
            'frame_height_fraction': number(0.02, 1),
        }), 1, 4),
        'perspective': dict(type='string', enum=['eye_level', 'high_angle', 'low_angle', 'neutral']),
        'background_object_ids': array(text(100), 0, 4),
        'preferred_view_id': text(100),
        'human_instructions': array(obj({'subject_id': text(100), 'instruction': text(240)}), 0, 3),
        'completion': obj({
            'max_center_error': number(0.01, 0.25),
            'max_scale_error': number(0.01, 0.3),
            'min_visible_fraction': number(0.1, 1),
            'headroom_fraction': number(0, 0.4),
            'headroom_tolerance': number(0.01, 0.2),
            'hold_seconds': number(0.2, 10),
        }),
        'feasibility_notes': text(),
    }), 1, 6),
})


# Older saved plans remain valid, with static motion and a level horizon by default.
_shot_schema = SHOT_PLAN_SCHEMA['properties']['shots']['items']
_shot_schema['properties'].update(
    motion=dict(type='string', enum=['static', 'sweep', 'push_in', 'pull_back', 'truck_left', 'truck_right', 'pedestal_up', 'pedestal_down']),
    horizon=dict(type='string', enum=['level', 'dutch']),
    motion_parameters=obj({'amplitude_m': number(0,1),
                           'direction': dict(type='string', enum=['none','auto','forward','backward','left','right','up','down']),
                           'subject_lock': text(100)}),
)


def scene_plan_schema(scene, allowed_views):
    """Constrain decoding to actual scene IDs, without mutating the base contract."""
    schema = copy.deepcopy(SHOT_PLAN_SCHEMA)
    shot = schema['properties']['shots']['items']['properties']
    object_ids = sorted(scene['objects'])
    people = sorted(key for key, value in scene['objects'].items() if value['category'] == 'person')
    if not object_ids or not allowed_views:
        raise ValueError('Planning requires observed objects and representative views')
    shot['targets']['items']['properties']['object_id']['enum'] = object_ids
    shot['motion_parameters']['properties']['subject_lock']['enum'] = ['none'] + object_ids
    shot['background_object_ids']['items']['enum'] = object_ids
    shot['preferred_view_id']['enum'] = sorted(set(allowed_views))
    if people:
        shot['human_instructions']['items']['properties']['subject_id']['enum'] = people
    else:
        shot['human_instructions']['maxItems'] = 0
    return schema


def _validate(value, schema, path='$'):
    kind = schema['type']
    valid = {'object': lambda: isinstance(value, dict),
             'array': lambda: isinstance(value, list),
             'string': lambda: isinstance(value, str),
             'integer': lambda: type(value) is int,
             'number': lambda: type(value) in (int, float) and math.isfinite(value)}[kind]()
    if not valid:
        raise PlanValidationError(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise PlanValidationError(f'{path}: must be one of {schema["enum"]}')
    if kind == 'object':
        if not set(schema['required']) <= set(value) or not set(value) <= set(schema['properties']):
            raise PlanValidationError(f'{path}: fields must be exactly {list(schema["properties"])}')
        for key in value:
            _validate(value[key], schema['properties'][key], f'{path}.{key}')
    elif kind == 'array':
        if not schema['minItems'] <= len(value) <= schema['maxItems']:
            raise PlanValidationError(f'{path}: invalid item count')
        for index, item in enumerate(value):
            _validate(item, schema['items'], f'{path}[{index}]')
    elif kind == 'string':
        if 'minLength' in schema and not schema['minLength'] <= len(value.strip()) <= schema['maxLength']:
            raise PlanValidationError(f'{path}: invalid text length')
    elif kind == 'number' and not schema['minimum'] <= value <= schema['maximum']:
        raise PlanValidationError(f'{path}: outside [{schema["minimum"]}, {schema["maximum"]}]')


def decode_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PlanValidationError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    try:
        plan = json.loads(content, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(PlanValidationError(f'Invalid number: {value}')))
    except (json.JSONDecodeError, TypeError, RecursionError) as error:
        raise PlanValidationError(f'Expected one JSON object: {error}') from error
    return plan


def parse_plan(content, scene, allowed_views):
    plan = decode_json(content)
    _validate(plan, SHOT_PLAN_SCHEMA)
    seen = set()
    for shot in plan['shots']:
        from .motion import validate
        validate(shot)
        if shot['shot_id'] in seen:
            raise PlanValidationError('Shot IDs must be unique')
        seen.add(shot['shot_id'])
        target_ids = [target['object_id'] for target in shot['targets']]
        background = shot['background_object_ids']
        if len(set(target_ids)) != len(target_ids) or len(set(background)) != len(background):
            raise PlanValidationError('Duplicate target/background IDs')
        if set(target_ids) & set(background):
            raise PlanValidationError('A target cannot also be a required background object')
        for identity in target_ids + background:
            if identity not in scene['objects']:
                raise PlanValidationError(f'Unknown object ID: {identity}')
        if shot['preferred_view_id'] not in allowed_views:
            raise PlanValidationError('Preferred view must be one of the supplied representative views')
        for index, instruction in enumerate(shot['human_instructions']):
            subject = scene['objects'].get(instruction['subject_id'])
            if subject is None or subject['category'] != 'person':
                people = sorted(key for key, value in scene['objects'].items() if value['category'] == 'person')
                raise PlanValidationError(
                    f"Shot {shot['shot_id']!r} human_instructions[{index}].subject_id={instruction['subject_id']!r} "
                    f"must reference a person. Allowed person IDs: {people}. "
                    'Address product-handling instructions to a person, or omit the instruction; use [] if no people exist.')
        if shot['completion']['hold_seconds'] > shot['duration_seconds']:
            raise PlanValidationError('Completion hold cannot exceed shot duration')
    return plan
