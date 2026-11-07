import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robopod.planning import parse_plan, PlanValidationError, scene_plan_schema, SHOT_PLAN_SCHEMA
from robopod.gemma import generate_plan, OllamaClient, PlannerError, prepare_context

SCENE = {'objects': {'body:person': {'category': 'person'}, 'body:product': {'category': 'instrument'}}}


def valid_plan():
    return dict(schema_version=1, title='Tech review', shots=[dict(
        shot_id='shot_1', shot_type='medium', description='Presenter and synthesizer', duration_seconds=8,
        targets=[dict(object_id='body:person', center_x=0.5, center_y=0.5, frame_height_fraction=0.65)],
        perspective='eye_level', background_object_ids=[], preferred_view_id='view_007',
        human_instructions=[dict(subject_id='body:person', instruction='Look at the camera.')],
        completion=dict(max_center_error=0.08, max_scale_error=0.1, min_visible_fraction=0.8,
                        headroom_fraction=0.08, headroom_tolerance=0.04, hold_seconds=1),
        feasibility_notes='Starting reference only; fixed rail limits perspective.')])


class ValidationTests(unittest.TestCase):
    def test_valid_and_reject_bad_semantics(self):
        self.assertEqual(parse_plan(json.dumps(valid_plan()), SCENE, ['view_007']), valid_plan())
        for field, value in [('duration_seconds', True), ('duration_seconds', float('nan')),
                             ('duration_seconds', -2), ('shot_type', 'orbit'),
                             ('preferred_view_id', 'missing')]:
            plan = valid_plan()
            plan['shots'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(PlanValidationError):
                parse_plan(json.dumps(plan), SCENE, ['view_007'])
        for mutate in [lambda s: s.update(pan=0.2),
                       lambda s: s['targets'][0].update(object_id='invented'),
                       lambda s: s['targets'][0].update(center_x=1.1),
                       lambda s: s['human_instructions'][0].update(subject_id='body:product'),
                       lambda s: s['completion'].update(hold_seconds=9)]:
            plan = valid_plan()
            mutate(plan['shots'][0])
            with self.assertRaises(PlanValidationError):
                parse_plan(json.dumps(plan), SCENE, ['view_007'])
        for content in ['```json\n{}\n```', '{"title":"a","title":"b"}', 'not JSON']:
            with self.assertRaises(PlanValidationError):
                parse_plan(content, SCENE, ['view_007'])

    def test_duplicate_shots_rejected(self):
        plan = valid_plan()
        plan['shots'].append(copy.deepcopy(plan['shots'][0]))
        with self.assertRaises(PlanValidationError):
            parse_plan(json.dumps(plan), SCENE, ['view_007'])


class GenerationTests(unittest.TestCase):
    def context(self):
        return SCENE, {'representative_views': [{'view_id': 'view_007'}]}, ['base64-image'], []

    def test_repair_once_and_save_only_valid_plan(self):
        class Client:
            def __init__(self): self.calls = []
            def check_model(self): return {'model': 'mock'}
            def chat(self, messages, *, schema=None):
                assert schema['properties']['shots']['items']['properties']['human_instructions']['items']['properties']['subject_id']['enum'] == ['body:person']
                self.calls.append(copy.deepcopy(messages))
                return '{}' if len(self.calls) == 1 else json.dumps(valid_plan())
        client = Client()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'scene.json'; source.write_text('{}')
            out = Path(tmp)/'plan'
            with patch('robopod.gemma.prepare_context', return_value=self.context()):
                plan = generate_plan(tmp, source, 'review', out, client=client)
            self.assertEqual(plan, valid_plan())
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0][1]['images'], ['base64-image'])
            self.assertIn('Validation error', client.calls[1][-1]['content'])
            self.assertTrue((out/'shot_plan.json').is_file())
            self.assertEqual(json.loads((out/'planning_report.json').read_text())['status'], 'validated')

    def test_two_invalid_responses_fail_without_plan(self):
        class Client:
            def check_model(self): return {}
            def chat(self, messages, *, schema=None): return '{}'
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'scene.json'; source.write_text('{}')
            out = Path(tmp)/'plan'
            with patch('robopod.gemma.prepare_context', return_value=self.context()), self.assertRaises(PlannerError):
                generate_plan(tmp, source, 'review', out, client=Client())
            self.assertFalse((out/'shot_plan.json').exists())
            self.assertEqual(json.loads((out/'planning_report.json').read_text())['status'], 'failed')

    def test_transport_contract(self):
        client = OllamaClient()
        with patch.object(client, '_request', return_value={'done': True, 'message': {'content': '{}'}}) as request:
            self.assertEqual(client.chat([{'role': 'user', 'content': 'test', 'images': ['abc']}], schema=scene_plan_schema(SCENE, ['view_007'])), '{}')
            endpoint, body = request.call_args.args
            self.assertEqual(endpoint, '/api/chat')
            self.assertEqual(body['model'], 'gemma4:e2b')
            self.assertFalse(body['stream'])
            self.assertEqual(body['format'], scene_plan_schema(SCENE, ['view_007']))
            self.assertEqual(body['messages'][0]['images'], ['abc'])
        for response in [{'done': False}, {'done': True, 'done_reason': 'length'}, {'done': True, 'message': {}}]:
            with patch.object(client, '_request', return_value=response), self.assertRaises(PlannerError):
                client.chat([])
        with patch.object(client, '_request', return_value={'models': []}), self.assertRaises(PlannerError):
            client.check_model()

    def test_unavailable_server_failure_report(self):
        class Client:
            def check_model(self): raise PlannerError('Ollama unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'scene.json'; source.write_text('{}')
            out = Path(tmp)/'plan'
            with patch('robopod.gemma.prepare_context', return_value=self.context()), self.assertRaises(PlannerError):
                generate_plan(tmp, source, 'review', out, client=Client())
            self.assertFalse((out/'shot_plan.json').exists())
            self.assertIn('unavailable', json.loads((out/'planning_report.json').read_text())['error'])


class SceneSchemaTests(unittest.TestCase):
    def test_scene_ids_constrain_generation_without_mutating_base(self):
        schema = scene_plan_schema(SCENE, ['view_007'])
        shot = schema['properties']['shots']['items']['properties']
        self.assertEqual(shot['human_instructions']['items']['properties']['subject_id']['enum'], ['body:person'])
        self.assertEqual(shot['targets']['items']['properties']['object_id']['enum'], ['body:person', 'body:product'])
        self.assertEqual(shot['preferred_view_id']['enum'], ['view_007'])
        base = SHOT_PLAN_SCHEMA['properties']['shots']['items']['properties']
        self.assertNotIn('enum', base['human_instructions']['items']['properties']['subject_id'])
        no_people = scene_plan_schema({'objects': {'product': {'category': 'instrument'}}}, ['v'])
        self.assertEqual(no_people['properties']['shots']['items']['properties']['human_instructions']['maxItems'], 0)

    def test_product_instruction_error_is_actionable(self):
        plan = valid_plan()
        plan['shots'][0]['human_instructions'][0]['subject_id'] = 'body:product'
        with self.assertRaises(PlanValidationError) as error:
            parse_plan(json.dumps(plan), SCENE, ['view_007'])
        self.assertIn('human_instructions[0].subject_id', str(error.exception))
        self.assertIn('body:product', str(error.exception))
        self.assertIn("Allowed person IDs: ['body:person']", str(error.exception))
