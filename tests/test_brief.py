import copy
import json
import unittest
from robopod.brief import requirements, validate_brief
from robopod.planning import PlanValidationError
from test_planning import valid_plan, SCENE

BRIEF='Show me and the synthesizer, then a close-up of its controls. Then zoom out to the presenter, follow the rule of thirds and add movement.'

class BriefTests(unittest.TestCase):
    def test_explicit_sequence_contract(self):
        self.assertEqual(requirements(BRIEF)['minimum_shots'],3)
        self.assertEqual(requirements('Start up close to the synthesizer then move out to the reviewer')['minimum_shots'],2)
        user_brief = (
            'Film a professional tech review: start as close as the robot can get to the synthesizer '
            'then move up and then move back make sure we have room of thirds once pulled back with the '
            'reviewer and synth in view move the camera around to keep the viewer interested while keeping '
            'the rule of thirds'
        )
        req = requirements(user_brief)
        self.assertEqual(req['minimum_shots'], 2)
        self.assertTrue(req['starts_close'])
        self.assertTrue(req['close_up_required'])
        self.assertTrue(req['return_to_presenter'])
        self.assertTrue(req['rule_of_thirds'])
        self.assertTrue(req['movement_requested'])
        self.assertEqual(requirements('A single medium shot')['minimum_shots'],1)
        with self.assertRaisesRegex(PlanValidationError,'at least 3'):
            validate_brief(valid_plan(),BRIEF,SCENE)

    def test_sequence_and_thirds_required(self):
        plan=valid_plan()
        plan['shots']=[copy.deepcopy(plan['shots'][0]) for _ in range(3)]
        for i,s in enumerate(plan['shots']):
            s['shot_id']=str(i)
            s['targets'][0]['center_x']=1/3
        plan['shots'][1]['shot_type']='close_up'
        plan['shots'][1]['targets'][0]['object_id']='body:product'
        validate_brief(plan,BRIEF,SCENE)
        plan['shots'][0]['targets'][0]['center_x']=.5
        with self.assertRaisesRegex(PlanValidationError,'rule of thirds'):
            validate_brief(plan,BRIEF,SCENE)

    def test_bottom_edge_is_not_valid_framing(self):
        plan=valid_plan()
        plan['shots'][0]['targets'][0]['center_y']=1
        with self.assertRaisesRegex(PlanValidationError,'beyond'):
            validate_brief(plan,'Medium shot',SCENE)

    def test_staged_generation_assembles_requested_shots(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from robopod.gemma import generate_plan
        class Client:
            def __init__(self): self.calls=0
            def check_model(self): return {'model':'test-only'}
            def chat(self,messages,*,schema):
                self.calls+=1
                props=schema['properties']['shots']['items']['properties']
                if 'target_ids' in props:
                    return json.dumps({'title': 'Review', 'shots': [
                        dict(shot_id=f'shot_{i+1}', shot_type=kind, description='Requested shot',
                             target_ids=ids, motion='static', horizon='level', motion_parameters=dict(amplitude_m=0.,subject_lock='none'))
                        for i,(kind,ids) in enumerate([
                            ('medium',['body:person','body:product']),
                            ('close_up',['body:product']), ('medium',['body:person'])])]})
                self_schema=schema['properties']['shots']
                assert self_schema['minItems']==self_schema['maxItems']==1
                p=valid_plan();s=p['shots'][0]
                s['shot_id']=props['shot_id']['enum'][0]
                s['shot_type']=props['shot_type']['enum'][0]
                s['targets']=[dict(object_id=k,center_x=.333333,center_y=.5,frame_height_fraction=.3)
                              for k in props['targets']['items']['properties']['object_id']['enum']]
                p['shots'][0]['motion_parameters'].pop('direction')
                return json.dumps(p)
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'map.json';source.write_text('{}')
            context=(SCENE,{'representative_views':[{'view_id':'view_007'}]},['image'],[])
            client=Client()
            with patch('robopod.gemma.prepare_context',return_value=context):
                plan=generate_plan(tmp,source,BRIEF,Path(tmp)/'plan',client=client)
            self.assertEqual(client.calls,4)
            self.assertEqual([s['shot_type'] for s in plan['shots']],['medium','close_up','medium'])
            self.assertEqual(len({s['shot_id'] for s in plan['shots']}),3)

    def test_static_and_negative_dutch_contract(self):
        plan = valid_plan()
        plan['shots'][0]['motion'] = 'sweep'
        for brief in ['Locked-off shot, no camera movement.', 'A professional tech review']:
            with self.assertRaisesRegex(PlanValidationError, 'static'):
                validate_brief(plan, brief, SCENE)
        plan['shots'][0]['motion'] = 'static'
        plan['shots'][0]['horizon'] = 'dutch'
        with self.assertRaisesRegex(PlanValidationError, 'level'):
            validate_brief(plan, 'No Dutch angle, please.', SCENE)
        validate_brief(plan, 'Use a Dutch angle.', SCENE)

    def test_explicit_shot_count_is_preserved(self):
        self.assertEqual(requirements('One medium shot of the presenter')['exact_shots'],1)
        self.assertEqual(requirements('One static close-up shot of the controls')['exact_shots'],1)
        self.assertTrue(requirements('One static close-up shot of the controls')['static_requested'])
        self.assertEqual(requirements('Four distinct shots of the room')['exact_shots'],4)
        with self.assertRaisesRegex(PlanValidationError,'exactly 4'):
            validate_brief(valid_plan(),'Four shots of the room',SCENE)


    def test_explicit_center_and_no_human_action_are_preserved(self):
        brief='One static close-up shot, centered. No human action is needed.'
        plan=valid_plan();s=plan['shots'][0]
        s['shot_type']='close_up'
        with self.assertRaisesRegex(PlanValidationError,'human actions'): validate_brief(plan,brief,SCENE)
        s['human_instructions']=[];s['targets'][0]['center_x']=.333
        with self.assertRaisesRegex(PlanValidationError,'centered'): validate_brief(plan,brief,SCENE)
        s['targets'][0]['center_x']=.5
        validate_brief(plan,brief,SCENE)
