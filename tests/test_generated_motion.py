import copy
import json
import unittest
from robopod.gemma import decode_generated
from robopod.planning import obj, array, text, number, PlanValidationError, parse_plan
from robopod.motion import validate
from test_planning import valid_plan, SCENE

class GeneratedMotionTests(unittest.TestCase):
    def test_every_motion_compiles_its_only_valid_direction(self):
        expected={'static':'none','sweep':'auto','push_in':'forward','pull_back':'backward',
                  'truck_left':'left','truck_right':'right','pedestal_up':'up','pedestal_down':'down'}
        schema=obj({'shots':array(obj({'motion':dict(type='string',enum=list(expected)),
                    'motion_parameters':obj({'amplitude_m':number(0,1),'subject_lock':text()})}),1,8)})
        raw={'shots':[{'motion':mode,'motion_parameters':{'amplitude_m':0 if mode=='static' else .2,'subject_lock':'none'}} for mode in expected]}
        compiled=decode_generated(json.dumps(raw),schema)
        for shot in compiled['shots']:
            self.assertEqual(shot['motion_parameters']['direction'],expected[shot['motion']])
            shot['targets']=[{'object_id':'body:person'}]
            validate(shot)
        self.assertNotIn('direction',raw['shots'][0]['motion_parameters'])
        self.assertNotIn('direction',schema['properties']['shots']['items']['properties']['motion_parameters']['properties'])
        # Invalid model fields and duplicate fields are rejected before compilation.
        bad=copy.deepcopy(raw);bad['shots'][1]['motion_parameters']['direction']='up'
        with self.assertRaises(PlanValidationError): decode_generated(json.dumps(bad),schema)
        with self.assertRaises(PlanValidationError): decode_generated('{"shots":[],"shots":[]}',schema)

    def test_imported_contradictory_direction_stays_invalid(self):
        plan=valid_plan();shot=plan['shots'][0]
        shot['motion']='sweep'
        shot['motion_parameters'].update(amplitude_m=.3,direction='up')
        with self.assertRaisesRegex(PlanValidationError,'requires direction auto'):
            parse_plan(json.dumps(plan),SCENE,['view_007'])

    def test_sequence_generation_compiles_direction_and_excludes_target_backgrounds(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from robopod.gemma import generate_plan
        modes=['push_in','sweep','pull_back']
        class Client:
            def check_model(self): return {}
            def chat(self,messages,*,schema):
                fields=schema['properties']['shots']['items']['properties']
                assert 'direction' not in fields['motion_parameters']['properties']
                if 'target_ids' in fields:
                    return json.dumps(dict(title='Moves',shots=[dict(shot_id=f's{i}',shot_type='medium',description='Camera move',
                        target_ids=['body:person'],motion=mode,horizon='level',
                        motion_parameters=dict(amplitude_m=.15,subject_lock='body:person')) for i,mode in enumerate(modes)]))
                assert 'body:person' not in fields['background_object_ids']['items']['enum']
                plan=valid_plan();s=plan['shots'][0]
                s['shot_id']=fields['shot_id']['enum'][0]
                s['motion']=fields['motion']['enum'][0]
                s['motion_parameters']=dict(amplitude_m=.15,subject_lock='body:person')
                return json.dumps(plan)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);scene=root/'scene.json';scene.write_text('{}')
            with patch('robopod.gemma.prepare_context',return_value=(SCENE,{'representative_views':[{'view_id':'view_007'}]},[],[])):
                result=generate_plan(root,scene,'Three shots with camera movement.',root/'plan',client=Client())
            self.assertEqual([s['motion'] for s in result['shots']],modes)
            self.assertEqual([s['motion_parameters']['direction'] for s in result['shots']],['forward','auto','backward'])
            parse_plan(json.dumps(result),SCENE,['view_007'])
