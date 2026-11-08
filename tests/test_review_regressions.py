"""Independent regressions for the seven repo-review findings."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robopod.execution import film_shot
from robopod.optimization import evaluate, plan_sweep, motion_matches
from robopod.simulation import Observation
from robopod.gemma import generate_plan, SYSTEM_PROMPT
from test_planning import valid_plan, SCENE


class ReviewRegressions(unittest.TestCase):
    def test_score_never_changes_primary_or_secondary_objectives(self):
        points = np.array([[-.4,-.8,2], [.4,.8,2], [-.4,-.8,4], [.4,.8,4]])
        obs = Observation(np.zeros((100,100,3), np.uint8), np.ones((100,100)),
                          np.zeros((100,100,2), int), np.array([[100,0,50],[0,100,50],[0,0,1]]),
                          np.eye(4), 0, {'dolly': 1.3, 'roll': 0})
        shot = valid_plan()['shots'][0]
        shot['targets'][0].update(frame_height_fraction=.5)
        shot['targets'].append(dict(object_id='body:product', center_x=.2, center_y=.5, frame_height_fraction=.6))
        surfaces = {'body:person': points, 'body:product': points*.2 + [0,0,2]}
        original = copy.deepcopy(shot)
        result = evaluate(shot, SCENE, surfaces, obs)
        for target in shot['targets']:
            metric = result['targets'][target['object_id']]
            self.assertAlmostEqual(metric['scale_error'], abs(metric['framing']['frame_height_fraction']-target['frame_height_fraction']))
            self.assertAlmostEqual(metric['center_error'], np.hypot(metric['framing']['center_x']-target['center_x'], metric['framing']['center_y']-target['center_y']))
        self.assertAlmostEqual(result['targets']['body:person']['scale_error'], .3)
        self.assertFalse(result['satisfies_observed_criteria'])
        self.assertEqual(shot, original)
        # Description text cannot opt out of the horizon contract.
        shot['description'] = 'No Dutch angle'
        obs.joints['roll'] = .2
        self.assertIn('horizon: roll exceeds level tolerance', evaluate(shot, SCENE, surfaces, obs)['failures'])
        shot['horizon'] = 'dutch'
        self.assertNotIn('horizon: roll exceeds level tolerance', evaluate(shot, SCENE, surfaces, obs)['failures'])

    def fake_sim(self):
        return SimpleNamespace(axes=['roll'], limits={'roll':(-.35,.35)}, data=SimpleNamespace(time=0., ctrl=[0.]),
                               model=SimpleNamespace(actuator=lambda n: SimpleNamespace(id=0)),
                               capture=lambda: SimpleNamespace(joints={'roll':0}))

    def test_static_path_does_not_search_or_capture(self):
        sim = self.fake_sim()
        sim.capture = Mock(side_effect=AssertionError('Static shots need no sweep search'))
        for shot in ({}, {'motion':'static', 'description':'camera movement in old prose'}):
            sweep = plan_sweep(sim, shot, {}, {})
            self.assertFalse(sweep['continuous'])
            self.assertEqual(sweep['start'], sweep['end'])
        sim.capture.assert_not_called()

    def monitor(self, passes, *, budget=2, duration=1., motion='static'):
        sim = self.fake_sim()
        sim.recovered = False
        perception = SimpleNamespace(detect=lambda obs: [], registry={})
        shot = dict(shot_id='s', motion=motion, duration_seconds=duration, completion={'hold_seconds':.3})
        def step(sim, start, end, elapsed, dt, duration, trace):
            sim.data.time = round(sim.data.time+dt, 6)
        def score(*args):
            passed = passes(sim)
            return dict(satisfies_observed_criteria=passed, failures=[] if passed else ['target: position'])
        def recover(*args, **kwargs):
            sim.recovered = True
            sim.data.time += 10  # Recovery must not count as recorded time or satisfied hold.
        sweep = dict(start={'roll':0.}, end={'roll':.01}, continuous=True)
        with patch('robopod.execution.plan_sweep', return_value=sweep), \
             patch('robopod.execution.step_sweep', side_effect=step), \
             patch('robopod.execution.evaluate', side_effect=score), \
             patch('robopod.execution.optimize', return_value={'target':{'roll':0}}) as optimize, \
             patch('robopod.execution.move_to', side_effect=recover):
            result = film_shot(sim, None, shot, {}, {}, perception, max_recoveries=budget)
        return result, optimize.call_count

    def test_final_drift_cannot_pass_on_an_earlier_hold(self):
        result, calls = self.monitor(lambda sim: sim.data.time < .9, budget=0)
        self.assertGreaterEqual(result['longest_satisfied_seconds'], .3)
        self.assertFalse(result['completed'])
        self.assertTrue(result['model_replan_needed'])
        self.assertEqual(calls,0)

    def test_drift_recovers_without_counting_recovery_as_footage(self):
        result, calls = self.monitor(lambda sim: sim.recovered)
        self.assertEqual(calls,1)
        self.assertTrue(result['completed'])
        self.assertAlmostEqual(result['samples'][-1]['recorded_seconds'],1.)
        self.assertAlmostEqual(result['terminal_satisfied_seconds'],.7)

    def test_failed_recovery_stops_at_budget(self):
        result, calls = self.monitor(lambda sim: False, duration=5.)
        self.assertEqual(calls,2)
        self.assertEqual(result['stop_reason'],'recovery_budget_exhausted')
        self.assertFalse(result['completed'])

    def test_unsupported_human_actions_remain_pending(self):
        from robopod.refinement import refine_shot
        sim = self.fake_sim()
        obs = sim.capture()
        perception = SimpleNamespace(detect=lambda obs: [], registry={})
        shot = valid_plan()['shots'][0]
        instructions = ['Do a backflip.', 'Juggle the product.', 'Sing a song.']
        shot['human_instructions'] = [dict(subject_id='body:person', instruction=t) for t in instructions]
        with TemporaryDirectory() as tmp, patch('robopod.refinement.SceneMap'), \
             patch('robopod.refinement.evaluate', return_value={'satisfies_observed_criteria':True}):
            result = refine_shot(sim, None, shot, SCENE, {}, perception, Path(tmp), max_adjustments=0, progress=None)
        self.assertEqual([a['instruction'] for a in result['pending_creative']], instructions)
        self.assertTrue(result['model_replan_needed'])

    def test_outline_preserves_four_shots_and_nondefault_subjects(self):
        scene = copy.deepcopy(SCENE)
        scene['objects'].update({'body:second_product': {'category':'instrument'}, 'body:plant': {'category':'plant'}})
        outline = dict(title='Different sequence', shots=[
            dict(shot_id=f's{i}', shot_type=kind, description=desc, target_ids=ids, motion='static', horizon='level', motion_parameters=dict(amplitude_m=0.,subject_lock='none'))
            for i,(kind,desc,ids) in enumerate([
                ('wide','Presenter with second product',['body:person','body:second_product']),
                ('detail','Second product controls',['body:second_product']),
                ('close_up','Plant',['body:plant']), ('medium','Return to presenter',['body:person'])])])
        class Client:
            def check_model(self): return {}
            def chat(self, messages, *, schema):
                fields = schema['properties']['shots']['items']['properties']
                if 'target_ids' in fields:
                    return json.dumps(outline)
                s = next(s for s in outline['shots'] if s['shot_id']==fields['shot_id']['enum'][0])
                plan = valid_plan()
                shot = plan['shots'][0]
                for name in ('shot_id','shot_type','description','motion','horizon','motion_parameters'):
                    shot[name] = s[name]
                shot['targets'] = [dict(object_id=k,center_x=.5,center_y=.5,frame_height_fraction=.4) for k in s['target_ids']]
                return json.dumps(plan)
        brief = 'Show the presenter and second product, then a detail shot of it, then the plant, then return to the presenter.'
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'map.json'; source.write_text('{}')
            with patch('robopod.gemma.prepare_context',return_value=(scene,{'representative_views':[{'view_id':'view_007'}]},[],[])):
                plan = generate_plan(tmp,source,brief,Path(tmp)/'plan',client=Client())
        self.assertEqual([s['shot_type'] for s in plan['shots']], ['wide','detail','close_up','medium'])
        self.assertEqual([[t['object_id'] for t in s['targets']] for s in plan['shots']], [s['target_ids'] for s in outline['shots']])
        self.assertNotIn('fixed rail cannot',SYSTEM_PROMPT)

    def test_requested_motion_direction_is_enforced(self):
        start = dict(truck=0., dolly=0., height=1.)
        end = dict(truck=.1, dolly=.2, height=.9)
        for motion in ('truck_right','push_in','pedestal_down','sweep'):
            self.assertTrue(motion_matches(start,end,motion))
        for motion in ('truck_left','pull_back','pedestal_up','static'):
            self.assertFalse(motion_matches(start,end,motion))
        self.assertFalse(motion_matches(start,start,'push_in'))

    def test_mixed_static_and_moving_sequence_is_allowed(self):
        from robopod.brief import requirements
        self.assertTrue(requirements('First a static shot, then dolly toward the plant')['movement_requested'])
        self.assertFalse(requirements('No camera movement')['movement_requested'])
        self.assertTrue(requirements('Another Dutch angle')['dutch_requested'])

    def test_recovered_sweep_does_not_claim_requested_motion_completed(self):
        result, calls = self.monitor(lambda sim: sim.recovered, motion='sweep')
        self.assertEqual(calls,1)
        self.assertFalse(result['motion_completed'])
        self.assertFalse(result['completed'])
