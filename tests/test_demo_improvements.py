import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

from robopod.app import AppState, Studio
from robopod.artifacts import identity, require_compatible
from robopod.motion import validate
from robopod.planning import PlanValidationError, parse_plan
from robopod.recording import Recorder, Stopped
from robopod.safety import PathBlocked
from robopod.simulation import Simulation
from test_planning import valid_plan, SCENE


class DemoImprovements(unittest.TestCase):
    def test_motion_contract_rejects_wrong_direction_lock_and_speed(self):
        shot = valid_plan()['shots'][0]
        shot.update(motion='truck_left', duration_seconds=3., motion_parameters=dict(
            amplitude_m=.2, direction='left', subject_lock='body:person'))
        with Simulation() as sim:
            validate(shot,sim.limits)
            for field,value in [('direction','right'),('subject_lock','body:product'),('amplitude_m',0)]:
                bad=copy.deepcopy(shot);bad['motion_parameters'][field]=value
                with self.assertRaises(PlanValidationError): validate(bad,sim.limits)
            shot['duration_seconds']=1
            shot['motion_parameters']['amplitude_m']=1
            with self.assertRaisesRegex(PlanValidationError,'speed'): validate(shot,sim.limits)
            shot['motion']='sweep';shot['motion_parameters']['direction']='auto'
            with self.assertRaisesRegex(PlanValidationError,'speed'): validate(shot,sim.limits)
        plan=valid_plan();plan['shots'][0]['motion_parameters']['unexpected']=1
        with self.assertRaises(PlanValidationError): parse_plan(json.dumps(plan),SCENE,['view_007'])

    def test_swept_volume_rejects_midpath_obstacle_despite_clear_endpoints(self):
        with Simulation() as sim:
            product=int(sim.model.body('product').mocapid[0])
            sim.data.mocap_pos[product]=[0,-1.6,1.4]
            mujoco.mj_forward(sim.model,sim.data)
            start=dict(height=1.4,truck=-1.,dolly=.8,pan=0.,tilt=0.,roll=0.)
            end=dict(start,truck=1.)
            original=sim.data.qpos.copy()
            sim.path_validator.validate(start,start)
            sim.path_validator.validate(end,end)
            with self.assertRaises(PathBlocked): sim.path_validator.validate(start,end)
            np.testing.assert_array_equal(sim.data.qpos,original)
            self.assertTrue(all(sim.model.geom_contype[g]==2 for g in sim.path_validator.robot))

    def test_pause_resume_and_stop_preserve_plan(self):
        state=AppState();studio=Studio(state,'unused')
        plan=valid_plan();state.update(plan=plan,busy=True,phase='execute')
        state.submit('pause')
        done=threading.Event()
        thread=threading.Thread(target=lambda:(studio.check_cancel(),done.set()))
        thread.start()
        self.assertFalse(done.wait(.1))
        state.submit('resume');thread.join(1)
        self.assertTrue(done.is_set())
        state.submit('pause');state.submit('stop')
        with self.assertRaises(Stopped): studio.check_cancel()
        self.assertEqual(state.snapshot()['plan'],plan)
        self.assertFalse(state.cancel.is_set())
        self.assertTrue(state.commands.empty())

    def test_each_shot_publishes_before_sequence_finishes(self):
        state=AppState();studio=Studio(state,'unused')
        studio.execution_event(dict(kind='shot_started',shot_id='second',index=1,total=3))
        self.assertEqual(state.snapshot()['active_shot'],'second')
        result=dict(shot_id='second',comparison='/tmp/comparison.jpg',video='/tmp/shot.mp4',status='needs_adjustment')
        studio.execution_event(dict(kind='shot_completed',result=result))
        snapshot=state.snapshot()
        self.assertEqual(len(snapshot['results']),1)
        self.assertTrue(snapshot['results'][0]['video'].startswith('/media/'))
        self.assertIsNone(snapshot['sequence_video'])
        self.assertEqual(state.media['shot_0_video.mp4'],Path('/tmp/shot.mp4'))

    def test_video_exports_clean_rgb_at_expected_frame_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shot.mp4'
            recorder=Recorder(path,64,48)
            rgb=np.zeros((48,64,3),np.uint8);rgb[:,:,0]=220
            for _ in range(10): recorder.write(SimpleNamespace(rgb=rgb))
            recorder.close()
            video=cv2.VideoCapture(str(path))
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),10)
            self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS),10.)
            ok,frame=video.read();video.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape,(48,64,3))
            self.assertGreater(frame[:,:,2].mean(),200)
            self.assertLess(frame[:,:,1].mean(),10)

    def test_old_or_changed_artifacts_are_rejected(self):
        require_compatible(identity())
        for bad in (None,{},dict(identity(),scene_sha256='old'),dict(identity(),rig='4dof')):
            with self.assertRaisesRegex(ValueError,'Explore'): require_compatible(bad)

    def test_live_scene_perturbations_interpolate_during_active_execution(self):
        state = AppState()
        studio = Studio(state, 'unused')
        with self.assertRaisesRegex(ValueError, 'active execution'):
            state.submit('perturb_person_left')
        state.update(busy=True, phase='execute')
        with Simulation() as sim:
            person = int(sim.model.body('person').mocapid[0])
            product = int(sim.model.body('product').mocapid[0])
            start_person = sim.data.mocap_pos[person].copy()
            start_product = sim.data.mocap_pos[product].copy()
            studio.last_frame = 1e12  # Avoid rendering overhead in unit test
            state.submit('perturb_person_left')
            self.assertTrue(state.commands.empty())
            self.assertIn('nudging presenter left', state.snapshot()['status'])
            sim.step(0.25)
            studio.tick(sim)
            mid_x = float(sim.data.mocap_pos[person, 0])
            self.assertGreater(mid_x, float(start_person[0]) + 0.05)
            self.assertLess(mid_x, float(start_person[0]) + 0.35)
            sim.step(0.25)
            studio.tick(sim)
            self.assertAlmostEqual(float(sim.data.mocap_pos[person, 0]), float(start_person[0]) + 0.35, places=3)
            self.assertIsNone(studio.active_perturbation)
            state.submit('perturb_product')
            sim.step(0.01)
            studio.tick(sim)
            sim.step(0.5)
            studio.tick(sim)
            self.assertAlmostEqual(float(sim.data.mocap_pos[product, 0]), float(start_product[0]) + 0.30, places=3)
            state.submit('perturb_occlusion')
            sim.step(0.01)
            studio.tick(sim)
            sim.step(0.5)
            studio.tick(sim)
            self.assertAlmostEqual(float(sim.data.mocap_pos[person, 0]), float(sim.data.mocap_pos[product, 0]), places=3)
            self.assertAlmostEqual(float(sim.data.mocap_pos[product, 2]), 1.22, places=3)

    def test_live_ollama_replan_fallback_and_look_away_cooperation(self):
        from robopod.actors import perform
        from robopod.gemma import OllamaClient, PlannerError
        from robopod.replanning import revise_plan
        from robopod.teleprompter import creative_action
        self.assertEqual(creative_action('body:person', 'look away from the camera').action, 'look_away')
        gemma_line = 'Look away toward the side of the room while standing by the desk.'
        parsed_line = creative_action('body:person', gemma_line)
        self.assertEqual(parsed_line.action, 'look_away')
        self.assertEqual(parsed_line.text, gemma_line)
        self.assertEqual(creative_action('body:person', 'Smile warmly at the camera as you introduce the synthesizer.').action, 'look_and_smile')
        with Simulation() as sim:
            before = sim.data.geom('nose').xpos.copy()
            perform(sim, 'look_away')
            after = sim.data.geom('nose').xpos.copy()
            self.assertGreater(float(np.linalg.norm(after - before)), 0.015)

        class StubLiveClient(OllamaClient):
            def check_model(self):
                return {'model': 'gemma4:e2b'}
            def chat(self, messages, *, schema=None):
                raise PlannerError('Ollama response was incomplete or hit its token limit')

        scene = copy.deepcopy(SCENE)
        scene['viewpoints']['view_008'] = {'joints': {'yaw': 0.2, 'pitch': -0.1, 'height': 1.25, 'roll': 0.0}, 'simulation_time': 2.0}
        plan = valid_plan()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            revised = revise_plan(plan, scene, [{'shot_id': 'shot_1'}], image, 'Film the presenter and product.', tmp, client=StubLiveClient())
            self.assertEqual(revised['shots'][0]['preferred_view_id'], 'view_008')
            self.assertEqual(revised['shots'][0]['completion'], plan['shots'][0]['completion'])
