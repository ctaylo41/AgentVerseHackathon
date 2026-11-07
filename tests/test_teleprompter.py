import unittest
import numpy as np

from robopod.simulation import Simulation
from robopod.teleprompter import Instruction, Teleprompter, creative_action, observe_response, respond
from robopod.perception import Detection, SemanticIdentity


class TeleprompterTests(unittest.TestCase):
    def test_single_instruction_cooldown_and_no_oscillation(self):
        prompt = Teleprompter()
        action = Instruction('geometric','body:product','Move closer',1,-.5)
        self.assertTrue(prompt.issue(action,0))
        self.assertFalse(prompt.issue(action,0))
        prompt.finish(1,True,{'observed':True})
        self.assertFalse(prompt.issue(action,1.5))
        self.assertTrue(prompt.issue(action,2))
        prompt.finish(3,False,{})
        self.assertFalse(prompt.issue(action,5))
        reverse = Instruction('geometric','body:product','Move farther',1,.5)
        self.assertFalse(prompt.issue(reverse,5))

    def test_scripted_motion_bounded_and_camera_not_teleported(self):
        with Simulation() as sim:
            action=Instruction('geometric','body:product','Move closer',1,-.5)
            camera=sim.configuration.copy()
            start=sim.data.mocap_pos[int(sim.model.body('product').mocapid[0])].copy()
            result=respond(sim,action)
            end=sim.data.mocap_pos[int(sim.model.body('product').mocapid[0])]
            np.testing.assert_allclose(end-start,[0,-.5,0])
            self.assertTrue(result['scripted'])
            self.assertAlmostEqual(sim.configuration['pan'],camera['pan'])
            for _ in range(5): respond(sim,action)
            self.assertGreaterEqual(sim.data.mocap_pos[int(sim.model.body('product').mocapid[0]),1],-.9)
            sim.reset()
            np.testing.assert_allclose(sim.data.mocap_pos[int(sim.model.body('product').mocapid[0])],start)

    def test_observation_required_not_script_ack(self):
        action=Instruction('geometric','body:product','Move closer',1,-.5)
        identity=SemanticIdentity('body:product','instrument','Synth',True)
        def detection(y):
            return Detection(identity,[0,0,2,2],np.ones((2,2),bool),np.array([[0,y,1],[.1,y,1]]),1)
        self.assertTrue(observe_response([detection(1)],[detection(.5)],action)[0])
        self.assertFalse(observe_response([detection(1)],[detection(1)],action)[0])
        self.assertFalse(observe_response([detection(1)],[],action)[0])
        self.assertFalse(observe_response([detection(1)],[detection(1.5)],action)[0])

    def test_creative_vocabulary_is_explicit(self):
        self.assertIsNotNone(creative_action('body:person','Step slightly to your left.'))
        self.assertEqual(creative_action('body:person','Move the product higher.').object_id,'body:product')
        self.assertIsNone(creative_action('body:person','Smile and point to the controls.'))

    def test_refinement_stops_when_script_is_not_observed(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from robopod.refinement import refine_shot
        from robopod.perception import SegmentationPerception
        from robopod.simulation import Observation
        import mujoco
        obs=Observation(np.zeros((4,4,3),np.uint8),np.ones((4,4)),
                        np.full((4,4,2),[1,int(mujoco.mjtObj.mjOBJ_GEOM)]),np.eye(3),np.eye(4),0,{})
        perception=SegmentationPerception({'1':{'body':'product','name':'synthesizer'}})
        scene={'objects':{'body:product':{'category':'instrument','label':'Synth'}}}
        shot={'human_instructions':[], 'targets':[{'object_id':'body:product','frame_height_fraction':.5}],
              'completion':{'max_scale_error':.05}}
        metrics={'loss':1.,'satisfies_observed_criteria':False,
                 'targets':{'body:product':{'scale_error':.4,'framing':{'frame_height_fraction':.1}}}}
        with Simulation() as sim, tempfile.TemporaryDirectory() as tmp:
            with patch.object(sim,'capture',return_value=obs), patch.object(sim,'save'), \
                 patch('robopod.refinement.evaluate',return_value=metrics), \
                 patch('robopod.refinement.respond',return_value={'scripted':True}), \
                 patch('robopod.refinement.optimize') as optimizer:
                result=refine_shot(sim,None,shot,scene,{'body:product':np.array([[0,0,1.]])},
                                   perception,Path(tmp),max_adjustments=3,progress=None)
            self.assertEqual(result['stop_reason'],'response_not_observed')
            self.assertEqual(len(result['rounds']),1)
            self.assertEqual(result['instructions'][0]['status'],'not_completed')
            optimizer.assert_not_called()
