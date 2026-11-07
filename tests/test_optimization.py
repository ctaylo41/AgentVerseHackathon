import copy
import unittest

import numpy as np

from robopod.control import MotionError, check_limits, move_to
from robopod.execution import sample_surface
from robopod.optimization import evaluate, safe_limits
from robopod.simulation import Simulation, Observation


def shot():
    return dict(targets=[dict(object_id='product', center_x=0.5, center_y=0.5, frame_height_fraction=0.2)],
                completion=dict(max_center_error=0.05, max_scale_error=0.05, min_visible_fraction=0.8,
                                headroom_fraction=0.1, headroom_tolerance=0.05),
                background_object_ids=[], perspective='neutral')


class OptimizationTests(unittest.TestCase):
    def test_projection_score_and_missing_subject(self):
        points = np.array([[-.2,-.2,2], [.2,-.2,2], [-.2,.2,2], [.2,.2,2]])
        obs = Observation(np.zeros((100,100,3),np.uint8), np.ones((100,100))*2,
                          np.zeros((100,100,2),int), np.array([[100,0,50],[0,100,50],[0,0,1]]), np.eye(4), 0, {})
        scene = {'objects': {'product': {'category': 'instrument'}}}
        score = evaluate(shot(), scene, {'product':points}, obs)
        self.assertTrue(score['satisfies_observed_criteria'])
        self.assertAlmostEqual(score['loss'],0)
        hidden = evaluate(shot(),scene,{'product':points},obs,[],{})
        self.assertFalse(hidden['satisfies_observed_criteria'])
        self.assertIn('product: visibility',hidden['failures'])
        obs.world_from_camera[0,3] = 10
        offscreen = evaluate(shot(),scene,{'product':points},obs)
        self.assertGreater(offscreen['loss'],score['loss'])

    def test_smooth_actuator_motion_limits_and_rate(self):
        with Simulation() as sim:
            before = sim.data.qpos.copy()
            with self.assertRaises(ValueError):
                move_to(sim, dict(height=10,pan=0,tilt=0,roll=0))
            np.testing.assert_array_equal(sim.data.qpos,before)
            trace=[]
            result=move_to(sim,dict(height=1.8,pan=.4,tilt=-.3,roll=.15),trace=trace)
            self.assertTrue(result['settled'])
            self.assertGreater(len(trace),20)
            for row in trace:
                for name,value in row['joints'].items():
                    self.assertTrue(sim.limits[name][0] <= value <= sim.limits[name][1])
            from robopod.control import RATES
            for a,b in zip(trace,trace[1:]):
                for name in sim.axes:
                    speed=abs(b['controls'][name]-a['controls'][name])/(b['time']-a['time'])
                    self.assertLessEqual(speed,RATES[name]+1e-6)
            bounds=safe_limits(sim)
            move_to(sim,{name:bounds[name][0] for name in sim.axes})
            move_to(sim,{name:bounds[name][1] for name in sim.axes})
            check_limits(sim)

    def test_sparse_reference_validation_preserves_extrema(self):
        points=np.random.default_rng(1).normal(size=(4096,3))
        sample=sample_surface(points)
        self.assertLessEqual(len(sample),256)
        np.testing.assert_array_equal(points.min(axis=0),sample.min(axis=0))
        np.testing.assert_array_equal(points.max(axis=0),sample.max(axis=0))
        with self.assertRaises(ValueError): sample_surface(np.array([[np.nan,0,0]]))
        with self.assertRaises(ValueError): sample_surface(np.empty((0,3)))
