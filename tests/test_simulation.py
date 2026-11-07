import unittest
import numpy as np
from robopod.simulation import Simulation


class SimulationTests(unittest.TestCase):
    def test_actuators_and_limits(self):
        with Simulation() as sim:
            self.assertEqual(sim.model.nq, 6)
            original = sim.data.ctrl.copy()
            for request in ({'height': 3}, {'truck': 5}, {'dolly': 5}, {'pan': float('nan')}, {'x': 0}):
                with self.assertRaises(ValueError):
                    sim.command(**request)
                np.testing.assert_array_equal(sim.data.ctrl, original)
            sim.command(height=1.8, truck=0.35, dolly=0.5, pan=0.3, tilt=-0.2, roll=0.1)
            self.assertAlmostEqual(sim.configuration['height'], 1.4)
            sim.step(4)
            for name, target in zip(sim.axes, [1.8, 0.35, 0.5, 0.3, -0.2, 0.1]):
                self.assertAlmostEqual(sim.configuration[name], target, delta=0.04)
            sim.reset()
            self.assertAlmostEqual(sim.configuration['height'], 1.4)
            self.assertAlmostEqual(sim.configuration['truck'], 0.0)
            self.assertAlmostEqual(sim.configuration['dolly'], 0.0)

    def test_camera_orientation(self):
        with Simulation() as sim:
            rotation = sim.data.cam_xmat[sim.camera_id].reshape(3, 3)
            np.testing.assert_allclose(-rotation[:, 2], [0, 1, 0], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
