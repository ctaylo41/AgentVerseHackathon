import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from robopod.exploration import SurfaceCoverage, explore, move_and_settle, survey_poses
from robopod.simulation import Observation, Simulation


def observation():
    return Observation(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)),
                       np.full((2, 2, 2), [4, int(mujoco.mjtObj.mjOBJ_GEOM)]),
                       np.eye(3), np.eye(4), 0.0, {})


class ExplorationTests(unittest.TestCase):
    def test_pattern_and_physical_reachability(self):
        with Simulation() as sim:
            poses = survey_poses(sim.limits)
            self.assertEqual(poses, survey_poses(sim.limits))
            self.assertEqual(len({tuple(p.values()) for p in poses}), 18)
            for target in poses:
                for name, value in target.items():
                    lo, hi = sim.limits[name]
                    self.assertTrue(lo <= value <= hi)
                result = move_and_settle(sim, target)
                self.assertTrue(result.settled, (target, result))
            self.assertGreater(sim.data.time, 1)

    def test_surface_projection_and_redundancy(self):
        coverage = SurfaceCoverage(voxel_size=0.1, stride=1)
        obs = observation()
        cells, ids = coverage.measure(obs)
        self.assertEqual(cells, {(0, 0, 10), (10, 0, 10), (0, 10, 10), (10, 10, 10)})
        coverage.add(cells, ids)
        self.assertFalse(coverage.measure(obs)[0] - coverage.cells)
        obs.world_from_camera[0, 3] = 2
        self.assertEqual(len(coverage.measure(obs)[0] - coverage.cells), 4)
        obs.segmentation[:] = -1
        self.assertEqual(coverage.measure(obs), (set(), set()))

    def test_invalid_depth_excluded(self):
        obs = observation()
        obs.depth_m[:] = [[np.nan, np.inf], [-1, 0]]
        self.assertEqual(SurfaceCoverage(stride=1).measure(obs), (set(), set()))

    def test_timeout_and_overwrite_protection(self):
        with Simulation() as sim, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                explore(sim, tmp)
            with patch('robopod.exploration.survey_poses', return_value=[dict(height=2.0, pan=0.5, tilt=0.2, roll=0.0)]):
                result = explore(sim, Path(tmp)/'survey', settle_timeout=0.02)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['views'][0]['status'], 'unsettled')
            self.assertEqual(result['saved_views'], 0)
            self.assertTrue((Path(tmp)/'survey/manifest.json').is_file())

    def test_save_uses_the_evaluated_observation(self):
        with Simulation() as sim, tempfile.TemporaryDirectory() as tmp:
            with patch.object(sim, 'capture', side_effect=AssertionError('Must not recapture')):
                sim.save(tmp, observation=observation())
            self.assertTrue((Path(tmp)/'geometry.npz').is_file())
