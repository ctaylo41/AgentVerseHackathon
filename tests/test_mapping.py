import tempfile
import json
import cv2
from pathlib import Path
import unittest
from dataclasses import replace

import mujoco
import numpy as np

from robopod.mapping import SceneMap, visibility, map_survey
from robopod.perception import PerceptionPipeline, SegmentationPerception
from robopod.simulation import Observation

GEOMS = {'1': {'body': 'person', 'name': 'head'},
         '2': {'body': 'person', 'name': 'torso'},
         '3': {'body': 'product', 'name': 'keys'},
         '4': {'body': 'product', 'name': 'synthesizer'},
         '5': {'body': 'desk', 'name': 'desktop'},
         '6': {'body': 'world', 'name': 'rail'}}


def observation(time=0):
    seg = np.full((4, 4, 2), -1, dtype=np.int32)
    seg[:, :, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    seg[:, :, 0] = [[1, 1, 3, 4], [2, 2, 3, 4], [5, 5, 5, 5], [6, -1, -1, -1]]
    return Observation(np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 2.0), seg,
                       np.array([[2, 0, 1.5], [0, 2, 1.5], [0, 0, 1.0]]),
                       np.eye(4), time, {})


class MappingTests(unittest.TestCase):
    def test_grouping_mask_geometry_and_pose_provenance(self):
        obs = observation()
        detections = SegmentationPerception(GEOMS).detect(obs)
        self.assertEqual({d.identity.object_id for d in detections}, {'body:person', 'body:product', 'body:desk'})
        person = next(d for d in detections if d.identity.category == 'person')
        self.assertEqual(person.bbox_xyxy, [0, 0, 2, 2])
        self.assertEqual(person.mask.sum(), 4)
        np.testing.assert_allclose(person.points_world[:, 2], 2)
        self.assertEqual(person.landmarks['head']['source'], 'visible_surface_proxy')
        obs.depth_m[:] = np.nan
        self.assertEqual(SegmentationPerception(GEOMS).detect(obs), [])

    def test_persistence_dynamic_geometry_and_replay_guard(self):
        scene = SceneMap(max_points=3)
        pipeline = SegmentationPerception(GEOMS)
        obs = observation()
        scene.update(pipeline.detect(obs), obs, 'a')
        obs2 = replace(observation(1), world_from_camera=np.eye(4))
        obs2.world_from_camera[0, 3] = 10
        scene.update(pipeline.detect(obs2), obs2, 'b')
        self.assertEqual(len(scene.objects), 3)
        person = scene.objects['body:person']
        self.assertEqual(person['observation_count'], 2)
        self.assertEqual(person['seen_from'], ['a', 'b'])
        self.assertGreater(person['observed_bounds_world']['min'][0], 8)
        self.assertLess(scene.objects['body:desk']['observed_bounds_world']['min'][0], 0)
        self.assertTrue(all(len(points) <= 3 for points in scene.points.values()))
        with self.assertRaises(ValueError):
            scene.update([], obs2, 'b')
        with self.assertRaises(ValueError):
            scene.update([], obs, 'old')
        with tempfile.TemporaryDirectory() as tmp:
            scene.save(tmp)
            self.assertTrue((Path(tmp)/'scene_map.json').exists())
            with np.load(Path(tmp)/'object_surfaces.npz', allow_pickle=False) as data:
                self.assertEqual(len(data.files), 3)

    def test_visibility_and_landmark_clearing(self):
        obs = observation()
        point = np.array([[0., 0., 2.]])
        self.assertEqual(visibility(point, obs), 'not_observed')
        obs.depth_m[:] = 1
        self.assertEqual(visibility(point, obs), 'occluded')
        self.assertEqual(visibility(np.array([[50., 0., 2.]]), obs), 'out_of_view')
        self.assertEqual(visibility(np.array([[0., 0., -2.]]), obs), 'out_of_view')
        obs.segmentation[:] = -1
        self.assertEqual(visibility(point, obs), 'not_observed')
        scene = SceneMap()
        first = observation()
        scene.update(SegmentationPerception(GEOMS).detect(first), first, 'a')
        scene.update([], replace(obs, simulation_time=1), 'b')
        self.assertEqual(scene.objects['body:person']['landmarks'], {})
        self.assertEqual(scene.objects['body:person']['last_seen'], 0)

    def test_optional_backends_independent_and_failure_safe(self):
        class BrokenDetector:
            def detect(self, rgb):
                raise RuntimeError('unavailable model')
        class Pose:
            def estimate(self, rgb):
                return [{'source': 'test', 'landmarks': []}]
        detections, extra = PerceptionPipeline(GEOMS, detector=BrokenDetector(), pose_estimator=Pose()).process(observation())
        self.assertEqual(len(detections), 3)
        self.assertEqual(len(extra['warnings']), 1)
        self.assertEqual(len(extra['learned_poses']), 1)
        _, disabled = PerceptionPipeline(GEOMS).process(observation())
        self.assertEqual(disabled, dict(learned_detections=[], learned_poses=[], warnings=[]))

    def test_offline_replay_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            survey = Path(tmp)/'survey'
            survey.mkdir()
            views = []
            for i in range(2):
                view_id = f'view_{i:03d}'
                folder = survey/view_id
                folder.mkdir()
                obs = observation(i)
                cv2.imwrite(str(folder/'rgb.png'), obs.rgb)
                np.savez_compressed(folder/'geometry.npz', depth_m=obs.depth_m,
                                    segmentation=obs.segmentation, intrinsics=obs.intrinsics,
                                    world_from_camera=obs.world_from_camera)
                (folder/'metadata.json').write_text(json.dumps(dict(geoms=GEOMS, simulation_time=i, joints={})))
                views.append(dict(view_id=view_id, status='saved'))
            (survey/'manifest.json').write_text(json.dumps(dict(views=views)))
            output = Path(tmp)/'map'
            scene = map_survey(survey, output)
            self.assertEqual(len(scene.objects), 3)
            self.assertEqual(scene.objects['body:person']['observation_count'], 2)
            self.assertEqual(json.loads((output/'mapping_report.json').read_text())['status'], 'completed')
            self.assertTrue((output/'view_001/masks.npz').is_file())
            with self.assertRaises(FileExistsError):
                map_survey(survey, output)
