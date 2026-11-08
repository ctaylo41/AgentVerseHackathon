"""Sparse observed-surface map; persistent simulator IDs, conservative visibility."""
from dataclasses import asdict
from pathlib import Path
import json

import cv2
import numpy as np

from .exploration import _write_json
from .perception import PerceptionPipeline, annotate
from .simulation import Observation


def _sparse(points, voxel_size, maximum):
    _, indices = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_index=True)
    selected = points[indices]
    if len(selected) > maximum:
        selected = selected[np.linspace(0, len(selected)-1, maximum, dtype=int)]
    return selected


def visibility(points, obs, depth_tolerance=0.08):
    """Classify only previous surface samples; occlusion is an inference.

    With no observed pixels, dynamic movement or missing depth remains ambiguous.
    Do not claim to know true geometry behind an occluder.
    """
    camera = (points - obs.world_from_camera[:3, 3]) @ obs.world_from_camera[:3, :3]
    front = camera[:, 2] > 1e-6
    camera = camera[front]
    if not len(camera):
        return 'out_of_view'
    projected = camera @ obs.intrinsics.T
    uv = np.rint(projected[:, :2] / projected[:, 2, None]).astype(int)
    h, w = obs.depth_m.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not inside.any():
        return 'out_of_view'
    uv, z = uv[inside], camera[inside, 2]
    depth = obs.depth_m[uv[:, 1], uv[:, 0]]
    ids = obs.segmentation[uv[:, 1], uv[:, 0], 0]
    valid = np.isfinite(depth) & (depth > 0) & (ids >= 0)
    if valid.all() and np.all(depth < z - depth_tolerance):
        return 'occluded'
    return 'not_observed'


class SceneMap:
    def __init__(self, voxel_size=0.08, max_points=2048):
        self.artifact_identity = None
        if not np.isfinite(voxel_size) or voxel_size <= 0 or max_points < 1:
            raise ValueError('Positive voxel size and point limit required')
        self.voxel_size, self.max_points = voxel_size, max_points
        self.objects = {}
        self.points = {}
        self.viewpoints = {}
        self.last_time = -float('inf')

    def update(self, detections, obs, view_id):
        if view_id in self.viewpoints:
            raise ValueError(f'Duplicate viewpoint: {view_id}')
        if obs.simulation_time < self.last_time:
            raise ValueError('Observations must be chronological; use a new map after reset')
        self.last_time = obs.simulation_time
        self.viewpoints[view_id] = dict(simulation_time=obs.simulation_time, joints=obs.joints,
                                       intrinsics=obs.intrinsics.tolist(),
                                       world_from_camera=obs.world_from_camera.tolist())
        for object_id, record in self.objects.items():
            state = visibility(self.points[object_id], obs)
            record.update(visibility=state, visibility_view_id=view_id,
                          visibility_basis='projection_of_last_observed_surfaces',
                          landmarks={})
        for detection in detections:
            identity = detection.identity
            object_id = identity.object_id
            record = self.objects.get(object_id)
            observed_min = detection.points_world.min(axis=0)
            observed_max = detection.points_world.max(axis=0)
            if record is None:
                record = dict(**asdict(identity), first_seen=obs.simulation_time,
                              seen_from=[], observation_count=0)
                self.objects[object_id] = record
            elif not identity.dynamic:
                observed_min = np.minimum(observed_min, record['observed_bounds_world']['min'])
                observed_max = np.maximum(observed_max, record['observed_bounds_world']['max'])
            # Dynamic objects use the latest observation, avoiding trails after movement.
            points = detection.points_world
            if object_id in self.points and not identity.dynamic:
                points = np.concatenate([self.points[object_id], points])
            self.points[object_id] = _sparse(points, self.voxel_size, self.max_points)
            record['seen_from'].append(view_id)
            record['observation_count'] += 1
            record.update(category=identity.category, label=identity.label, parent_id=identity.parent_id)
            record.update(last_seen=obs.simulation_time, visibility='visible', visibility_view_id=view_id,
                          visibility_basis='segmented_pixels', confidence=detection.confidence,
                          confidence_meaning='learned label confidence' if detection.source.startswith('learned') else 'unclassified instance; category unknown' if identity.category=='unknown' else 'configured simulator identity; not geometry completeness',
                          source=detection.source, bbox_xyxy=detection.bbox_xyxy,
                          last_bbox_view_id=view_id,
                          position_world=((observed_min + observed_max)/2).tolist(),
                          position_meaning='observed surface bounding-box centre',
                          observed_bounds_world=dict(min=observed_min.tolist(), max=observed_max.tolist()),
                          bounds_complete=False, landmarks=detection.landmarks,
                          geometry_policy='latest_observation' if identity.dynamic else 'accumulated_observations',
                          sparse_point_count=len(self.points[object_id]))
        return self.to_dict()

    def to_dict(self):
        return dict(artifact_identity=self.artifact_identity, schema_version=2, coordinate_frame='world, metres, Z up',
                    voxel_size_m=self.voxel_size, max_points_per_object=self.max_points,
                    latest_simulation_time=self.last_time if self.viewpoints else None,
                    objects=self.objects, viewpoints=self.viewpoints)

    def save(self, directory):
        directory = Path(directory)
        _write_json(directory / 'scene_map.json', self.to_dict())
        # Stable IDs are also the array keys; no pickle/object arrays.
        np.savez_compressed(directory / 'object_surfaces.npz', **self.points)
        if not cv2.imwrite(str(directory / 'map_topdown.png'), self.topdown()):
            raise OSError('Could not save map preview')

    def topdown(self):
        canvas = np.full((720, 900, 3), (30, 27, 24), dtype=np.uint8)
        cv2.putText(canvas, 'Observed surface map | XY top view | metres', (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
        if not self.points:
            return canvas
        all_xy = np.concatenate(list(self.points.values()))[:, :2]
        low, high = all_xy.min(axis=0)-0.3, all_xy.max(axis=0)+0.3
        scale = min(800 / max(high[0]-low[0], 1), 570 / max(high[1]-low[1], 1))
        def pixel(xy):
            coords = (np.asarray(xy)-low)*scale
            return np.array([50+coords[0], 650-coords[1]], dtype=int)
        for object_id, points in self.points.items():
            record = self.objects[object_id]
            infrastructure = record['category'] in ('wall', 'floor')
            color = (75, 75, 75) if infrastructure else (110, 210, 240)
            for point in points:
                cv2.circle(canvas, tuple(pixel(point[:2])), 1, color, -1)
            if not infrastructure:
                p = pixel(record['position_world'][:2])
                cv2.putText(canvas, record['label'], tuple(p), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'X: {low[0]:.1f} to {high[0]:.1f} m   Y: {low[1]:.1f} to {high[1]:.1f} m; +Y up',
                    (20, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas


def load_observation(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    rgb = cv2.imread(str(directory / 'rgb.png'))
    if rgb is None:
        raise ValueError(f'Missing/unreadable RGB: {directory}')
    with np.load(directory / 'geometry.npz', allow_pickle=False) as arrays:
        obs = Observation(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), arrays['depth_m'], arrays['segmentation'],
                          arrays['intrinsics'], arrays['world_from_camera'],
                          metadata['simulation_time'], metadata['joints'])
    return obs, metadata


def map_survey(survey, output, *, detector=None, pose_estimator=None):
    """Offline replay of retained observations; no GL context or model downloads."""
    survey, output = Path(survey), Path(output)
    manifest = json.loads((survey / 'manifest.json').read_text())
    views = [v for v in manifest['views'] if v['status'] == 'saved']
    if not views:
        raise ValueError('Survey has no saved observations')
    output.mkdir(parents=True, exist_ok=False)
    scene = SceneMap()
    scene.artifact_identity = manifest.get('artifact_identity')
    report = dict(status='running', source_survey=str(survey.resolve()), views=[])
    pipeline = None
    try:
        for view in views:
            view_id = view['view_id']
            # Generated survey IDs are plain directory names, never arbitrary paths.
            if Path(view_id).name != view_id or view_id in ('.', '..'):
                raise ValueError('Invalid survey view ID')
            obs, metadata = load_observation(survey / view_id)
            if metadata.get('artifact_identity') != scene.artifact_identity:
                raise ValueError('Mixed scene/robot versions in survey observations')
            if pipeline is None:
                pipeline = PerceptionPipeline(metadata['geoms'], detector=detector, pose_estimator=pose_estimator)
            detections, supplemental = pipeline.process(obs)
            scene.update(detections, obs, view_id)
            target = output / view_id
            target.mkdir()
            masks = {d.identity.object_id: d.mask for d in detections}
            np.savez_compressed(target / 'masks.npz', **masks)
            _write_json(target / 'detections.json', dict(detections=[d.summary() for d in detections], **supplemental))
            if not cv2.imwrite(str(target / 'annotated.jpg'), annotate(obs, detections)):
                raise OSError('Could not save annotated image')
            report['views'].append(dict(view_id=view_id, objects=[d.identity.object_id for d in detections],
                                        visibility={k: v['visibility'] for k, v in scene.objects.items()}))
        scene.save(output)
        report.update(status='completed', object_count=len(scene.objects))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write_json(output / 'mapping_report.json', report)
    return scene
