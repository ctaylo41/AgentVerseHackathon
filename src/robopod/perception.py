"""Visible-pixel perception with explicit simulator identity provenance.

Optional learned backends implement the protocols below and are injected separately;
no model package is imported or downloaded by the fallback path.
"""
from dataclasses import dataclass, field
import json
from typing import Protocol

import cv2
import mujoco
import numpy as np

from .simulation import Observation


@dataclass(frozen=True)
class SemanticIdentity:
    object_id: str
    category: str
    label: str
    dynamic: bool = False


# Names encode only identity/category. No simulator transforms or hidden geometry.
BODY_LABELS = {'person': ('person', 'Presenter', True),
               'product': ('instrument', 'Synthesizer', True),
               'desk': ('furniture', 'Desk', False),
               'plant': ('plant', 'Plant', False)}
WORLD_LABELS = {'floor': ('floor', 'Floor'), 'back_wall': ('wall', 'Back wall'),
                'side_wall': ('wall', 'Side wall'), 'art': ('decoration', 'Wall art')}


def semantic_registry(geoms):
    registry = {}
    for geom_id, info in geoms.items():
        body, name = info['body'], info['name']
        if body in BODY_LABELS:
            category, label, dynamic = BODY_LABELS[body]
            registry[int(geom_id)] = SemanticIdentity(f'body:{body}', category, label, dynamic)
        elif body == 'world' and name in WORLD_LABELS:
            category, label = WORLD_LABELS[name]
            registry[int(geom_id)] = SemanticIdentity(f'geom:{name}', category, label)
        # Rig geometry and unconfigured identities do not become semantic objects.
    return registry


def backproject(obs, u, v):
    pixels = np.stack([u, v, np.ones(len(u))])
    camera = np.linalg.solve(obs.intrinsics, pixels) * obs.depth_m[v, u]
    return (obs.world_from_camera[:3, :3] @ camera + obs.world_from_camera[:3, 3, None]).T


@dataclass
class Detection:
    identity: SemanticIdentity
    bbox_xyxy: list[int]  # exclusive right and bottom
    mask: np.ndarray
    points_world: np.ndarray
    confidence: float
    source: str = 'mujoco_segmentation'
    landmarks: dict = field(default_factory=dict)

    def summary(self):
        return dict(object_id=self.identity.object_id, category=self.identity.category,
                    label=self.identity.label, bbox_xyxy=self.bbox_xyxy,
                    visible_pixels=int(self.mask.sum()), confidence=self.confidence,
                    source=self.source, landmarks=self.landmarks,
                    sampled_points=len(self.points_world))


class ObjectDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Return JSON-safe learned 2D detections, separately from simulator IDs."""
        ...


class PoseEstimator(Protocol):
    def estimate(self, rgb: np.ndarray) -> list[dict]:
        """Return JSON-safe learned poses with their own confidence/provenance."""
        ...


class SegmentationPerception:
    def __init__(self, geoms, max_points=4096):
        if max_points < 1:
            raise ValueError('max_points must be positive')
        self.geoms = {int(k): v for k, v in geoms.items()}
        self.registry = semantic_registry(geoms)
        self.max_points = max_points

    def detect(self, obs: Observation):
        seg = obs.segmentation
        valid = ((seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                 & (seg[:, :, 0] >= 0) & np.isfinite(obs.depth_m) & (obs.depth_m > 0))
        grouped = {}
        for geom_id in np.unique(seg[:, :, 0][valid]):
            identity = self.registry.get(int(geom_id))
            if identity is not None:
                grouped.setdefault(identity.object_id, (identity, []))[1].append(int(geom_id))
        detections = []
        for object_id in sorted(grouped):
            identity, ids = grouped[object_id]
            mask = valid & np.isin(seg[:, :, 0], ids)
            v, u = np.nonzero(mask)
            sample = np.linspace(0, len(u)-1, min(len(u), self.max_points), dtype=int)
            points = backproject(obs, u[sample], v[sample])
            landmarks = {}
            if identity.category == 'person':
                for geom_id in ids:
                    name = self.geoms[geom_id]['name']
                    if name not in ('head', 'torso', 'left_arm', 'right_arm'):
                        continue
                    lv, lu = np.nonzero(valid & (seg[:, :, 0] == geom_id))
                    select = np.linspace(0, len(lu)-1, min(len(lu), 512), dtype=int)
                    landmarks[name] = dict(pixel_uv=[float(np.median(lu)), float(np.median(lv))],
                                           world_position=np.median(backproject(obs, lu[select], lv[select]), axis=0).tolist(),
                                           source='visible_surface_proxy', visible=True)
            detections.append(Detection(identity, [int(u.min()), int(v.min()), int(u.max()+1), int(v.max()+1)],
                                        mask, points, 1.0, landmarks=landmarks))
        return detections


class PerceptionPipeline:
    """Independent optional hooks; learned failures never disable fallback mapping.

    Learned output remains supplemental until an explicit association strategy exists.
    Confidence 1.0 on fallback detections means exact configured identity, not full
    surface completeness or learned recognition accuracy.
    """
    def __init__(self, geoms, *, detector: ObjectDetector | None = None,
                 pose_estimator: PoseEstimator | None = None):
        self.fallback = SegmentationPerception(geoms)
        self.detector, self.pose_estimator = detector, pose_estimator

    def process(self, obs):
        result = dict(learned_detections=[], learned_poses=[], warnings=[])
        for backend, method, key in ((self.detector, 'detect', 'learned_detections'),
                                     (self.pose_estimator, 'estimate', 'learned_poses')):
            if backend is not None:
                try:
                    value = getattr(backend, method)(obs.rgb)
                    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                        raise ValueError('Backend must return a list of JSON-safe dictionaries')
                    json.dumps(value, allow_nan=False)
                    result[key] = value
                except Exception as error:
                    result['warnings'].append(f'{key}: {type(error).__name__}: {error}')
        return self.fallback.detect(obs), result


def annotate(obs, detections):
    image = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    grid_color = (90, 95, 100)
    for frac in (1/3, 2/3):
        gx, gy = round(w * frac), round(h * frac)
        cv2.line(image, (gx, 0), (gx, h - 1), grid_color, 1, cv2.LINE_AA)
        cv2.line(image, (0, gy), (w - 1, gy), grid_color, 1, cv2.LINE_AA)
    for fx in (1/3, 2/3):
        for fy in (1/3, 2/3):
            cv2.circle(image, (round(w * fx), round(h * fy)), 4, (120, 185, 230), 1, cv2.LINE_AA)
    for detection in detections:
        if detection.identity.category in ('wall', 'floor'):
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = (100, 230, 100)
        image[detection.mask] = (0.8 * image[detection.mask] + 0.2 * np.array(color)).astype(np.uint8)
        cv2.rectangle(image, (x1, y1), (x2-1, y2-1), color, 1)
        cv2.putText(image, detection.identity.label, (x1, max(14, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for landmark in detection.landmarks.values():
            cv2.circle(image, tuple(round(v) for v in landmark['pixel_uv']), 3, (0, 200, 255), -1)
    return image
