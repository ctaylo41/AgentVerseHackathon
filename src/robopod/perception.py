"""Visible-pixel perception with explicit simulator identity provenance.

Optional learned backends implement the protocols below and are injected separately;
no model package is imported or downloaded by the fallback path.
"""
from dataclasses import dataclass, field, replace
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
    parent_id: str | None = None
    detail_ids: tuple = ()


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
        body, name = info.get('semantic_body',info['body']), info['name']
        if info.get('robot') or body in ('carriage','shoulder_boom','forearm_boom','pan_frame','tilt_frame','camera_head') or name=='rail':
            continue
        if body in BODY_LABELS:
            category, label, dynamic = BODY_LABELS[body]
            registry[int(geom_id)] = SemanticIdentity(f'body:{body}', category, label, dynamic,
                detail_ids=(f'part:product:{name}',) if body=='product' and name in ('controls','keys') else ())
        elif body == 'world' and name in WORLD_LABELS:
            category, label = WORLD_LABELS[name]
            registry[int(geom_id)] = SemanticIdentity(f'geom:{name}', category, label)
        else:
            key = f'geom:{name or geom_id}' if body=='world' else f'body:{body}'
            registry[int(geom_id)] = SemanticIdentity(key, 'unknown', body if body!='world' else name or 'Unclassified object', True)
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
        for geom_id in np.unique(seg[:,:,0][valid]):
            identity = self.registry.get(int(geom_id))
            if identity:
                for part in identity.detail_ids:
                    detail = SemanticIdentity(part, 'component', 'Synthesizer '+part.split(':')[-1], True, parent_id=identity.object_id)
                    grouped.setdefault(part,(detail,[]))[1].append(int(geom_id))
        detections = []
        for object_id in sorted(grouped):
            identity, ids = grouped[object_id]
            mask = valid & np.isin(seg[:, :, 0], ids)
            v, u = np.nonzero(mask)
            sample = np.linspace(0, len(u)-1, min(len(u), self.max_points), dtype=int)
            points = backproject(obs, u[sample], v[sample])
            landmarks = {}
            if identity.category in ('person','unknown'):
                for geom_id in ids:
                    name = self.geoms[geom_id]['name']
                    if name not in ('head', 'nose', 'torso', 'left_arm', 'right_arm', 'left_eye', 'right_eye', 'mouth_left', 'mouth_right', 'right_hand', 'left_hand', 'left_shoulder', 'right_shoulder'):
                        continue
                    lv, lu = np.nonzero(valid & (seg[:, :, 0] == geom_id))
                    select = np.linspace(0, len(lu)-1, min(len(lu), 512), dtype=int)
                    landmarks[name] = dict(pixel_uv=[float(np.median(lu)), float(np.median(lv))],
                                           world_position=np.median(backproject(obs, lu[select], lv[select]), axis=0).tolist(),
                                           source='visible_surface_proxy', visible=True)
            detections.append(Detection(identity, [int(u.min()), int(v.min()), int(u.max()+1), int(v.max()+1)],
                                        mask, points, 0.0 if identity.category=='unknown' else 1.0,
                                        source='unclassified_segmentation' if identity.category=='unknown' else 'mujoco_segmentation', landmarks=landmarks))
        return detections


class PerceptionPipeline:
    """Independent optional hooks; learned failures never disable fallback mapping.

    Learned masks associate to persistent simulator instances by image-space IoU.
    Confidence 1.0 on fallback detections means exact configured identity, not full
    surface completeness or learned recognition accuracy.
    """
    def __init__(self, geoms, *, detector: ObjectDetector | None = None,
                 pose_estimator: PoseEstimator | None = None):
        self.fallback = SegmentationPerception(geoms)
        self.detector, self.pose_estimator = detector, pose_estimator
        self.labels = {}
        self.warnings = []
        self.last_report = None

    @property
    def registry(self):
        return self.fallback.registry

    def detect(self, obs):
        detections, report = self.process(obs)
        self.warnings = report['warnings']
        self.last_report = report
        return detections

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
                    for item in value:
                        if key=='learned_detections':
                            if not isinstance(item.get('category'),str) or not isinstance(item.get('confidence'),(int,float)) or not 0<=item['confidence']<=1:
                                raise ValueError('Detection requires category and confidence in [0,1]')
                            if 'bbox_xyxy' in item and np.asarray(item['bbox_xyxy'],dtype=float).shape!=(4,):
                                raise ValueError('Invalid detection box')
                            if item.get('polygon') and (np.asarray(item['polygon'],dtype=float).ndim!=2 or np.asarray(item['polygon']).shape[1]!=2):
                                raise ValueError('Invalid detection polygon')
                        else:
                            for point in item.get('landmarks',[]):
                                if not all(isinstance(point.get(k),(int,float)) for k in ('x','y','index','visibility')):
                                    raise ValueError('Invalid pose landmark')
                    result[key] = value
                except Exception as error:
                    result['warnings'].append(f'{key}: {type(error).__name__}: {error}')
        detections = self.fallback.detect(obs)
        # Associate learned masks to observed simulator instances; never duplicate an existing ID.
        associated = set()
        for learned in sorted(result['learned_detections'],key=lambda d:d.get('confidence',0),reverse=True):
            if learned.get('confidence',0) < .35 or not learned.get('category'):
                continue
            mask = np.zeros(obs.depth_m.shape,np.uint8)
            if learned.get('polygon'):
                polygon=np.asarray(learned['polygon'],dtype=float)
                if polygon.ndim!=2 or polygon.shape[1]!=2 or not np.isfinite(polygon).all():
                    continue
                cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
            elif 'bbox_xyxy' in learned:
                x1,y1,x2,y2=np.asarray(learned['bbox_xyxy'],dtype=int)
                mask[np.clip(y1,0,mask.shape[0]):np.clip(y2,0,mask.shape[0]),np.clip(x1,0,mask.shape[1]):np.clip(x2,0,mask.shape[1])]=1
            scored=[(float(np.logical_and(mask,d.mask).sum()/max(1,np.logical_or(mask,d.mask).sum())),i)
                    for i,d in enumerate(detections) if not d.identity.parent_id and d.identity.object_id not in associated
                    and d.identity.category not in ('floor','wall')]
            if not scored:
                continue
            overlap,index=max(scored)
            if overlap < .25:
                continue
            d=detections[index];associated.add(d.identity.object_id)
            if d.identity.category=='unknown':
                self.labels[d.identity.object_id]=(learned['category'],float(learned['confidence']))
            # Configured identities remain authoritative; evidence is retained on every match.
            d.landmarks['learned_detection'] = dict(category=learned['category'],confidence=learned['confidence'],
                                                    association_iou=overlap,source=learned.get('source','learned'))
        for d in detections:
            if d.identity.object_id in self.labels:
                category,confidence=self.labels[d.identity.object_id]
                d.identity=replace(d.identity,category=category,label=category)
                d.confidence=confidence;d.source='learned_associated_segmentation'
        for pose in result['learned_poses']:
            landmarks=[p for p in pose.get('landmarks',[]) if p.get('visibility',0)>=.5 and p.get('presence',1)>=.5]
            if not landmarks:
                continue
            h,w=obs.depth_m.shape
            u=np.clip([round(p['x']*(w-1)) for p in landmarks],0,w-1).astype(int)
            v=np.clip([round(p['y']*(h-1)) for p in landmarks],0,h-1).astype(int)
            people=[d for d in detections if d.identity.category=='person']
            if not people:
                continue
            person=max(people,key=lambda d:int(d.mask[v,u].sum()))
            if person.mask[v,u].mean()<.3:
                continue
            for point,x,y in zip(landmarks,u,v):
                if not person.mask[y,x]:
                    continue
                person.landmarks[f'pose_{point["index"]}']=dict(pixel_uv=[int(x),int(y)],
                    world_position=backproject(obs,np.array([x]),np.array([y]))[0].tolist(),
                    confidence=point['visibility'],source='mediapipe_local',visible=True)
                if point['index']==0:
                    person.landmarks['head']=person.landmarks['pose_0']
        return detections, result


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
            if 'pixel_uv' not in landmark:
                continue
            cv2.circle(image, tuple(round(v) for v in landmark['pixel_uv']), 3, (0, 200, 255), -1)
    return image
