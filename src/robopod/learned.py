"""Optional local-file adapters. No packages or model weights are fetched at runtime."""
from pathlib import Path
import os
import cv2
import numpy as np


def local_model(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Local model file not found: {path}. Download it separately, then configure its path.')
    return str(path)


class YoloDetector:
    def __init__(self, model_path, *, device='cpu'):
        path = local_model(model_path)
        if Path(path).suffix.lower() != '.pt':
            raise ValueError('YOLO requires local .pt segmentation weights')
        # Ultralytics honors these before importing optional integrations/checks.
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_OFFLINE'] = 'true'
        from ultralytics import YOLO
        self.model = YOLO(path, task='segment')
        self.device = device

    def detect(self, rgb):
        result = self.model.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), device=self.device,
                                    verbose=False, conf=.35, retina_masks=True)[0]
        detections = []
        if result.boxes is None:
            return detections
        for i,box in enumerate(result.boxes):
            record = dict(category=result.names[int(box.cls.item())], confidence=float(box.conf.item()),
                          bbox_xyxy=box.xyxy[0].cpu().numpy().tolist(), source='yolo_local')
            if result.masks is not None:
                record['polygon'] = result.masks.xy[i].tolist()
            detections.append(record)
        return detections


class MediaPipePose:
    def __init__(self, model_path):
        path = local_model(model_path)
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=4)
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def estimate(self, rgb):
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                               data=np.ascontiguousarray(rgb)))
        return [dict(source='mediapipe_local', landmarks=[dict(index=i, x=p.x, y=p.y,
                     visibility=p.visibility, presence=p.presence) for i,p in enumerate(pose)])
                for pose in result.pose_landmarks]

    def close(self):
        self.model.close()


def build_backends(config):
    """Each independently enabled backend falls back with a visible configuration error."""
    detector = pose = None
    warnings = []
    for key, factory in (('yolo_model', YoloDetector), ('pose_model', MediaPipePose)):
        if config.get(key):
            try:
                backend = factory(config[key])
                if key == 'yolo_model': detector = backend
                else: pose = backend
            except Exception as error:
                warnings.append(f'{key}: {error}')
    return detector, pose, warnings


def validate_config(config):
    config = dict(config or {})
    allowed = {'yolo_model', 'pose_model', 'adaptive_views', 'replan_budget'}
    if set(config)-allowed:
        raise ValueError('Unknown configuration option')
    for key, default, maximum in (('adaptive_views',6,12),('replan_budget',1,2)):
        value=config.get(key,default)
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f'{key} must be an integer from 0 to {maximum}')
        config[key]=value
    for key in ('yolo_model','pose_model'):
        value=config.get(key,'')
        if not isinstance(value,str) or len(value)>2000:
            raise ValueError(f'{key} must be a local file path')
        config[key]=value.strip()
    return config


def close_backends(detector, pose):
    for backend in (detector,pose):
        if backend is not None and hasattr(backend,'close'):
            backend.close()
