"""Deterministic geometric search with rendered checks of shortlisted candidates."""
from itertools import product
import math

import mujoco
import numpy as np

from .control import move_to
from .perception import SegmentationPerception


def geometry_catalog(sim):
    return {str(i): dict(name=sim.model.geom(i).name,
                        body=sim.model.body(int(sim.model.geom_bodyid[i])).name)
            for i in range(sim.model.ngeom)}


def project(points, transform, intrinsics):
    camera = (points-transform[:3, 3]) @ transform[:3, :3]
    uv = np.full((len(points), 2), np.nan)
    front = camera[:, 2] > 1e-4
    pixels = camera[front] @ intrinsics.T
    uv[front] = pixels[:, :2]/pixels[:, 2, None]
    return uv, camera[:, 2]


def framing(uv, width, height):
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        return None
    low, high = uv[valid].min(axis=0), uv[valid].max(axis=0)
    return dict(center_x=float((low[0]+high[0])/2/width),
                center_y=float((low[1]+high[1])/2/height),
                frame_height_fraction=float((high[1]-low[1])/height))


def evaluate(shot, scene, surfaces, obs, detections=None, registry=None):
    """Visibility is a proxy over known surface samples, never full-object truth."""
    width, height = obs.rgb.shape[1], obs.rgb.shape[0]
    detected = None if detections is None else {d.identity.object_id: d for d in detections}
    criteria = shot['completion']
    metrics, reasons, loss = {}, [], 0.0
    for target in shot['targets']:
        key = target['object_id']
        uv, z = project(surfaces[key], obs.world_from_camera, obs.intrinsics)
        inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        visible = inside.copy()
        box = framing(uv, width, height)
        headroom = float(np.nanmin(uv[:, 1])/height) if np.isfinite(uv[:, 1]).any() else None
        if detected is not None:
            # Nearest rendered depth and semantic ID check detects occlusion, including self-occlusion.
            visible[:] = False
            for idx in np.flatnonzero(inside):
                u, v = np.floor(uv[idx]).astype(int)
                patch = obs.segmentation[max(0,v-1):min(height,v+2), max(0,u-1):min(width,u+2)]
                depth = obs.depth_m[max(0,v-1):min(height,v+2), max(0,u-1):min(width,u+2)]
                same = np.zeros(depth.shape, dtype=bool)
                for geom_id in np.unique(patch[:, :, 0]):
                    identity = registry.get(int(geom_id))
                    if identity is not None and identity.object_id == key:
                        same |= (patch[:, :, 0] == geom_id) & (patch[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                visible[idx] = np.any(same & np.isfinite(depth) & (np.abs(depth-z[idx]) < 0.12))
            detection = detected.get(key)
            box = None
            headroom = None
            if detection is not None:
                x1, y1, x2, y2 = detection.bbox_xyxy
                box = dict(center_x=(x1+x2)/2/width, center_y=(y1+y2)/2/height,
                           frame_height_fraction=(y2-y1)/height)
                # Top of person mask is a headroom approximation; require observed head proxy.
                if 'head' in detection.landmarks:
                    headroom = y1/height
        fraction = float(visible.mean())
        center_error = math.hypot(box['center_x']-target['center_x'], box['center_y']-target['center_y']) if box else 2.0
        scale_error = abs(box['frame_height_fraction']-target['frame_height_fraction']) if box else 1.0
        head_error = 0.0
        if scene['objects'][key]['category'] == 'person':
            head_error = abs(headroom-criteria['headroom_fraction']) if headroom is not None else 1.0
        metrics[key] = dict(framing=box, center_error=center_error, scale_error=scale_error,
                            visible_surface_fraction=fraction, headroom_error=head_error)
        loss += 4*center_error + 3*scale_error + 2*(1-fraction) + 2*head_error
        for failed, label in [(center_error > criteria['max_center_error'], 'position'),
                              (scale_error > criteria['max_scale_error'], 'scale'),
                              (fraction < criteria['min_visible_fraction'], 'visibility'),
                              (head_error > criteria['headroom_tolerance'], 'headroom')]:
            if failed:
                reasons.append(f'{key}: {label}')
    backgrounds = {}
    for key in shot['background_object_ids']:
        if detected is None:
            uv, _ = project(surfaces[key], obs.world_from_camera, obs.intrinsics)
            seen = bool(np.any((uv[:,0] >= 0) & (uv[:,0] < width) & (uv[:,1] >= 0) & (uv[:,1] < height)))
        else:
            seen = key in detected and detected[key].mask.sum() >= 16
        backgrounds[key] = bool(seen)
        if not seen:
            loss += 1
            reasons.append(f'{key}: background missing')
    # Perspective is a soft preference; actual frame metrics govern completion.
    target_z = np.mean([surfaces[t['object_id']][:,2].max() for t in shot['targets']])
    delta = float(obs.world_from_camera[2,3]-target_z)
    perspective_loss = {'neutral': 0.0, 'eye_level': abs(delta)*0.1,
                        'high_angle': max(0., -delta)*0.2, 'low_angle': max(0., delta)*0.2}[shot['perspective']]
    return dict(loss=float(loss+perspective_loss), targets=metrics, backgrounds=backgrounds,
                satisfies_observed_criteria=not reasons, failures=reasons,
                visibility_metric='fraction of reference surface samples; incomplete-geometry proxy')


def safe_limits(sim):
    # Leave room for gravity sag and soft joint-limit dynamics.
    return {name: (lo+(0.06 if name == 'height' else 0.03), hi-0.03)
            for name, (lo, hi) in sim.limits.items()}


def optimize(sim, probe, shot, scene, surfaces, *, shortlist=8, progress=None):
    """Forward-kinematic proposals use private MjData, never the execution state.

    Shortlisted configurations are then reached through actuators in a separate probe
    simulator, and ranked using actual rendered segmentation/depth and framing.
    """
    from .simulation import Observation
    reference = sim.capture()
    perception = SegmentationPerception(geometry_catalog(sim))
    baseline = evaluate(shot, scene, surfaces, reference, perception.detect(reference), perception.registry)
    limits = safe_limits(sim)
    axes = sim.axes
    current = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in axes}
    scratch = mujoco.MjData(sim.model)
    scratch.mocap_pos[:] = sim.data.mocap_pos
    scratch.mocap_quat[:] = sim.data.mocap_quat
    ranked = {}

    def rank(target):
        target = {name: float(np.clip(target[name], *limits[name])) for name in axes}
        key = tuple(round(target[name], 7) for name in axes)
        if key in ranked:
            return
        for name, value in target.items():
            # Predict the measured vertical position of the existing gravity-loaded servo.
            scratch.joint(name).qpos[0] = value - (current[name] - sim.configuration[name])
        mujoco.mj_forward(sim.model, scratch)
        transform = np.eye(4)
        transform[:3,:3] = scratch.cam_xmat[sim.camera_id].reshape(3,3) @ np.diag([1,-1,-1])
        transform[:3,3] = scratch.cam_xpos[sim.camera_id]
        obs = Observation(reference.rgb, reference.depth_m, reference.segmentation, reference.intrinsics,
                          transform, float(scratch.time), target)
        score = evaluate(shot, scene, surfaces, obs)['loss']
        travel = sum(abs(target[n]-current[n])/(limits[n][1]-limits[n][0]) for n in axes)
        ranked[key] = (score + 0.05*travel, target)

    rank(current)
    preferred = scene['viewpoints'][shot['preferred_view_id']]['joints']
    rank(preferred)
    for values in product(np.linspace(*limits['height'], 4), np.linspace(*limits['pan'], 7),
                          np.linspace(*limits['tilt'], 7), [-0.15, 0., 0.15]):
        rank(dict(zip(('height', 'pan', 'tilt', 'roll'), values)))
    for step in (0.12, 0.04):
        best = min(ranked.values(), key=lambda item: item[0])[1]
        for offsets in product([-step, 0, step], repeat=4):
            rank({name: best[name]+offset for name, offset in zip(axes, offsets)})
    candidates = sorted(ranked.values(), key=lambda item: item[0])[:shortlist]
    probe.data.mocap_pos[:] = sim.data.mocap_pos
    probe.data.mocap_quat[:] = sim.data.mocap_quat
    best_target, best_score = current, baseline
    best_objective = (not baseline['satisfies_observed_criteria'], baseline['loss'])
    checked = []
    for index, (predicted, target) in enumerate(candidates):
        move_to(probe, target)
        obs = probe.capture()
        result = evaluate(shot, scene, surfaces, obs, perception.detect(obs), perception.registry)
        travel = sum(abs(target[n]-current[n])/(limits[n][1]-limits[n][0]) for n in axes)
        objective = (not result['satisfies_observed_criteria'], result['loss'] + 0.05*travel)
        checked.append(dict(target=target, predicted_loss=predicted, measured=result))
        # Never choose a prettier prediction over a better measured initial view.
        if objective < best_objective:
            best_target, best_score, best_objective = target, result, objective
        if progress:
            progress(f"{shot['shot_id']}: checked candidate {index+1}/{len(candidates)}")
    return dict(target=best_target, before=baseline, predicted_after=best_score,
                geometric_candidates=len(ranked), rendered_candidates=checked)
