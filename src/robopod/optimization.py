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
    allow_dutch = 'dutch' in shot.get('description', '').lower()
    roll_loss = 0.0 if allow_dutch else abs(float(obs.joints.get('roll', 0.0))) * 0.5
    return dict(loss=float(loss+perspective_loss+roll_loss), targets=metrics, backgrounds=backgrounds,
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
    allow_dutch = 'dutch' in shot.get('description', '').lower()
    scratch = mujoco.MjData(sim.model)
    scratch.mocap_pos[:] = sim.data.mocap_pos
    scratch.mocap_quat[:] = sim.data.mocap_quat
    ranked = {}

    def rank(target):
        target = {name: float(np.clip(target.get(name, current.get(name, 0.0)), *limits[name])) for name in axes}
        if not allow_dutch and 'roll' in target:
            target['roll'] = 0.0
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
    rolls = [-0.15, 0., 0.15] if allow_dutch else [0.0]
    for values in product(np.linspace(*limits['height'], 4), np.linspace(*limits['pan'], 7),
                          np.linspace(*limits['tilt'], 7), rolls):
        rank(dict(zip(('height', 'pan', 'tilt', 'roll'), values), truck=0.0, dolly=0.0))
    if {'truck', 'dolly'} <= set(axes) and shot['targets']:
        primary = shot['targets'][0]
        pts = surfaces[primary['object_id']]
        center = pts.mean(axis=0)
        fovy_rad = math.radians(float(sim.model.cam_fovy[sim.camera_id]))
        fovx_rad = 2 * math.atan(math.tan(fovy_rad / 2) * (sim.width / sim.height))
        for truck in np.linspace(*limits['truck'], 5):
            for dolly in np.linspace(*limits['dolly'], 5):
                for height in np.linspace(*limits['height'], 4):
                    dx = float(center[0] - truck)
                    dy = float(center[1] - (-2.31 + dolly))
                    dz = float(center[2] - height)
                    dist_xy = max(1e-3, math.hypot(dx, dy))
                    pan_look = math.atan2(-dx, dy) + (primary['center_x'] - 0.5) * fovx_rad
                    tilt_look = math.atan2(dz, dist_xy) + (primary['center_y'] - 0.5) * fovy_rad
                    rank(dict(height=height, truck=truck, dolly=dolly, pan=pan_look, tilt=tilt_look, roll=0.0))
    for step in (0.12, 0.04):
        best = min(ranked.values(), key=lambda item: item[0])[1]
        for dh, dp, dt in product([-step, 0, step], repeat=3):
            candidate = dict(best, height=best['height']+dh, pan=best['pan']+dp, tilt=best['tilt']+dt)
            if not allow_dutch:
                candidate['roll'] = 0.0
            rank(candidate)
        if {'truck', 'dolly'} <= set(axes):
            best = min(ranked.values(), key=lambda item: item[0])[1]
            for d_truck, d_dolly, dp, dt in product([-step*2, 0, step*2], [-step*2, 0, step*2], [-step, 0, step], [-step, 0, step]):
                rank(dict(best, truck=best['truck']+d_truck, dolly=best['dolly']+d_dolly,
                          pan=best['pan']+dp, tilt=best['tilt']+dt, roll=0.0))
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


def plan_sweep(sim, shot, scene, surfaces):
    """Select a smooth, framing-preserving continuous camera sweep across the shot duration."""
    from .control import RATES
    from .simulation import Observation
    reference = sim.capture()
    perception = SegmentationPerception(geometry_catalog(sim))
    measured_start = evaluate(shot, scene, surfaces, reference, perception.detect(reference), perception.registry)
    limits = safe_limits(sim)
    axes = sim.axes
    start = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in axes}
    duration = float(shot.get('duration_seconds', 2.0))
    max_delta = {n: RATES[n] * duration / 1.875 * 0.85 for n in axes}
    scratch = mujoco.MjData(sim.model)
    scratch.mocap_pos[:] = sim.data.mocap_pos
    scratch.mocap_quat[:] = sim.data.mocap_quat
    sag = {n: start[n] - sim.configuration[n] for n in axes}

    def eval_pose(pose):
        for name, value in pose.items():
            scratch.joint(name).qpos[0] = value - sag[name]
        mujoco.mj_forward(sim.model, scratch)
        transform = np.eye(4)
        transform[:3, :3] = scratch.cam_xmat[sim.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = scratch.cam_xpos[sim.camera_id]
        obs = Observation(reference.rgb, reference.depth_m, reference.segmentation, reference.intrinsics,
                          transform, float(scratch.time), pose)
        return evaluate(shot, scene, surfaces, obs)

    start_eval = eval_pose(start)
    criteria = shot['completion']
    tight_margin = any(
        m['scale_error'] > 0.75 * criteria['max_scale_error'] or m['center_error'] > 0.75 * criteria['max_center_error']
        for m in measured_start['targets'].values()
    )
    loss_budget = 0.005 if tight_margin else 0.05
    best_end = dict(start)
    best_score = -float('inf')

    dh_vals = [-0.14, -0.08, 0.0, 0.08, 0.14]
    dp_vals = [-0.05, -0.025, 0.0, 0.025, 0.05]
    dt_vals = np.linspace(-0.06, 0.06, 9)

    for dh, dp in product(dh_vals, dp_vals):
        if abs(dh) < 1e-4 and abs(dp) < 1e-4:
            continue
        h_end = float(np.clip(start['height'] + np.clip(dh, -max_delta['height'], max_delta['height']), *limits['height']))
        p_end = float(np.clip(start['pan'] + np.clip(dp, -max_delta['pan'], max_delta['pan']), *limits['pan']))
        for dt in dt_vals:
            t_end = float(np.clip(start['tilt'] + np.clip(dt, -max_delta['tilt'], max_delta['tilt']), *limits['tilt']))
            end = dict(start, height=h_end, pan=p_end, tilt=t_end, roll=0.0)
            if 'truck' in axes:
                end['truck'] = float(np.clip(start['truck'] + np.clip(-dp * 1.2, -max_delta['truck'], max_delta['truck']), *limits['truck']))
            if 'dolly' in axes:
                d_step = max(0.0, dh * 0.3) if tight_margin else dh * 0.3
                end['dolly'] = float(np.clip(start['dolly'] + np.clip(d_step, -max_delta['dolly'], max_delta['dolly']), *limits['dolly']))
            disp = (abs(end['height'] - start['height'])
                    + abs(end.get('truck', 0.0) - start.get('truck', 0.0))
                    + abs(end.get('dolly', 0.0) - start.get('dolly', 0.0))
                    + 0.6 * abs(end['pan'] - start['pan']) + 0.6 * abs(end['tilt'] - start['tilt']))
            if disp < 0.02:
                continue
            valid = True
            losses = []
            for frac in (0.35, 0.7, 1.0):
                wp = {n: start[n] + frac * (end[n] - start[n]) for n in axes}
                res = eval_pose(wp)
                if start_eval['satisfies_observed_criteria'] and not res['satisfies_observed_criteria']:
                    valid = False
                    break
                if res['loss'] > start_eval['loss'] + loss_budget:
                    valid = False
                    break
                losses.append(res['loss'])
            if not valid:
                continue
            mean_drift = max(0.0, float(np.mean(losses)) - start_eval['loss'])
            score = min(disp, 0.18) - 2.0 * mean_drift
            if score > best_score:
                best_score = score
                best_end = end

    displacement = {n: round(best_end[n] - start[n], 5) for n in axes}
    return dict(start=start, end=best_end, displacement=displacement,
                continuous=any(abs(v) > 1e-4 for v in displacement.values()))

