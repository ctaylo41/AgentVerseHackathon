"""Deterministic geometric search with rendered checks of shortlisted candidates."""
from itertools import product
import math

import mujoco
import numpy as np

from .control import move_to
from .perception import SegmentationPerception


def geometry_catalog(sim):
    return sim.geometry_catalog()


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
    for raw_target in shot['targets']:
        key = raw_target['object_id']
        uv, z = project(surfaces[key], obs.world_from_camera, obs.intrinsics)
        inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        visible = inside.copy()
        proj_box = framing(uv, width, height)
        box = proj_box
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
                    if identity is not None and (identity.object_id == key or key in identity.detail_ids):
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
        target = raw_target  # Score the validated objective without moving the goalposts.
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
    allow_dutch = shot.get('horizon', 'level') == 'dutch'
    roll_error = abs(float(obs.joints.get('roll', 0.0)))
    if not allow_dutch and roll_error > 0.01:
        reasons.append('horizon: roll exceeds level tolerance')
    roll_loss = 0.0 if allow_dutch else roll_error * 0.5
    return dict(loss=float(loss+perspective_loss+roll_loss), targets=metrics, backgrounds=backgrounds,
                satisfies_observed_criteria=not reasons, failures=reasons,
                visibility_metric='fraction of reference surface samples; incomplete-geometry proxy')


def safe_limits(sim):
    # Leave room for gravity sag and soft joint-limit dynamics.
    return {name: (lo+(0.06 if name == 'height' else 0.03), hi-0.03)
            for name, (lo, hi) in sim.limits.items()}


def _discover_workspace(sim, scratch, perception, limits, surfaces, shot):
    """Dynamically determine unoccluded camera bounds and base offset via MuJoCo kinematics and raycasts."""
    scratch.qpos[:] = 0.0
    mujoco.mj_forward(sim.model, scratch)
    base_cam = scratch.cam_xpos[sim.camera_id].copy()
    geomid = np.zeros(1, dtype=np.int32)
    min_safe_dolly = limits['dolly'][0] if 'dolly' in limits else 0.0
    if 'dolly' in limits:
        mid_h = 0.5 * (limits['height'][0] + limits['height'][1])
        for d in np.linspace(*limits['dolly'], 25):
            scratch.qpos[:] = 0.0
            scratch.joint('height').qpos[0] = mid_h
            scratch.joint('dolly').qpos[0] = d
            mujoco.mj_forward(sim.model, scratch)
            cam_pos = scratch.cam_xpos[sim.camera_id]
            cam_rot = scratch.cam_xmat[sim.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            origin = cam_pos + 0.04 * cam_rot[:, 2]
            hit = mujoco.mj_ray(sim.model, scratch, origin, cam_rot[:, 2], None, 1, -1, geomid)
            if not (0.0 <= hit < 0.25 and int(geomid[0]) not in perception.registry):
                min_safe_dolly = float(d)
                break
    elevated_tops = [
        float(pts[:, 2].max()) for pts in surfaces.values()
        if len(pts) and float(pts[:, 2].min()) > 0.30 and float(pts[:, 2].max() - pts[:, 2].min()) < 0.45
    ]
    min_safe_height = limits['height'][0]
    if elevated_tops and shot.get('perspective') != 'low_angle':
        min_safe_height = max(limits['height'][0], min(limits['height'][1] - 0.1, max(elevated_tops) + 0.04))
    return base_cam, min_safe_dolly, min_safe_height


def _ray_occlusion_penalty(sim, scratch, perception, cam_pos, cam_forward, target_sample_points):
    """Cast rays from camera lens to target sample points to penalize rig or obstacle occlusion."""
    origin = cam_pos + 0.04 * cam_forward
    geomid = np.zeros(1, dtype=np.int32)
    penalty = 0.0
    for key, sample_pts in target_sample_points.items():
        for pt in sample_pts:
            vec = pt - origin
            dist_pt = float(np.linalg.norm(vec))
            if dist_pt <= 0.05:
                continue
            hit = mujoco.mj_ray(sim.model, scratch, origin, vec / dist_pt, None, 1, -1, geomid)
            if 0.0 <= hit < dist_pt - 0.14:
                ident = perception.registry.get(int(geomid[0]))
                if ident is None:
                    penalty += 8.0
                elif ident.object_id != key and key not in ident.detail_ids:
                    penalty += 1.5
    return penalty


def optimize(sim, probe, shot, scene, surfaces, *, shortlist=8, progress=None):
    """Forward-kinematic proposals use private MjData, never the execution state.

    Shortlisted configurations are then reached through actuators in a separate probe
    simulator, and ranked using actual rendered segmentation/depth and framing.
    """
    from .simulation import Observation
    if shortlist < 1:
        raise ValueError('At least one rendered candidate is required')
    reference = sim.capture()
    perception = SegmentationPerception(geometry_catalog(sim))
    baseline = evaluate(shot, scene, surfaces, reference, perception.detect(reference), perception.registry)
    limits = safe_limits(sim)
    axes = sim.axes
    current = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in axes}
    allow_dutch = shot.get('horizon', 'level') == 'dutch'
    scratch = mujoco.MjData(sim.model)
    scratch.mocap_pos[:] = sim.data.mocap_pos
    scratch.mocap_quat[:] = sim.data.mocap_quat
    ranked = {}

    base_cam, min_safe_dolly, min_safe_height = _discover_workspace(
        sim, scratch, perception, limits, surfaces, shot
    )
    target_sample_points = {}
    for t in shot['targets']:
        pts = surfaces[t['object_id']]
        center_pt = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        target_sample_points[t['object_id']] = [
            center_pt,
            pts[pts[:, 2].argmax()],
            pts[pts[:, 2].argmin()],
            pts[pts[:, 0].argmin()],
            pts[pts[:, 0].argmax()],
        ]
    is_closeup_shot = shot.get('shot_type') in ('close_up', 'detail') or 'as close' in shot.get('description', '').lower()

    def rank(target):
        sim.checkpoint()
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
        eval_res = evaluate(shot, scene, surfaces, obs)
        score = eval_res['loss']
        if not eval_res['satisfies_observed_criteria']:
            score += 4.0 + 1.5 * len(eval_res['failures'])
        score += _ray_occlusion_penalty(sim, scratch, perception, transform[:3, 3], transform[:3, 2], target_sample_points)
        if target['height'] < min_safe_height:
            score += 10.0 * (min_safe_height - target['height'])
        if 'dolly' in target and target['dolly'] < min_safe_dolly:
            score += 20.0 * (min_safe_dolly - target['dolly'])
        if is_closeup_shot and 'dolly' in axes:
            dolly_gap = (limits['dolly'][1] - target['dolly']) / (limits['dolly'][1] - limits['dolly'][0])
            score += 0.45 * dolly_gap
        travel = sum(abs(target[n]-current[n])/(limits[n][1]-limits[n][0]) for n in axes)
        ranked[key] = (score + 0.05*travel, target)

    rank(current)
    preferred = scene['viewpoints'][shot['preferred_view_id']]['joints']
    rank(preferred)
    rolls = [-0.15, 0., 0.15] if allow_dutch else [0.0]
    heights_grid = np.linspace(min_safe_height, limits['height'][1], 4)
    for values in product(heights_grid, np.linspace(*limits['pan'], 7),
                          np.linspace(*limits['tilt'], 7), rolls):
        rank(dict(zip(('height', 'pan', 'tilt', 'roll'), values), truck=0.0, dolly=max(0.0, min_safe_dolly)))
    fovy_rad = math.radians(float(sim.model.cam_fovy[sim.camera_id]))
    fovx_rad = 2 * math.atan(math.tan(fovy_rad / 2) * (sim.width / sim.height))
    tan_half_y = math.tan(fovy_rad / 2)
    tan_half_x = math.tan(fovx_rad / 2)
    if {'truck', 'dolly'} <= set(axes) and shot['targets']:
        aim_targets = list(shot['targets'])
        for raw_aim in aim_targets:
            aim = raw_aim
            pts = surfaces[aim['object_id']]
            center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            span_z = max(0.12, float(pts[:, 2].max() - pts[:, 2].min()))
            ideal_dist = span_z / max(0.08, 2.0 * aim['frame_height_fraction'] * tan_half_y)
            dolly_samples = set(np.round(np.linspace(min_safe_dolly, limits['dolly'][1], 6), 4))
            is_person = scene.get('objects', {}).get(aim['object_id'], {}).get('category') == 'person'
            cy_offsets = (0.0, -0.025, 0.025) if is_person else (0.0,)
            max_search_height = (
                min(limits['height'][1], max(min_safe_height + 0.15, float(center[2]) + 0.55))
                if is_closeup_shot else limits['height'][1]
            )
            for truck in np.linspace(*limits['truck'], 5):
                dx = float(center[0] - (base_cam[0] + truck))
                rem = max(0.2, ideal_dist * ideal_dist - dx * dx)
                ideal_dolly = float(np.clip(center[1] - math.sqrt(rem) - base_cam[1], min_safe_dolly, limits['dolly'][1]))
                for dolly in sorted(dolly_samples | {round(ideal_dolly, 4), round(limits['dolly'][1], 4)}):
                    for height in np.linspace(min_safe_height, max_search_height, 5):
                        dy = float(center[1] - (base_cam[1] + dolly))
                        dz = float(center[2] - (base_cam[2] + height))
                        dist_xy = max(1e-3, math.hypot(dx, dy))
                        pan_look = math.atan2(-dx, dy) + math.atan((aim['center_x'] - 0.5) * 2.0 * tan_half_x)
                        for cy_off in cy_offsets:
                            tilt_look = math.atan2(dz, dist_xy) + math.atan((aim['center_y'] + cy_off - 0.5) * 2.0 * tan_half_y)
                            rank(dict(height=height, truck=truck, dolly=dolly, pan=pan_look, tilt=tilt_look, roll=0.0))
    for step in (0.12, 0.04):
        best = min(ranked.values(), key=lambda item: item[0])[1]
        for dh, dp, dt in product([-step, 0, step], repeat=3):
            candidate = dict(best, height=max(min_safe_height, best['height']+dh), pan=best['pan']+dp, tilt=best['tilt']+dt)
            if not allow_dutch:
                candidate['roll'] = 0.0
            rank(candidate)
        if {'truck', 'dolly'} <= set(axes):
            best = min(ranked.values(), key=lambda item: item[0])[1]
            for d_truck, d_dolly, dp, dt in product([-step*2, 0, step*2], [-step*2, 0, step*2], [-step, 0, step], [-step, 0, step]):
                rank(dict(best, truck=best['truck']+d_truck, dolly=max(min_safe_dolly, best['dolly']+d_dolly),
                          pan=best['pan']+dp, tilt=best['tilt']+dt, roll=0.0))
    candidates = sorted(ranked.values(), key=lambda item: item[0])[:shortlist]
    probe.model.body_quat[:] = sim.model.body_quat
    probe.model.geom_pos[:] = sim.model.geom_pos
    probe.data.mocap_pos[:] = sim.data.mocap_pos
    probe.data.mocap_quat[:] = sim.data.mocap_quat
    best_target, best_score = current, baseline
    best_objective = (not baseline['satisfies_observed_criteria'], baseline['loss'] + (0.45 if is_closeup_shot else 0.0))
    if not allow_dutch and (abs(current['roll']) > 1e-8 or abs(reference.joints['roll']) > 0.01):
        best_objective = (True, float('inf'))
    checked = []

    def check_rendered(target, predicted):
        nonlocal best_target, best_score, best_objective
        from .safety import PathBlocked
        try:
            sim.path_validator.validate(sim.configuration, target)
            move_to(probe, target)
        except PathBlocked as error:
            checked.append(dict(target=target, rejected_path=str(error)))
            return None, None
        obs = probe.capture()
        result = evaluate(shot, scene, surfaces, obs, perception.detect(obs), perception.registry)
        travel = sum(abs(target[n]-current[n])/(limits[n][1]-limits[n][0]) for n in axes)
        closeup_penalty = (
            0.45 * (limits['dolly'][1] - target.get('dolly', 0.0)) / (limits['dolly'][1] - limits['dolly'][0])
            if is_closeup_shot and 'dolly' in axes else 0.0
        )
        objective = (not result['satisfies_observed_criteria'], result['loss'] + closeup_penalty + 0.05*travel)
        checked.append(dict(target=target, predicted_loss=predicted, measured=result))
        # Never choose a prettier prediction over a better measured initial view.
        if objective < best_objective:
            best_target, best_score, best_objective = target, result, objective
        return obs, result

    for index, (predicted, target) in enumerate(candidates):
        check_rendered(target, predicted)
        if progress:
            progress(f"{shot['shot_id']}: checked candidate {index+1}/{len(candidates)}")

    # Closed-loop pan/tilt servo correction in probe to eliminate residual framing offset
    # between sparse surface sample projection and the full rendered segmentation mask bbox.
    if shot['targets']:
        primary_spec = max(
            shot['targets'],
            key=lambda t: (
                float(np.linalg.norm(surfaces[t['object_id']].max(axis=0) - surfaces[t['object_id']].min(axis=0)))
                if t['object_id'] in surfaces and len(surfaces[t['object_id']]) else 0.0
            ),
        )
        m_box = best_score['targets'].get(primary_spec['object_id'], {}).get('framing')
        if m_box is not None and (
            abs(m_box['center_x'] - primary_spec['center_x']) > 0.002
            or abs(m_box['center_y'] - primary_spec['center_y']) > 0.002
        ):
            d_pan = math.atan((primary_spec['center_x'] - m_box['center_x']) * 2.0 * tan_half_x)
            d_tilt = math.atan((primary_spec['center_y'] - m_box['center_y']) * 2.0 * tan_half_y)
            refined = dict(
                best_target,
                pan=float(np.clip(best_target['pan'] + d_pan, *limits['pan'])),
                tilt=float(np.clip(best_target['tilt'] + d_tilt, *limits['tilt'])),
            )
            if not allow_dutch and 'roll' in refined:
                refined['roll'] = 0.0
            check_rendered(refined, best_score['loss'])

    if not allow_dutch and abs(best_target['roll']) > 1e-8:
        from .safety import PathBlocked
        raise PathBlocked('No collision-free level-horizon candidate is reachable')
    return dict(target=best_target, before=baseline, predicted_after=best_score,
                geometric_candidates=len(ranked), rendered_candidates=checked)


def motion_matches(start, end, motion):
    """Named translations have explicit rig-coordinate directions; sweep is free-form."""
    directions = {'push_in': ('dolly', 1), 'pull_back': ('dolly', -1),
                  'truck_right': ('truck', 1), 'truck_left': ('truck', -1),
                  'pedestal_up': ('height', 1), 'pedestal_down': ('height', -1)}
    if motion not in directions:
        return motion == 'sweep'
    axis, sign = directions[motion]
    return axis in start and sign * (end[axis]-start[axis]) >= .015


def plan_sweep(sim, shot, scene, surfaces):
    """Select a smooth, framing-preserving continuous camera sweep across the shot duration."""
    from .control import RATES
    from .simulation import Observation
    from .motion import validate as validate_motion
    contract = validate_motion(shot, sim.limits)
    start = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in sim.axes}
    if shot.get('motion', 'static') == 'static':
        return dict(start=start, end=dict(start), displacement={n: 0.0 for n in start},
                    continuous=False, reason='static_requested')
    reference = sim.capture()
    perception = SegmentationPerception(geometry_catalog(sim))
    measured_start = evaluate(shot, scene, surfaces, reference, perception.detect(reference), perception.registry)
    if not measured_start['satisfies_observed_criteria']:
        return dict(start=start, end=dict(start), displacement={n: 0.0 for n in start},
                    continuous=False, reason='starting_composition_unsatisfied')
    limits = safe_limits(sim)
    axes = sim.axes
    start = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in axes}
    duration = float(shot.get('duration_seconds', 2.0))
    max_delta = {n: RATES[n] * duration / 1.875 * 0.85 for n in axes}
    scratch = mujoco.MjData(sim.model)
    scratch.mocap_pos[:] = sim.data.mocap_pos
    scratch.mocap_quat[:] = sim.data.mocap_quat
    sag = {n: start[n] - sim.configuration[n] for n in axes}
    base_cam, min_safe_dolly, min_safe_height = _discover_workspace(
        sim, scratch, perception, limits, surfaces, shot
    )
    target_sample_points = {
        t['object_id']: [0.5 * (surfaces[t['object_id']].min(axis=0) + surfaces[t['object_id']].max(axis=0)),
                         surfaces[t['object_id']][surfaces[t['object_id']][:, 2].argmax()]]
        for t in shot['targets']
    }
    criteria = shot['completion']
    framing_bias = {}

    def raw_eval_pose(pose):
        for name, value in pose.items():
            scratch.joint(name).qpos[0] = value - sag[name]
        mujoco.mj_forward(sim.model, scratch)
        transform = np.eye(4)
        transform[:3, :3] = scratch.cam_xmat[sim.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = scratch.cam_xpos[sim.camera_id]
        obs = Observation(reference.rgb, reference.depth_m, reference.segmentation, reference.intrinsics,
                          transform, float(scratch.time), pose)
        res = evaluate(shot, scene, surfaces, obs)
        occ = _ray_occlusion_penalty(sim, scratch, perception, transform[:3, 3], transform[:3, 2], target_sample_points)
        res['occlusion_clear'] = occ == 0
        if occ > 0:
            res['loss'] += occ
            res['satisfies_observed_criteria'] = False
        return res, obs

    raw_start, _ = raw_eval_pose(start)
    for raw_t in shot['targets']:
        key = raw_t['object_id']
        m_t = measured_start['targets'].get(key, {})
        p_t = raw_start['targets'].get(key, {})
        if m_t.get('framing') is not None and p_t.get('framing') is not None:
            framing_bias[key] = (
                m_t['framing']['center_x'] - p_t['framing']['center_x'],
                m_t['framing']['center_y'] - p_t['framing']['center_y'],
                m_t['framing']['frame_height_fraction'] - p_t['framing']['frame_height_fraction'],
                m_t['headroom_error'] - p_t['headroom_error'],
            )

    def eval_pose(pose):
        res, obs = raw_eval_pose(pose)
        if not framing_bias:
            return res
        reasons = [r for r in res['failures'] if r.endswith('background missing') or r.endswith('visibility') or r.startswith('horizon:')]
        occluded = not res['occlusion_clear']
        loss_delta = 0.0
        for raw_t in shot['targets']:
            key = raw_t['object_id']
            if key not in framing_bias or res['targets'][key]['framing'] is None:
                continue
            bx, by, bh, bhead = framing_bias[key]
            p_box = res['targets'][key]['framing']
            cal_box = dict(
                center_x=p_box['center_x'] + bx,
                center_y=p_box['center_y'] + by,
                frame_height_fraction=p_box['frame_height_fraction'] + bh,
            )
            eff_t = raw_t
            old_m = res['targets'][key]
            c_err = math.hypot(cal_box['center_x'] - eff_t['center_x'], cal_box['center_y'] - eff_t['center_y'])
            s_err = abs(cal_box['frame_height_fraction'] - eff_t['frame_height_fraction'])
            h_err = max(0.0, old_m['headroom_error'] + bhead) if scene['objects'][key]['category'] == 'person' else 0.0
            loss_delta += (4 * (c_err - old_m['center_error'])
                           + 3 * (s_err - old_m['scale_error'])
                           + 2 * (h_err - old_m['headroom_error']))
            res['targets'][key].update(framing=cal_box, center_error=c_err, scale_error=s_err, headroom_error=h_err)
            if c_err > criteria['max_center_error']:
                reasons.append(f'{key}: position')
            if s_err > criteria['max_scale_error']:
                reasons.append(f'{key}: scale')
            if h_err > criteria['headroom_tolerance']:
                reasons.append(f'{key}: headroom')
        res['loss'] = float(max(0.0, res['loss'] + loss_delta))
        res['failures'] = reasons
        res['satisfies_observed_criteria'] = not reasons and not occluded
        return res

    start_eval = eval_pose(start)
    tight_margin = any(
        m['scale_error'] > 0.70 * criteria['max_scale_error'] or m['center_error'] > 0.70 * criteria['max_center_error']
        for m in measured_start['targets'].values()
    )
    ultra_tight = min(criteria['max_center_error'], criteria['max_scale_error']) <= 0.025
    loss_budget = 0.005 if (tight_margin or ultra_tight) else 0.05
    best_end = dict(start)
    best_score = -float('inf')
    is_close_up = shot.get('shot_type') in ('close_up', 'detail')

    def consider(end):
        nonlocal best_score, best_end
        if not motion_matches(start, end, shot.get('motion', 'static')):
            return
        if 'dolly' in end and end['dolly'] < min_safe_dolly:
            return
        if end['height'] < min_safe_height:
            return
        disp = (abs(end['height'] - start['height'])
                + abs(end.get('truck', 0.0) - start.get('truck', 0.0))
                + abs(end.get('dolly', 0.0) - start.get('dolly', 0.0))
                + 0.6 * abs(end['pan'] - start['pan']) + 0.6 * abs(end['tilt'] - start['tilt']))
        if disp < 0.015:
            return
        if ultra_tight and disp > 0.12:
            return
        losses = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            wp = {n: start[n] + frac * (end[n] - start[n]) for n in axes}
            res = eval_pose(wp)
            if start_eval['satisfies_observed_criteria'] and not res['satisfies_observed_criteria']:
                return
            if measured_start['satisfies_observed_criteria']:
                tol_factor = 0.65 if ultra_tight else 0.82
                for m in res['targets'].values():
                    if (m['center_error'] > tol_factor * criteria['max_center_error']
                            or m['scale_error'] > tol_factor * criteria['max_scale_error']
                            or m['headroom_error'] > 0.85 * criteria['headroom_tolerance']):
                        return
            if res['loss'] > start_eval['loss'] + loss_budget:
                return
            losses.append(res['loss'])
        mean_drift = max(0.0, float(np.mean(losses)) - start_eval['loss'])
        bonus = 0.0
        if is_close_up:
            # Prefer rising and pulling back on close-up reveal sweeps.
            if end['height'] > start['height'] + 0.04:
                bonus += 0.06
            if end.get('dolly', 0.0) < start.get('dolly', 0.0) - 0.04:
                bonus += 0.06
        else:
            if 'truck' in axes and abs(end['truck'] - start['truck']) > 0.12:
                bonus += 0.05
            if abs(end['height'] - start['height']) > 0.04:
                bonus += 0.04
        score = min(disp, 0.52) + bonus - 2.0 * mean_drift
        if score > best_score:
            best_score = score
            best_end = end

    from .motion import DIRECTIONS
    kind = shot.get('motion', 'static')
    directions = list(DIRECTIONS.values()) if kind == 'sweep' else [DIRECTIONS[kind]]
    lock = contract['subject_lock']
    center = None if lock == 'none' else .5*(surfaces[lock].min(axis=0)+surfaces[lock].max(axis=0))
    for axis, sign, _ in directions:
        if axis not in start:
            continue
        end = dict(start)
        end[axis] += sign*contract['amplitude_m']
        if not limits[axis][0] <= end[axis] <= limits[axis][1]:
            continue  # Never silently shrink a requested amplitude.
        if abs(end[axis]-start[axis]) > max_delta[axis]:
            continue
        if center is not None:
            old_pos = reference.world_from_camera[:3,3]
            new_pos = old_pos.copy()
            new_pos[{'truck':0,'dolly':1,'height':2}[axis]] += sign*contract['amplitude_m']
            old_vec, new_vec = center-old_pos, center-new_pos
            end['pan'] += math.atan2(-new_vec[0],new_vec[1])-math.atan2(-old_vec[0],old_vec[1])
            end['tilt'] += math.atan2(new_vec[2],np.linalg.norm(new_vec[:2]))-math.atan2(old_vec[2],np.linalg.norm(old_vec[:2]))
        if any(not limits[n][0] <= end[n] <= limits[n][1] or abs(end[n]-start[n]) > max_delta[n] for n in axes):
            continue
        consider(end)
    if best_end != start:
        from .safety import PathBlocked
        try:
            sim.path_validator.validate(sim.configuration,best_end)
            path = sim.path_validator.validate(start,best_end)
        except PathBlocked:
            best_end = dict(start)
            path = dict(certified=False, reason='requested_path_blocked')
    else:
        path = dict(certified=False, reason='no_framing_preserving_path_at_requested_amplitude')
    displacement = {n: round(best_end[n] - start[n], 5) for n in axes}
    return dict(start=start, end=best_end, displacement=displacement,
                continuous=any(abs(v) > 1e-4 for v in displacement.values()),
                reason='framing_preserving_search', contract=contract, path_validation=path)

