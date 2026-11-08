"""Validated shot execution and measured completion reports; no model calls."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from .control import move_to, step_sweep
from .exploration import _write_json
from .optimization import evaluate, geometry_catalog, optimize, plan_sweep
from .perception import SegmentationPerception
from .planning import parse_plan
from .simulation import Simulation


def sample_surface(points, maximum=256):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Each object needs finite Nx3 observed surface points')
    indices = np.unique(np.concatenate([np.linspace(0, len(points)-1, min(maximum-6, len(points)), dtype=int),
                                        points.argmin(axis=0), points.argmax(axis=0)]))
    return points[indices].copy()


def film_shot(sim, probe, shot, scene, surfaces, perception, *, trace=None, show=None,
              progress=None, max_recoveries=2, on_sample=None):
    """Sample the shot, interrupt drifting motion, and allow bounded actuator recovery.

    Recovery time is excluded from recording time. Completion requires a satisfied
    terminal hold, not a good interval earlier in a subsequently spoiled shot.
    """
    initial_joints = sim.capture().joints.copy()
    sweep = plan_sweep(sim, shot, scene, surfaces)
    initial_sweep = sweep
    recorded = stable = longest = failed_for = 0.0
    samples, recoveries = [], []
    motion_requested = shot.get('motion', 'static') != 'static'
    motion_executed = False
    stop = 'duration_complete'
    while recorded < shot['duration_seconds'] - 1e-8:
        dt = min(0.1, shot['duration_seconds'] - recorded)
        before_time = float(sim.data.time)
        step_sweep(sim, sweep['start'], sweep['end'], recorded, dt,
                   shot['duration_seconds'], trace)
        elapsed = float(sim.data.time) - before_time
        recorded += elapsed
        motion_executed |= bool(sweep['continuous'])
        obs = sim.capture()
        result = evaluate(shot, scene, surfaces, obs, perception.detect(obs), perception.registry)
        passed = result['satisfies_observed_criteria']
        stable = stable + elapsed if passed else 0.0
        failed_for = 0.0 if passed else failed_for + elapsed
        longest = max(longest, stable)
        samples.append(dict(time=float(sim.data.time), recorded_seconds=recorded,
                            joints=obs.joints, metrics=result))
        if on_sample:
            on_sample(obs, result, recorded)
        if show:
            show(sim)
        if failed_for + 1e-8 < 0.3:
            continue
        if len(recoveries) >= max_recoveries:
            stop = 'recovery_budget_exhausted'
            break
        if progress:
            progress(f"{shot['shot_id']}: framing drift detected; recovering camera ({len(recoveries)+1}/{max_recoveries})")
        # Freeze setpoints, refresh observed moving subjects, then reoptimise the same objective.
        for detection in perception.detect(obs):
            if detection.identity.dynamic and detection.identity.object_id in surfaces:
                surfaces[detection.identity.object_id] = sample_surface(detection.points_world)
        search = optimize(sim, probe, shot, scene, surfaces, progress=progress)
        move_to(sim, search['target'], trace=trace, callback=show)
        recoveries.append(dict(recorded_seconds=recorded, failures=result['failures'], search=search))
        controls = {n: float(sim.data.ctrl[sim.model.actuator(n).id]) for n in sim.axes}
        # Stop the failed sweep; recovered footage holds the subject instead of repeating the path.
        sweep = dict(start=controls, end=dict(controls), continuous=False)
        stable = failed_for = 0.0
    final = sim.capture()
    measured = evaluate(shot, scene, surfaces, final, perception.detect(final), perception.registry)
    translation_axes = [n for n in ('height','truck','dolly') if n in initial_joints]
    translation_matches = all(
        abs((final.joints[n]-initial_joints[n])-(initial_sweep['end'][n]-initial_sweep['start'][n]))
        <= .005 + .1*abs(initial_sweep['end'][n]-initial_sweep['start'][n]) for n in translation_axes)
    motion_completed = (not motion_requested or (motion_executed and not recoveries and translation_matches and all(
        abs(final.joints[n]-initial_sweep['end'][n]) <= (.04 if n=='height' else .02)
        for n in initial_sweep['end'])))
    completed = (stop == 'duration_complete' and measured['satisfies_observed_criteria']
                 and stable + 1e-8 >= shot['completion']['hold_seconds']
                 and motion_completed)
    return dict(sweep=initial_sweep, samples=samples, recoveries=recoveries,
                terminal_satisfied_seconds=stable, longest_satisfied_seconds=longest,
                completed=bool(completed), after=measured, stop_reason=stop,
                motion_requested=motion_requested, motion_executed=motion_executed, motion_completed=motion_completed,
                model_replan_needed=not completed)


def execute_plan(sim, plan_path, scene_path, output, *, progress=print, preview=False, refine=False, max_adjustments=3, on_frame=None, on_event=None, perception=None):
    if not 0 <= max_adjustments <= 6:
        raise ValueError('Refinement budget must be 0–6')
    plan_path, scene_path, output = Path(plan_path), Path(scene_path), Path(output)
    scene = json.loads(scene_path.read_text())
    from .artifacts import require_compatible
    require_compatible(scene.get('artifact_identity'))
    plan = parse_plan(plan_path.read_text(), scene, list(scene['viewpoints']))
    from .motion import validate as validate_motion
    for shot in plan['shots']:
        validate_motion(shot, sim.limits)
    report_path = plan_path.parent/'planning_report.json'
    if report_path.exists():
        planning_report = json.loads(report_path.read_text())
        expected = planning_report.get('scene_map_sha256')
        if expected and hashlib.sha256(scene_path.read_bytes()).hexdigest() != expected:
            raise ValueError('Plan was generated from a different scene map')
    needed = {t['object_id'] for shot in plan['shots'] for t in shot['targets']}
    needed.update(key for shot in plan['shots'] for key in shot['background_object_ids'])
    with np.load(scene_path.parent/'object_surfaces.npz', allow_pickle=False) as arrays:
        surfaces = {key: sample_surface(arrays[key]) for key in needed}
    perception = perception or SegmentationPerception(geometry_catalog(sim))
    if not needed <= {key for identity in perception.registry.values() for key in (identity.object_id,*identity.detail_ids)}:
        raise ValueError('Plan references objects not configured in this simulator')
    if not {'height', 'pan', 'tilt', 'roll'} <= set(sim.axes):
        raise ValueError('Rig must provide at least height, pan, tilt and roll axes')
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'validated_plan.json', plan)
    report = dict(status='running', shots=[],
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  scene_map_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                  physical_limits=sim.limits,
                  completion_scope='observed geometry only; visibility is a sparse-surface proxy, not full object area',
                  human_instructions_executed=False)
    trace = []
    from .recording import Recorder, Stopped
    sequence_video = shot_video = None
    def event(kind, **values):
        if on_event:
            on_event(dict(kind=kind, **values))
    def show(simulator, instruction=''):
        if on_frame is not None:
            on_frame(simulator, instruction)
        if preview:
            obs = simulator.capture()
            image = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
            if instruction:
                import textwrap
                for line, text in enumerate(textwrap.wrap(instruction, 58)):
                    cv2.putText(image, text, (10, 30+line*25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 2)
            cv2.imshow('Robopod shot execution', image)
            if cv2.waitKey(1) & 0xff == ord('q'):
                raise KeyboardInterrupt('Execution cancelled in preview')
    try:
        sequence_video = Recorder(output/'sequence.mp4', sim.width, sim.height)
        # Refresh movable targets from the current, usually broad, initial view.
        # Do not accumulate different-time dynamic geometry or query hidden body poses.
        move_to(sim, {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in sim.axes}, trace=trace)
        initial = sim.capture()
        for detection in perception.detect(initial):
            key = detection.identity.object_id
            if key in surfaces and detection.identity.dynamic:
                surfaces[key] = sample_surface(detection.points_world)
        np.savez_compressed(output/'reference_surfaces.npz', **surfaces)
        with Simulation(sim.width, sim.height, float(sim.model.cam_fovy[sim.camera_id])) as probe:
            probe.checkpoint = sim.checkpoint
            for index, shot in enumerate(plan['shots']):
                folder = output/f'shot_{index:03d}'
                folder.mkdir()
                event('shot_started', shot_id=shot['shot_id'], index=index, total=len(plan['shots']))
                before = sim.save(folder/'before')
                if progress:
                    progress(f"{shot['shot_id']}: searching reachable configurations")
                search = optimize(sim, probe, shot, scene, surfaces, progress=progress)
                motion = move_to(sim, search['target'], trace=trace, callback=show)
                refinement = None
                if refine:
                    from .refinement import refine_shot
                    refinement = refine_shot(sim, probe, shot, scene, surfaces, perception, folder,
                                             max_adjustments=max_adjustments, progress=progress,
                                             show=show, trace=trace)
                    report['human_instructions_executed'] |= bool(refinement['rounds'])
                    show(sim)
                shot_video = Recorder(folder/'shot.mp4', sim.width, sim.height)
                def sample(obs, metrics, seconds):
                    shot_video.write(obs)
                    sequence_video.write(obs)
                    event('measurement', shot_id=shot['shot_id'], metrics=metrics,
                          seconds=seconds, duration=shot['duration_seconds'], criteria=shot['completion'])
                try:
                    filming = film_shot(sim, probe, shot, scene, surfaces, perception,
                                        trace=trace, show=show, progress=progress, on_sample=sample)
                finally:
                    shot_video.close()
                    shot_video = None
                motion['continuous_sweep'] = filming['sweep']
                after = sim.save(folder/'after')
                measured = filming['after']
                completed = filming['completed']
                if not refine and shot['human_instructions']:
                    completed = False  # Unexecuted human actions cannot count as completed.
                if refinement and refinement['pending_creative']:
                    completed = False
                status = 'satisfied_observed_criteria' if completed else 'needs_adjustment'
                advice = []
                if not completed:
                    for key, metric in measured['targets'].items():
                        actual = metric['framing']
                        t_spec = next(t for t in shot['targets'] if t['object_id'] == key)
                        label = scene['objects'][key]['label']
                        if metric['scale_error'] > shot['completion']['max_scale_error']:
                            direction = 'closer to' if actual is None or actual['frame_height_fraction'] < t_spec['frame_height_fraction'] else 'farther from'
                            advice.append(f'Move {label} {direction} the camera, then reobserve.')
                        if metric['center_error'] > shot['completion']['max_center_error'] and actual is not None:
                            side = 'left' if actual['center_x'] < t_spec['center_x'] else 'right'
                            advice.append(f'Shift {label} laterally ({side} in frame) toward ({t_spec["center_x"]:.2f}, {t_spec["center_y"]:.2f}), then reobserve.')
                        if metric['visible_surface_fraction'] < shot['completion']['min_visible_fraction']:
                            advice.append(f'Clear occlusion on {label} by separating foreground/background subjects.')
                    if not advice:
                        advice.append('Reposition the subject/product or relax the composition requirements, then reobserve.')
                item = dict(shot_id=shot['shot_id'], status=status, search=search, motion=motion,
                            after=measured, hold_samples=filming['samples'],
                            longest_satisfied_seconds=filming['longest_satisfied_seconds'],
                            terminal_satisfied_seconds=filming['terminal_satisfied_seconds'],
                            camera_recoveries=filming['recoveries'], filming_stop_reason=filming['stop_reason'],
                            model_replan_needed=not completed, motion_completed=filming['motion_completed'],
                            subject_repositioning_requests=advice,
                            creative_instructions=shot['human_instructions'], refinement=refinement,
                            limitation='Bounded search cannot prove global infeasibility. Scripted actions are limited to the documented vocabulary; full human biomechanics are not simulated.')
                _write_json(folder/'result.json', item)
                comparison = np.concatenate([before.rgb, after.rgb], axis=1)
                comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                for x, label in [(10, 'Before'), (sim.width+10, 'After')]:
                    cv2.putText(comparison, label, (x,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                if not cv2.imwrite(str(folder/'comparison.jpg'), comparison):
                    raise OSError('Failed to save comparison')
                report['shots'].append(dict(shot_id=shot['shot_id'], status=status, directory=folder.name,
                                           before_loss=search['before']['loss'], after_loss=measured['loss'],
                                           comparison=str((folder/'comparison.jpg').resolve()),
                                           video=str((folder/'shot.mp4').resolve())))
                event('shot_completed', result=report['shots'][-1])
                _write_json(output/'execution_report.json', report)
                if progress:
                    progress(f"{shot['shot_id']}: {status}; framing loss {search['before']['loss']:.3f} -> {measured['loss']:.3f}")
        report['status'] = 'completed' if all(s['status'] == 'satisfied_observed_criteria' for s in report['shots']) else 'needs_adjustment'
    except Stopped:
        report.update(status='stopped', error='Stopped by user; completed shots and partial footage are preserved')
        raise
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if shot_video is not None:
            shot_video.close()
        if sequence_video is not None:
            sequence_video.close()
            report['video'] = str((output/'sequence.mp4').resolve())
            report['video_frames'] = sequence_video.frames
            event('execution_finished', status=report['status'], video=report['video'], frames=sequence_video.frames)
        _write_json(output/'execution_report.json', report)
        _write_json(output/'trajectory.json', trace)
        if preview:
            cv2.destroyAllWindows()
    return report
