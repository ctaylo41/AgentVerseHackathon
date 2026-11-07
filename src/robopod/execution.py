"""Validated shot execution and measured completion reports; no model calls."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from .control import advance, move_to
from .exploration import _write_json
from .optimization import evaluate, geometry_catalog, optimize
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


def execute_plan(sim, plan_path, scene_path, output, *, progress=print, preview=False, refine=False, max_adjustments=3, on_frame=None):
    if not 0 <= max_adjustments <= 6:
        raise ValueError('Refinement budget must be 0–6')
    plan_path, scene_path, output = Path(plan_path), Path(scene_path), Path(output)
    scene = json.loads(scene_path.read_text())
    plan = parse_plan(plan_path.read_text(), scene, list(scene['viewpoints']))
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
    perception = SegmentationPerception(geometry_catalog(sim))
    if not needed <= {identity.object_id for identity in perception.registry.values()}:
        raise ValueError('Plan references objects not configured in this simulator')
    if set(sim.axes) != {'height', 'pan', 'tilt', 'roll'}:
        raise ValueError('This search currently supports the four-axis rail rig')
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'validated_plan.json', plan)
    report = dict(status='running', shots=[],
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  scene_map_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                  physical_limits=sim.limits,
                  completion_scope='observed geometry only; visibility is a sparse-surface proxy, not full object area',
                  human_instructions_executed=False)
    trace = []
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
            for index, shot in enumerate(plan['shots']):
                folder = output/f'shot_{index:03d}'
                folder.mkdir()
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
                after = sim.save(folder/'after')
                measured = evaluate(shot, scene, surfaces, after, perception.detect(after), perception.registry)
                stable, max_stable = 0.0, 0.0
                measurements = []
                # Duration is simulated time. Sample framing at 10 Hz throughout the shot.
                start = float(sim.data.time)
                while sim.data.time-start < shot['duration_seconds'] - 1e-8:
                    previous_time = float(sim.data.time)
                    advance(sim, min(0.1, shot['duration_seconds']-(sim.data.time-start)), trace)
                    obs = sim.capture()
                    result = evaluate(shot, scene, surfaces, obs, perception.detect(obs), perception.registry)
                    stable = stable + float(sim.data.time)-previous_time if result['satisfies_observed_criteria'] else 0.0
                    max_stable = max(max_stable, stable)
                    measurements.append(dict(time=float(sim.data.time), metrics=result))
                    show(sim)
                completed = max_stable + 1e-8 >= shot['completion']['hold_seconds']
                if refinement and refinement['pending_creative']:
                    completed = False
                status = 'satisfied_observed_criteria' if completed else 'needs_adjustment'
                advice = []
                if not completed:
                    for key, metric in measured['targets'].items():
                        if metric['scale_error'] > shot['completion']['max_scale_error']:
                            actual = metric['framing']
                            wanted = next(t['frame_height_fraction'] for t in shot['targets'] if t['object_id'] == key)
                            direction = 'closer to' if actual is None or actual['frame_height_fraction'] < wanted else 'farther from'
                            advice.append(f'Move {scene["objects"][key]["label"]} {direction} the camera, then reobserve; fixed rail/FOV cannot provide arbitrary shot scale.')
                    if not advice:
                        advice.append('Reposition the subject/product or relax the composition requirements, then reobserve.')
                item = dict(shot_id=shot['shot_id'], status=status, search=search, motion=motion,
                            after=measured, hold_samples=measurements, longest_satisfied_seconds=max_stable,
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
                                           before_loss=search['before']['loss'], after_loss=measured['loss']))
                _write_json(output/'execution_report.json', report)
                if progress:
                    progress(f"{shot['shot_id']}: {status}; framing loss {search['before']['loss']:.3f} -> {measured['loss']:.3f}")
        report['status'] = 'completed' if all(s['status'] == 'satisfied_observed_criteria' for s in report['shots']) else 'needs_adjustment'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write_json(output/'execution_report.json', report)
        _write_json(output/'trajectory.json', trace)
        if preview:
            cv2.destroyAllWindows()
    return report
