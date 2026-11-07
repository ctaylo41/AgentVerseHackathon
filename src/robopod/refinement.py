"""Bounded camera/subject cooperation; no hidden-state completion or LLM calls."""
import numpy as np

from .control import advance, move_to
from .exploration import _write_json
from .mapping import SceneMap
from .optimization import evaluate, optimize
from .teleprompter import Teleprompter, creative_action, correction, respond, observe_response


def refine_shot(sim, probe, shot, scene, surfaces, perception, folder, *, max_adjustments=3,
                progress=print, show=None, trace=None):
    from .execution import sample_surface
    if not 0 <= max_adjustments <= 6:
        raise ValueError('Refinement budget must be 0–6 adjustments')
    prompt = Teleprompter()
    local_map = SceneMap()
    rounds, unsupported = [], []
    creative = []
    for entry in shot['human_instructions']:
        action = creative_action(entry['subject_id'], entry['instruction'])
        if action is None:
            unsupported.append(dict(**entry, status='requires_manual_response'))
            if progress:
                progress(f"TELEPROMPTER [manual response required]: {entry['instruction']}")
            if show:
                show(sim, entry['instruction'])
        else:
            creative.append(action)
    def observe(tag):
        obs = sim.capture()
        detections = perception.detect(obs)
        local_map.update(detections, obs, tag)
        return obs, detections
    obs, detections = observe('initial')
    metrics = evaluate(shot, scene, surfaces, obs, detections, perception.registry)
    initial_metrics = metrics
    stop = 'budget_exhausted'
    for index in range(max_adjustments):
        if creative:
            instruction = creative.pop(0)
        elif metrics['satisfies_observed_criteria']:
            stop = 'criteria_satisfied'
            break
        else:
            instruction = correction(shot, metrics, scene)
        if instruction is None:
            stop = 'no_supported_correction'
            break
        if sim.data.time < prompt.ready_at:
            advance(sim, prompt.ready_at-sim.data.time, trace)
        if not prompt.issue(instruction, float(sim.data.time)):
            stop = 'repeat_or_direction_change_blocked'
            break
        if progress:
            progress(f'TELEPROMPTER [{instruction.source}]: {instruction.text}')
        _write_json(folder/'teleprompter.json', dict(active=prompt.history[-1], history=prompt.history,
                                                   unsupported=unsupported))
        old = detections
        old_metrics = metrics
        def display(state):
            if show:
                show(state, instruction.text)
        applied = respond(sim, instruction, callback=display, trace=trace)
        obs, detections = observe(f'response_{index}')
        observed, evidence = observe_response(old, detections, instruction)
        prompt.finish(float(sim.data.time), observed, evidence)
        # Replace only changed geometry with fresh observed pixels, never scripted target coordinates.
        for detection in detections:
            key = detection.identity.object_id
            if key in surfaces and detection.identity.dynamic:
                surfaces[key] = sample_surface(detection.points_world)
                scene['objects'][key] = local_map.objects[key].copy()
        sim.save(folder/f'response_{index}', observation=obs)
        response_metrics = evaluate(shot, scene, surfaces, obs, detections, perception.registry)
        if observed:
            search = optimize(sim, probe, shot, scene, surfaces, progress=progress)
            move_to(sim, search['target'], trace=trace, callback=lambda state: display(state))
            obs, detections = observe(f'camera_{index}')
            metrics = evaluate(shot, scene, surfaces, obs, detections, perception.registry)
            sim.save(folder/f'refined_{index}', observation=obs)
        else:
            metrics = response_metrics
        rounds.append(dict(instruction=prompt.history[-1].copy(), scripted_response=applied,
                           before=old_metrics, after_response=response_metrics, after_camera=metrics))
        if not observed:
            stop = 'response_not_observed'
            break
        if instruction.source == 'geometric' and metrics['loss'] >= old_metrics['loss']-.01:
            stop = 'no_progress'
            break
    if metrics['satisfies_observed_criteria'] and not creative:
        stop = 'criteria_satisfied'
    pending = unsupported + [dict(subject_id=a.object_id, instruction=a.text, status='budget_exhausted') for a in creative]
    if any(h['source']=='creative' and h['status']!='completed' for h in prompt.history):
        pending += [dict(instruction=h['text'], status=h['status']) for h in prompt.history if h['source']=='creative' and h['status']!='completed']
    result = dict(stop_reason=stop, rounds=rounds, instructions=prompt.history, pending_creative=pending,
                  before=initial_metrics, after=metrics,
                  changed_scene=bool(rounds), model_replan_needed=not metrics['satisfies_observed_criteria'])
    _write_json(folder/'teleprompter.json', dict(active=None, history=prompt.history, pending_creative=pending))
    _write_json(folder/'refinement.json', result)
    local_map.save(folder)
    np.savez_compressed(folder/'refined_reference_surfaces.npz', **surfaces)
    return result
