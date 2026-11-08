"""Bounded local-model escalation with current images and immutable shot objectives."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import queue
import threading

import cv2
import numpy as np

from .exploration import _write_json
from .gemma import OllamaClient, SYSTEM_PROMPT
from .mapping import SceneMap
from .planning import array, obj, parse_plan, scene_plan_schema, text


def _fallback_viewpoint(old_view, scene):
    views = sorted(scene['viewpoints'])
    alternatives = [v for v in views if v != old_view]
    if not alternatives:
        return old_view
    old_joints = (scene['viewpoints'].get(old_view) or {}).get('joints') or {}
    if old_joints:
        return min(
            alternatives,
            key=lambda vid: sum(
                abs(float(((scene['viewpoints'].get(vid) or {}).get(k, 0.0)) - float(old_joints.get(k, 0.0)))
                for k in ('yaw', 'pitch', 'height', 'roll')
            ),
        )
    return alternatives[0]


def _live_revision_schema(plan, scene, no_human_actions):
    views = sorted(scene['viewpoints'])
    dynamic = sorted(k for k, v in scene['objects'].items() if v.get('category') in ('person', 'dynamic_object'))
    instruction_item = (
        obj({'subject_id': dict(type='string', enum=dynamic), 'instruction': text(240)})
        if dynamic else obj({'subject_id': text(40), 'instruction': text(240)})
    )
    shot_ids = [shot['shot_id'] for shot in plan['shots']]
    return obj({
        'shots': array(
            obj({
                'shot_id': dict(type='string', enum=shot_ids),
                'preferred_view_id': dict(type='string', enum=views),
                'human_instructions': array(instruction_item, 0, 0 if (no_human_actions or not dynamic) else 2),
                'feasibility_notes': text(240),
            }),
            len(plan['shots']),
            len(plan['shots']),
        )
    })


def _merge_live_revision(plan, raw_text, scene, no_human_actions):
    merged = copy.deepcopy(plan)
    by_id = {}
    if raw_text:
        try:
            payload = json.loads(raw_text)
            for item in payload.get('shots') or []:
                if isinstance(item, dict) and item.get('shot_id'):
                    by_id[item['shot_id']] = item
        except Exception:
            by_id = {}
    views = set(scene['viewpoints'])
    dynamic = sorted(k for k, v in scene['objects'].items() if v.get('category') in ('person', 'dynamic_object'))
    for shot in merged['shots']:
        old_view = shot['preferred_view_id']
        old_instructions = copy.deepcopy(shot.get('human_instructions', []))
        candidate = by_id.get(shot['shot_id']) or {}
        new_view = candidate.get('preferred_view_id')
        if new_view in views:
            shot['preferred_view_id'] = new_view
        if not no_human_actions:
            combined = copy.deepcopy(old_instructions)
            for entry in candidate.get('human_instructions') or []:
                if (isinstance(entry, dict)
                        and entry.get('subject_id') in dynamic
                        and isinstance(entry.get('instruction'), str)
                        and entry['instruction'].strip()
                        and entry not in combined):
                    combined.append({'subject_id': entry['subject_id'], 'instruction': entry['instruction'].strip()})
            shot['human_instructions'] = combined[:4]
        else:
            shot['human_instructions'] = []
        notes = candidate.get('feasibility_notes')
        if isinstance(notes, str) and notes.strip():
            shot['feasibility_notes'] = notes.strip()[:240]
        if shot['preferred_view_id'] == old_view and shot['human_instructions'] == old_instructions:
            alt_view = _fallback_viewpoint(old_view, scene)
            if alt_view != old_view:
                shot['preferred_view_id'] = alt_view
                shot['feasibility_notes'] = f'Switched preferred viewpoint from {old_view} to {alt_view} after observed failure.'[:240]
            elif not no_human_actions and dynamic:
                subject_id = 'body:person' if 'body:person' in dynamic else dynamic[0]
                fallback_inst = {'subject_id': subject_id, 'instruction': 'step closer to the camera'}
                if fallback_inst not in shot['human_instructions']:
                    shot['human_instructions'].append(fallback_inst)
                    shot['feasibility_notes'] = 'Added cooperative subject adjustment after observed failure.'[:240]
    return json.dumps(merged, allow_nan=False)


def revise_plan(plan, scene, failures, image, brief, output, *, client=None, limits=None):
    """Revise execution strategy, never relax original composition/motion/completion values."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    client=client or OllamaClient()
    live_ollama=isinstance(client,OllamaClient)
    from .brief import requirements
    no_human_actions=requirements(brief or '')['human_actions_forbidden']
    if live_ollama:
        schema=_live_revision_schema(plan,scene,no_human_actions)
        scene_fields=('label','category','position_world','observed_bounds_world','visibility','confidence','source','parent_id')
    else:
        schema=scene_plan_schema(scene,list(scene['viewpoints']))
        shot_schema=schema['properties']['shots']
        shot_schema.update(minItems=len(plan['shots']),maxItems=len(plan['shots']))
        shot_schema['items']['required'] += ['motion','horizon','motion_parameters']
        mutable={'preferred_view_id','human_instructions','feasibility_notes'}
        if no_human_actions:
            shot_schema['items']['properties']['human_instructions']['maxItems']=0
        for key,field in shot_schema['items']['properties'].items():
            if key not in mutable and all(key in shot for shot in plan['shots']):
                field['enum']=[shot[key] for shot in plan['shots']]
        scene_fields=('label','category','position_world','observed_bounds_world','visibility','confidence','source','parent_id','landmarks')
    compact_scene=dict(objects={key:{k:v for k,v in item.items() if k in scene_fields}
        for key,item in scene['objects'].items()},
        viewpoints={key:dict(joints=value.get('joints'),simulation_time=value.get('simulation_time'))
                    for key,value in scene['viewpoints'].items()})
    context=dict(creative_brief=brief, failed_plan=plan, failure_evidence=failures,
                 current_scene=compact_scene, physical_limits=limits,
                 allowed_changes=['preferred_view_id','human_instructions','feasibility_notes'],
                 instruction='Keep every original human instruction; add supported cooperation if useful. '
                             'All other shot fields and objectives must remain exactly unchanged. '
                             'Return a full plan containing only these failed shots. Change the preferred viewpoint '
                             'or add a supported human instruction to address the failure. Advice only in '
                             'feasibility_notes does not execute an action. Do not claim success.')
    _write_json(output/'context.json',context)
    cv2.imwrite(str(output/'current.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    if not ok: raise ValueError('Cannot encode replan evidence')
    model=client.check_model()
    if live_ollama:
        try:
            raw_reply=client.chat([dict(role='system',content=SYSTEM_PROMPT),dict(role='user',
                content=json.dumps(context,allow_nan=False), images=[base64.b64encode(encoded).decode()])],
                schema=schema)
        except Exception as exc:
            raw_reply=json.dumps({'error':str(exc),'shots':[]})
        raw=_merge_live_revision(plan,raw_reply,scene,no_human_actions)
    else:
        raw=client.chat([dict(role='system',content=SYSTEM_PROMPT),dict(role='user',
            content=json.dumps(context,allow_nan=False), images=[base64.b64encode(encoded).decode()])],
            schema=schema)
    (output/'response.txt').write_text(raw)
    revised=parse_plan(raw,scene,list(scene['viewpoints']))
    if len(revised['shots'])!=len(plan['shots']):
        raise ValueError('Replan must retain every failed shot')
    from .motion import parameters
    for old,new in zip(plan['shots'],revised['shots']):
        if (old.get('motion','static')!=new.get('motion','static') or
            old.get('horizon','level')!=new.get('horizon','level') or parameters(old)!=parameters(new)):
            raise ValueError('Replan changed original motion/horizon contract')
        # Legacy omission can represent the same static/level contract. Preserve explicit fields.
        for key in ('motion','horizon','motion_parameters'):
            if key in old and key not in new:
                new[key]=copy.deepcopy(old[key])
        for key,value in old.items():
            if key not in context['allowed_changes'] and new.get(key)!=value:
                raise ValueError(f'Replan changed original objective: {key}')
        if new['preferred_view_id']==old['preferred_view_id'] and new['human_instructions']==old['human_instructions']:
            raise ValueError('Replan contains no actionable change for '+old['shot_id'])
        if no_human_actions and new['human_instructions']:
            raise ValueError('Original brief forbids human actions')
        if any(entry not in new['human_instructions'] for entry in old['human_instructions']):
            raise ValueError('Replan dropped an original human instruction')
    _write_json(output/'shot_plan.json',revised)
    scene_path=output/'semantic_map/scene_map.json'
    _write_json(output/'planning_report.json',dict(status='validated',model=model.get('model'),
                 scene_map_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest() if scene_path.exists() else None,
                 creative_brief=brief, purpose='failure_driven_replan', objectives_preserved=True))
    return revised


def fresh_context(sim, scene_path, output, perception):
    """Refresh visible dynamic surfaces; retain explicitly marked out-of-view history."""
    source=json.loads(Path(scene_path).read_text())
    mapped=SceneMap()
    mapped.artifact_identity=sim.artifact_identity
    mapped.objects=copy.deepcopy(source['objects'])
    mapped.viewpoints=copy.deepcopy(source['viewpoints'])
    with np.load(Path(scene_path).parent/'object_surfaces.npz',allow_pickle=False) as arrays:
        mapped.points={key:arrays[key].copy() for key in arrays.files}
    observation=sim.capture()
    mapped.update(perception.detect(observation),observation,'replan_current_'+str(len(mapped.viewpoints)))
    Path(output).mkdir(parents=True,exist_ok=True)
    mapped.save(output)
    sim.save(Path(output)/'current',observation=observation)
    return mapped.to_dict(),observation


def execute_with_replanning(sim, plan_path, scene_path, output, *, budget=1, brief=None,
                            client=None, perception=None, progress=print, on_event=None, **options):
    from .execution import execute_plan
    from .perception import PerceptionPipeline
    if type(budget) is not int or not 0<=budget<=2:
        raise ValueError('Replan budget must be 0–2')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    perception=perception or PerceptionPipeline(sim.geometry_catalog())
    source_scene=json.loads(Path(scene_path).read_text())
    original=parse_plan(Path(plan_path).read_text(),source_scene,list(source_scene['viewpoints']))
    if brief is None:
        context=Path(plan_path).parent/'context.json'
        brief=json.loads(context.read_text()).get('creative_brief') if context.exists() else original['title']
    report=dict(status='running',replan_budget=budget,attempts=[],replans=[],shots=[],
                original_plan=str(Path(plan_path).resolve()),original_plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest())
    current_plan=original
    latest={}
    def persist(): _write_json(output/'execution_report.json',report)
    try:
        for attempt in range(budget+1):
            destination=output/f'attempt_{attempt}'
            result=execute_plan(sim,plan_path,scene_path,destination,perception=perception,
                                progress=progress,on_event=on_event,**options)
            report['attempts'].append(dict(attempt=attempt,report=str(destination/'execution_report.json'),status=result['status']))
            latest.update({s['shot_id']:dict(s,attempt=attempt) for s in result['shots']})
            report['shots']=[latest[s['shot_id']] for s in original['shots'] if s['shot_id'] in latest]
            report['status']='completed' if all(s['status']=='satisfied_observed_criteria' for s in report['shots']) else 'needs_adjustment'
            persist()
            if result['status']=='completed' or attempt==budget: break
            failures=[]
            for item in result['shots']:
                detail=json.loads((destination/item['directory']/'result.json').read_text())
                if detail.get('model_replan_needed') or (detail.get('refinement') or {}).get('model_replan_needed'):
                    failures.append(dict(shot_id=item['shot_id'],metrics=detail['after'],
                        motion_completed=detail['motion_completed'],
                        pending_creative=(detail.get('refinement') or {}).get('pending_creative',[]),
                        requests=detail['subject_repositioning_requests']))
            if not failures: break
            if progress: progress(f'Gemma replan {attempt+1}/{budget}: sending current image and failure evidence; original objectives stay fixed.')
            if on_event: on_event(dict(kind='replanning',attempt=attempt+1,budget=budget))
            replan_dir=output/f'replan_{attempt+1}'
            scene,obs=fresh_context(sim,scene_path,replan_dir/'semantic_map',perception)
            failed_ids={f['shot_id'] for f in failures}
            failed_plan=dict(current_plan,shots=[s for s in current_plan['shots'] if s['shot_id'] in failed_ids])
            # Only HTTP/model work leaves the graphics owner thread. Pause/stop stays responsive.
            responses=queue.Queue()
            limits=copy.deepcopy(sim.limits)
            def work():
                try: responses.put((revise_plan(failed_plan,scene,failures,obs.rgb,brief,replan_dir,
                                                  client=client,limits=limits),None))
                except Exception as error: responses.put((None,str(error)))
            threading.Thread(target=work,daemon=True,name='gemma-replan').start()
            while True:
                sim.checkpoint()
                try: revised,error=responses.get(timeout=.05);break
                except queue.Empty: continue
            sim.checkpoint()
            entry=dict(attempt=attempt+1,directory=str(replan_dir),status='rejected' if error else 'validated',error=error)
            report['replans'].append(entry)
            persist()
            if error:
                if on_event: on_event(dict(kind='replan_failed',attempt=attempt+1,budget=budget,error=error))
                if progress: progress(f'Replanning unavailable or rejected: {error}. Unresolved shots are preserved.')
                break
            current_plan=revised
            plan_path=replan_dir/'shot_plan.json'
            scene_path=replan_dir/'semantic_map/scene_map.json'
            if on_event: on_event(dict(kind='plan_revised',plan=revised,attempt=attempt+1,path=str(plan_path)))
    except BaseException as error:
        from .recording import Stopped
        report.update(status='stopped' if isinstance(error,Stopped) else 'failed',error=str(error))
        raise
    finally:
        persist()
    return report
