"""Loopback browser app. Only the main owner thread touches MuJoCo/OpenGL."""
import copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import queue
import secrets
import threading
import time
from urllib.parse import urlsplit

import cv2
import mujoco
import numpy as np

from .simulation import Simulation
from .recording import Stopped


PERTURB_ACTIONS = {
    'perturb_person_left': ('person', (0.35, 0.0, 0.0), 'Scene perturbation: nudging presenter left…'),
    'perturb_person_right': ('person', (-0.35, 0.0, 0.0), 'Scene perturbation: nudging presenter right…'),
    'perturb_person_back': ('person', (0.0, 0.35, 0.0), 'Scene perturbation: nudging presenter back…'),
    'perturb_product': ('product', (0.30, 0.0, 0.0), 'Scene perturbation: sliding synthesizer…'),
    'perturb_occlusion': ('occlusion', (0.0, 0.0, 0.0), 'Scene perturbation: stepping into occlusion…'),
}


class Cancelled(Exception):
    pass


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)
        self.commands = queue.Queue()
        self.perturbations = queue.Queue()
        self.cancel = threading.Event()
        self.paused = threading.Event()
        self.stop = threading.Event()
        self.media = {}
        self.images = {}
        self.data = dict(busy=False, phase='starting', status='Starting simulation…', error=None,
                         survey_ready=False, plan=None, plan_source=None, plan_brief=None, plan_version=None, plan_notes=[], exploration=0, coverage=0,
                         objects=[], joints={}, teleprompter='Stand by. Prepare your scene.',
                         events=[], results=[], output=None, frame=0, map_frame=0, paused=False,
                         active_shot=None, shot_metrics=None, sequence_video=None)

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def snapshot(self):
        with self.lock:
            return dict(copy.deepcopy(self.data), token=self.token)

    def log(self, message):
        with self.lock:
            self.data['status'] = message
            self.data['events'].append(message)
            self.data['events'] = self.data['events'][-80:]

    def image(self, name, content):
        with self.lock:
            self.images[name] = content

    def submit(self, action, brief='', config=None):
        if action not in ('restart', 'explore', 'plan', 'demo', 'execute', 'pause', 'resume', 'stop', 'configure', *PERTURB_ACTIONS):
            raise ValueError('Unknown action')
        if action == 'plan' and (not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 4000):
            raise ValueError('Enter a creative brief of 1–4000 characters')
        if action == 'configure':
            from .learned import validate_config
            brief = validate_config(config)
        with self.lock:
            if action in PERTURB_ACTIONS:
                if self.data['phase'] != 'execute' or not self.data['busy']:
                    raise ValueError('Scene perturbations require an active execution')
                _, _, message = PERTURB_ACTIONS[action]
                self.perturbations.put(action)
                self.data['status'] = message
                self.data['events'].append(message)
                self.data['events'] = self.data['events'][-80:]
                return
            if action in ('pause','resume','stop'):
                if self.data['phase'] != 'execute' or not self.data['busy']:
                    raise ValueError('Playback controls require an active execution')
                if action == 'pause':
                    self.paused.set()
                else:
                    self.paused.clear()
                if action == 'stop':
                    self.stop.set()
                self.data.update(paused=self.paused.is_set(), status={'pause':'Paused. Simulation time is frozen.',
                                 'resume':'Resuming execution…','stop':'Stopping execution…'}[action])
                return
            if action == 'restart':
                if self.data['phase'] == 'restarting':
                    raise ValueError('Restart already requested')
                self.cancel.set()
                self.paused.clear()
                self.data.update(busy=True, phase='restarting', status='Restart requested…')
            else:
                if self.data['busy']:
                    raise ValueError('An operation is already running')
                if action in ('plan', 'demo') and not self.data['survey_ready']:
                    raise ValueError('Explore the scene first')
                if action == 'execute' and not self.data['plan']:
                    raise ValueError('Generate or load a plan first')
                self.stop.clear()
                self.paused.clear()
                self.data.update(paused=False, busy=True, phase=action, error=None, status=f'Starting {action}…')
            self.commands.put((action, brief))


class OwnerSimulation(Simulation):
    def __init__(self, tick):
        self.tick = tick
        super().__init__()

    def checkpoint(self):
        self.tick.__self__.check_cancel()

    def step(self, seconds=1/30):
        self.checkpoint()
        start = time.monotonic()
        super().step(seconds)
        self.tick(self)
        # Pace only executing simulation; search probes remain fast.
        delay = min(seconds, .02) - (time.monotonic()-start)
        if delay > 0:
            time.sleep(delay)


class Studio:
    def __init__(self, state, output, config=None):
        from .learned import validate_config
        self.config = validate_config(config)
        self.detector = self.pose = None
        self.state = state
        self.output = Path(output)
        self.sim = None
        self.survey = self.scene_path = self.plan_path = None
        self.epoch = 0
        self.planner = None
        self.plan_results = queue.Queue()
        self.active_perturbation = None
        self.last_sim_time = 0.0
        self.last_frame = 0
        self.frame_number = 0

    def clear_perturbations(self):
        self.active_perturbation = None
        self.last_sim_time = 0.0
        while not self.state.perturbations.empty():
            try:
                self.state.perturbations.get_nowait()
            except queue.Empty:
                break

    def _perturbation_target(self, sim, action):
        person = int(sim.model.body('person').mocapid[0])
        product = int(sim.model.body('product').mocapid[0])
        start = sim.data.mocap_pos.copy()
        end = start.copy()
        person_low, person_high = np.array([-1.2, 0.1, 0.0]), np.array([1.2, 2.2, 0.0])
        product_low, product_high = np.array([-0.85, 0.45, 0.82]), np.array([0.85, 1.15, 1.45])
        target_kind, delta, _ = PERTURB_ACTIONS[action]
        if target_kind == 'person':
            end[person] = np.clip(start[person] + np.asarray(delta, dtype=float), person_low, person_high)
        elif target_kind == 'product':
            dx = -abs(delta[0]) if start[product, 0] > 0.15 else abs(delta[0])
            end[product] = np.clip(start[product] + np.array([dx, delta[1], delta[2]], dtype=float), product_low, product_high)
        elif target_kind == 'occlusion':
            end[person] = np.clip(
                np.array([start[product, 0], max(0.95, start[product, 1] + 0.35), start[person, 2]], dtype=float),
                person_low, person_high,
            )
            end[product] = np.clip(
                np.array([end[person, 0], start[product, 1], 1.22], dtype=float),
                product_low, product_high,
            )
            if np.linalg.norm(end - start) < 0.05:
                shift = -0.35 if start[person, 0] > 0.0 else 0.35
                end[person, 0] = float(np.clip(start[person, 0] + shift, person_low[0], person_high[0]))
                end[product, 0] = float(np.clip(end[person, 0], product_low[0], product_high[0]))
        return end

    def apply_perturbation(self, sim):
        now = float(sim.data.time)
        if now < self.last_sim_time:
            self.last_sim_time = now
        prev_time = self.last_sim_time
        self.last_sim_time = now
        if self.active_perturbation is None:
            try:
                action = self.state.perturbations.get_nowait()
            except queue.Empty:
                return
            self.active_perturbation = dict(
                action=action,
                start_time=prev_time if now - prev_time <= 0.5 else now,
                duration=0.5,
                start_mocap=sim.data.mocap_pos.copy(),
                end_mocap=self._perturbation_target(sim, action),
            )
        pert = self.active_perturbation
        t = min(1.0, max(0.0, (now - pert['start_time']) / max(pert['duration'], 1e-6)))
        blend = 3.0 * t * t - 2.0 * t * t * t
        sim.data.mocap_pos[:] = pert['start_mocap'] + blend * (pert['end_mocap'] - pert['start_mocap'])
        mujoco.mj_forward(sim.model, sim.data)
        if t >= 1.0 - 1e-9:
            self.active_perturbation = None

    def check_cancel(self):
        if self.state.cancel.is_set():
            raise Cancelled('Restart requested')
        if self.state.stop.is_set():
            raise Stopped('Execution stopped')
        while self.state.paused.is_set():
            if self.state.cancel.wait(.05):
                raise Cancelled('Restart requested')
            if self.state.stop.is_set():
                raise Stopped('Execution stopped')

    def tick(self, sim):
        self.check_cancel()
        self.apply_perturbation(sim)
        if time.monotonic()-self.last_frame >= .12:
            self.frame(sim)

    def frame(self, sim, instruction=None):
        from .perception import annotate
        self.check_cancel()
        obs = sim.capture()
        started = time.monotonic()
        detections = self.perception.detect(obs)
        learned = self.perception.last_report or {}
        self.state.update(perception_status=dict(
            yolo_enabled=self.detector is not None, pose_enabled=self.pose is not None,
            detections=len(learned.get('learned_detections', [])),
            poses=len(learned.get('learned_poses', [])),
            associated_objects=sum('learned_detection' in d.landmarks for d in detections),
            milliseconds=round((time.monotonic()-started)*1000)))
        if self.perception.warnings:
            self.state.update(perception_warnings=self.perception.warnings)
        self.frame_number += 1
        if obs.simulation_time != self.live_map.last_time:
            self.live_map.update(detections, obs, f'live_{self.frame_number}')
            # The UI map is a rolling preview, not an unbounded recording.
            while len(self.live_map.viewpoints) > 256:
                del self.live_map.viewpoints[next(iter(self.live_map.viewpoints))]
            for record in self.live_map.objects.values():
                record['seen_from'] = record['seen_from'][-256:]
        clean_ok, clean = cv2.imencode('.jpg', cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
        if clean_ok:
            self.state.image('clean.jpg', clean.tobytes())
        ok, encoded = cv2.imencode('.jpg', annotate(obs, detections), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.state.image('camera.jpg', encoded.tobytes())
        if hasattr(sim, 'capture_overview'):
            ov_ok, ov_encoded = cv2.imencode('.jpg', sim.capture_overview(), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ov_ok:
                self.state.image('overview.jpg', ov_encoded.tobytes())
        self.state.update(frame=self.frame_number, joints=obs.joints, simulation_time=obs.simulation_time,
                          objects=[dict(object_id=k, label=v['label'], category=v['category'],
                                        visibility=v['visibility'], observations=v['observation_count'])
                                   for k,v in self.live_map.objects.items()])
        if instruction:
            self.state.update(teleprompter=instruction)
        if self.frame_number % 8 == 0:
            ok, encoded = cv2.imencode('.png', self.live_map.topdown())
            if ok:
                self.state.image('map.png', encoded.tobytes())
                self.state.update(map_frame=self.frame_number)
        self.last_frame = time.monotonic()

    def reset(self):
        from .mapping import SceneMap
        from .optimization import geometry_catalog
        from .perception import PerceptionPipeline
        from .learned import build_backends, close_backends
        close_backends(self.detector,self.pose)
        self.detector,self.pose,warnings = build_backends(self.config)
        self.state.update(config=self.config, perception_warnings=warnings)
        self.epoch += 1
        self.state.cancel.clear()
        self.state.stop.clear()
        self.state.paused.clear()
        self.clear_perturbations()
        if self.sim:
            self.sim.close()
        self.sim = OwnerSimulation(self.tick)
        self.perception = PerceptionPipeline(geometry_catalog(self.sim),detector=self.detector,pose_estimator=self.pose)
        self.live_map = SceneMap()
        self.frame_number = 0
        self.survey = self.scene_path = self.plan_path = None
        with self.state.lock:
            self.state.images.clear()
            self.state.media.clear()
        self.state.update(busy=True, phase='restarting', survey_ready=False, plan=None, plan_source=None, plan_brief=None, plan_version=None, plan_notes=[],
                          exploration=0, coverage=0, results=[], output=None, error=None,
                          objects=[], map_frame=0, paused=False, active_shot=None, shot_metrics=None, sequence_video=None, teleprompter='Stand by. Explore the room to begin.')
        self.frame(self.sim)
        self.state.update(busy=False, phase='ready', status='Ready. Explore the scene.')

    def new_path(self, prefix):
        self.output.mkdir(parents=True, exist_ok=True)
        return self.output/f'{prefix}-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'

    def progress(self, message):
        self.check_cancel()
        self.state.log(message)
        if message.startswith('TELEPROMPTER'):
            self.state.update(teleprompter=message.split(': ',1)[-1])

    def action(self, action, brief):
        if action == 'configure':
            self.config = brief
            self.reset()
            self.progress('Settings applied. Explore again to build a map with these backends.')
            return
        if action == 'restart':
            self.reset()
            return
        self.check_cancel()
        if action == 'explore':
            from .exploration import explore
            from .mapping import map_survey, SceneMap
            # A new survey always starts from home and invalidates an old plan.
            self.sim.reset()
            self.live_map = SceneMap()
            self.state.update(plan=None, plan_source=None, survey_ready=False, exploration=0, coverage=0, results=[])
            self.plan_path = None
            self.survey = self.new_path('survey')
            def survey_progress(view, total):
                self.progress(f"Survey {int(view['view_id'].split('_')[-1])+1}/{total}: {view['status']}")
                self.state.update(exploration=(int(view['view_id'].split('_')[-1])+1)/total,
                                  coverage=view['cumulative_voxels'])
                self.frame(self.sim)
            result = explore(self.sim, self.survey, progress=survey_progress, adaptive_views=self.config['adaptive_views'], perception=self.perception)
            self.check_cancel()
            self.progress('Building semantic scene map…')
            mapped = map_survey(self.survey, self.survey/'semantic_map', detector=self.detector, pose_estimator=self.pose)
            self.scene_path = self.survey/'semantic_map/scene_map.json'
            self.state.image('map.png', (self.survey/'semantic_map/map_topdown.png').read_bytes())
            self.state.update(survey_ready=True, exploration=1, map_frame=self.frame_number+1, output=str(self.survey.resolve()),
                              teleprompter='Survey complete. Describe the film you want to make.')
            self.progress(f"Survey complete: {len(mapped.objects)} objects; {result['saved_views']} saved views.")
        elif action == 'plan':
            from .gemma import generate_plan
            if self.planner and self.planner.is_alive():
                raise ValueError('A previous Ollama request is still finishing. Please retry shortly.')
            output = self.new_path('plan')
            epoch, survey, scene_path = self.epoch, self.survey, self.scene_path
            self.state.update(plan=None, plan_source=None, plan_brief=brief, plan_notes=[], plan_version=None)
            self.plan_path = None
            self.progress('Gemma is planning locally. The first request may take a few minutes…')
            def work():
                try:
                    def planning_progress(message):
                        if epoch != self.epoch or self.state.cancel.is_set():
                            raise Cancelled('Superseded planning request')
                        self.state.log(message)
                    plan = generate_plan(survey, scene_path, brief, output, image_count=2, progress=planning_progress)
                    self.plan_results.put((epoch, output, plan, None))
                except Exception as error:
                    self.plan_results.put((epoch, output, None, str(error)))
            self.planner = threading.Thread(target=work, daemon=True, name='local-gemma')
            self.planner.start()
            return
        elif action == 'demo':
            from .planning import parse_plan
            scene = json.loads(self.scene_path.read_text())
            source = Path(__file__).parent/'assets/cooperative_demo_plan.json'
            plan = parse_plan(source.read_text(), scene, list(scene['viewpoints']))
            self.plan_path = self.new_path('demo-plan')
            self.plan_path.mkdir()
            self.plan_path = self.plan_path/'shot_plan.json'
            self.plan_path.write_text(json.dumps(plan, indent=2))
            self.state.update(plan=plan, plan_brief=None, plan_version=self.plan_path.parent.name, plan_notes=['Hand-authored demo: this sequence does not use your creative brief.'], plan_source='Hand-authored cooperative demo (not Gemma)', output=str(self.plan_path.parent.resolve()))
            self.progress('Cooperative demo loaded. Ready to execute.')
        elif action == 'execute':
            from .replanning import execute_with_replanning
            from .mapping import SceneMap
            # Plans are based on the surveyed original scene, so restore that initial scene.
            self.clear_perturbations()
            self.sim.reset()
            self.live_map = SceneMap()
            current_plan = self.state.snapshot().get('plan') or {}
            first_cue = current_plan.get('shots', [{}])[0].get('description', 'Moving camera to first shot…') if current_plan.get('shots') else 'Moving camera to first shot…'
            self.state.update(results=[], active_shot=None, shot_metrics=None, sequence_video=None, teleprompter=first_cue)
            output = self.new_path('execution')
            self.state.update(output=str(output.resolve()))
            self.frame(self.sim)
            with self.state.lock:
                self.state.images['before.jpg'] = self.state.images.get('camera.jpg', b'')
            try:
                result = execute_with_replanning(self.sim, self.plan_path, self.scene_path, output, refine=False,
                                      budget=self.config['replan_budget'], brief=self.state.snapshot().get('plan_brief'), perception=self.perception,
                                      progress=self.progress, on_frame=self.frame, on_event=self.execution_event)
            except Stopped:
                self.state.stop.clear()
                self.state.paused.clear()
                self.state.update(busy=False, phase='ready', paused=False, active_shot=None,
                                  status='Stopped. Plan and completed shots are preserved.',
                                  teleprompter='Stopped. Resume by starting a new take.')
                return
            self.check_cancel()
            self.frame(self.sim)
            with self.state.lock:
                self.state.images['after.jpg'] = self.state.images.get('camera.jpg', b'')
            self.state.update(output=str(output.resolve()),
                              teleprompter='Sequence complete.' if result['status']=='completed' else 'Some shots need adjustment. Review the results below.')
            self.progress(f"Execution {result['status'].replace('_',' ')}.")
        self.state.update(busy=False, phase='ready', paused=False, active_shot=None)

    def execution_event(self, event):
        kind = event['kind']
        if kind == 'replanning':
            self.state.update(active_shot=None,teleprompter=f"Replanning {event['attempt']}/{event['budget']} from observed failure evidence…")
        elif kind == 'plan_revised':
            self.state.update(last_replan=event['plan'], teleprompter=f"Executing replan {event['attempt']} with adjusted viewpoint/cooperation…", plan_notes=[f"Gemma revision {event['attempt']} is executing failed shots with unchanged objectives. Starting a new take uses the original plan."])
        elif kind == 'replan_failed':
            self.state.update(active_shot=None, teleprompter='Replan could not revise unresolved shots. Review results below.', plan_notes=[f"Replan {event['attempt']} was rejected: {event['error']}"])
        elif kind == 'shot_started':
            updates = dict(active_shot=event['shot_id'], shot_metrics=None)
            if event.get('teleprompter'):
                updates['teleprompter'] = event['teleprompter']
            self.state.update(**updates)
        elif kind == 'measurement':
            self.state.update(shot_metrics=event)
        elif kind == 'shot_completed':
            result = dict(event['result'])
            with self.state.lock:
                index = len(self.state.data['results'])
                for key, extension in (('comparison','jpg'),('video','mp4')):
                    name = f'shot_{index}_{key}.{extension}'
                    self.state.media[name] = Path(result[key])
                    result[key] = '/media/'+name
                self.state.data['results'].append(result)
        elif kind == 'execution_finished':
            if event['frames']:
                with self.state.lock:
                    self.state.media['sequence.mp4'] = Path(event['video'])
                self.state.update(sequence_video='/media/sequence.mp4')
            self.state.update(active_shot=None)

    def poll_plans(self):
        try:
            epoch, output, plan, error = self.plan_results.get_nowait()
        except queue.Empty:
            return
        if epoch != self.epoch or self.state.cancel.is_set():
            return
        self.state.update(busy=False, phase='ready', output=str(output.resolve()))
        if error:
            self.state.update(error=error)
            self.state.log('Planning failed. Check the error and retry.')
        else:
            self.plan_path = output/'shot_plan.json'
            brief = json.loads((output/'context.json').read_text()).get('creative_brief', '') if (output/'context.json').exists() else ''
            notes = ['Motion is executed only as specified per shot. Legacy plans default to static; optical zoom is unavailable.']
            first_cue = next((i['instruction'] for s in plan.get('shots', []) for i in s.get('human_instructions', []) if i.get('instruction')),
                             plan['shots'][0]['description'] if plan.get('shots') else 'Gemma plan ready.')
            self.state.update(plan=plan, plan_source='Local Gemma 4 E2B • validated', plan_brief=brief, plan_version=output.name, plan_notes=notes, teleprompter=first_cue)
            self.state.log(f"Validated {len(plan['shots'])} shots. Ready to execute.")


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, kind='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.local_host():
                return self.send(403, b'{}')
            path = urlsplit(self.path).path
            if path == '/':
                return self.send(200, (Path(__file__).parent/'assets/studio.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/state':
                return self.send(200, json.dumps(state.snapshot(), allow_nan=False).encode())
            if path in ('/camera.jpg','/clean.jpg','/overview.jpg','/map.png','/before.jpg','/after.jpg'):
                with state.lock:
                    content = state.images.get(path[1:])
                return self.send(200 if content else 404, content or b'', 'image/png' if path.endswith('.png') else 'image/jpeg')
            if path.startswith('/media/'):
                with state.lock:
                    media = state.media.get(path.removeprefix('/media/'))
                if media is not None and media.is_file():
                    return self.send(200, media.read_bytes(), 'video/mp4' if media.suffix=='.mp4' else 'image/jpeg')
            self.send(404, b'{}')

        def do_POST(self):
            origin = self.headers.get('Origin')
            if (not self.local_host() or (origin and origin != f'http://127.0.0.1:{self.server.server_port}')
                    or self.headers.get('X-Robopod-Token') != state.token):
                return self.send(403, b'{"error":"Local session required"}')
            if self.path != '/api/action':
                return self.send(404, b'{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20000:
                    raise ValueError('Invalid request length')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
                state.submit(data.get('action'), data.get('brief',''), data.get('config'))
                self.send(202, b'{"accepted":true}')
            except (ValueError, TypeError) as error:
                self.send(409, json.dumps({'error':str(error)}).encode())
    return Handler


def run_app(port=8765, output=Path('outputs/studio'), survey=None, plan_path=None, config=None):
    state = AppState()
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name='local-ui').start()
    studio = Studio(state, output, config)
    print(f'Robopod Studio: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        studio.reset()
        if survey is not None:
            studio.survey = Path(survey)
            studio.scene_path = studio.survey/'semantic_map/scene_map.json'
            saved_scene = json.loads(studio.scene_path.read_text())
            manifest = json.loads((studio.survey/'manifest.json').read_text())
            from .artifacts import require_compatible
            require_compatible(manifest.get('artifact_identity'))
            require_compatible(saved_scene.get('artifact_identity'))
            state.update(survey_ready=True, exploration=1, coverage=manifest['covered_voxels'],
                         status=f"Loaded survey: {len(saved_scene['objects'])} objects. Generate a new plan.",
                         output=str(studio.survey.resolve()))
        if plan_path is not None:
            if studio.scene_path is None:
                raise ValueError('Loading an app plan requires --survey')
            from .planning import parse_plan
            scene = json.loads(studio.scene_path.read_text())
            studio.plan_path = Path(plan_path)
            restored = parse_plan(studio.plan_path.read_text(), scene, list(scene['viewpoints']))
            report_path = studio.plan_path.parent/'planning_report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if report.get('scene_map_sha256') and report['scene_map_sha256'] != hashlib.sha256(studio.scene_path.read_bytes()).hexdigest():
                raise ValueError('Loaded plan was generated from a different scene map')
            source = 'Local Gemma 4 E2B • loaded' if report.get('model') == 'gemma4:e2b' and report.get('status') == 'validated' else 'Imported plan • schema validated'
            state.update(plan=restored, plan_source=source, plan_version=studio.plan_path.parent.name,
                         plan_brief=report.get('creative_brief'), plan_notes=report.get('capability_notes', []),
                         status=f"Loaded {len(restored['shots'])} validated shots. Ready to execute.")
        while True:
            studio.poll_plans()
            try:
                action, brief = state.commands.get(timeout=.1)
            except queue.Empty:
                if not state.cancel.is_set() and time.monotonic()-studio.last_frame > .5:
                    studio.frame(studio.sim)
                continue
            try:
                studio.action(action, brief)
            except Stopped:
                state.stop.clear()
                state.paused.clear()
                state.update(busy=False, phase='ready', paused=False, active_shot=None,
                             status='Stopped. Plan and completed shots are preserved.')
            except Cancelled:
                state.log('Stopping current operation; restarting…')
            except Exception as error:
                state.update(busy=False, phase='ready', error=str(error))
                state.log('Operation failed. Restart or retry after correcting the error.')
    except KeyboardInterrupt:
        print('\nStopping Robopod Studio.', flush=True)
    finally:
        state.cancel.set()
        from .learned import close_backends
        close_backends(studio.detector,studio.pose)
        if studio.sim:
            studio.sim.close()
        server.shutdown()
        server.server_close()
