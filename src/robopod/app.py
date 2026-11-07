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

from .simulation import Simulation


class Cancelled(Exception):
    pass


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)
        self.commands = queue.Queue()
        self.cancel = threading.Event()
        self.images = {}
        self.data = dict(busy=False, phase='starting', status='Starting simulation…', error=None,
                         survey_ready=False, plan=None, plan_source=None, plan_brief=None, plan_version=None, plan_notes=[], exploration=0, coverage=0,
                         objects=[], joints={}, teleprompter='Stand by. Prepare your scene.',
                         events=[], results=[], output=None, frame=0, map_frame=0)

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

    def submit(self, action, brief=''):
        if action not in ('restart', 'explore', 'plan', 'demo', 'execute'):
            raise ValueError('Unknown action')
        if action == 'plan' and (not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 4000):
            raise ValueError('Enter a creative brief of 1–4000 characters')
        with self.lock:
            if action == 'restart':
                if self.data['phase'] == 'restarting':
                    raise ValueError('Restart already requested')
                self.cancel.set()
                self.data.update(busy=True, phase='restarting', status='Restart requested…')
            else:
                if self.data['busy']:
                    raise ValueError('An operation is already running')
                if action in ('plan', 'demo') and not self.data['survey_ready']:
                    raise ValueError('Explore the scene first')
                if action == 'execute' and not self.data['plan']:
                    raise ValueError('Generate or load a plan first')
                self.data.update(busy=True, phase=action, error=None, status=f'Starting {action}…')
            self.commands.put((action, brief))


class OwnerSimulation(Simulation):
    def __init__(self, tick):
        self.tick = tick
        super().__init__()

    def step(self, seconds=1/30):
        start = time.monotonic()
        super().step(seconds)
        self.tick(self)
        # Pace only executing simulation; search probes remain fast.
        delay = min(seconds, .02) - (time.monotonic()-start)
        if delay > 0:
            time.sleep(delay)


class Studio:
    def __init__(self, state, output):
        self.state = state
        self.output = Path(output)
        self.sim = None
        self.survey = self.scene_path = self.plan_path = None
        self.epoch = 0
        self.planner = None
        self.plan_results = queue.Queue()
        self.last_frame = 0
        self.frame_number = 0

    def check_cancel(self):
        if self.state.cancel.is_set():
            raise Cancelled('Restart requested')

    def tick(self, sim):
        self.check_cancel()
        if time.monotonic()-self.last_frame >= .12:
            self.frame(sim)

    def frame(self, sim, instruction=None):
        from .perception import annotate
        self.check_cancel()
        obs = sim.capture()
        detections = self.perception.detect(obs)
        self.frame_number += 1
        if obs.simulation_time != self.live_map.last_time:
            self.live_map.update(detections, obs, f'live_{self.frame_number}')
            # The UI map is a rolling preview, not an unbounded recording.
            while len(self.live_map.viewpoints) > 256:
                del self.live_map.viewpoints[next(iter(self.live_map.viewpoints))]
            for record in self.live_map.objects.values():
                record['seen_from'] = record['seen_from'][-256:]
        ok, encoded = cv2.imencode('.jpg', annotate(obs, detections), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.state.image('camera.jpg', encoded.tobytes())
        self.state.update(frame=self.frame_number, joints=obs.joints, simulation_time=obs.simulation_time,
                          objects=[dict(object_id=k, label=v['label'], category=v['category'],
                                        visibility=v['visibility'], observations=v['observation_count'])
                                   for k,v in self.live_map.objects.items()])
        if instruction is not None:
            self.state.update(teleprompter=instruction or 'Hold position. The camera is working.')
        if self.frame_number % 8 == 0:
            ok, encoded = cv2.imencode('.png', self.live_map.topdown())
            if ok:
                self.state.image('map.png', encoded.tobytes())
                self.state.update(map_frame=self.frame_number)
        self.last_frame = time.monotonic()

    def reset(self):
        from .mapping import SceneMap
        from .optimization import geometry_catalog
        from .perception import SegmentationPerception
        self.epoch += 1
        self.state.cancel.clear()
        if self.sim:
            self.sim.close()
        self.sim = OwnerSimulation(self.tick)
        self.perception = SegmentationPerception(geometry_catalog(self.sim))
        self.live_map = SceneMap()
        self.frame_number = 0
        self.survey = self.scene_path = self.plan_path = None
        with self.state.lock:
            self.state.images.clear()
        self.state.update(busy=True, phase='restarting', survey_ready=False, plan=None, plan_source=None, plan_brief=None, plan_version=None, plan_notes=[],
                          exploration=0, coverage=0, results=[], output=None, error=None,
                          objects=[], map_frame=0, teleprompter='Stand by. Explore the room to begin.')
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
            result = explore(self.sim, self.survey, progress=survey_progress)
            self.check_cancel()
            self.progress('Building semantic scene map…')
            mapped = map_survey(self.survey, self.survey/'semantic_map')
            self.scene_path = self.survey/'semantic_map/scene_map.json'
            self.state.image('map.png', (self.survey/'semantic_map/map_topdown.png').read_bytes())
            self.state.update(survey_ready=True, map_frame=self.frame_number+1, output=str(self.survey.resolve()),
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
            from .execution import execute_plan
            from .mapping import SceneMap
            # Plans are based on the surveyed original scene, so restore that initial scene.
            self.sim.reset()
            self.live_map = SceneMap()
            self.state.update(results=[], teleprompter='Hold position. Preparing the first shot.')
            output = self.new_path('execution')
            self.frame(self.sim)
            with self.state.lock:
                self.state.images['before.jpg'] = self.state.images.get('camera.jpg', b'')
            result = execute_plan(self.sim, self.plan_path, self.scene_path, output, refine=True,
                                  progress=self.progress, on_frame=self.frame)
            self.check_cancel()
            self.frame(self.sim)
            with self.state.lock:
                self.state.images['after.jpg'] = self.state.images.get('camera.jpg', b'')
            self.state.update(results=result['shots'], output=str(output.resolve()),
                              teleprompter='Sequence complete.' if result['status']=='completed' else 'Some shots need adjustment. Review the results below.')
            self.progress(f"Execution {result['status'].replace('_',' ')}.")
        self.state.update(busy=False, phase='ready')

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
            from .brief import requirements
            brief = json.loads((output/'context.json').read_text()).get('creative_brief', '') if (output/'context.json').exists() else ''
            notes = ['Continuous in-shot movement and optical zoom are not yet supported; execution transitions between compositions.'] if requirements(brief)['movement_requested'] else []
            self.state.update(plan=plan, plan_source='Local Gemma 4 E2B • validated', plan_brief=brief, plan_version=output.name, plan_notes=notes)
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
            if path in ('/camera.jpg','/map.png','/before.jpg','/after.jpg'):
                with state.lock:
                    content = state.images.get(path[1:])
                return self.send(200 if content else 404, content or b'', 'image/png' if path.endswith('.png') else 'image/jpeg')
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
                state.submit(data.get('action'), data.get('brief',''))
                self.send(202, b'{"accepted":true}')
            except (ValueError, TypeError) as error:
                self.send(409, json.dumps({'error':str(error)}).encode())
    return Handler


def run_app(port=8765, output=Path('outputs/studio'), survey=None, plan_path=None):
    state = AppState()
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name='local-ui').start()
    studio = Studio(state, output)
    print(f'Robopod Studio: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        studio.reset()
        if survey is not None:
            studio.survey = Path(survey)
            studio.scene_path = studio.survey/'semantic_map/scene_map.json'
            saved_scene = json.loads(studio.scene_path.read_text())
            manifest = json.loads((studio.survey/'manifest.json').read_text())
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
            except Cancelled:
                state.log('Stopping current operation; restarting…')
            except Exception as error:
                state.update(busy=False, phase='ready', error=str(error))
                state.log('Operation failed. Restart or retry after correcting the error.')
    except KeyboardInterrupt:
        print('\nStopping Robopod Studio.', flush=True)
    finally:
        state.cancel.set()
        if studio.sim:
            studio.sim.close()
        server.shutdown()
        server.server_close()
