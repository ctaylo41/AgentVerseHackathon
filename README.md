# Robopod — local AI cinematographer

A local macOS demo with a six-axis MuJoCo camera, deterministic scene exploration,
segmentation-based semantic mapping, local Gemma planning, geometric optimisation
and scripted subject cooperation. No cloud services or runtime model downloads.

## Run

Python 3.11 on Apple Silicon, with a logged-in macOS graphics session:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m robopod app
```

Open http://127.0.0.1:8765. Explore, enter a brief, generate a plan and execute it.
Gemma requires a separately installed Ollama service and local `gemma4:e2b` weights.
The labelled cooperative demo requires no model. `--port 8766` selects another port.
`mjpython -m robopod viewer` opens the native viewer on macOS.

## Studio controls

- **Pause / Resume:** freezes simulation time, including scripted responses and
  camera search. The current take continues after resume.
- **Stop take:** stops at a controller checkpoint, preserves the survey, plan,
  completed shot results and partial sequence video. Execute starts a new take.
- **Restart simulation:** stops the workflow and resets the session. Saved files remain.
- **Perception overlays:** switches between annotated and clean live RGB.
- Active shot highlighting, recorded time, centre/scale errors and their tolerances
  update during filming. Results and per-shot comparisons appear immediately after
  each shot, not only at sequence completion.
- Download individual clean shot MP4s or the clean sequence MP4. Video is 10 fps,
  silent, fixed resolution and includes recorded shot frames only. Search, recovery,
  pause and between-shot transitions are excluded. A stopped take may be partial.
  MPEG-4 Part 2 (`mp4v`) uses the installed OpenCV encoder; player support varies.

## Motion and composition contract

Six axes: height (Z), truck (X), dolly (Y), pan, tilt and roll. Translation units are
metres; angular units are radians. FOV is fixed during a run; there is no optical zoom.

Each new generated shot has:

```json
{
  "motion": "truck_left",
  "motion_parameters": {
    "direction": "left",
    "amplitude_m": 0.15,
    "subject_lock": "body:person"
  },
  "horizon": "level"
}
```

This fragment belongs inside a complete validated shot. Supported modes:

| Motion | Direction | Translation |
|---|---|---|
| static | none | amplitude 0 |
| push_in / pull_back | forward / backward | +Y / −Y |
| truck_left / truck_right | left / right | −X / +X |
| pedestal_up / pedestal_down | up / down | +Z / −Z |
| sweep | auto | controller chooses a feasible signed translation |

During plan generation, Gemma selects the motion mode, amplitude and subject lock.
The app derives the corresponding direction deterministically and saves the complete
contract above. Imported plans still reject contradictory mode/direction combinations.
Directions refer to rig/world axes, not screen space at an arbitrary pan angle.
Moving amplitude is 0.02–1 m. The controller validates target IDs, directions, axis
availability, travel and command speed. `subject_lock` selects a shot target for
compensating pan/tilt, or `none` to keep orientation fixed. All target composition
criteria still apply. Amplitude is never silently shortened to fit an obstacle or
framing constraint; an infeasible request remains unsatisfied.

Horizon is `level` by default, with a 0.01-radian measured roll tolerance. `dutch`
requires explicit structured intent. Descriptive prose cannot override motor policy.
Legacy plan JSON without motion fields remains readable: static/level is the default;
legacy moving plans use a 0.15 m amplitude and lock their first target.

Gemma first outlines multi-shot sequences, then expands individual shots. Subjects,
shot count and order are not forced into a presenter/product template. Output is
validated before execution; natural-language interpretation remains model-dependent.

Scores always use the original requested target positions and sizes. No automatic
relaxation of goals is counted as success. Unsupported human instructions stay pending.
During filming, 0.3 s of sustained framing failure triggers camera recovery, at most
twice per shot. Recovery time is not recorded. Success requires a satisfied final
frame and terminal hold, plus requested camera movement and resolved human actions.

## Physical path checks

All modeled rig solids participate in contact with the environment. Intentional
intersections between connected gimbal parts are excluded from self collision.
Before motion, signed MuJoCo geom distances and a conservative joint-displacement
bound certify continuous clearance along the interpolated joint path (2 mm minimum).
This checks the whole modeled rig against obstacles, including paths whose endpoints
are clear but whose middle crosses an object. It is independent of image visibility.
Uncertified paths are rejected; there is no automatic obstacle-routing planner.

Measured rig clearances and joint limits are checked each physics step. Position
servos and soft contacts are a numerical simulation, not a hardware safety guarantee.
The path certificate assumes the current scene snapshot; scripted scene changes are
checked during stepping. It covers the MJCF solids, not unmodeled cables or mechanisms.

## Artifact compatibility

New captures, survey manifests and scene maps carry an artifact version, rig identifier,
perception registry version and SHA-256 of the scene/robot XML. Planning, app restore
and execution reject missing/mismatched identities with a request to explore again.
Old artifacts remain available for inspection/offline mapping, but cannot silently
supply stale rig capabilities to a new run. Plan reports additionally bind to the
exact scene-map hash. After changing the robot/room, run a fresh survey and plan.

## CLI workflows

```sh
python -m robopod capture --truck 0.2 --dolly 0.3 --camera-height 1.6
python -m robopod explore --output outputs/my-survey
python -m robopod map --survey outputs/my-survey
python -m robopod plan --survey outputs/my-survey --brief "Show me and the synthesizer, then its controls."
python -m robopod execute --survey outputs/my-survey --plan PATH/shot_plan.json --refine
python -m robopod app --survey outputs/my-survey --plan PATH/shot_plan.json
python -m unittest discover -s tests -v
```

Use a new output directory for each survey, map, plan or execution. `--preview` shows
CLI execution; Q cancels. Exit 0 means observed criteria passed; exit 2 means shots
need adjustment. Rendering tests require macOS graphics access; a sandbox can block it.

Outputs include calibrated RGB/depth/segmentation, sparse surface maps, planning
context/responses, immutable validated objectives, trajectory samples, per-shot metrics,
path checks, refinement evidence, before/after comparisons and MP4 footage.

## Prototype limits

The map represents partial observed surfaces; visibility remains a sampled-surface proxy.
Scripted actors and learned recognition have the limits described below. This is a local
simulation demo, with fixed FOV and no physical grasp controller. See [architecture](docs/architecture.md).

## Adaptive discovery and optional local perception

Exploration starts with the reproducible 18-pose baseline, then ranks additional
reachable viewpoints using observed object uncertainty, projected visibility and view
diversity. It stops after two low-novelty observations or the configured budget (default
6, maximum 12). `--adaptive-views 0` retains only the baseline. Decisions, blocked paths,
coverage gains and calibrated captures remain in the manifest. It cannot promise to
find an object that no camera view reveals.

Unknown simulator bodies/geoms receive stable instance IDs and an `unknown` category.
Optional YOLO segmentation labels associate by mask overlap to those IDs instead of
creating duplicate objects. Configured demo identities remain authoritative. MediaPipe
pose landmarks associate to visible person masks and back-project valid observed depth.
Model errors retain the segmentation fallback and produce warnings. Recognition covers
the chosen model's vocabulary; this is not open-world recognition of arbitrary products.

Install only the extras you want, and download compatible weights separately:

```sh
python -m pip install -e '.[yolo]'
python -m pip install -e '.[pose]'
python -m robopod app --yolo-model /absolute/path/segmentation.pt --pose-model /absolute/path/pose_landmarker.task
python -m robopod map --survey outputs/my-survey --yolo-model /absolute/path/segmentation.pt
```

The Studio settings panel accepts these local paths, the adaptive view budget and the
replan budget. Applying settings resets the session and requires a fresh survey. Empty
paths disable learned backends. Runtime never fetches weights. Model packages/weights
are optional and inference quality on stylised MuJoCo imagery depends on the model.
Adapter APIs: [Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/)
and [MediaPipe PoseLandmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker).

## Detail shots and expressive cooperation

`body:product` remains the whole synthesizer. `part:product:controls` and
`part:product:keys` have separate masks, sparse surfaces and framing/visibility criteria.
A controls close-up must target the controls when that part has been observed.

The demo presenter supports these exact phrases (case-insensitive, final punctuation optional):

- `Turn your shoulders toward the camera` / `Turn toward the instrument`
- `Look at the camera` / `Smile` / `Look at the camera and smile`
- `Point to the controls` / `Point to the product`
- `Hold the product near your face` / `Hold the product closer to your face`
- Existing steps left/right/closer/back and product raise/lower/move phrases.

These animate torso/head/arm orientation, visible mouth features and the held prop.
They are scripted kinematic cooperation, not physically realistic grasping or human
biomechanics. Verification uses visible facial/shoulder/hand landmarks and segmented
depth: missing evidence remains unresolved. Unknown people are mapped but have no
scripted actor; their instructions require manual response. There is no facial emotion
classifier or eyeball tracking: smile and gaze are visible geometric proxies.

## Bounded Gemma escalation

CLI execution and Studio now consume failed-shot `model_replan_needed` flags. The
default budget is one revised plan (`--replan-budget 0` disables, maximum 2). After a
failed take, the owner thread captures a fresh image and updates visible object surfaces.
Gemma receives the original brief, failed shots, measured framing/motion failures,
pending human actions, current semantic map and physical limits. Only failed shots retry;
previous successes remain in the aggregate report.

The revision may change the preferred viewpoint, add cooperation instructions and explain
feasibility. It cannot change targets, framing, tolerances, duration, motion or horizon,
or drop original human instructions. Notes-only revisions are rejected because they do not change execution. Invalid/unavailable model responses stop escalation
with unresolved results; no retry loop claims success by weakening the goal. Pause/stop
remain responsive while the HTTP worker runs. A cancelled HTTP request can finish in
its background thread, but cannot execute a stale result.

Execution folders contain `attempt_0`, optional `replan_1` / `attempt_1`, and a root
`execution_report.json` linking all attempts. Each attempt has its own sequence video;
the UI's sequence download refers to the latest attempt, not an edited montage of retries.
Starting a new take uses the original loaded plan. Learned inference is synchronous on
the simulation owner thread; large optional models can reduce live preview responsiveness.
