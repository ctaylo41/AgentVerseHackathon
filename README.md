# Robopod — local AI cinematographer

Milestones 1–6 are implemented: a furnished MuJoCo studio, synthetic person and synthesizer,
actuated vertical rail plus pan/tilt/roll gimbal, native controls, and calibrated RGB,
metric depth and geometry-ID segmentation captures, plus deterministic exploration
and observed-surface coverage, fallback perception and sparse semantic mapping.
Local Gemma shot-planning integration supplies validated objectives for execution.
Geometric shot optimisation and actuator execution are implemented; cooperative
teleprompter refinement is now available with `--refine`.
No model downloads or cloud calls occur at runtime.

## Run on Apple Silicon (Python 3.11, macOS 14+ baseline)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m robopod capture
mjpython -m robopod viewer
```

In the native viewer, expand the right-side **Control** panel to move the actuators.
The separate RGB window shows the actual rig view: **S** saves a capture, **R** resets,
**Q** exits. Keep that window focused for these keys. Native viewer mouse navigation
only changes the inspection view; it does not teleport the rig camera.

```sh
python -m robopod capture --camera-height 1.8 --pan 0.15 --tilt -0.15 --roll 0.05
python -m robopod capture --width 1280 --height 720 --fovy 55 --output outputs/wide
python -m unittest discover -s tests -v
```

Angles are radians; positive tilt looks up, positive pan turns toward world -X.
Height is metres. Out-of-range commands fail before updating controls. Position
servos move the rig via physics; reset alone writes the home configuration. Physical
limits are soft MuJoCo constraints, not a guarantee of zero transient overshoot.
The vertical servo has a small gravity-dependent steady-state error; metadata always
records measured positions. Camera FOV is configured per run, not a fifth actuator.

Each capture writes `rgb.png`, `geometry.npz` and `metadata.json`. The NPZ contains
`depth_m` (optical-axis depth), `segmentation` (object ID, object type), `intrinsics`
and `world_from_camera`. Segmentation IDs identify geoms, not semantic instances;
the metadata maps geoms to bodies for the future fallback perception module.
Camera coordinates are OpenCV right/down/forward; world coordinates are Z-up.
Use `p_world = T_world_camera @ [depth * inv(K) @ [u,v,1], 1]`.

The room is intentionally primitive, so learned detectors may miss the person or
product. MuJoCo geometry identities will provide a dependable semantic fallback.
The person is currently a stationary mocap body, ready for scripted movement later.

## macOS rendering

Use `mjpython` for the passive native viewer (required by MuJoCo on macOS).
The pinned renderer uses OpenGL, not Metal. Do not set Linux EGL/OSMesa backends.
Offscreen capture uses `python` and still needs a working macOS graphics session.
Rendering belongs to one thread; a future Gradio interface must send commands to
that simulation owner rather than call the renderer from arbitrary web callbacks.

See [architecture and dependency assessment](docs/architecture.md) for interfaces,
compatibility references, data flow and milestone acceptance checks.

## Verification on this Mac

Python 3.11 arm64 installation, two physics/control tests, and a 640x480 RGB/depth/
segmentation capture passed. The saved RGB was visually inspected. Interactive viewer
controls have not yet been manually exercised. macOS emitted an `ARB_clip_control`
warning: this driver uses reduced depth precision. Background pixels (`segmentation`
ID -1) contain far-plane depths and must be excluded from reconstruction. The near/far
settings are scaled by MuJoCo's model extent, not literal metre distances.

For the exact tested transitive environment, install `requirements-lock.txt` before
`pip install -e .`. A sandbox may block CoreGraphics; run captures from a normal
local terminal in a logged-in desktop session.

## Milestone 2 — deterministic exploration

```sh
source .venv/bin/activate
python -m robopod explore
# Optional explicit NEW directory and coverage settings:
python -m robopod explore --output outputs/my-survey --voxel-size 0.1 --min-novelty 0.03
```

The default output is a new timestamped survey directory. Explicit output directories
must not already exist, so earlier captures cannot be overwritten. Exploration starts
from the simulator's home state and visits 18 interior poses: three heights, three
pan angles, two tilts, zero roll, with serpentine pan ordering. All movement uses
position actuators. No exploration step writes joint positions or teleports the camera.
The capture/viewer pose flags do not configure the exploration grid.

Each pose must remain near its target with low velocity for 0.2 simulated seconds
before capture. Height tolerance is 4 cm to accommodate the current servo's gravity
sag; angular tolerance is 0.015 radians and velocity threshold is 0.015 in each axis's
SI units per second. A five-second simulated timeout prevents hangs. Use
`--settle-timeout` to change it. Failed poses are recorded without saving observations;
a run with settling failures exits nonzero. Physics runs faster than wall-clock time.

Outputs:
- `view_000/`, etc.: RGB PNG, compressed depth/segmentation/calibration NPZ and JSON
  metadata. IDs correspond to candidate order, so skipped views leave gaps.
- `manifest.json`: every candidate, target/measured joints, simulation timestamp,
  settling errors, visible geom IDs, novelty, cumulative coverage and relative paths.
  It is updated after every candidate and records failures/interruption status.
- `coverage.npz`: sorted integer world-space voxel indices, voxel size in metres,
  and observed geom IDs. Cell centres are `(voxel_indices + 0.5) * voxel_size_m`.
- `contact_sheet.jpg`: labelled thumbnails of retained observations.

Coverage is the union of **observed surface voxels**, not a percentage of the whole
room or free-space occupancy. Every fourth pixel is back-projected through measured
camera calibration. Background/non-geom pixels and invalid depths are excluded.
The default cell size is 10 cm. A frame is redundant if fewer than 3% of its cells
are new and it reveals no new sampled geom IDs. Redundant frames are evaluated and
logged but not stored; coverage refers to stored observations only. Set
`--min-novelty 0` to retain every nonempty settled candidate. These are geometric
coverage measures, not semantic mapping or adaptive exploration (later milestones).
Resolution, pixel stride and voxel size affect counts; compare runs with identical
settings. Occluded surfaces remain unknown, and the fixed rail cannot see everywhere.

Milestone 2 verification on this Mac: seven unit tests passed; two full 18-pose
surveys produced identical manifests, RGB files and geometry arrays. Each retained
12 views, skipped six redundant views and covered 9,895 surface cells at the default
settings. Recomputing coverage from saved observations exactly matched coverage.npz.
The contact sheet was visually inspected. Counts are scene/configuration specific.

## Milestone 3 — fallback perception and semantic mapping

Map any completed survey without rendering again or downloading models:

```sh
python -m robopod map --survey outputs/milestone2-survey
# To rerun, choose a NEW output directory:
python -m robopod map --survey outputs/milestone2-survey --output outputs/my-map
```

By default, mapping writes a new `semantic_map` directory inside the survey. Existing
output directories are refused. This stage runs offline on the saved RGB, depth,
segmentation and calibration. It does not need macOS graphics access.

The fallback registry in `perception.py` supplies known simulation identities: presenter,
synthesizer, desk, plant, floor, walls and art. It groups geoms belonging to one object
before producing masks and bounding boxes. Only objects with valid visible depth are
added. The rig and unconfigured names are excluded. Change the explicit registry for
new scenes; names provide labels, never hidden positions or dimensions.

Outputs:
- `scene_map.json`: persistent IDs, categories, approximate world positions, partial
  observed bounds, confidence/source, first/last seen times, seen-from view IDs,
  last pixel boxes, pose proxies and calibrated viewpoints.
- `object_surfaces.npz`: up to 2,048 surface points per object, keyed by object ID,
  sparsified into 8 cm cells. All geometry comes from depth back-projection.
- `map_topdown.png`: XY map preview of observed surfaces.
- `view_*/detections.json`, `masks.npz`, `annotated.jpg`: per-view detections, binary
  instance masks and visual overlays. Boxes use exclusive right/bottom coordinates.
- `mapping_report.json`: processing status and visibility history by viewpoint.

Repeated observations share stable scene-local IDs (`body:person`, `body:product`,
etc.). Static objects accumulate observed bounds; movable presenter/product geometry
uses the latest observation to avoid motion trails. Positions are partial surface
bounding-box centres, not guaranteed physical centres. Do not treat these bounds as
complete collision volumes. Starting a different scene/reset requires a new map.

Visibility is `visible` when detected. Otherwise prior surface samples are projected
into the current image: `out_of_view` when none project inside, `occluded` when all
in-image samples have a nearer valid depth, and `not_observed` when ambiguous.
These missing-object states refer to last-known geometry; a moving unseen subject
could have moved elsewhere. They are estimates, not ground-truth hidden-state access.
Old positions retain their `last_seen` timestamp. Current landmarks clear on absence.

Human head/torso/arm markers are **visible-surface proxies**, not anatomical joints,
shoulder estimates or a full MediaPipe skeleton. They are only emitted for observed
parts. Identity confidence 1.0 means exact segmentation/registry association; it does
not imply complete shape or reliable learned classification.

### Optional learned perception

No new dependencies are needed. `PerceptionPipeline(geoms, detector=None,
pose_estimator=None)` independently accepts an object backend implementing
`detect(rgb)` and a pose backend implementing `estimate(rgb)`. Both return lists of
JSON-safe dictionaries. With `None`, nothing is imported or downloaded. Backend errors
are recorded as warnings while segmentation still works. Supplemental learned output
is saved separately; it cannot overwrite simulator IDs or create duplicate map objects.
The same injection points are available on `map_survey`.

Actual YOLO/MediaPipe adapters and model weights remain deferred; this milestone ships
and tests the fallback plus optional extension interfaces, not learned-model inference.
`SceneMap.update(detections, observation, view_id)` and `PerceptionPipeline.process`
can also be used in a future live loop. The CLI currently replays saved survey views.

## Milestone 4 — local Gemma shot planning

Once you have installed Ollama, start it and download the explicit local model tag:

```sh
ollama pull gemma4:e2b
# If the Ollama app is not already serving:
ollama serve
```

In another terminal, from this repository:

```sh
source .venv/bin/activate
python -m robopod plan \
  --survey outputs/milestone2-survey \
  --brief "I want to film a professional tech review. Show me and the product, then capture a close-up of the product while I explain it."
```

The scene map defaults to `SURVEY/semantic_map/scene_map.json`. Override it with
`--scene-map PATH`. `--images 3` selects 1–4 representative images; `--timeout 180`
sets the per-request timeout (maximum 600 seconds). `--output PATH` selects a NEW
output directory; the default is a timestamped `plan-*` directory inside the survey.
No graphics window is needed. No dependency installation is needed beyond M1.

The planner uses only `http://127.0.0.1:11434` and `gemma4:e2b`; environment proxies
and HTTP redirects are disabled. It checks that the local model is installed and
advertises vision support. It does not download models, start services, or switch to
cloud inference. An unavailable service/model produces an actionable error and no
validated plan. Model quality/performance still needs verification with your install.

Input preparation verifies survey/map camera metadata matches, chooses views by
semantic coverage and target prominence with a small viewpoint-diversity preference,
and encodes up to four JPEGs at a maximum 640-pixel side. Images are ordered and labelled
by view ID in the prompt. The model sees a compact eight-object summary for the sample
scene, partial geometry caveats, the brief and measured rig limits—not depth arrays,
all survey frames or full sparse point clouds. Selection is deterministic and currently
scene-based, not a learned interpretation of the brief.

`planning.py` defines the strict schema and validates both structure and semantics:
- Ordered 1–6 shot objectives; unique shot IDs; wide/medium/close-up/detail types.
- Known targets with normalized centre and frame-height composition goals.
- Perspective preference, required background object IDs and a supplied starting view.
- Durations, concise instructions addressed to known people, and feasibility notes.
- Measurable centre/scale errors, visible fraction, headroom and stable-hold criteria.

All fields are required; extra fields (including motor commands), nonfinite numbers,
unknown IDs, duplicate targets and invalid ranges are rejected. The hold interval
cannot exceed the shot duration. Headroom applies to person targets only. Visibility
fraction is a desired metric, not a claim that complete object shape is already known.
The model receives the schema via Ollama's `format` field. One invalid response gets
one corrective retry; transport errors are not automatically retried. Inference is
occasional and outside the simulation control loop.

Outputs:
- `shot_plan.json`: written **only after validation succeeds**.
- `context.json`: compact input and selected viewpoint references, without base64 images.
- `shot_plan.schema.json`: exact contract sent to Ollama.
- `response_1.txt` (and optionally `response_2.txt`): raw model JSON text for debugging.
- `planning_report.json`: validation/errors, model digest/capabilities, source map hash
  and selected image paths/hashes. Failure never creates a valid-looking fallback plan.

Validation checks objective structure and references, not physical achievability or
creative quality. Milestone 5 will optimise reachable poses and evaluate these goals;
Milestone 4 never moves actuators. The fixed rail/FOV limitations are explicit in the
prompt, and requested close-ups may require subject/product repositioning.

Verification so far: 18 tests pass, including malformed output, semantic validation,
one-retry repair, unavailable model/server, and request schema/image wiring. The actual
survey selects three images and produces a compact context successfully. Live Gemma
inference is intentionally pending your Ollama/model installation; simulated responses
in tests are not represented as real Gemma output.

API references: [Ollama chat](https://docs.ollama.com/api/chat),
[structured output](https://docs.ollama.com/capabilities/structured-outputs),
[Gemma 4 E2B](https://ollama.com/library/gemma4:e2b).

The generation schema now constrains object/view references to the current scene and
human-instruction subject IDs to known people. Product-handling instructions must be
addressed to the presenter, not the product. With no people, human instructions must
be empty. Repair errors identify the offending field and allowed person IDs. The
saved schema is the exact scene-specific schema used for that request.

## Milestone 5 — geometric optimisation and actuator execution

Run a validated plan against its scene map (no Ollama calls):

```sh
python -m robopod execute \
  --survey outputs/milestone2-survey \
  --plan outputs/milestone2-survey/plan-20261107-154753-305722/shot_plan.json
```

Replace the plan path with your generated `shot_plan.json`. `--scene-map PATH` can
replace `--survey`. Optional `--output NEW_DIRECTORY` controls output location;
`--preview` opens an RGB preview of the executing rig (Q cancels). Preview renders
selected motion frames; execution runs in simulation time, not real-time pacing.
`--width`, `--height`, `--fovy` configure the execution camera; FOV is fixed throughout
the run. This mode uses offscreen macOS rendering and can be run with normal `python`.

A hand-authored diagnostic example (not Gemma output) is included:

```sh
python -m robopod execute --survey outputs/milestone2-survey \
  --plan examples/actuator_demo_plan.json
```

The optimiser samples an interior height/pan/tilt/roll grid, then refines its best
geometric candidate. It projects sparse observed surfaces using forward kinematics
in private scratch data. This hypothetical search never writes the executing rig's
joint positions. Scores include target position/scale, headroom, field-of-view
coverage, required backgrounds, a soft perspective preference and movement cost.
The current pose and the plan's preferred survey pose are also considered.

Eight shortlisted poses are reached through smooth actuator trajectories in a
separate probe simulator and evaluated with rendered RGB/depth/segmentation. These
checks measure visible boxes and occlusion of reference surface points. The best
measured pose is selected, preferring satisfied observed criteria; the current pose
is retained if candidates do not improve it. The executing rig then reaches that
configuration through position actuators. No execution step teleports the camera.

The quintic actuator setpoints have bounded rates: 0.35 m/s vertical, 0.6 rad/s pan,
0.5 rad/s tilt and 0.35 rad/s roll. These are command-rate limits, not guarantees on
instantaneous physical velocity. Search margins leave 6 cm above the rail's lower
limit and 3 cm/radians at other boundaries. Every physics step checks actual joint
positions; violations stop the run. Settling permits existing servo gravity sag.
Predicted sag uses the current measured actuator offset, not an LLM estimate.
This is a four-axis rig implementation; extra axes need a search-policy extension.

For each shot the controller records before/after captures, then samples the framing
criteria at 10 Hz for the requested duration. Completion requires a consecutive
sampled hold interval. The result is `satisfied_observed_criteria` or `needs_adjustment`.
The latter includes subject/product repositioning suggestions, but does not perform
human movement or claim that a finite search proves global infeasibility. Creative
human instructions are retained in the report and not executed until Milestone 6.

Important geometry limitation: complete object silhouettes are unavailable from the
sparse map. `min_visible_fraction` is currently evaluated as a **visible reference
surface sample fraction**, not true full-object projected area. Reports explicitly
label this proxy. Movable-object references are refreshed from the initial camera
observation; this milestone assumes the subject/product then remain stationary.
Headroom uses the top of a detected person mask and requires a visible head proxy.
Required background visibility needs at least 16 segmented pixels. Perspective is a
soft preference, and shot type is descriptive; numeric criteria govern execution.

Outputs:
- `execution_report.json`: overall/per-shot status and before/after scores.
- `trajectory.json`: measured joints and actuator commands along the executed motion
  and holds (checks run every physics step; saved samples are less frequent).
- `reference_surfaces.npz`, `validated_plan.json`: the actual reference geometry and
  revalidated objectives used in the run.
- `shot_000/result.json`, etc.: candidate scores, chosen commands, measured framing,
  hold samples and repositioning requests.
- `shot_000/before/`, `after/`: full calibrated captures.
- `shot_000/comparison.jpg`: labelled before/after view.

Input plans are revalidated before motion. If the plan's planning report includes a
scene-map hash, it must match. Output directories cannot already exist. Exit codes:
0 = all observed criteria satisfied; 2 = execution finished but some shots need
adjustment; 1 = input, motion or I/O failure. No failed/unsatisfied shot is silently
reported as a completed cinematic objective.

Milestone 5 verification on this Mac: 23 tests pass. The two-shot diagnostic run
satisfied the presenter framing for a two-second sampled hold (loss 1.506 → 0.309).
The oversized product close-up improved but correctly requested moving the synthesizer
closer. All 336 saved trajectory samples stayed within measured and commanded joint
limits; runtime limit checks additionally ran every physics step. The existing Gemma
plan improved from loss 3.738 → 3.037 and remained explicitly unsatisfied. Rendered
before/after images were visually inspected. No human instructions were executed.

## Milestone 6 — teleprompter and closed-loop cooperation

Enable scripted subject responses and bounded refinement:

```sh
python -m robopod execute --survey outputs/milestone2-survey \
  --plan examples/cooperative_demo_plan.json --refine --preview
```

Use your own validated plan with the same `--refine` flag. The cooperative example is
hand-authored, not Gemma output: a presenter step, followed by a product shot requiring
repositioning. `--max-adjustments 3` sets the shared per-shot creative/geometric action
budget (0–6). Without `--refine`, execution retains the Milestone 5 camera-only path.

Each shot first optimises the camera. The teleprompter then presents one supported
creative instruction at a time; when framing is still wrong, geometric scale errors
can generate a closer/farther request. The console labels the source, and `--preview`
overlays the current instruction on the executing camera image. `teleprompter.json`
provides an active instruction snapshot and final history for a future web interface.

Supported scripted creative phrases (case-insensitive, optional final punctuation):
- “Step to your left/right” or “Step slightly to your left/right”: 25 cm.
- “Step closer to the camera” or “Step back”: 30 cm.
- “Raise the product” or “Move the product higher”: 15 cm.

Left/right is the presenter's left/right while facing the initial camera (world +X/-X).
The actor does not currently turn or articulate joints. Other creative instructions,
such as smiling, pointing or looking at a specific object, are displayed as requiring
manual response and recorded as pending. They are never silently marked completed;
pending creative actions prevent a shot from being reported fully satisfied. There
is not yet a manual acknowledgement/control UI to resolve them within a run.

The existing person and now the product are kinematic MuJoCo mocap bodies. A scripted
response interpolates their positions smoothly over one simulated second within a
bounded demo workspace. The product may appear carried/floating: walking, grasping,
contact-safe human motion and hand articulation are not physically simulated. Camera
movement continues exclusively through its actuators and per-step joint-limit checks.
Mocap movement is explicitly labelled in reports and never confused with robot motion.

Completion uses fresh segmentation and calibrated depth before/after movement:
the requested object must be observed in both frames and its visible surface median
must shift at least 5 cm in the requested world-axis direction (or 25% for requests
under 20 cm). A script acknowledgement alone cannot complete an instruction. This is
an observed-motion check, not a claim to infer true body pose behind occlusion.

After each response, dynamic geometry is replaced from fresh observations, the local
semantic map is updated, and camera optimisation is repeated using the changed scene.
The probe simulator receives the current mocap scene, so it cannot score the original
subject placement by mistake. Framing is remeasured; the final shot must still pass
the sampled hold check. No language model runs inside this loop.

Loop safeguards:
- One active supported instruction, with a one-second simulation-time cooldown.
- Repeated same-direction adjustments require observed prior completion; failed
  requests and direction reversals are blocked for the rest of the shot.
- Stop on missing response evidence, less than 0.01 framing-loss improvement after
  a geometric adjustment, no supported correction, or the action budget.
- Unresolved framing sets `model_replan_needed` in the report; it does not automatically
  invoke Gemma or claim success. Geometric corrections currently handle scale only;
  other unresolved framing errors remain explicit.

Additional per-shot outputs:
- `teleprompter.json`: instruction source, lifecycle, timestamps and completion evidence.
- `refinement.json`: each response, before/after/camera metrics, stop reason and pending
  creative requests. `result.json` includes this summary too.
- `response_*/`, `refined_*/`: full calibrated captures after subject response and
  subsequent camera correction.
- `scene_map.json`, `object_surfaces.npz`, `map_topdown.png`: fresh local map from the
  refinement observations, with dynamic geometry replaced rather than trailed.
- `refined_reference_surfaces.npz`: geometry used for final framing evaluation.

Verified on this Mac: 28 tests pass. The cooperative demo completed both shots; a
creative presenter step and two geometric product responses were verified from depth.
The product-shot framing loss improved from 0.918 to 0.191. A stricter 50%-height
close-up improved through three responses but stopped at the budget as unsatisfied,
confirming that bounded refinement does not manufacture completion.
