# Architecture and dependency assessment

Verified against upstream documentation on 7 November 2026. Milestone 1 intentionally
uses a conservative pinned renderer API rather than adopting newer rendering systems.

## Dependencies

| Component | Choice | Compatibility / decision |
|---|---|---|
| Runtime | Python 3.11, native arm64 | Use an isolated venv; do not mix Rosetta packages. |
| Physics and rendering | mujoco==3.3.7 | arm64 wheel installed locally; native passive viewer requires mjpython. OpenGL renderer. |
| Arrays and geometry | numpy==2.2.6 | Exact minimal baseline; no SciPy needed for a small sampled candidate search. |
| Images and preview | opencv-python==4.12.0.88 | arm64 wheel; RGB/BGR conversion at I/O boundaries. |
| Detection, optional later | Ultralytics, yolo26n-seg.pt | Model exists. Adds PyTorch and substantial dependencies. Start CPU; test MPS before enabling it. COCO classes will not identify a synthesizer specifically. |
| Pose, optional later | MediaPipe Tasks Pose Landmarker Lite | Current setup documents Apple Silicon and Python 3.11 support. Use a local .task asset and CPU baseline; do not assume Python GPU delegation on Mac. |
| Creative planner, later | Ollama + gemma4:e2b | Explicit local model tag supports images. macOS 14+; Apple M CPU/GPU supported. Fetch weights once, then run locally. Download size is not total runtime RAM. |
| UI, later | Gradio bound to 127.0.0.1, share=False | Deferred until exploration works; native viewer already demonstrates M1. |
| Validation, later | Pydantic with the UI, or explicit schema validation | Validate both JSON structure and semantics before accepting plans. |

Only the three pinned Python packages above are required now (plus their transitive
dependencies). YOLO and MediaPipe are independently disabled by default in the planned
configuration. Do not install multiple OpenCV wheel variants into one environment;
resolve their optional dependencies together when adding learned perception. A combined
learned stack has not yet been installed/tested. If those modules cannot run, use
segmentation/body identities and synthetic skeleton landmarks. No replacement model
is necessary for M1. If Gemma does not fit available memory, retain a clearly labelled
rule-based demo planner while reducing context/images or selecting a smaller local
vision model after measuring it. Do not silently substitute a cloud endpoint.

ROS, Isaac, Modal, torch in the base environment, an Ollama Python SDK, a database,
SLAM, SciPy and a web framework are unnecessary at this stage. Later use the standard
library HTTP client to call loopback Ollama with a timeout and limited context.

Upstream references:
- [MuJoCo Python / macOS viewer](https://mujoco.readthedocs.io/en/stable/python.html)
- [YOLO26 supported models](https://docs.ultralytics.com/models/yolo26/)
- [Segmentation and COCO classes](https://docs.ultralytics.com/tasks/segment/)
- [MediaPipe Python platforms](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
- [Pose Landmarker Python](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [Gemma on Ollama, explicit E2B tag](https://ai.google.dev/gemma/docs/integrations/ollama)
- [Ollama macOS requirements](https://docs.ollama.com/macos)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

## Repository and contracts

Implemented:
- `assets/room.xml`: physical rig, camera, semantic body/geom names, static scenery.
- `simulation.py`: Simulation owns model/data/renderer; command(**axis_targets),
  step(seconds), reset(), capture() -> Observation, save(path), close().
- `Observation`: RGB uint8 HxWx3, depth metres HxW, segmentation int HxWx2,
  K 3x3, T_world_camera 4x4, simulation time, measured named joint positions.
- `__main__.py`: capture and native interactive demonstration.
- `tests/`: actuator validation, dynamic movement, reset and optical-axis convention.

Add these modules incrementally, not empty abstractions now:

| Module | Input → output contract |
|---|---|
| exploration.py | Axis limits + coverage state → bounded named joint targets; observations → viewpoint records |
| perception.py | Observation → Detection(object ID, category, bbox, mask, score, source, landmarks) |
| mapping.py | Detections + calibrated depth → SceneMap with persistent IDs, world bounds, timestamps, seen-from IDs and visibility state |
| gemma.py | Brief + compact map + 2–4 images + limits → raw structured response |
| planning.py | Raw response → validated ShotPlan with ordered ShotObjective records |
| optimization.py | ShotObjective + SceneMap + current joints → feasible target configuration + measured errors |
| control.py | Target configuration → rate-limited actuator commands + settling status |
| teleprompter.py | Objective + errors + instruction history → one instruction, completion condition and cooldown |
| app.py | Local UI events → simulation-owner queue; immutable snapshots → live UI |

Named joint targets allow future X/Y rails without changing planners to depend on
four-element positional vectors. The current rig only translates vertically: pan,
tilt and roll cannot provide an orbit or arbitrary close-up. FOV remains fixed during
optimisation. Reject infeasible requests or request subject/product repositioning.

## Data flow

1. Simulation owner steps physics and records calibrated frames at reachable poses.
2. Deterministic height/pan/tilt sampling records frames after settling; later choose
   additional views using observed surface/semantic coverage and novelty thresholds.
3. Geometry-ID fallback groups visible geoms by semantic body. Optional learned
   detections augment it and preserve provenance. Back-project valid segmented depth
   pixels through K and T; associate IDs across views. Never treat unseen simulator
   geometry as if the robot observed it. Partial surface bounds are not full object bounds.
4. SceneMap retains last-seen objects, observed bounds, confidence, landmarks and
   viewpoint references; distinguish out-of-view, occluded and currently visible states.
5. Brief + compact map + representative RGB + rig limits go to local Gemma once.
6. Validate shot types, known target IDs, finite durations, normalized composition
   coordinates, background IDs, instructions and measurable error thresholds. Reject
   motor values from the model. Retry once, then show an actionable local error/fallback.
7. Score reachable candidates for target visibility, position, projected scale,
   headroom, background, occlusion and motion cost; move through actuators and settle.
8. Observe again. If camera corrections cannot satisfy thresholds, issue one subject
   instruction, apply an explicitly scripted mocap response, then reobserve before
   acknowledging completion. Cooldowns, progress checks and an iteration budget avoid
   repeated/contradictory instructions. Replan only on persistent infeasibility.

Keep simulation time for reproducible observations; add wall time/session IDs when
persisting exploration. Save a small number of full observations; UI may receive
throttled RGB snapshots. Start at 640x480, 30 Hz preview target, slower perception,
and occasional LLM calls; these are targets, not measured performance guarantees.

## Milestone gates

1. Load four-axis rig; reject invalid commands; move using actuators; inspect RGB;
   verify finite metric depth, visible segmentation IDs and camera calibration.
2. Repeat deterministic survey; save unique view IDs/metadata and report coverage.
3. Map one person/product across several frames without duplicate instances; demonstrate
   segmentation fallback with learned modules disabled.
4. Local image-conditioned Gemma response passes schema and semantic validation;
   malformed/unavailable responses fail gracefully.
5. Reachable shot improves measured framing, with explicit infeasibility handling.
6. A scripted subject adjustment changes observations and satisfies an instruction
   completion check; refinement terminates within a bounded budget.
7. Local UI orchestrates restart, survey, brief, plan and execution and displays
   before/after captures and actual completion metrics.

## Milestone 2 implementation

`exploration.py` now provides `survey_poses(limits)`, `move_and_settle(sim, target)`,
`SurfaceCoverage.measure(observation)` and `explore(sim, directory, progress=...)`.
The progress callback receives one candidate record and the candidate count, allowing
a later UI to display progress without owning the renderer. The current CLI reports
status and cumulative observed voxels after each candidate. `Simulation.save` can
persist an already-evaluated Observation, avoiding a second render or mismatched data.

Coverage uses only rendered depths and segmentation, never hidden world geometry.
Geom IDs are diagnostic provenance, not semantic object IDs. The survey manifest
records unsaved candidates as redundant, empty or unsettled. Only retained frames
contribute to the persisted coverage union. No adaptive next-view policy is added yet.

## Milestone 3 implementation

`perception.py` implements SemanticIdentity, Detection, SegmentationPerception and
PerceptionPipeline. Geometry-ID masks are grouped by configured identity and converted
to world-space sampled surface points. Optional ObjectDetector/PoseEstimator protocols
are independent; failure is isolated and output remains supplemental. No learned
adapters or model packages have been added to the minimal dependency set.

`mapping.py` implements SceneMap.update/save, conservative prior-surface visibility,
saved-observation loading and map_survey. Persistent IDs prevent duplicates; dynamic
geometry is replaced rather than merged across movement. Per-object storage is capped,
and partial bounds/provenance/timestamps are explicit. SceneMap rejects duplicate view
IDs and backwards simulation time. The `map` CLI operates on M2 datasets without
opening a GL context and writes map JSON, sparse arrays, masks and visual diagnostics.

## Milestone 4 implementation

`planning.py` owns SHOT_PLAN_SCHEMA and parse_plan(content, scene, allowed_views).
A small strict validator covers the schema's used types/ranges/required fields without
adding a dependency. Semantic checks cover known object/person/view IDs, unique IDs,
non-conflicting target/background references and duration/hold consistency.

`gemma.py` owns prepare_context, OllamaClient and generate_plan. Input preparation
matches saved calibration before selecting representative RGBs. The HTTP client uses
standard-library urllib, fixed loopback/model, bounded response size and timeout,
nonstreaming schema-guided generation, temperature zero, a fixed seed and a 16K context
budget. This improves repeatability but does not guarantee deterministic LLM output.
No cloud or automatic model installation path exists. Raw responses and provenance
are saved; only locally validated JSON is released as a shot plan. Invalid model
content is retried once; unavailable services fail explicitly. No simulation coupling
or continuous inference is introduced. Live model testing awaits the user's install.

## Milestone 5 implementation

`control.py`: move_to uses bounded-rate quintic setpoints and physics stepping,
checks measured joint ranges every step and requires settling before capture.

`optimization.py`: project/evaluate compute normalized framing errors and observed
surface visibility; optimize ranks deterministic scratch-data forward-kinematic
proposals, then renders actuator-reached candidates in a separate probe simulation.
No hypothetical configuration is written into execution MjData. Required backgrounds
and headroom are checked; perspective and motion cost are soft score terms. Sparse
visibility is an explicitly named proxy for the plan's full-area visibility objective.

`execution.py`: revalidates the plan/map relationship, loads bounded sparse references,
executes chosen targets, samples the hold criteria over each shot duration and writes
audit data and before/after images. Creative instructions remain pending for M6.
Completion is scoped to observed geometry and sampled holds; feasibility is not
certified by this bounded search. Current implementation assumes a stationary scene
and the supplied four-axis MJCF rig. No language model calls are made in this loop.

## Milestone 6 implementation

`teleprompter.py` owns Instruction/Teleprompter state, a conservative creative phrase
adapter, geometric scale corrections, bounded mocap responses and independent
segmented-depth movement checks. Script completion is not observation completion.

`refinement.py` owns the per-shot action budget, one-at-a-time lifecycle, cooldown,
progress/oscillation guards, fresh dynamic surfaces and SceneMap updates. After a
verified response it reuses the geometric optimiser and actuator controller against
the updated mocap scene. Unsupported creative actions remain pending. Replanning is
reported as needed rather than making an unbounded model call.

Execution enables this workflow with --refine; --preview overlays the instruction.
The product is now a mocap body as well as the person; geom/body identities and the
four camera DOFs are preserved, so saved surveys remain readable. Scripted actor
motion is kinematic demo behaviour without articulated grasp/walk physics. The local
map uses observed geometry; mocap coordinates are used only to animate the response.
