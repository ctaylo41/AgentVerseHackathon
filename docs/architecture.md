# Current architecture

## Local runtime

Python 3.11, mujoco 3.3.7, NumPy 2.2.6 and OpenCV 4.12.0.88. The application uses
stdlib HTTP and threads, with no frontend build step or cloud service. Ollama receives
occasional loopback image-conditioned `gemma4:e2b` requests. macOS rendering uses
OpenGL; the native passive viewer requires `mjpython`. Learned perception is optional
through lazy local-file YOLO segmentation and MediaPipe Tasks pose adapters.

## Modules and contracts

| Module | Responsibility |
|---|---|
| simulation.py / assets/room.xml | Six named actuators, RGB/depth/segmentation and calibration; sole model/data/renderer owner |
| artifacts.py | Versioned scene/robot identity and strict compatibility checks |
| safety.py | Full modeled rig/environment clearance, continuous path certification and runtime geometry checks |
| control.py | Bounded-rate quintic setpoints, settling, joint/clearance checks at physics rate |
| exploration.py / adaptive.py | Eighteen baseline poses plus bounded observation-driven next views, coverage/novelty and saved evidence |
| perception.py / learned.py | Persistent segmentation instances, unknown objects, part targets, optional learned label/pose association |
| mapping.py | Persistent IDs, partial surface bounds, visibility inference and versioned sparse maps |
| planning.py / motion.py | Strict shot schema, semantic references and motion intent/feasibility validation |
| gemma.py / brief.py | Compact context, sequence outline, per-shot generation, repair and brief constraints |
| optimization.py | Scratch-data pose proposals, actuator/rendered shortlist checks, exact-amplitude motion candidates |
| teleprompter.py / actors.py / refinement.py | One-at-a-time scripted responses, observed completion and bounded camera/subject refinement |
| replanning.py | Bounded Gemma failure escalation, immutable objectives, fresh visual evidence and retry lineage |
| execution.py / recording.py | Shot monitoring, recovery, reports, progress events and clean MP4 exports |
| app.py / assets/studio.html | Local UI, playback control, copied snapshots, current shot metrics and allowlisted exports |

## Ownership and playback

The main thread owns executing MuJoCo state and OpenGL. HTTP threads never access
simulation objects. They read copied JSON/images, enqueue workflow actions and set
pause/stop/restart events. A checkpoint on the owner thread freezes simulation time
while paused; resume unblocks it. Stop raises a distinct workflow exception and closes
video writers/reports without resetting the plan or survey. Restart additionally
invalidates the planner epoch and reconstructs the session. Probe simulations inherit
the owner's checkpoint, so long searches also respond to pause/stop.

Gemma runs in a separate worker using saved artifacts only. Stale results cannot
replace the plan after restart. A stopped execution can be run again as a new take;
pause/resume is the mechanism for continuing the same take.

## Data and compatibility

Observations use OpenCV camera axes (right/down/forward), Z-up world metres, optical-axis
depth and exact capture intrinsics/extrinsics. Semantic identity never implies complete
shape knowledge. Static geometry accumulates; dynamic objects retain latest observed
surfaces. Every new persisted survey/capture/map carries a matching artifact identity.
The simulator captures its identity when loading the XML, so editing the file while
an old process runs cannot relabel that old simulator as a new version.

Planning and execution require current identities. The planner also checks survey/map
calibration; generated plans bind to the exact map hash. Unknown legacy versions must
be resurveyed, not guessed. Offline inspection/mapping of old captures remains possible.

## Motion and safety

Motion intent contains a mode, signed rig direction, amplitude in metres, a subject
lock ID (or none) and horizon policy. The schema rejects inconsistent directions and
unknown locks. Execution additionally checks axes, travel and duration/rate feasibility.
Named translation candidates use the requested amplitude exactly. A subject lock adds
pan/tilt compensation; original framing objectives are never rewritten during scoring.

PathValidator uses MuJoCo signed primitive-geometry distance, independent of contact
masks or image occlusion. It recursively certifies joint-space intervals using a
conservative bound on translation plus rotational surface displacement. Each certified
interval has at least 2 mm modeled clearance even between samples. Clear endpoints
alone are insufficient. All executing motions receive preflight certification and
measured per-physics-step clearance checks. Rig/environment contact is enabled;
connected assembly self-overlaps are intentional and excluded. Certificates concern
the modeled solids in the current scene snapshot, not hardware or unseen mechanisms.

## Execution evidence and footage

The executor publishes shot_started, measurement, shot_completed and execution_finished
events. The UI highlights the active shot, shows original error thresholds, publishes
completed results immediately and serves only media registered by these events.
There is no arbitrary file path endpoint. An overlay switch selects copied annotated
or clean live images. Per-shot comparisons compare the same objective before/after.

Clean MP4 uses the same sampled observations as shot evaluation, 10 fps at the simulation
camera resolution. It excludes planning, transitions, recovery and pauses; there is no
audio. Writers close on success, stop and failure. Partial sequence footage remains
available after a user stop. Playback codec support depends on the local player.

Completion requires a good final frame and consecutive terminal hold, all human actions
resolved, and requested movement executed. Persistent errors interrupt motion and allow
at most two camera recoveries. Unsupported actions and impossible framing stay explicit;
no scoring adjustment manufactures success. Replanning sends measured failure evidence and a
fresh image to Gemma under a 0–2 request budget, then retries only failed shots. The revision
validator preserves all original geometric/motion objectives and original human requests.
Tests independently compare requested values with rendered boxes, exercise
mid-path obstacles, playback flags, artifact rejection and decoded video frames.


## Learned perception, scene discovery and actors

Unknown scene instances remain mapped with stable IDs. YOLO masks associate by IoU
(at least 0.25 and confidence 0.35) to segmentation instances; configured identities
stay authoritative. Unknown categories can acquire persistent learned labels. MediaPipe
landmarks vote for a person mask, then use valid in-mask depth. Neither learned backend
is imported unless a local model path is enabled. Backend failures preserve fallback
perception; user settings and reports expose warnings. No runtime weight downloads occur.

Adaptive ranking uses only observed centres, uncertainty, image area and previous views.
Forward kinematics predicts camera visibility; full-rig path checks remain a separate
constraint. New voxel gain controls convergence. It is deterministic for the same
observations/backends, bounded by budget, and cannot infer complete hidden geometry.

Product controls and keys are child semantic targets with independent pixel masks and
surface arrays; their parent still represents the whole product. Scripted actors retain
six camera joints: pose animation changes rigid child transforms and face-feature
geometry, with no extra camera axes or hidden-state completion checks. Probe models copy
these transforms before evaluating shots. Reset restores all actor and rig state.
Shoulder direction, face alignment, mouth corners, pointing direction and product-to-face
proximity are evaluated from visible segmented depth, with unresolved results when
landmarks are missing. Actors are kinematic demonstration scripts, not learned behavior.
