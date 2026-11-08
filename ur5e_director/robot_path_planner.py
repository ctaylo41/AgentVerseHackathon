"""Robot Path Planning and Real-Time Mink IK Control Module for Antigravity.

This module is the primary physics, inverse kinematics (IK), and visualization
engine for the Antigravity autonomous robotic cameraman system. It integrates:
  1. High-level shot plans parsed by Gemma 4 (`orbit`, `dolly_in`,
     `low_angle_hero`, `tracking`).
  2. Real-time 3D user tracking (simulating ToF LiDAR, IMU, or MediaPipe streams)
     mapped onto a MuJoCo mocap humanoid (`humanoid_target`).
  3. Relative camera trajectory generation (`user_pos - base_offset`) with
     spherical workspace reach clamping (`max_reach = 0.75m`) to prevent UR5e
     arm singularities.
  4. Differential Inverse Kinematics solved via `mink` (`FrameTask` for the
     `camera_site` end-effector + `PostureTask` for joint regularization).
  5. Split-view visualization: MuJoCo's interactive 3D passive viewer alongside
     an OpenCV Robot POV viewfinder window overlaid with live Digital Twin
     coaching feedback.
  6. Optional bi-directional WebSocket streaming (`--bridge`) via
     `AntigravityDirectorBridge`.

Typical usage example:
  ```bash
  python robot_path_planner.py --shot orbit --duration 20 --bridge
  ```
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import mink
import mujoco
import mujoco.viewer
import numpy as np

# Ensure local imports work whether executed from project root or inside ur5e_director/
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from user_director_twin import HUMANOID_TORSO_Z, UserDirectorDigitalTwin

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_mocap_z(pos: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Normalizes the humanoid mocap Z coordinate so its feet rest on the floor.

    In the MuJoCo `humanoid.xml` model, the mocap body origin is located at the
    torso center, which is at `Z = 1.282m` when the feet soles touch the floor
    at `Z = 0.0m`. If an external tracking stream provides floor-referenced
    (`Z < 0.4m`) or waist/pelvis-referenced (`0.4m <= Z < 1.1m`) coordinates,
    this function shifts `Z` into the torso frame.

    Args:
        pos: A 3-element sequence or NumPy array `[X, Y, Z]` representing the
            raw user position in meters.

    Returns:
        A 1D NumPy float array `[X, Y, Z]` with `Z` normalized to torso height.
    """
    p = np.asarray(pos, dtype=float).copy()
    if p[2] < 0.4:
        p[2] += HUMANOID_TORSO_Z  # Convert floor-referenced Z to torso frame.
    elif p[2] < 1.1:
        p[2] += HUMANOID_TORSO_Z - 0.857  # Convert pelvis-referenced Z to torso.
    return p


class RobotPathPlanner:
    """Translates shot plans and live user tracking into UR5e camera trajectories.

    Loads the master MuJoCo scene (`scene.xml`), configures `mink` differential
    IK tasks for the UR5e `camera_site`, computes look-at rotation matrices, and
    executes the real-time control loop.

    Attributes:
        model: Loaded `mujoco.MjModel` instance representing the studio, UR5e
            robot arm, and humanoid subject.
        data: `mujoco.MjData` instance holding dynamic simulation state.
        configuration: `mink.Configuration` wrapper around the model's joint
            coordinates for differential IK solving.
        end_effector_task: `mink.FrameTask` tracking the 6-DoF SE(3) target pose
            of the `camera_site` attached to the UR5e wrist.
        posture_task: `mink.PostureTask` regularizing joint angles toward the
            `"home"` keyframe to avoid unnatural elbow flips.
        tasks: List of active `mink` IK tasks passed to `mink.solve_ik`.
        renderer: Offscreen `mujoco.Renderer` (640x480) for capturing the live
            view through the robot-mounted `director_cam`, or `None` if headless.
    """

    def __init__(self, model_path: str = "scene.xml") -> None:
        """Initializes the MuJoCo simulation model, Mink IK tasks, and renderer.

        Args:
            model_path: File path to the master MuJoCo XML scene. Defaults to
                `'scene.xml'` inside `ur5e_director/`.
        """
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            resolved_path = MODULE_DIR / model_path
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(resolved_path))
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # Initialize Mink configuration and reset arm to the 'home' keyframe.
        self.configuration: mink.Configuration = mink.Configuration(self.model)
        key_names = [
            self.model.key(i).name for i in range(self.model.nkey)
        ]
        if "home" in key_names:
            self.configuration.update_from_keyframe("home")
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, self.model.key("home").id
            )

        # Primary 6-DoF task for positioning and orienting the wrist camera site.
        self.end_effector_task: mink.FrameTask = mink.FrameTask(
            frame_name="camera_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.6,
            lm_damping=1.0,
        )
        # Secondary posture task to keep the arm near a well-conditioned pose.
        self.posture_task: mink.PostureTask = mink.PostureTask(
            self.model, cost=1e-2
        )
        self.posture_task.set_target_from_configuration(self.configuration)

        self.tasks: List[Any] = [self.end_effector_task, self.posture_task]
        self.renderer: Optional[mujoco.Renderer] = None
        try:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception:
            self.renderer = None

    def compute_look_at(
        self,
        cam_pos: np.ndarray,
        target_pos: np.ndarray,
        up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        """Computes a 3x3 rotation matrix orienting the camera toward `target_pos`.

        In MuJoCo's camera convention, a camera looks down its local `-Z` axis,
        with `+X` pointing right and `+Y` pointing up in the image plane.

        Args:
            cam_pos: 3D position `[X, Y, Z]` of the camera in world coordinates.
            target_pos: 3D position `[X, Y, Z]` of the subject to look at.
            up: World up-vector used to keep the camera horizon level. Defaults
                to `[0.0, 0.0, 1.0]`.

        Returns:
            A `(3, 3)` orthonormal rotation matrix `R = [x_axis | y_axis | z_axis]`
            suitable for constructing a `mink.SO3` orientation target.
        """
        forward = target_pos - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        z_axis = -forward  # MuJoCo cameras look along local -Z.
        x_axis = np.cross(up, z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            # Handle degenerate collinear case when looking straight up/down.
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= norm_x
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def get_shot_offset(self, shot_type: str, t: float) -> np.ndarray:
        """Computes the time-parameterized camera offset relative to the user.

        The returned offset vector `base_offset` is subtracted from `user_pos`
        (`desired_cam_pos = user_pos - base_offset`) to maintain relative
        subject framing as the user moves.

        Args:
            shot_type: Active shot primitive (`'orbit'`, `'dolly_in'`,
                `'low_angle_hero'`, or `'tracking'`).
            t: Elapsed shot time in seconds.

        Returns:
            A 1D NumPy float array `[dx, dy, dz]` representing the relative
            camera displacement in meters.
        """
        if shot_type == "orbit":
            radius = 0.6
            return np.array(
                [
                    0.4 + radius * np.cos(t * 0.5),
                    radius * np.sin(t * 0.5),
                    0.55,
                ]
            )
        elif shot_type == "dolly_in":
            dolly_pos = max(0.2, 0.8 - (t * 0.05))
            return np.array([dolly_pos, 0.0, 0.50])
        elif shot_type == "low_angle_hero":
            # Large Z offset places the camera low (Z ≈ 0.43m) looking up at the torso.
            return np.array([0.5, 0.0, 0.85])
        elif shot_type == "tracking":
            return np.array([0.45, 0.15 * np.sin(0.4 * t), 0.55])
        else:
            return np.array([0.4, 0.0, 0.5])

    def clamp_to_workspace(
        self,
        cam_pos: np.ndarray,
        max_reach: float = 0.75,
        min_z: float = 0.25,
    ) -> np.ndarray:
        """Clamps a desired 3D camera position to the UR5e safe workspace envelope.

        Prevents the relative camera target `user_pos - base_offset` from pulling
        the UR5e arm beyond its physical reach (`0.85m` nominal max reach) or
        colliding with the floor plane (`Z < 0.25m`).

        Args:
            cam_pos: Unclamped 3D desired camera position `[X, Y, Z]` in meters.
            max_reach: Maximum radial distance from the robot base origin in
                meters. Defaults to `0.75`.
            min_z: Minimum allowed height above the floor plane in meters.
                Defaults to `0.25`.

        Returns:
            A 1D NumPy float array `[X, Y, Z]` guaranteed to lie inside the
            reachable forward hemisphere of the UR5e arm.
        """
        pos = np.asarray(cam_pos, dtype=float).copy()
        pos[0] = max(0.20, pos[0])  # Keep end-effector in front of base column.
        pos[2] = max(min_z, pos[2])
        dist = np.linalg.norm(pos)
        if dist > max_reach:
            pos = pos * (max_reach / dist)
        return pos

    def render_pov_hud(
        self,
        active_shot: str,
        user_pos: np.ndarray,
        cam_pos: np.ndarray,
        report: Dict[str, Any],
        t: float,
    ) -> bool:
        """Renders the Robot Camera POV window with a live Digital Twin HUD.

        Captures an RGB image from the wrist-mounted `director_cam`, overlays
        the current shot mode, recording timer, digital twin deviation score,
        and real-time coaching tip, and displays it via OpenCV.

        Args:
            active_shot: Name of the currently executing shot primitive.
            user_pos: Current 3D subject position `[X, Y, Z]` in meters.
            cam_pos: Current 3D camera target position `[X, Y, Z]` in meters.
            report: Evaluation dictionary from
                `UserDirectorDigitalTwin.evaluate_user_performance`.
            t: Elapsed simulation time in seconds.

        Returns:
            `True` if the window remains open and simulation should continue;
            `False` if the user pressed `Esc` or `'q'` to quit.
        """
        if self.renderer is None:
            return True
        try:
            import cv2

            self.renderer.update_scene(self.data, camera="director_cam")
            rgb = self.renderer.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]

            # Top HUD banner: Recording indicator, elapsed time, and active shot.
            cv2.rectangle(frame, (0, 0), (w, 58), (20, 20, 20), -1)
            cv2.circle(frame, (22, 22), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"REC {t:05.1f}s | SHOT: {active_shot.upper()}",
                (38, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            dev = report.get("deviation_score", 0.0)
            dev_color = (0, 255, 128) if dev <= 0.15 else (0, 165, 255)
            cv2.putText(
                frame,
                f"TWIN DEV: {dev:.3f}m",
                (w - 210, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                dev_color,
                1,
            )

            # Sub-banner: Real-time Digital Twin coaching tip.
            tips = report.get("coaching_tips", [])
            tip_text = tips[0] if tips else ""
            cv2.putText(
                frame,
                tip_text[:78],
                (18, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                dev_color,
                1,
            )

            # Bottom telemetry bar: Subject and Camera 3D coordinates.
            cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)
            cv2.putText(
                frame,
                (
                    f"USER: [{user_pos[0]:.2f}, {user_pos[1]:.2f}, {user_pos[2]:.2f}]  "
                    f"CAM: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
                ),
                (15, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Antigravity: Robot POV & Digital Twin Coach", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return False
        except Exception:
            pass
        return True

    def execute_path_plan(
        self,
        shot_type: str = "orbit",
        duration: float = 30.0,
        rate: float = 0.005,
        show_viewer: bool = True,
        show_pov: bool = True,
        bridge: Optional[Any] = None,
    ) -> None:
        """Runs the main MuJoCo control loop tracking the user and shot plan.

        On every control tick:
          1. Checks if the WebSocket bridge received a dynamic shot change from
             Gemma 4.
          2. Updates the tracked user position (`humanoid_target` mocap body).
          3. Computes the relative camera waypoint and look-at orientation.
          4. Evaluates user performance against the Digital Twin trajectory.
          5. Solves differential IK via `mink` and steps the MuJoCo physics.

        Args:
            shot_type: Initial shot primitive identifier. Defaults to `'orbit'`.
            duration: Total simulation run time in seconds (`<= 0` runs
                indefinitely until the viewer window is closed). Defaults to `30.0`.
            rate: Integration and physics timestep in seconds. Defaults to
                `0.005` (200 Hz).
            show_viewer: Whether to launch the interactive 3D MuJoCo passive
                viewer window. Defaults to `True`.
            show_pov: Whether to display the OpenCV Robot Camera POV window with
                coaching HUD overlays. Defaults to `True`.
            bridge: Optional `AntigravityDirectorBridge` instance for live
                WebSocket telemetry broadcasting and dynamic shot switching.
        """
        coach = UserDirectorDigitalTwin(shot_type=shot_type)
        if bridge:
            bridge.set_shot(shot_type)

        print("=" * 65)
        print(f"🎬 AGENT DIRECTOR SHOT PLAN: '{shot_type.upper()}'")
        print(f"📋 INSTRUCTION: {coach.instruction}")
        if bridge:
            print(f"🌐 WEBSOCKET BRIDGE ACTIVE: ws://{bridge.host}:{bridge.port}")
        print("=" * 65)

        mocap_id = self.model.body("humanoid_target").mocapid[0]

        t = 0.0
        step_count = 0
        active_shot = shot_type
        last_report: Dict[str, Any] = {}

        def step_ik(dt: float) -> bool:
            """Solves one differential IK step with automatic QP solver fallback."""
            for solver in ["daqp", "quadprog"]:
                try:
                    vel = mink.solve_ik(
                        self.configuration,
                        self.tasks,
                        dt,
                        solver=solver,
                        damping=1e-3,
                    )
                    self.configuration.integrate_inplace(vel, dt)
                    return True
                except Exception:
                    continue
            return False

        try:
            if show_viewer:
                with mujoco.viewer.launch_passive(
                    self.model, self.data
                ) as viewer:
                    print(
                        f"Starting Robot Path Plan for Shot Type: '{active_shot}'... (Esc to exit)"
                    )
                    while viewer.is_running() and (
                        duration <= 0 or t < duration
                    ):
                        step_start = time.time()

                        # Check for dynamic shot transitions pushed via WebSocket.
                        if bridge and bridge.current_shot != active_shot:
                            active_shot = bridge.current_shot
                            coach.set_shot_type(active_shot)
                            print(
                                f"\n[Dynamic Transition] Switched to shot: {active_shot.upper()}"
                            )
                            print(f"📋 NEW INSTRUCTION: {coach.instruction}")

                        # 1. Simulate live user tracking (in production: ToF/IMU/MediaPipe).
                        ideal_pos = coach.get_ideal_twin_position(t)
                        raw_user_pos = ideal_pos + np.array(
                            [
                                0.04 * np.sin(t * 1.5),
                                0.03 * np.cos(t * 1.2),
                                -0.02,
                            ]
                        )
                        user_pos = normalize_mocap_z(raw_user_pos)
                        self.data.mocap_pos[mocap_id] = user_pos

                        # 2. Compute desired camera pose using relative subject framing.
                        base_offset = self.get_shot_offset(active_shot, t)
                        desired_cam_pos = self.clamp_to_workspace(
                            user_pos - base_offset
                        )
                        desired_cam_rot = self.compute_look_at(
                            desired_cam_pos, user_pos
                        )

                        # Evaluate performance & push telemetry to WebSocket clients.
                        last_report = coach.evaluate_user_performance(
                            user_pos, t
                        )
                        if bridge:
                            bridge.update_telemetry(
                                user_pos, desired_cam_pos, t
                            )

                        # Print console coaching summary once per simulated second.
                        if step_count % max(1, int(1.0 / rate)) == 0:
                            tips = " | ".join(last_report["coaching_tips"])
                            print(
                                f"[{t:4.1f}s | {active_shot.upper()}] "
                                f"Dev: {last_report['deviation_score']:.3f}m | {tips}"
                            )

                        # 3. Solve differential IK via Mink and apply joint controls.
                        target_se3 = mink.SE3.from_rotation_and_translation(
                            mink.SO3.from_matrix(desired_cam_rot),
                            desired_cam_pos,
                        )
                        self.end_effector_task.set_target(target_se3)

                        step_ik(rate)
                        self.data.qpos[: self.model.nq] = self.configuration.q
                        self.data.ctrl[:] = self.configuration.q[
                            : self.model.nu
                        ]

                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()

                        # Render Robot POV + Coaching HUD at ~30 FPS.
                        if (
                            show_pov
                            and step_count % max(1, int(0.033 / rate)) == 0
                        ):
                            if not self.render_pov_hud(
                                active_shot,
                                user_pos,
                                desired_cam_pos,
                                last_report,
                                t,
                            ):
                                break

                        t += rate
                        step_count += 1
                        elapsed = time.time() - step_start
                        if elapsed < rate:
                            time.sleep(rate - elapsed)
            else:
                # Headless simulation mode for automated testing and CI verification.
                while duration <= 0 or t < duration:
                    if bridge and bridge.current_shot != active_shot:
                        active_shot = bridge.current_shot
                        coach.set_shot_type(active_shot)
                        print(
                            f"\n[Dynamic Transition] Switched to shot: {active_shot.upper()}"
                        )

                    ideal_pos = coach.get_ideal_twin_position(t)
                    raw_user_pos = ideal_pos + np.array(
                        [
                            0.04 * np.sin(t * 1.5),
                            0.03 * np.cos(t * 1.2),
                            -0.02,
                        ]
                    )
                    user_pos = normalize_mocap_z(raw_user_pos)
                    self.data.mocap_pos[mocap_id] = user_pos

                    base_offset = self.get_shot_offset(active_shot, t)
                    desired_cam_pos = self.clamp_to_workspace(
                        user_pos - base_offset
                    )
                    desired_cam_rot = self.compute_look_at(
                        desired_cam_pos, user_pos
                    )

                    last_report = coach.evaluate_user_performance(user_pos, t)
                    if bridge:
                        bridge.update_telemetry(user_pos, desired_cam_pos, t)

                    if step_count % max(1, int(1.0 / rate)) == 0:
                        tips = " | ".join(last_report["coaching_tips"])
                        print(
                            f"[{t:4.1f}s | {active_shot.upper()}] "
                            f"Dev: {last_report['deviation_score']:.3f}m | {tips}"
                        )

                    target_se3 = mink.SE3.from_rotation_and_translation(
                        mink.SO3.from_matrix(desired_cam_rot), desired_cam_pos
                    )
                    self.end_effector_task.set_target(target_se3)

                    step_ik(rate)
                    self.data.qpos[: self.model.nq] = self.configuration.q
                    self.data.ctrl[:] = self.configuration.q[: self.model.nu]
                    mujoco.mj_step(self.model, self.data)

                    t += rate
                    step_count += 1
        finally:
            if show_pov:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if bridge:
                bridge.stop()

        print("Execution completed cleanly.")


def main() -> None:
    """Parses CLI arguments and launches the Antigravity Robot Path Planner."""
    parser = argparse.ArgumentParser(
        description="Antigravity: Robot Path Planner & Digital Twin Tracking"
    )
    parser.add_argument(
        "--shot",
        default="orbit",
        choices=["orbit", "dolly_in", "low_angle_hero", "tracking"],
        help="Initial cinematic shot primitive",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration in seconds (0 = infinite)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without the interactive 3D MuJoCo viewer",
    )
    parser.add_argument(
        "--no-pov",
        action="store_true",
        help="Disable the OpenCV Robot POV viewfinder window",
    )
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="Launch the WebSocket bridge (ws://127.0.0.1:8765) for Gemma 4 UI control",
    )
    args = parser.parse_args()

    bridge = None
    if args.bridge:
        from director_server import start_bridge_background

        bridge = start_bridge_background()

    planner = RobotPathPlanner("scene.xml")
    planner.execute_path_plan(
        shot_type=args.shot,
        duration=args.duration,
        show_viewer=not args.no_viewer,
        show_pov=not args.no_pov,
        bridge=bridge,
    )


if __name__ == "__main__":
    main()
