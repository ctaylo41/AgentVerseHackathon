"""Standalone Interactive Keyboard-Controlled Cameraman Demo (`main.py`).

This script provides a self-contained interactive demo of the UR5e robotic
cameraman tracking a randomly wandering subject (`RandomWalker`) with manual
keyboard shot switching in an OpenCV viewfinder HUD (`[T]rack`, `[O]rbit`,
`[D]olly`, `[H]igh Angle`, `[C]lose-Up`).

Note on Architecture:
  - For the full AI-directed pipeline with Gemma 4 natural-language prompt
    parsing, ideal Digital Twin choreography, real-time posture/deviation
    coaching, and WebSocket streaming, use `robot_path_planner.py` instead.
  - This file is preserved as a standalone manual control & random-walk
    stress-test benchmark for the Mink differential IK solver.
"""

import argparse
import time
from typing import List, Sequence
import cv2
import mink
import mujoco
import mujoco.viewer
import numpy as np

# Humanoid torso center height (in meters) when standing upright on the floor (Z=0.0).
HUMANOID_TORSO_Z: float = 1.282


class RandomWalker:
    """Generates smooth, bounded 3D Brownian motion with momentum for the humanoid.

    Simulates an unscripted human creator wandering inside the studio safe zone
    to test the responsiveness and stability of the robot's look-at IK tracking.

    Attributes:
        pos: Current 3D position `[X, Y, Z]` of the walker in meters.
        vel: Current 3D velocity `[vx, vy, vz]` in meters per step.
        bounds: Nested list `[[x_min, x_max], [y_min, y_max], [z_min, z_max]]`
            defining the rectangular walking boundary.
    """

    def __init__(
        self, start_pos: Sequence[float], bounds: Sequence[Sequence[float]]
    ) -> None:
        """Initializes the random walker at `start_pos` within `bounds`.

        Args:
            start_pos: Initial `[X, Y, Z]` position in meters.
            bounds: Axis-aligned bounding box limits `[[x0, x1], [y0, y1], [z0, z1]]`.
        """
        self.pos: np.ndarray = np.array(start_pos, dtype=float)
        self.vel: np.ndarray = np.zeros(3)
        self.bounds: Sequence[Sequence[float]] = bounds

    def step(self) -> np.ndarray:
        """Advances the walker by one timestep using damped Brownian acceleration.

        Returns:
            A copy of the updated 3D position vector `[X, Y, Z]` in meters.
        """
        # 1. Apply horizontal random impulse while keeping Z grounded.
        acc = np.random.randn(3) * 0.0025
        acc[2] = 0.0

        # 2. Integrate velocity with viscous damping (0.98).
        self.vel += acc
        self.vel *= 0.98
        self.pos += self.vel

        # 3. Soft elastic reflection off the room boundaries.
        for i in range(3):
            if self.pos[i] < self.bounds[i][0] or self.pos[i] > self.bounds[i][1]:
                self.vel[i] *= -0.5
                self.pos[i] = np.clip(
                    self.pos[i], self.bounds[i][0], self.bounds[i][1]
                )

        return self.pos.copy()


def compute_look_at(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Computes a 3x3 look-at rotation matrix pointing camera `-Z` at `target_pos`.

    Args:
        cam_pos: 3D position `[X, Y, Z]` of the camera in world coordinates.
        target_pos: 3D position `[X, Y, Z]` of the target subject.
        up: World up-vector `[0, 0, 1]` used to orthogonalize the camera axes.

    Returns:
        A `(3, 3)` orthonormal rotation matrix `[right | ortho_up | z_axis]`.
    """
    forward = target_pos - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-6)

    # In MuJoCo, cameras look down their local -Z axis.
    z_axis = -forward
    right = np.cross(up, z_axis)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= norm_r

    ortho_up = np.cross(z_axis, right)
    ortho_up /= np.linalg.norm(ortho_up) + 1e-6

    return np.column_stack((right, ortho_up, z_axis))


class ShotDirector:
    """Controls camera trajectory primitives for interactive keyboard switching.

    Attributes:
        MODES: List of supported camera trajectory modes.
        mode: Currently active camera mode string.
        t: Cumulative elapsed time in seconds.
    """

    MODES: List[str] = ["tracking", "orbit", "dolly", "high_angle", "close_up"]

    def __init__(self, default_mode: str = "tracking") -> None:
        """Initializes the shot director with a starting camera mode.

        Args:
            default_mode: Initial mode from `ShotDirector.MODES`. Defaults to
                `'tracking'`.
        """
        self.mode: str = default_mode
        self.t: float = 0.0

    def compute_cam_pos(
        self, person_pos: np.ndarray, dt: float = 0.01
    ) -> np.ndarray:
        """Computes the desired world-frame camera position for the current mode.

        Args:
            person_pos: Current 3D position `[X, Y, Z]` of the tracked person.
            dt: Simulation timestep in seconds. Defaults to `0.01`.

        Returns:
            A 1D NumPy array `[X, Y, Z]` of the target camera position in meters.
        """
        del person_pos  # Base trajectories are anchored in the UR5e workspace.
        self.t += dt
        if self.mode == "orbit":
            angle = 0.5 * self.t
            return np.array(
                [0.42 + 0.12 * np.cos(angle), 0.22 * np.sin(angle), 0.70]
            )
        elif self.mode == "dolly":
            d = 0.12 * np.sin(0.8 * self.t)
            return np.array([0.45 + d, 0.08 * np.sin(0.4 * self.t), 0.72])
        elif self.mode == "high_angle":
            return np.array([0.40, 0.06 * np.sin(0.5 * self.t), 0.90])
        elif self.mode == "close_up":
            return np.array([0.55, 0.08 * np.sin(0.5 * self.t), 0.78])
        else:
            return np.array([0.46, 0.10 * np.sin(0.3 * self.t), 0.72])


def draw_director_hud(
    frame: np.ndarray,
    shot_mode: str,
    person_pos: np.ndarray,
    cam_pos: np.ndarray,
    fps: float,
    elapsed_time: float,
) -> np.ndarray:
    """Overlays a cinematic director's viewfinder HUD onto a BGR video frame.

    Args:
        frame: OpenCV BGR image array of shape `(H, W, 3)`.
        shot_mode: Active shot mode name.
        person_pos: 3D position `[X, Y, Z]` of the subject in meters.
        cam_pos: 3D position `[X, Y, Z]` of the camera in meters.
        fps: Measured simulation/rendering framerate in frames per second.
        elapsed_time: Total elapsed run time in seconds.

    Returns:
        The annotated BGR image frame.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Top-left REC indicator and timecode.
    cv2.circle(frame, (28, 28), 7, (0, 0, 255), -1)
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    millis = int((elapsed_time * 100) % 100)
    timecode = f"REC  {mins:02d}:{secs:02d}:{millis:02d}"
    cv2.putText(
        frame,
        timecode,
        (45, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    mode_text = f"SHOT MODE: {shot_mode.upper()}"
    cv2.putText(
        frame,
        mode_text,
        (45, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 180),
        1,
    )

    # Center crosshairs and framing guide box.
    cross_size = 14
    cv2.line(
        frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 255), 1
    )
    cv2.line(
        frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 255), 1
    )
    bw, bh = 50, 40
    cv2.rectangle(
        frame, (cx - bw, cy - bh), (cx + bw, cy + bh), (0, 255, 255), 1
    )

    # Bottom-left telemetry readout.
    dist = np.linalg.norm(person_pos - cam_pos)
    telem1 = f"DIST: {dist:.2f}m | FPS: {fps:.1f}"
    telem2 = (
        f"SUBJ: [{person_pos[0]:.2f}, {person_pos[1]:.2f}, {person_pos[2]:.2f}]"
    )
    cv2.putText(
        frame,
        telem1,
        (20, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame,
        telem2,
        (20, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
    )

    # Bottom-right keyboard shortcut legend.
    guide = "[T]rack [O]rbit [D]olly [H]igh [C]lose"
    cv2.putText(
        frame,
        guide,
        (w - 310, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
    )

    return frame


def main() -> None:
    """Entry point for the standalone interactive keyboard cameraman simulation."""
    parser = argparse.ArgumentParser(
        description="Antigravity: Autonomous Robotic Cameraman & Director Simulation"
    )
    parser.add_argument(
        "--scene", default="scene.xml", help="Path to master MuJoCo scene XML"
    )
    parser.add_argument(
        "--mode",
        default="tracking",
        choices=ShotDirector.MODES,
        help="Initial shot mode",
    )
    parser.add_argument(
        "--no-pov",
        action="store_true",
        help="Disable the OpenCV POV camera preview window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Max simulation steps (0 = infinite)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ANTIGRAVITY: AI Robotic Cameraman & Director Simulation")
    print("=" * 60)

    print(f"[1/4] Loading master scene from '{args.scene}'...", flush=True)
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    print("[2/4] Initializing Mink Differential IK Solver...", flush=True)
    configuration = mink.Configuration(model)

    key_names = [model.key(i).name for i in range(model.nkey)]
    if "home" in key_names:
        home_id = model.key("home").id
        configuration.update_from_keyframe("home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        print("      Initialized robot to 'home' keyframe pose.")

    end_effector_task = mink.FrameTask(
        frame_name="camera_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.8,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)
    tasks = [end_effector_task, posture_task]

    print("[3/4] Configuring Humanoid Walker and Shot Director...", flush=True)
    room_bounds = [
        [0.4, 2.0],
        [-1.0, 1.0],
        [HUMANOID_TORSO_Z, HUMANOID_TORSO_Z],
    ]
    target_walker = RandomWalker(
        start_pos=[1.0, 0.0, HUMANOID_TORSO_Z], bounds=room_bounds
    )
    director = ShotDirector(default_mode=args.mode)

    mocap_id = 0
    for i in range(model.nbody):
        if model.body(i).name == "humanoid_target":
            mocap_id = model.body(i).mocapid[0]
            break

    renderer = None
    show_pov = not args.no_pov
    if show_pov:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            print("      Camera POV renderer initialized (640x480).")
        except Exception as exc:
            print(f"      POV renderer disabled ({exc}).")
            show_pov = False

    print("[4/4] Launching MuJoCo Passive Viewer...", flush=True)
    print("      Controls: Press Esc in viewer to exit.")
    print(
        "      In POV window: [T]rack, [O]rbit, [D]olly, [H]igh Angle, [C]lose-Up, [Q]uit"
    )
    print("-" * 60, flush=True)

    step_dt = 0.01  # 100 Hz control loop.
    step_count = 0
    t_start = time.time()
    last_fps_time = time.time()
    fps = 60.0
    frames_since_fps = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                person_pos = target_walker.step()
                data.mocap_pos[mocap_id] = person_pos

                cam_target_pos = director.compute_cam_pos(
                    person_pos, dt=step_dt
                )
                cam_rot = compute_look_at(cam_target_pos, person_pos)

                target_se3 = mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(cam_rot), cam_target_pos
                )
                end_effector_task.set_target(target_se3)

                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    dt=step_dt,
                    solver="quadprog",
                    damping=1e-3,
                )
                configuration.integrate_inplace(vel, dt=step_dt)

                data.qpos[: model.nq] = configuration.q
                data.ctrl[:] = configuration.q[: model.nu]

                mujoco.mj_step(model, data)
                viewer.sync()

                step_count += 1
                frames_since_fps += 1

                if frames_since_fps >= 30:
                    now = time.time()
                    fps = frames_since_fps / (now - last_fps_time + 1e-6)
                    last_fps_time = now
                    frames_since_fps = 0

                if show_pov and renderer is not None and step_count % 3 == 0:
                    try:
                        renderer.update_scene(data, camera="director_cam")
                        rgb_frame = renderer.render()
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                        hud_frame = draw_director_hud(
                            bgr_frame,
                            director.mode,
                            person_pos,
                            cam_target_pos,
                            fps,
                            time.time() - t_start,
                        )
                        cv2.imshow(
                            "Robot Cameraman Viewfinder (Antigravity)",
                            hud_frame,
                        )

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                        elif key == ord("t"):
                            director.mode = "tracking"
                        elif key == ord("o"):
                            director.mode = "orbit"
                        elif key == ord("d"):
                            director.mode = "dolly"
                        elif key == ord("h"):
                            director.mode = "high_angle"
                        elif key == ord("c"):
                            director.mode = "close_up"
                    except Exception:
                        pass

                if args.steps > 0 and step_count >= args.steps:
                    print(
                        f"Reached specified step limit ({args.steps}). Exiting."
                    )
                    break

                elapsed = time.time() - step_start
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)
    finally:
        if show_pov:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("Simulation finished cleanly.")


if __name__ == "__main__":
    main()
