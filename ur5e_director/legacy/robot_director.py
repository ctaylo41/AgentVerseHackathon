"""Legacy Prototype: Basic Humanoid Random-Walk Camera Tracker (`robot_director.py`).

Note:
  This file is an earlier prototype superseded by `robot_path_planner.py` (for
  Gemma 4 shot planning + Digital Twin coaching) and `main.py` (for interactive
  HUD control). It is retained for reference as a minimal example of combining
  a `RandomWalker` mocap humanoid with `mink` differential IK on the UR5e arm.
"""

from typing import Sequence
import cv2
import mink
import mujoco
import mujoco.viewer
import numpy as np

# Standing torso height (in meters) that places humanoid feet at Z = 0.0m.
HUMANOID_TORSO_Z: float = 1.282


class RandomWalker:
    """Generates smooth, bounded 3D random walk motion for the humanoid mocap target.

    Attributes:
        pos: Current 3D position `[X, Y, Z]` in meters.
        vel: Current 3D velocity `[vx, vy, vz]` in meters per step.
        bounds: Axis-aligned `[min, max]` bounds for `X`, `Y`, and `Z`.
    """

    def __init__(
        self, start_pos: Sequence[float], bounds: Sequence[Sequence[float]]
    ) -> None:
        """Initializes the walker with a starting position and room bounds.

        Args:
            start_pos: Initial `[X, Y, Z]` coordinates in meters.
            bounds: 3x2 sequence `[[x_min, x_max], [y_min, y_max], [z_min, z_max]]`.
        """
        self.pos: np.ndarray = np.array(start_pos, dtype=float)
        self.vel: np.ndarray = np.zeros(3)
        self.bounds: Sequence[Sequence[float]] = bounds

    def step(self) -> np.ndarray:
        """Steps the random walker forward one tick and returns its 3D position.

        Returns:
            A 1D NumPy float array `[X, Y, Z]` representing the updated position.
        """
        acc = np.random.randn(3) * 0.001
        acc[2] = 0.0  # Keep subject grounded at constant standing height.

        self.vel += acc
        self.vel *= 0.98
        self.pos += self.vel

        for i in range(3):
            if self.pos[i] < self.bounds[i][0] or self.pos[i] > self.bounds[i][1]:
                self.vel[i] *= -0.5
                self.pos[i] = np.clip(
                    self.pos[i], self.bounds[i][0], self.bounds[i][1]
                )

        return self.pos.copy()


def look_at_matrix(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Calculates a 3x3 rotation matrix orienting camera `-Z` toward `target_pos`.

    Args:
        cam_pos: 3D position `[X, Y, Z]` of the camera in meters.
        target_pos: 3D position `[X, Y, Z]` of the humanoid target in meters.
        up: World up-vector `[0, 0, 1]`.

    Returns:
        A `(3, 3)` orthonormal rotation matrix `[x_axis | y_axis | z_axis]`.
    """
    forward = target_pos - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-6)

    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def run_prototype() -> None:
    """Runs the minimal UR5e look-at tracking loop with a wandering humanoid."""
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)
    configuration.update_from_keyframe("home")
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    end_effector_task = mink.FrameTask(
        frame_name="camera_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)
    tasks = [end_effector_task, posture_task]

    renderer = mujoco.Renderer(model, height=480, width=640)

    room_bounds = [
        [0.5, 2.0],
        [-1.0, 1.0],
        [HUMANOID_TORSO_Z, HUMANOID_TORSO_Z],
    ]
    humanoid = RandomWalker(
        start_pos=[1.0, 0.0, HUMANOID_TORSO_Z], bounds=room_bounds
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        rate = 0.005
        t = 0.0
        mocap_id = model.body("humanoid_target").mocapid[0]

        while viewer.is_running():
            person_pos = humanoid.step()
            data.mocap_pos[mocap_id] = person_pos

            desired_cam_pos = np.array([0.35, 0.15 * np.sin(t), 0.70])
            desired_cam_rot = look_at_matrix(desired_cam_pos, person_pos)

            target_se3 = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(desired_cam_rot), desired_cam_pos
            )
            end_effector_task.set_target(target_se3)

            vel = mink.solve_ik(
                configuration, tasks, rate, solver="daqp", damping=1e-3
            )
            configuration.integrate_inplace(vel, rate)
            data.ctrl[:] = configuration.q[: model.nu]
            data.qpos[: model.nq] = configuration.q

            mujoco.mj_step(model, data)
            viewer.sync()

            if int(t / rate) % 6 == 0:
                renderer.update_scene(data, camera="director_cam")
                rgb_frame = renderer.render()
                cv2.imshow(
                    "Robot Cameraman POV",
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)

            t += rate


if __name__ == "__main__":
    run_prototype()
