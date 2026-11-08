"""Initial UR5e Mink IK Sandbox Prototype (`main_robo.py`).

Note:
  This file was the earliest proof-of-concept test for driving the standalone
  UR5e arm (`universal_robots_ur5e/scene.xml`) with `mink` differential IK
  before adding the humanoid target, AI-generated room boundaries, and Digital
  Twin Coach. For the complete application, run `robot_path_planner.py`.
"""

import cv2
import mink
import mujoco
import mujoco.viewer
import numpy as np


def look_at_matrix(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Computes a 3x3 rotation matrix orienting camera `-Z` toward `target_pos`.

    Args:
        cam_pos: 3D camera position `[X, Y, Z]` in meters.
        target_pos: 3D target point `[X, Y, Z]` in meters.
        up: World up-vector `[0.0, 0.0, 1.0]`.

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


def run_minimal_ur5e_demo() -> None:
    """Launches the minimal UR5e sinusoidal tracking demo."""
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        rate = 0.005  # 200 Hz control loop.
        t = 0.0

        while viewer.is_running():
            person_pos = np.array([0.8, 0.3 * np.sin(t), 0.5])
            data.mocap_pos[0] = person_pos

            desired_cam_pos = np.array([0.4, 0.15 * np.sin(t), 0.4])
            desired_cam_rot = look_at_matrix(desired_cam_pos, person_pos)

            target_se3 = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(desired_cam_rot), desired_cam_pos
            )
            end_effector_task.set_target(target_se3)

            vel = mink.solve_ik(
                configuration,
                tasks,
                rate,
                solver="daqp",
                damping=1e-3,
            )
            configuration.integrate_inplace(vel, rate)
            data.ctrl[:] = configuration.q[: model.nu]

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
    run_minimal_ur5e_demo()
