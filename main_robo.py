import mujoco
import mujoco.viewer
import mink
import numpy as np
import cv2

# 1. Load your modified scene.xml
model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
data = mujoco.MjData(model)

# 2. Setup Mink IK Configuration
configuration = mink.Configuration(model)
configuration.update_from_keyframe("home")
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

end_effector_task = mink.FrameTask(
    frame_name="camera_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.5, # Slightly lower orientation cost so it prioritizes looking over exact roll
    lm_damping=1.0,       # Helps stabilize when target is slightly out of reach
)
posture_task = mink.PostureTask(model, cost=1e-2)
posture_task.set_target_from_configuration(configuration)

tasks = [end_effector_task, posture_task]

# 3. Setup Renderer for the Robot's Camera POV
renderer = mujoco.Renderer(model, height=480, width=640)

def look_at_matrix(cam_pos, target_pos, up=np.array([0.0, 0.0, 1.0])):
    forward = target_pos - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    
    return np.column_stack((x_axis, y_axis, z_axis))

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    rate = 0.005 # 200Hz control loop
    t = 0.0
    
    while viewer.is_running():
        # Simulate person moving side to side
        person_pos = np.array([0.8, 0.3 * np.sin(t), 0.5])
        data.mocap_pos[0] = person_pos
        
        # Desired camera position (Agent can dynamically change this for 'dolly', 'orbit', etc.)
        desired_cam_pos = np.array([0.4, 0.15 * np.sin(t), 0.4])
        desired_cam_rot = look_at_matrix(desired_cam_pos, person_pos)
        
        # Set target pose for Mink IK
        target_se3 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(desired_cam_rot), desired_cam_pos
        )
        end_effector_task.set_target(target_se3)
        
        # Solve IK and apply to joints
        vel = mink.solve_ik(
            configuration, 
            tasks, 
            rate, 
            solver="daqp", 
            damping=1e-3
        )
        configuration.integrate_inplace(vel, rate)
        data.ctrl[:] = configuration.q[:model.nu]
        
        # Step physics
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Render the Robot Camera POV every few frames (~30 FPS)
        if int(t / rate) % 6 == 0:
            renderer.update_scene(data, camera="director_cam")
            rgb_frame = renderer.render()
            cv2.imshow("Robot Cameraman POV", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
        t += rate