"""Single-thread-owned MuJoCo rig. SI units; angles are radians."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import cv2
import mujoco
import numpy as np


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray  # H,W,2: MuJoCo object ID, object type; background -1
    intrinsics: np.ndarray
    world_from_camera: np.ndarray  # OpenCV axes: right, down, forward
    simulation_time: float
    joints: dict[str, float]


class Simulation:
    def __init__(self, width=640, height=480, fovy=45.0):
        if not (16 <= width <= 1920 and 16 <= height <= 1080):
            raise ValueError("Resolution must fit 1920x1080 and be at least 16x16")
        if not math.isfinite(fovy) or not 10 <= fovy <= 120:
            raise ValueError("Vertical FOV must be 10–120 degrees")
        from .artifacts import identity
        self.artifact_identity = identity()
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / 'assets/room.xml'))
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera('rig').id
        self.model.cam_fovy[self.camera_id] = fovy
        self.width, self.height = width, height
        self.renderer = None
        self.axes = tuple(self.model.joint(i).name for i in range(self.model.njnt))
        self._actor_body_quat = self.model.body_quat.copy()
        self._actor_geom_pos = self.model.geom_pos.copy()
        self.reset()
        from .safety import PathValidator
        self.path_validator = PathValidator(self)

    def checkpoint(self):
        """Owner-thread cancellation/pause hook; headless simulations do not block."""
        pass

    @property
    def limits(self):
        return {name: tuple(self.model.joint(name).range) for name in self.axes}

    @property
    def configuration(self):
        return {name: float(self.data.joint(name).qpos[0]) for name in self.axes}

    def reset(self):
        if hasattr(self, '_actor_body_quat'):
            self.model.body_quat[:] = self._actor_body_quat
            self.model.geom_pos[:] = self._actor_geom_pos
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def command(self, **targets):
        # Validate the whole request before changing any actuator.
        for name, value in targets.items():
            if name not in self.axes:
                raise ValueError(f'Unknown axis: {name}')
            lo, hi = self.limits[name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} must be in [{lo}, {hi}]')
        for name, value in targets.items():
            self.data.ctrl[self.model.actuator(name).id] = value

    def step(self, seconds=1/30):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Duration must be finite and nonnegative')
        if seconds == 0:
            return
        mujoco.mj_step(self.model, self.data, nstep=max(1, round(seconds / self.model.opt.timestep)))
        mujoco.mj_forward(self.model, self.data)

    def capture(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        r = self.renderer
        try:
            r.update_scene(self.data, camera='rig')
            rgb = r.render().copy()
            r.enable_depth_rendering()
            depth = r.render().copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            segmentation = r.render().copy()
        finally:
            r.disable_depth_rendering()
            r.disable_segmentation_rendering()
        focal = self.height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2))
        k = np.array([[focal, 0, (self.width-1)/2], [0, focal, (self.height-1)/2], [0, 0, 1]])
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = self.data.cam_xpos[self.camera_id]
        return Observation(rgb, depth, segmentation, k, transform, float(self.data.time), self.configuration)

    def capture_overview(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, -0.55, 1.2]
        cam.distance = 4.1
        cam.azimuth = 135.0
        cam.elevation = -20.0
        self.renderer.update_scene(self.data, camera=cam)
        scn = self.renderer.scene

        def add_rod(p1, p2, radius, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            dist = float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))
            if dist < 1e-4:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(p1, dtype=float), np.asarray(p2, dtype=float))
            scn.ngeom += 1

        def add_sphere(pos, radius, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
                np.asarray(pos, dtype=float), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
            )
            scn.ngeom += 1

        carriage_pos = self.data.body('carriage').xpos.copy()
        shoulder_pos = self.data.body('shoulder_boom').xpos.copy()
        forearm_pos = self.data.body('forearm_boom').xpos.copy()
        pan_pos = self.data.body('pan_frame').xpos.copy()
        rig_pos = self.data.cam_xpos[self.camera_id].copy()
        rig_rot = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])

        # 3D Gantry & Telescoping Robotic Arm Links
        base_floor = np.array([carriage_pos[0], carriage_pos[1], 0.0])
        base_top = np.array([carriage_pos[0], carriage_pos[1], 2.25])
        add_rod(base_floor, base_top, 0.035, [0.42, 0.46, 0.50, 1.0])
        add_rod([shoulder_pos[0] - 0.15, carriage_pos[1], carriage_pos[2]],
                [carriage_pos[0] + 0.15, carriage_pos[1], carriage_pos[2]], 0.028, [0.55, 0.60, 0.65, 1.0])
        add_rod(carriage_pos, shoulder_pos, 0.032, [0.91, 0.73, 0.40, 1.0])
        add_rod(shoulder_pos, pan_pos, 0.028, [0.91, 0.73, 0.40, 1.0])
        add_sphere(carriage_pos, 0.055, [0.25, 0.28, 0.32, 1.0])
        add_sphere(shoulder_pos, 0.052, [0.91, 0.73, 0.40, 1.0])
        add_sphere(pan_pos, 0.058, [0.38, 0.82, 0.72, 1.0])
        add_sphere(rig_pos, 0.045, [0.95, 0.80, 0.45, 1.0])

        # 3D Camera Frustum & Optical Sightline
        f_dist = 0.65
        f_w, f_h = 0.28, 0.21
        corners = [
            rig_pos + f_dist * rig_rot[:, 2] + sx * f_w * rig_rot[:, 0] + sy * f_h * rig_rot[:, 1]
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
        ]
        for idx in range(4):
            add_rod(rig_pos, corners[idx], 0.008, [0.62, 0.85, 0.75, 0.85])
            add_rod(corners[idx], corners[(idx + 1) % 4], 0.008, [0.62, 0.85, 0.75, 0.85])
        add_rod(rig_pos, rig_pos + 2.6 * rig_rot[:, 2], 0.010, [0.95, 0.75, 0.38, 0.90])

        rgb = self.renderer.render().copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cfg = self.configuration
        hud = (f"RIG POSE  H:{cfg.get('height',0):.2f}m  T:{cfg.get('truck',0):+.2f}m  "
               f"D:{cfg.get('dolly',0):+.2f}m  Pan:{math.degrees(cfg.get('pan',0)):+.1f}deg  "
               f"Tilt:{math.degrees(cfg.get('tilt',0)):+.1f}deg")
        cv2.rectangle(bgr, (0, self.height - 30), (self.width, self.height), (17, 20, 22), -1)
        cv2.putText(bgr, hud, (12, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (233, 185, 103), 1, cv2.LINE_AA)
        return bgr

    def geometry_catalog(self):
        result = {}
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            root = body
            while self.model.body_parentid[root] > 0:
                root = int(self.model.body_parentid[root])
            result[str(i)] = dict(name=self.model.geom(i).name, body=self.model.body(body).name,
                                  semantic_body=self.model.body(root).name or f'unnamed_{root}',
                                  robot=self.model.body(root).name == 'carriage')
        return result

    def save(self, directory, *, observation=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        obs = observation if observation is not None else self.capture()
        if not cv2.imwrite(str(directory / 'rgb.png'), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)):
            raise OSError('Could not save RGB image')
        np.savez_compressed(directory / 'geometry.npz', depth_m=obs.depth_m,
                            segmentation=obs.segmentation, intrinsics=obs.intrinsics,
                            world_from_camera=obs.world_from_camera)
        metadata = dict(artifact_identity=self.artifact_identity, simulation_time=obs.simulation_time, joints=obs.joints,
                        intrinsics=obs.intrinsics.tolist(), world_from_camera=obs.world_from_camera.tolist(),
                        width=self.width, height=self.height, fovy=float(self.model.cam_fovy[self.camera_id]),
                        axes='OpenCV: x right, y down, z forward; world z up; metres',
                        segmentation_channels=['object_id', 'object_type'],
                        geoms=self.geometry_catalog())
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        return obs

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
