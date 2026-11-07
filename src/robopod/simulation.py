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
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / 'assets/room.xml'))
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera('rig').id
        self.model.cam_fovy[self.camera_id] = fovy
        self.width, self.height = width, height
        self.renderer = None
        self.axes = tuple(self.model.joint(i).name for i in range(self.model.njnt))
        self.reset()

    @property
    def limits(self):
        return {name: tuple(self.model.joint(name).range) for name in self.axes}

    @property
    def configuration(self):
        return {name: float(self.data.joint(name).qpos[0]) for name in self.axes}

    def reset(self):
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

    def save(self, directory, *, observation=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        obs = observation if observation is not None else self.capture()
        if not cv2.imwrite(str(directory / 'rgb.png'), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)):
            raise OSError('Could not save RGB image')
        np.savez_compressed(directory / 'geometry.npz', depth_m=obs.depth_m,
                            segmentation=obs.segmentation, intrinsics=obs.intrinsics,
                            world_from_camera=obs.world_from_camera)
        metadata = dict(simulation_time=obs.simulation_time, joints=obs.joints,
                        intrinsics=obs.intrinsics.tolist(), world_from_camera=obs.world_from_camera.tolist(),
                        width=self.width, height=self.height, fovy=float(self.model.cam_fovy[self.camera_id]),
                        axes='OpenCV: x right, y down, z forward; world z up; metres',
                        segmentation_channels=['object_id', 'object_type'],
                        geoms={str(i): {'name': self.model.geom(i).name,
                                       'body': self.model.body(int(self.model.geom_bodyid[i])).name}
                               for i in range(self.model.ngeom)})
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
