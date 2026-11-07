"""Deterministic reachable surveys and observed-surface voxel coverage."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import cv2
import mujoco
import numpy as np

from .simulation import Observation, Simulation


def survey_poses(limits):
    """18 interior poses, serpentine pan order to reduce travel; fixed roll."""
    fractions = {'height': (0.2, 0.5, 0.8), 'pan': (0.2, 0.5, 0.8),
                 'tilt': (0.35, 0.55)}
    values = {axis: [limits[axis][0] + f * (limits[axis][1] - limits[axis][0])
                     for f in fs] for axis, fs in fractions.items()}
    poses = []
    for height in values['height']:
        for tilt in values['tilt']:
            pans = values['pan'] if len(poses) % 2 == 0 else values['pan'][::-1]
            for pan in pans:
                poses.append(dict(height=height, pan=pan, tilt=tilt, roll=0.0))
    return poses


@dataclass(frozen=True)
class SettleResult:
    settled: bool
    elapsed: float
    max_position_error: float
    max_velocity: float


def move_and_settle(sim, target, timeout=5.0):
    """Require low velocity and target proximity for 0.2 simulated seconds.

    Height tolerance allows the existing position servo's ~3 cm gravity sag.
    No qpos writes, resets, teleportation or wall-clock sleeps.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Settle timeout must be finite and positive')
    sim.command(**target)
    start = sim.data.time
    stable = 0
    errors, speed = [float('inf')], float('inf')
    while sim.data.time - start < timeout:
        sim.step(0.02)
        errors = [abs(sim.configuration[name] - value) for name, value in target.items()]
        close = all(error <= (0.04 if name == 'height' else 0.015)
                    for name, error in zip(target, errors))
        speed = max(abs(float(sim.data.joint(name).qvel[0])) for name in target)
        inside = all(lo <= sim.configuration[name] <= hi for name, (lo, hi) in sim.limits.items())
        stable = stable + 1 if close and inside and speed < 0.015 else 0
        if stable >= 10:
            return SettleResult(True, float(sim.data.time-start), max(errors), speed)
    return SettleResult(False, float(sim.data.time-start), max(errors), speed)


class SurfaceCoverage:
    """Unique world-space surface cells; never claims percent of unknown room.

    Only rendered geom pixels count. Sky/far-plane background is excluded.
    Sampling every fourth pixel keeps CPU/memory cost small at 640x480.
    """
    def __init__(self, voxel_size=0.1, stride=4):
        if not math.isfinite(voxel_size) or voxel_size <= 0 or stride < 1 or int(stride) != stride:
            raise ValueError('Positive finite voxel size and integer stride required')
        self.voxel_size, self.stride = voxel_size, int(stride)
        self.cells = set()
        self.geom_ids = set()

    def measure(self, observation):
        depth = observation.depth_m[::self.stride, ::self.stride]
        seg = observation.segmentation[::self.stride, ::self.stride]
        valid = (np.isfinite(depth) & (depth > 0) & (seg[:, :, 0] >= 0)
                 & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        v, u = np.nonzero(valid)
        pixels = np.stack([u*self.stride, v*self.stride, np.ones(len(u))])
        camera = np.linalg.solve(observation.intrinsics, pixels) * depth[valid]
        transform = observation.world_from_camera
        world = transform[:3, :3] @ camera + transform[:3, 3, None]
        cells = set(map(tuple, np.floor(world.T / self.voxel_size).astype(np.int64).tolist()))
        ids = set(map(int, seg[:, :, 0][valid]))
        return cells, ids

    def add(self, cells, ids):
        self.cells.update(cells)
        self.geom_ids.update(ids)


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def explore(sim: Simulation, directory, *, voxel_size=0.1, min_novelty=0.03,
            settle_timeout=5.0, progress=None):
    """Save a reproducible survey in a new directory, refusing overwrites.

    Coverage includes retained observations only. Skip redundant candidates unless
    they reveal a new geom ID. Every candidate and settling failure stays in manifest.
    """
    if not math.isfinite(min_novelty) or not 0 <= min_novelty <= 1:
        raise ValueError('Minimum novelty must be in [0, 1]')
    if not math.isfinite(settle_timeout) or settle_timeout <= 0:
        raise ValueError('Settle timeout must be finite and positive')
    coverage = SurfaceCoverage(voxel_size)
    poses = survey_poses(sim.limits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status='running', pattern='serpentine-3x3x2-v1',
                    candidate_count=len(poses), voxel_size_m=voxel_size, pixel_stride=coverage.stride,
                    min_novelty=min_novelty, coverage_definition='unique observed surface voxels; no total-room percentage',
                    initial_joints=sim.configuration, limits=sim.limits, views=[])
    thumbnails = []
    try:
        for index, target in enumerate(poses):
            result = move_and_settle(sim, target, settle_timeout)
            record = dict(view_id=f'view_{index:03d}', target_joints=target,
                          measured_joints=sim.configuration, simulation_time=float(sim.data.time),
                          settling=vars(result), status='unsettled', files=None)
            if result.settled:
                obs = sim.capture()
                cells, ids = coverage.measure(obs)
                new_cells, new_ids = cells - coverage.cells, ids - coverage.geom_ids
                novelty = len(new_cells) / max(1, len(cells))
                record.update(visible_geom_ids=sorted(ids), observed_voxels=len(cells),
                              new_voxels=len(new_cells), novelty_fraction=novelty)
                if cells and (not coverage.cells or new_ids or novelty >= min_novelty):
                    relative = Path(record['view_id'])
                    sim.save(directory / relative, observation=obs)
                    coverage.add(cells, ids)
                    record.update(status='saved', files={name: str(relative / name) for name in
                                  ('rgb.png', 'geometry.npz', 'metadata.json')})
                    thumb = cv2.resize(cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR), (240, 180))
                    cv2.putText(thumb, record['view_id'], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    thumbnails.append(thumb)
                else:
                    record['status'] = 'redundant' if cells else 'empty'
            record['cumulative_voxels'] = len(coverage.cells)
            manifest['views'].append(record)
            _write_json(directory / 'manifest.json', manifest)
            if progress:
                progress(record, len(poses))
        manifest['status'] = 'completed' if all(v['status'] != 'unsettled' for v in manifest['views']) else 'completed_with_failures'
        if not coverage.cells:
            manifest['status'] = 'failed'
        voxels = np.asarray(sorted(coverage.cells), dtype=np.int64).reshape(-1, 3)
        np.savez_compressed(directory / 'coverage.npz', voxel_indices=voxels,
                            voxel_size_m=voxel_size, visible_geom_ids=sorted(coverage.geom_ids))
        if thumbnails:
            sheet = np.zeros((math.ceil(len(thumbnails)/3)*180, 720, 3), dtype=np.uint8)
            for i, thumb in enumerate(thumbnails):
                row, col = divmod(i, 3)
                sheet[row*180:(row+1)*180, col*240:(col+1)*240] = thumb
            if not cv2.imwrite(str(directory / 'contact_sheet.jpg'), sheet):
                raise OSError('Could not save contact sheet')
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(saved_views=sum(v['status'] == 'saved' for v in manifest['views']),
                        covered_voxels=len(coverage.cells), visible_geom_ids=sorted(coverage.geom_ids))
        _write_json(directory / 'manifest.json', manifest)
    return manifest
