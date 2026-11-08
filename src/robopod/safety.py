"""Conservative swept-volume checks against MuJoCo primitive collision geometry.

Uses signed geom distances, independent of renderer visibility and contact masks.
A Lipschitz displacement bound certifies each straight joint-space segment between
samples. This covers the modeled rig solids against all non-rig solids; overlapping
parts of the connected gimbal assembly are intentionally excluded from self checks.
"""
import math
import mujoco
import numpy as np


class PathBlocked(RuntimeError):
    pass


class PathValidator:
    clearance = .002  # 2 mm modeled surface clearance

    def __init__(self, sim):
        self.sim = sim
        self.model = sim.model
        self.scratch = mujoco.MjData(self.model)
        root = self.model.body('carriage').id
        def robot(body):
            while body:
                if body == root:
                    return True
                body = int(self.model.body_parentid[body])
            return False
        self.robot = [g for g in range(self.model.ngeom) if robot(int(self.model.geom_bodyid[g]))]
        self.obstacles = [g for g in range(self.model.ngeom) if g not in self.robot]
        self.pairs = [(a,b) for a in self.robot for b in self.obstacles]
        self.pair_a = np.array([a for a,b in self.pairs], dtype=int)
        self.pair_b = np.array([b for a,b in self.pairs], dtype=int)
        self.radii = {}
        # Per-joint triangle-inequality lever bounds over downstream solids. Axes
        # upstream of a hinge do not increase its rotational lever arm.
        for name in sim.axes:
            joint = self.model.joint(name)
            root_body = int(joint.bodyid[0])
            bounds = []
            for g in self.robot:
                body = int(self.model.geom_bodyid[g])
                bound = np.linalg.norm(self.model.geom_pos[g]) + self.model.geom_rbound[g] + np.linalg.norm(joint.pos)
                while body and body != root_body:
                    bound += np.linalg.norm(self.model.body_pos[body])
                    for j in range(int(self.model.body_jntadr[body]), int(self.model.body_jntadr[body]+self.model.body_jntnum[body])):
                        bound += 2*np.linalg.norm(self.model.jnt_pos[j])
                        if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                            bound += max(abs(self.model.jnt_range[j]))
                    body = int(self.model.body_parentid[body])
                if body == root_body:
                    bounds.append(bound)
            self.radii[name] = max(bounds, default=0.)

    def displacement_bound(self, start, end):
        return sum(abs(end[n]-start[n]) * (1. if self.model.joint(n).type[0] == mujoco.mjtJoint.mjJNT_SLIDE else self.radii[n])
                   for n in self.sim.axes)

    def check(self, data, margin=None):
        margin = self.clearance if margin is None else margin
        a_ids, b_ids = self.pair_a, self.pair_b
        radii = self.model.geom_rbound
        separation = np.linalg.norm(data.geom_xpos[a_ids]-data.geom_xpos[b_ids],axis=1)
        near = (radii[b_ids] == 0) | (separation <= radii[a_ids]+radii[b_ids]+margin)
        for a,b in zip(a_ids[near],b_ids[near]):
            distance = mujoco.mj_geomDistance(self.model, data, a, b, margin + .001, None)
            if not np.isfinite(distance) or distance <= margin:
                raise PathBlocked(f'Camera rig path blocked by {self.model.geom(b).name or b} (rig geom {a}, clearance {distance:.4f} m)')

    def validate(self, start, end):
        for pose in (start,end):
            for n in self.sim.axes:
                lo,hi = self.sim.limits[n]
                if not math.isfinite(pose[n]) or not lo <= pose[n] <= hi:
                    raise PathBlocked(f'Path endpoint exceeds {n} limits')
        self.scratch.mocap_pos[:] = self.sim.data.mocap_pos
        self.scratch.mocap_quat[:] = self.sim.data.mocap_quat
        samples = 0
        def certify(left, right, depth=0):
            nonlocal samples
            self.sim.checkpoint()
            for n in self.sim.axes:
                self.scratch.joint(n).qpos[0] = left[n]
            mujoco.mj_kinematics(self.model,self.scratch)
            samples += 1
            self.check(self.scratch)  # Real penetration cannot be fixed by subdivision.
            try:
                self.check(self.scratch, self.clearance+self.displacement_bound(left,right))
            except PathBlocked:
                if depth >= 20:
                    raise PathBlocked('Could not certify continuous clearance near an obstacle')
                mid = {n: (left[n]+right[n])/2 for n in self.sim.axes}
                certify(left,mid,depth+1)
                certify(mid,right,depth+1)
        certify(start,end)
        return dict(certified=True, samples=samples, clearance_m=self.clearance,
                    method='signed_geom_distance_with_inter_sample_displacement_bound',
                    scope='modeled rig versus environment; connected assembly self-overlaps excluded')
