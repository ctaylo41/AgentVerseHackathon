"""Bounded next-view selection from observed coverage and semantic uncertainty only."""
from itertools import product
import mujoco
import numpy as np


class AdaptiveViews:
    def __init__(self, sim):
        self.sim = sim
        self.scratch = mujoco.MjData(sim.model)
        self.visited = []
        self.objects = {}
        self.candidates = []
        limits = sim.limits
        for h,p,t,x,y in product((.25,.55,.8),(.25,.5,.75),(.35,.55),(-.65,0,.65),(.2,.8,1.2)):
            pose = dict(height=limits['height'][0]+h*np.ptp(limits['height']),
                        pan=limits['pan'][0]+p*np.ptp(limits['pan']),
                        tilt=limits['tilt'][0]+t*np.ptp(limits['tilt']), roll=0.)
            for axis,value in (('truck',x),('dolly',y)):
                if axis in limits:
                    pose[axis] = float(np.clip(value,limits[axis][0]+.04,limits[axis][1]-.04))
            self.candidates.append(pose)

    def observe(self, observation, detections):
        self.visited.append(observation.joints)
        h,w=observation.depth_m.shape
        for d in detections:
            if d.identity.category in ('floor','wall'):
                continue
            old=self.objects.get(d.identity.object_id,{})
            self.objects[d.identity.object_id]=dict(center=np.median(d.points_world,axis=0),
                uncertainty=(1.5 if d.identity.category=='unknown' else .5)+(1-d.confidence),
                best_area=max(old.get('best_area',0),float(d.mask.sum()/(h*w))),
                views=old.get('views',0)+1)

    def choose(self):
        if not self.visited or not self.candidates:
            return None
        ranked=[]
        for index,pose in enumerate(self.candidates):
            distance=min(sum(abs(pose[n]-v.get(n,0))/np.ptp(self.sim.limits[n]) for n in pose)
                         for v in self.visited)
            if distance < .12:
                continue
            for n,value in pose.items():
                self.scratch.joint(n).qpos[0]=value
            mujoco.mj_kinematics(self.sim.model,self.scratch)
            mujoco.mj_camlight(self.sim.model,self.scratch)
            pos=self.scratch.cam_xpos[self.sim.camera_id]
            rot=self.scratch.cam_xmat[self.sim.camera_id].reshape(3,3) @ np.diag([1,-1,-1])
            gain=0.
            for record in self.objects.values():
                xyz=(record['center']-pos) @ rot
                if xyz[2]<=.1:
                    continue
                tan=np.tan(np.deg2rad(self.sim.model.cam_fovy[self.sim.camera_id])/2)
                if abs(xyz[1]/xyz[2])<tan and abs(xyz[0]/xyz[2])<tan*self.sim.width/self.sim.height:
                    gain+=(record['uncertainty']+1/(record['views']+1))/(1+15*record['best_area'])
            ranked.append((gain+.6*min(distance,2),index))
        if not ranked:
            return None
        score,index=max(ranked,key=lambda item:(item[0],-item[1]))
        return self.candidates.pop(index), dict(score=float(score), reason='underobserved_or_uncertain_objects_and_viewpoint_diversity')
