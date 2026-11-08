"""Explicit kinematic performance scripts; completion is evaluated from rendered evidence."""
import math
import mujoco
import numpy as np
from .control import advance


def yaw(angle):
    return np.array([math.cos(angle/2),0.,0.,math.sin(angle/2)])


def perform(sim, action, callback=None, trace=None):
    person=int(sim.model.body('person').mocapid[0])
    product=int(sim.model.body('product').mocapid[0])
    upper=sim.model.body('upper_body').id
    head=sim.model.body('head_frame').id
    right=sim.model.body('right_arm_frame').id
    left=sim.model.body('left_arm_frame').id
    body_start=sim.model.body_quat.copy()
    geom_start=sim.model.geom_pos.copy()
    mocap_start=sim.data.mocap_pos.copy()
    body_end=body_start.copy();geom_end=geom_start.copy();mocap_end=mocap_start.copy()
    center=sim.data.mocap_pos[person]
    camera=sim.data.cam_xpos[sim.camera_id]
    facing=math.atan2(camera[0]-center[0],-(camera[1]-center[1]))
    root_yaw=2*math.atan2(sim.data.mocap_quat[person,3],sim.data.mocap_quat[person,0])
    if action in ('turn_to_camera','turn_to_product'):
        target=camera if action=='turn_to_camera' else sim.data.mocap_pos[product]
        angle=math.atan2(target[0]-center[0],-(target[1]-center[1]))-root_yaw
        body_end[upper]=yaw(float(np.clip(angle,-1.2,1.2)))
    if action=='look_away':
        away=float(np.clip(facing-root_yaw+(.9 if facing<=root_yaw else -.9),-1.2,1.2))
        body_end[upper]=yaw(away*.6)
        body_end[head]=yaw(away*.8)
    if action in ('look_at_camera','look_and_smile'):
        upper_yaw=2*math.atan2(body_start[upper,3],body_start[upper,0])
        body_end[head]=yaw(float(np.clip(facing-root_yaw-upper_yaw,-1.1,1.1)))
    if action in ('smile','look_and_smile'):
        geom_end[sim.model.geom('mouth_left').id,2]=-.035
        geom_end[sim.model.geom('mouth_right').id,2]=-.035
    if action=='hold_near_face':
        face=sim.data.body('head_frame').xpos.copy()
        mocap_end[product]=face+np.array([0,-.38,-.22])
    if action in ('point_to_product','hold_near_face'):
        # Kinematic arm poses; contact forces and grasp stability are not simulated.
        for arm,sign in ((right,1),(left,-1)):
            if action=='point_to_product' and sign==-1: continue
            shoulder=sim.data.xpos[arm]
            target=sim.data.geom('controls').xpos.copy() if action=='point_to_product' else mocap_end[product]+np.array([sign*.3,0,0])
            desired=sim.data.body('upper_body').xmat.reshape(3,3).T @ (target-shoulder)
            original=np.array([sign*.25,-.2,-.44]);original/=np.linalg.norm(original)
            desired/=max(np.linalg.norm(desired),1e-8)
            cross=np.cross(original,desired);dot=float(np.dot(original,desired))
            quat=np.array([1+dot,*cross]);quat/=max(np.linalg.norm(quat),1e-8)
            body_end[arm]=quat
    for i in range(1,51):
        t=i/50;blend=3*t*t-2*t*t*t
        for body in (upper,head,right,left):
            quat=(1-blend)*body_start[body]+blend*body_end[body]
            sim.model.body_quat[body]=quat/np.linalg.norm(quat)
        sim.model.geom_pos[:]=geom_start+(geom_end-geom_start)*blend
        sim.data.mocap_pos[:]=mocap_start+(mocap_end-mocap_start)*blend
        advance(sim,.02,trace)
        if callback and i%5==0: callback(sim)
    return dict(scripted=True, action=action, mode='kinematic_pose_animation')


def observed(before, after, action, camera_position=None):
    def person(detections):
        return next((d for d in detections if d.identity.object_id=='body:person'),None)
    old,new=person(before),person(after)
    if old is None or new is None:
        return False,dict(reason='Presenter must be visible before and after the action')
    def landmark(d,key):
        item=d.landmarks.get(key)
        return None if item is None else np.asarray(item['world_position'])
    def evidence(key):
        a,b=landmark(old,key),landmark(new,key)
        return None if a is None or b is None else float(np.linalg.norm(b-a))
    if action=='hold_near_face':
        prop=next((d for d in after if d.identity.object_id=='body:product'),None)
        head=landmark(new,'head')
        if prop is None or head is None:
            return False,dict(reason='Head and held product must both be visible')
        distance=float(np.linalg.norm(np.median(prop.points_world,axis=0)-head))
        return distance<.65,dict(observed_face_product_distance_m=distance,threshold_m=.65)
    if action=='look_away':
        moved=evidence('nose') or evidence('head') or 0.0
        eyes=[new.landmarks.get(k) for k in ('left_eye','right_eye')]
        nose_uv=(new.landmarks.get('nose') or {}).get('pixel_uv')
        turned_away=False
        if all(e is not None for e in eyes) and nose_uv is not None:
            lo,hi=sorted(e['pixel_uv'][0] for e in eyes)
            span=max(hi-lo,1.0)
            turned_away=not (lo+.2*span<=nose_uv[0]<=hi-.2*span)
        return bool(moved>.012 or turned_away),dict(observed_head_turn_m=float(moved),turned_away=bool(turned_away))
    if action in ('smile','look_and_smile'):
        head=landmark(new,'head');left=landmark(new,'mouth_left');right=landmark(new,'mouth_right')
        if head is None or left is None or right is None:
            return False,dict(reason='Mouth corners must be observed to verify smile')
        # Use visible facial feature positions, never the animation state.
        smile=bool(min(left[2],right[2])>head[2]-.055)
        if action=='smile':
            return smile,dict(observed_smile_corners=smile)
    if action in ('look_at_camera','look_and_smile'):
        head=landmark(new,'head');nose=landmark(new,'nose')
        if head is None or nose is None:
            return False,dict(reason='Head and nose must be observed to verify gaze')
        # Image-space nose between eyes indicates the face points approximately toward the lens.
        eyes=[new.landmarks.get(k) for k in ('left_eye','right_eye')]
        if any(e is None for e in eyes):
            return False,dict(reason='Both eyes must be visible')
        lo,hi=sorted(e['pixel_uv'][0] for e in eyes)
        aligned=lo<=new.landmarks['nose']['pixel_uv'][0]<=hi
        return bool(aligned and (action!='look_and_smile' or smile)),dict(observed_face_alignment=bool(aligned))
    prop=next((d for d in after if d.identity.object_id=='body:product'),None)
    if action=='point_to_product':
        hand=landmark(new,'right_hand');arm=landmark(new,'right_arm')
        if hand is None or arm is None or prop is None:
            return False,dict(reason='Arm, hand and product must be observed')
        detail=next((d for d in after if d.identity.object_id=='part:product:controls'),prop)
        direction=hand-arm;target=np.median(detail.points_world,axis=0)-hand
        cosine=float(np.dot(direction,target)/max(np.linalg.norm(direction)*np.linalg.norm(target),1e-8))
        return cosine>.8,dict(observed_pointing_cosine=cosine,minimum=.8)
    left=landmark(new,'left_shoulder');right=landmark(new,'right_shoulder')
    if left is None or right is None:
        return False,dict(reason='Both shoulders must be observed to verify orientation')
    center=(left+right)/2
    target=camera_position if action=='turn_to_camera' else (None if prop is None else np.median(prop.points_world,axis=0))
    if target is None:
        return False,dict(reason='Orientation target is not observed')
    across=right-left
    forward=np.array([across[1],-across[0]])
    desired=np.asarray(target)[:2]-center[:2]
    cosine=float(np.dot(forward,desired)/max(np.linalg.norm(forward)*np.linalg.norm(desired),1e-8))
    return cosine>np.cos(np.deg2rad(20)),dict(observed_shoulder_facing_cosine=cosine,tolerance_degrees=20)
