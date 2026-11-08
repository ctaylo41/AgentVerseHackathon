import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import numpy as np

from robopod.adaptive import AdaptiveViews
from robopod.actors import perform, observed
from robopod.learned import build_backends, validate_config
from robopod.perception import PerceptionPipeline, SegmentationPerception
from robopod.replanning import revise_plan
from robopod.simulation import Simulation
from test_mapping import GEOMS, observation
from test_planning import valid_plan, SCENE


class VisionExtensions(unittest.TestCase):
    def test_unknown_instance_is_associated_without_duplicate(self):
        geoms=copy.deepcopy(GEOMS)
        geoms['3']['body']=geoms['4']['body']='unfamiliar'
        backend=Mock()
        backend.detect.return_value=[dict(category='keyboard',confidence=.8,bbox_xyxy=[2,0,4,2])]
        pipeline=PerceptionPipeline(geoms,detector=backend)
        ds,_=pipeline.process(observation())
        found=[d for d in ds if d.identity.object_id=='body:unfamiliar']
        self.assertEqual(len(found),1)
        self.assertEqual(found[0].identity.category,'keyboard')
        self.assertEqual(found[0].source,'learned_associated_segmentation')
        backend.detect.side_effect=RuntimeError('offline')
        ds,extra=pipeline.process(observation(1))
        self.assertTrue(extra['warnings'])
        self.assertEqual(next(d for d in ds if d.identity.object_id=='body:unfamiliar').identity.category,'keyboard')

    def test_malformed_backend_does_not_disable_fallback(self):
        backend=Mock()
        for value in ([{'confidence':'oops'}],[dict(category='thing',confidence=.8,bbox_xyxy=[1])]):
            backend.detect.return_value=value
            ds,extra=PerceptionPipeline(GEOMS,detector=backend).process(observation())
            self.assertEqual(len(ds),4)
            self.assertTrue(extra['warnings'])
        detector,pose,warnings=build_backends({'yolo_model':'/does/not/exist.pt'})
        self.assertIsNone(detector)
        self.assertTrue(warnings)
        with self.assertRaises(ValueError): validate_config({'adaptive_views':13})

    def test_detail_mask_is_exact_geom_and_parent_retained(self):
        ds={d.identity.object_id:d for d in SegmentationPerception(GEOMS).detect(observation())}
        self.assertEqual(ds['part:product:keys'].mask.sum(),2)
        self.assertEqual(ds['body:product'].mask.sum(),4)
        self.assertEqual(ds['part:product:keys'].identity.parent_id,'body:product')

    def test_adaptive_selection_is_repeatable_and_observation_driven(self):
        with Simulation(160,120) as sim:
            obs=sim.capture()
            ds=SegmentationPerception(sim.geometry_catalog()).detect(obs)
            a,b=AdaptiveViews(sim),AdaptiveViews(sim)
            a.observe(obs,ds);b.observe(obs,ds)
            pose,decision=a.choose()
            self.assertEqual(pose,b.choose()[0])
            self.assertGreater(decision['score'],0)
            # Moving an observed ambiguous object's centre changes predicted information gain.
            c=AdaptiveViews(sim);c.observe(obs,ds)
            c.objects={'unseen_edge':dict(center=np.array([2.,0.,1.5]),uncertainty=100.,views=1,best_area=0)}
            self.assertNotEqual(c.choose()[0],pose)
            self.assertNotEqual(pose,obs.joints)

    def test_expressive_actions_have_visible_evidence(self):
        with Simulation() as sim:
            perception=SegmentationPerception(sim.geometry_catalog())
            for action in ('smile','look_at_camera','point_to_product','hold_near_face','turn_to_product','turn_to_camera'):
                sim.reset()
                before=perception.detect(sim.capture())
                perform(sim,action)
                after=perception.detect(sim.capture())
                passed,evidence=observed(before,after,action,sim.capture().world_from_camera[:3,3])
                self.assertTrue(passed,(action,evidence))
            sim.reset()
            ds=perception.detect(sim.capture())
            self.assertFalse(observed(ds,ds,'point_to_product')[0])

    def test_replan_rejects_goal_relaxation_and_dropped_actions(self):
        plan=valid_plan()
        scene=dict(SCENE,viewpoints={'view_007':{}})
        client=Mock();client.check_model.return_value={'model':'gemma4:e2b'}
        for mutate in ('scale','instruction','notes_only','valid'):
            revised=copy.deepcopy(plan)
            plan['shots'][0]['human_instructions']=[{'subject_id':'body:person','instruction':'smile'}]
            revised=copy.deepcopy(plan)
            if mutate=='scale': revised['shots'][0]['targets'][0]['frame_height_fraction']=.1
            elif mutate=='instruction': revised['shots'][0]['human_instructions']=[]
            elif mutate=='notes_only': revised['shots'][0]['feasibility_notes']='Try again.'
            else: revised['shots'][0]['human_instructions'].append({'subject_id':'body:person','instruction':'Step closer to the camera.'})
            client.chat.return_value=json.dumps(revised)
            with TemporaryDirectory() as tmp:
                call=lambda:revise_plan(plan,scene,[],np.zeros((10,10,3),np.uint8),'original brief',tmp,client=client)
                if mutate=='valid': self.assertEqual(call()['shots'][0]['targets'],plan['shots'][0]['targets'])
                else:
                    with self.assertRaises(ValueError): call()

    def test_pose_landmarks_associate_to_existing_person(self):
        backend=Mock()
        backend.estimate.return_value=[dict(landmarks=[dict(index=0,x=0,y=0,visibility=.9,presence=.9)])]
        ds,extra=PerceptionPipeline(GEOMS,pose_estimator=backend).process(observation())
        person=next(d for d in ds if d.identity.object_id=='body:person')
        self.assertIn('pose_0',person.landmarks)
        self.assertEqual(person.landmarks['pose_0']['source'],'mediapipe_local')
        self.assertEqual(person.landmarks['pose_0']['world_position'],[-1.5,-1.5,2.])

    def test_local_adapter_api_contracts(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from robopod.learned import YoloDetector, MediaPipePose
        with TemporaryDirectory() as tmp:
            weights=Path(tmp)/'weights.pt';weights.touch()
            box=NS(cls=NS(item=lambda:0),conf=NS(item=lambda:.9),xyxy=[NS(cpu=lambda:NS(numpy=lambda:np.array([1,2,3,4])))])
            model=Mock();model.predict.return_value=[NS(boxes=[box],names={0:'cup'},masks=NS(xy=[np.array([[1,2],[3,2],[3,4]])]))]
            module=NS(YOLO=Mock(return_value=model))
            with patch.dict('sys.modules',{'ultralytics':module}):
                detector=YoloDetector(weights)
                rgb=np.array([[[1,2,3]]],np.uint8)
                found=detector.detect(rgb)
                self.assertEqual(found[0]['category'],'cup')
                np.testing.assert_array_equal(model.predict.call_args.kwargs['source'],[[[3,2,1]]])
                module.YOLO.assert_called_once_with(str(weights.resolve()),task='segment')
            pose_model=Mock();pose_model.detect.return_value=NS(pose_landmarks=[[NS(x=.1,y=.2,visibility=.9,presence=.8)]])
            vision=NS(PoseLandmarkerOptions=Mock(),RunningMode=NS(IMAGE='IMAGE'),PoseLandmarker=NS(create_from_options=Mock(return_value=pose_model)))
            mp=NS(tasks=NS(vision=vision,BaseOptions=Mock()),Image=Mock(),ImageFormat=NS(SRGB='SRGB'))
            with patch.dict('sys.modules',{'mediapipe':mp}):
                pose=MediaPipePose(weights)
                self.assertEqual(pose.estimate(rgb)[0]['landmarks'][0]['index'],0)
                mp.tasks.BaseOptions.assert_called_once_with(model_asset_path=str(weights.resolve()))
                pose.close();pose_model.close.assert_called_once()

    def test_bounded_replanning_consumes_failures_and_keeps_successful_shots(self):
        from unittest.mock import patch
        from robopod.replanning import execute_with_replanning
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=valid_plan()
            second=copy.deepcopy(plan['shots'][0]);second['shot_id']='shot_2'
            plan['shots'].append(second)
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'scene.json').write_text(json.dumps(dict(SCENE,viewpoints={'view_007':{}})))
            sim=Mock();sim.limits={'height':(.5,2.)}
            calls=[]
            def execute(sim,plan_path,scene_path,output,**kwargs):
                shots=json.loads(Path(plan_path).read_text())['shots']
                calls.append([s['shot_id'] for s in shots])
                results=[]
                for i,shot in enumerate(shots):
                    folder=output/f'shot_{i}';folder.mkdir(parents=True)
                    failed=shot['shot_id']=='shot_2'
                    (folder/'result.json').write_text(json.dumps(dict(model_replan_needed=failed,
                        after={'failures':['scale']},motion_completed=True,refinement={'model_replan_needed':failed,'pending_creative':[]},subject_repositioning_requests=[])))
                    results.append(dict(shot_id=shot['shot_id'],directory=folder.name,status='needs_adjustment' if failed else 'satisfied_observed_criteria'))
                return dict(status='needs_adjustment',shots=results)
            def revise(plan,scene,failures,image,brief,output,**kwargs):
                self.assertEqual([f['shot_id'] for f in failures],['shot_2'])
                output.mkdir(parents=True,exist_ok=True)
                (output/'shot_plan.json').write_text(json.dumps(plan))
                return plan
            with patch('robopod.execution.execute_plan',side_effect=execute), \
                 patch('robopod.replanning.fresh_context',return_value=({},Mock(rgb=np.zeros((2,2,3))))), \
                 patch('robopod.replanning.revise_plan',side_effect=revise) as replan:
                result=execute_with_replanning(sim,root/'plan.json',root/'scene.json',root/'out',budget=1,perception=Mock(),progress=None)
            self.assertEqual(calls,[['shot_1','shot_2'],['shot_2']])
            self.assertEqual(replan.call_count,1)
            self.assertEqual(len(result['shots']),2)
            self.assertEqual(result['status'],'needs_adjustment')
            self.assertTrue((root/'out/execution_report.json').exists())


    def test_configuration_is_queued_and_invalid_settings_do_not_reset(self):
        from robopod.app import AppState
        state=AppState()
        state.update(phase='ready')
        with self.assertRaises(ValueError): state.submit('configure',config={'replan_budget':3})
        self.assertFalse(state.snapshot()['busy'])
        state.submit('configure',config={'adaptive_views':4,'replan_budget':0})
        action,config=state.commands.get_nowait()
        self.assertEqual(action,'configure')
        self.assertEqual(config['adaptive_views'],4)
        self.assertEqual(config['replan_budget'],0)
        self.assertTrue(state.snapshot()['busy'])

    def test_replan_cannot_add_motion_to_legacy_static_plan(self):
        legacy=valid_plan()
        for key in ('motion','horizon','motion_parameters'): legacy['shots'][0].pop(key)
        changed=copy.deepcopy(legacy)
        changed['shots'][0].update(motion='truck_right',motion_parameters=dict(amplitude_m=.1,direction='right',subject_lock='body:person'))
        client=Mock();client.check_model.return_value={'model':'gemma4:e2b'}
        client.chat.return_value=json.dumps(changed)
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'motion/horizon'):
            revise_plan(legacy,dict(SCENE,viewpoints={'view_007':{}}),[],np.zeros((2,2,3),np.uint8),'static',tmp,client=client)


    def test_detail_correction_moves_parent_only_when_requested(self):
        from robopod.teleprompter import correction
        shot=valid_plan()['shots'][0]
        shot['targets'][0]['object_id']='part:product:controls'
        scene=copy.deepcopy(SCENE)
        scene['objects']['part:product:controls']=dict(category='component',label='Controls',parent_id='body:product')
        measured={'targets':{'part:product:controls':dict(framing={'frame_height_fraction':.1},scale_error=.5)}}
        self.assertIsNone(correction(shot,measured,scene))
        shot['description']='Bring the product closer for the controls shot.'
        response=correction(shot,measured,scene)
        self.assertEqual(response.object_id,'body:product')
        self.assertLess(response.displacement,0)
