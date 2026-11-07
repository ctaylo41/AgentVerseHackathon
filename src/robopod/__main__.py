"""Local cinematographer: capture, exploration, semantic mapping, Gemma planning and actuator execution."""
import argparse
import time
from pathlib import Path

from .simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'viewer', 'explore', 'map', 'plan', 'execute', 'app'], nargs='?', default='capture')
    parser.add_argument('--port', type=int, default=8765, help='Local app port')
    parser.add_argument('--refine', action='store_true', help='Enable teleprompter and scripted closed-loop responses')
    parser.add_argument('--max-adjustments', type=int, default=3, help='Subject-response budget per shot (0–6)')
    parser.add_argument('--plan', type=Path, help='Validated shot_plan.json for execute/app modes')
    parser.add_argument('--preview', action='store_true', help='Show the executing camera; Q cancels')
    parser.add_argument('--brief', help='Creative brief for local Gemma planning')
    parser.add_argument('--scene-map', type=Path, help='Defaults to SURVEY/semantic_map/scene_map.json')
    parser.add_argument('--images', type=int, default=3, help='Representative images, 1–4')
    parser.add_argument('--timeout', type=float, default=180, help='Ollama request timeout in seconds')
    parser.add_argument('--survey', type=Path, help='Existing survey directory for map/plan modes')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fovy', type=float, default=45)
    parser.add_argument('--camera-height', type=float, default=1.4)
    parser.add_argument('--truck', type=float, default=0)
    parser.add_argument('--dolly', type=float, default=0)
    parser.add_argument('--pan', type=float, default=0)
    parser.add_argument('--tilt', type=float, default=0)
    parser.add_argument('--roll', type=float, default=0)
    parser.add_argument('--voxel-size', type=float, default=0.1)
    parser.add_argument('--min-novelty', type=float, default=0.03)
    parser.add_argument('--settle-timeout', type=float, default=5.0)
    args = parser.parse_args()
    if args.mode == 'app':
        from .app import run_app
        run_app(args.port, args.output or Path('outputs/studio'), survey=args.survey, plan_path=args.plan)
        return
    if args.mode == 'execute':
        if args.plan is None or (args.scene_map is None and args.survey is None):
            parser.error('execute requires --plan and either --scene-map or --survey')
        from datetime import datetime
        from .execution import execute_plan
        from .control import MotionError
        scene_path = args.scene_map or args.survey / 'semantic_map/scene_map.json'
        output = args.output or Path('outputs') / datetime.now().strftime('execution-%Y%m%d-%H%M%S-%f')
        try:
            with Simulation(args.width, args.height, args.fovy) as sim:
                report = execute_plan(sim, args.plan, scene_path, output, preview=args.preview, refine=args.refine, max_adjustments=args.max_adjustments)
        except (ValueError, OSError, MotionError) as error:
            parser.exit(1, f'Execution failed: {error}\n')
        print(f"{report['status']}: {output.resolve() / 'execution_report.json'}")
        raise SystemExit(0 if report['status'] == 'completed' else 2)
    if args.mode == 'plan':
        if args.survey is None or args.brief is None:
            parser.error('plan requires --survey and --brief')
        from datetime import datetime
        from .gemma import generate_plan, OllamaClient, PlannerError
        output = args.output or args.survey / datetime.now().strftime('plan-%Y%m%d-%H%M%S-%f')
        scene_path = args.scene_map or args.survey / 'semantic_map/scene_map.json'
        try:
            plan = generate_plan(args.survey, scene_path, args.brief, output,
                                 client=OllamaClient(args.timeout), image_count=args.images)
        except (PlannerError, ValueError, OSError) as error:
            parser.exit(1, f'Planning failed: {error}\n')
        print(f"Validated {len(plan['shots'])} shots: {output.resolve() / 'shot_plan.json'}")
        print('High-level objectives validated. Use execute to test geometric feasibility and move the rig.')
        return
    if args.mode == 'map':
        if args.survey is None:
            parser.error('map requires --survey')
        from .mapping import map_survey
        output = args.output or args.survey / 'semantic_map'
        scene = map_survey(args.survey, output)
        print(f'Mapped {len(scene.objects)} persistent objects from {len(scene.viewpoints)} views to {output.resolve()}')
        return
    if args.output is None:
        from datetime import datetime
        args.output = (Path('outputs') / datetime.now().strftime('survey-%Y%m%d-%H%M%S-%f')
                       if args.mode == 'explore' else Path('outputs/capture'))
    with Simulation(args.width, args.height, args.fovy) as sim:
        if args.mode == 'explore':
            from .exploration import explore
            def report(view, total):
                print(f"{view['view_id']} / {total}: {view['status']}; "
                      f"coverage={view['cumulative_voxels']} surface voxels", flush=True)
            result = explore(sim, args.output, voxel_size=args.voxel_size,
                             min_novelty=args.min_novelty, settle_timeout=args.settle_timeout,
                             progress=report)
            print(f"{result['status']}: {result['saved_views']} views saved to {args.output.resolve()}")
            if result['status'] != 'completed':
                raise SystemExit(1)
            return
        sim.command(height=args.camera_height, truck=args.truck, dolly=args.dolly,
                    pan=args.pan, tilt=args.tilt, roll=args.roll)
        sim.step(3)
        if args.mode == 'capture':
            sim.save(args.output)
            print(f'Saved RGB, depth, segmentation and calibration to {args.output.resolve()}')
            print(sim.configuration)
            return
        import mujoco
        import mujoco.viewer
        import cv2
        print('Viewer: right panel > Control for actuator sliders. Camera window: S saves; R resets; Q exits.')
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = sim.camera_id
            try:
                while viewer.is_running():
                    start = time.monotonic()
                    with viewer.lock():
                        sim.step()
                        obs = sim.capture()
                    viewer.sync()
                    cv2.imshow('Robopod RGB | S save | R reset | Q quit', cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xff
                    if key == ord('q'):
                        break
                    if key == ord('r'):
                        with viewer.lock():
                            sim.reset()
                    if key == ord('s'):
                        sim.save(args.output / f'{sim.data.time:.3f}')
                    time.sleep(max(0, 1/30 - (time.monotonic() - start)))
            finally:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
