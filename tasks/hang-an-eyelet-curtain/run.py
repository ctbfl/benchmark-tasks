"""Run the offline scene check, the scripted demo or one Dream-1 trial."""
import json
import sys
import traceback
from datetime import datetime, timezone

from config import CONFIG, HERE


def check(experiment, out):
    import numpy as np
    from PIL import Image
    from runtime import write_json
    from inspect_robots import Action, Scene
    from dream_eval import CurtainEmbodiment
    from config import INSTRUCTION
    adapter = CurtainEmbodiment(experiment)
    observation = adapter.reset(Scene(id='hang-an-eyelet-curtain', instruction=INSTRUCTION), seed=0)
    assert set(observation.images) == {'left_cam', 'top_cam', 'right_cam'}
    assert set(observation.state) == {'joint_pos'} and not observation.extra
    assert observation.state['joint_pos'].shape == (14,)
    links = experiment.link_poses()
    print('CURTAIN nodes', len(experiment.rest), 'triangles', len(experiment.faces), 'rings', experiment.links, flush=True)
    for name, row in zip(experiment.links, links):
        print('LINK', name, np.round(row, 4).tolist(), flush=True)
    print('ROD', np.round(experiment.pose(experiment.rod), 4).tolist(), 'STAND', np.round(experiment.pose(experiment.stand), 4).tolist(), flush=True)
    print('SCORE0', experiment.result(), flush=True)
    images = experiment.capture()
    for name, pixels in images.items():
        if name == 'observer':
            continue
        assert pixels.shape == (360, 640, 3) and pixels.std() > 10
        Image.fromarray(pixels).save(out/'frames'/(name+'.png'))
    camera = experiment.base.scene['screenshot']
    camera.update(1/20, force_recompute=True)
    top = experiment.base.scene['top_cam']
    k = camera.data.intrinsic_matrices.torch[0].cpu().numpy()
    top_k = top.data.intrinsic_matrices.torch[0].cpu().numpy()
    assert np.allclose(k[:2], 3*top_k[:2], atol=1e-3)
    assert np.allclose(camera.data.pos_w.torch.cpu(), top.data.pos_w.torch.cpu())
    pixels = camera.data.output['rgb'].torch[0, ..., :3].cpu().numpy()
    assert pixels.shape == (1080, 1920, 3)
    Image.fromarray(pixels).save(out/'frames/screenshot.png')
    write_json(out/'calibration.json', dict(top_intrinsics=top_k.tolist(), screenshot_intrinsics=k.tolist(),
        position=camera.data.pos_w.torch[0].cpu().tolist(), rotation=list(camera.cfg.offset.rot),
        screenshot_resolution=[1920, 1080]))
    initial = images['left_cam'].copy()
    action = experiment.ready.copy()
    action[0] += .12
    for _ in range(20):
        result = adapter.step(Action(data=action))
    moved = experiment.capture()['left_cam']
    change = float(np.abs(moved.astype(float)-initial).mean())
    assert change > 1.
    assert result.observation.state_time == 1. and experiment.ticks == 240
    assert abs(result.observation.state['joint_pos'][0]-observation.state['joint_pos'][0]) > .05
    write_json(out/'check.json', dict(passed=True, api_calls=0, left_wrist_image_change=change,
                                      score=experiment.result()))


def main():
    modes = ('check', 'demo', 'dream')
    assert len(sys.argv) == 2 and sys.argv[1] in modes, 'Usage: python run.py check|demo|dream'
    mode = sys.argv[1]
    out = HERE/'runs'/(mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    (out/'logs').mkdir(parents=True)
    (out/'frames').mkdir()
    print('RUN_DIRECTORY='+str(out), flush=True)
    (out/'config.json').write_text(json.dumps(CONFIG, indent=2)+'\n')
    from isaaclab.app import AppLauncher
    frames = mode in ('check', 'dream')
    app = AppLauncher(headless=True, enable_cameras=frames, kit_args=
        '--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 '
        f'--/log/file={out}/logs/kit.log').app
    experiment, code = None, 0
    try:
        from assets import build
        from runtime import Experiment
        build()
        experiment = Experiment(out, cameras=frames, screenshot=mode == 'check')
        if mode == 'check':
            check(experiment, out)
        elif mode == 'dream':
            from dream_eval import run
            run(experiment)
        else:
            from demo import run as demo
            demo(experiment)
        print('RESULT='+json.dumps({k: experiment.result()[k] for k in
              ('score', 'rings_on_rod', 'rod_on_stand', 'control_steps')}), flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
        (out/'FAILED').write_text(traceback.format_exc())
    finally:
        if experiment is not None:
            experiment.save()
            experiment.env.close()
        sys.stdout.flush()
        sys.stderr.flush()
        app.close(exit_code=code)


if __name__ == '__main__':
    main()
