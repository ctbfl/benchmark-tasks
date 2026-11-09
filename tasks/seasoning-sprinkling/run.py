"""Load-check or replay fixed 20 Hz actions."""
import json
import sys
import traceback
from datetime import datetime, timezone
from config import CONFIG,HERE


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('check','demo'), 'Usage: python run.py check|demo'
    mode=sys.argv[1]
    out=HERE/'runs'/(mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    (out/'logs').mkdir(parents=True)
    (out/'frames').mkdir()
    print('RUN_DIRECTORY='+str(out),flush=True)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=mode=='check',kit_args=
        '--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 '
        f'--/log/file={out}/logs/kit.log').app
    experiment,code=None,0
    try:
        from assets import build
        from runtime import Experiment,write_json
        if mode=='check': build()
        experiment=Experiment(out,cameras=mode=='check',screenshot=mode=='check')
        if mode=='check':
            import numpy as np
            from PIL import Image
            images=experiment.capture()
            for name,pixels in images.items():
                assert pixels.shape==(360,640,3) and pixels.std()>10
                Image.fromarray(pixels).save(out/'frames'/(name+'.png'))
            camera=experiment.base.scene['screenshot']
            camera.update(1/20,force_recompute=True)
            top=experiment.base.scene['top_cam']
            k=camera.data.intrinsic_matrices.torch[0].cpu().numpy()
            top_k=top.data.intrinsic_matrices.torch[0].cpu().numpy()
            assert np.allclose(k[:2],3*top_k[:2],atol=1e-3)
            assert np.allclose(camera.data.pos_w.torch,top.data.pos_w.torch)
            pixels=camera.data.output['rgb'].torch[0,...,:3].cpu().numpy()
            assert pixels.shape==(1080,1920,3)
            Image.fromarray(pixels).save(out/'frames/screenshot.png')
            write_json(out/'calibration.json',dict(top_intrinsics=top_k.tolist(),screenshot_intrinsics=k.tolist()))
            print('INITIAL',experiment.view.get_root_transforms().tolist(),experiment.angles().tolist(),experiment.result,flush=True)
            write_json(out/'check.json',dict(passed=True,api_calls=0,score=experiment.result))
        else:
            import numpy as np
            for action in np.load(HERE/'demo_actions.npy',allow_pickle=False): experiment.step(action)
            assert experiment.result['success'],experiment.result
        print('RESULT='+json.dumps(experiment.result),flush=True)
    except BaseException:
        code=1
        traceback.print_exc()
    finally:
        if experiment is not None:
            experiment.save()
            experiment.env.close()
        app.close(exit_code=code)


if __name__=='__main__': main()
