"""Inspect the fixed scene or replay the open-loop demo."""
from datetime import datetime, timezone
import sys
import traceback
from config import HERE


def main(mode):
    import numpy as np
    from PIL import Image
    from simulation import Simulation, archive_source, write_json
    out=HERE/'runs'/(mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True)
    archive_source(out)
    print('RUN_DIRECTORY',out,flush=True)
    sim=Simulation(out,cameras=mode=='check',screenshot=mode=='check')
    error=None
    try:
        if mode=='check':
            for _ in range(12): sim.capture()
            camera=sim.base.scene['screenshot']
            camera.update(.05,force_recompute=True)
            image=camera.data.output['rgb'].torch[0,...,:3].cpu().numpy()
            Image.fromarray(image).save(out/'screenshot.png')
            write_json(out/'initial.json',{name:sim.object_state(name).tolist() for name in sim.objects})
            print('INITIAL', {name:sim.object_state(name).round(4).tolist() for name in sim.objects}, flush=True)
            for _ in range(100): sim.step(sim.action)
            assert sim.scorer.insertion_s is None, sim.state()
        elif mode=='demo':
            actions=np.load(HERE/'demo_actions.npy')
            assert actions.ndim==2 and actions.shape[1]==14
            for i,action in enumerate(actions):
                sim.step(action)
                if i%100==99: print('DEMO',sim.state(),flush=True)
            assert sim.score==1.,sim.state()
        else:
            raise ValueError(mode)
        print('RESULT',sim.state(),flush=True)
    except Exception:
        error=traceback.format_exc()
        raise
    finally:
        sim.save('success' if error is None else 'failed',error)
        sim.close()


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('check','demo'), 'Usage: python run.py check|demo'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cpu',enable_cameras=sys.argv[1]=='check',
        kit_args='--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8').app
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
