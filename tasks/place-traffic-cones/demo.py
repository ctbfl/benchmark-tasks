"""Replay fixed 20 Hz robot actions, independently measuring the physical outcome."""
from datetime import datetime, timezone
import json
import time
import traceback
import numpy as np
from config import HERE, MAX_STEPS, CONTROL_HZ


def main():
    actions=np.load(HERE/'demo_actions.npy')
    assert actions.ndim==2 and actions.shape[1]==14 and 0<len(actions)<=MAX_STEPS
    assert np.isfinite(actions).all()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=HERE/'runs'/f'demo-{stamp}'
    out.mkdir(parents=True)
    print('RUN_DIRECTORY',out,flush=True)
    from archive import snapshot
    snapshot(out)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8').app
    runtime=None
    exit_code=0
    started=time.perf_counter()
    try:
        from runtime import Runtime
        runtime=Runtime(log_dir=out/'logs')
        # The only policy input is the fixed action array and its integer time index.
        for i,action in enumerate(actions):
            runtime.step(action)
            if (i+1)%100==0:
                print('REPLAY', (i+1)/CONTROL_HZ, 'seconds',flush=True)
        assert runtime.measure()['score']==1., runtime.measure()
        assert all(m['score']==1. for m in runtime.measurements[-CONTROL_HZ:])
    except Exception:
        exit_code=1
        traceback.print_exc()
    finally:
        if runtime is not None:
            runtime.save(out)
            (out/'metrics.json').write_text(json.dumps(dict(exit_code=exit_code,
                wall_time_s=time.perf_counter()-started,**runtime.measure()),indent=2)+'\n')
            runtime.close()
        app.close(exit_code=exit_code)


if __name__=='__main__':
    main()
