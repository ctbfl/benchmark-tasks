"""Evaluate Dream-1 in the shared environment; --check never calls the model."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import sys
import time
import traceback

import numpy as np
from inspect_robots import Action, Scene, Task, StepResult, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment
from config import HERE, CONFIG, CONTROL_HZ, MAX_STEPS, INSTRUCTION

CAMERAS=tuple(CONFIG['observation']['camera_names'])


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append_json(file,value):
    file.write(json.dumps(value,allow_nan=False)+'\n')
    file.flush()
    os.fsync(file.fileno())


class SavedDreamSession(DreamSession):
    def __init__(self,out,physics_tick):
        super().__init__()
        self.physics_tick=physics_tick
        self.out=out/'dream'
        self.out.mkdir()
        for name in CAMERAS:
            (self.out/name).mkdir()
        self.tick=0
        self.observations=(self.out/'observations.jsonl').open('x')
        self.replies=(self.out/'replies.jsonl').open('x')

    def observe(self,frames,state,prompt,timestamp):
        assert set(frames)==set(CAMERAS) and prompt==INSTRUCTION
        paths={}
        for name,jpeg in frames.items():
            path=self.out/name/f'{self.tick:06}.jpg'
            path.write_bytes(base64.b64decode(jpeg))
            paths[name]=str(path.relative_to(self.out))
        append_json(self.observations,dict(tick=self.tick,time_s=timestamp,frames=paths,
            joint_pos=state.tolist(),prompt=prompt))
        self.tick+=1
        super().observe(frames,state,prompt,timestamp)

    def act(self,commanded):
        before=self.physics_tick()
        chunk=super().act(commanded)
        assert self.physics_tick()==before, "Physics advanced during API wait"
        append_json(self.replies,dict(observation_tick=self.tick-1,physics_tick=before,
            commanded=None if commanded is None else commanded.tolist(),**chunk.to_json()))
        print('DREAM REPLY',self.tick-1,len(chunk.rows),'actions',chunk.rtt_s,'seconds',flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class ConeEmbodiment(YamArenaEmbodiment):
    def __init__(self,out):
        super().__init__(device=CONFIG['simulation']['device'],cam_width=640,cam_height=360,
                         control_hz=CONTROL_HZ,scene='station',object_jitter=0)
        self.info=replace(self.info,name='place_traffic_cones',supported_target_kinds=frozenset(),
            environment_id='yam-place-traffic-cones-v1',docs='\n\n'.join((JOINT_GUIDE,ROOM_DOCS['station'],INSTRUCTION)))
        self.out=out
        self.runtime=None
        self.last_info={}

    def reset(self,scene,*,seed=None):
        from runtime import Runtime
        assert self.runtime is None, 'One episode per process'
        write_json(self.out/'reset.json',dict(framework_seed=seed,layout_seed=CONFIG['episode']['seed']))
        self.runtime=Runtime(cameras=True,log_dir=self.out/'logs')
        self._env,self._base=self.runtime.env,self.runtime.base
        self._torch=self.runtime.torch
        self._ids={name:(self._base.scene[name].find_joints(ARM_JOINTS,preserve_order=True)[0],
                         self._base.scene[name].find_joints(FINGER_JOINTS,preserve_order=True)[0]) for name in ARMS}
        self._instruction=scene.instruction
        self._t=0.
        self.last_info=self.runtime.measure()
        for _ in range(4):
            obs=self.capture()
        assert self.last_info['score']==0.
        return obs

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        count=self._base.sim._physics_step_count
        self._base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self._base.scene[name].update(1/CONTROL_HZ,force_recompute=True)
        assert count==self._base.sim._physics_step_count
        obs=self._observation()
        assert set(obs.images)==set(CAMERAS) and set(obs.state)=={'joint_pos'} and not obs.extra
        return obs

    def step(self,action):
        requested=np.asarray(action.data,dtype=float)
        assert requested.shape==(14,) and np.isfinite(requested).all()
        self.last_info=self.runtime.step(np.clip(requested,self._low,self._high))
        self._t=self.runtime.steps/CONTROL_HZ
        if self.runtime.steps%100==0:
            self.runtime.save(self.out)
            print('PROGRESS',self._t,'score',self.last_info['score'],flush=True)
        # Score only the final state; never terminate on a fleeting intermediate success.
        return StepResult(observation=self.capture(),truncated=self.runtime.steps>=MAX_STEPS,info=self.last_info)

    def close(self):
        if self.runtime is not None:
            self.runtime.save(self.out)
            self.runtime.close()


class FinalScore:
    name='traffic_cones_final_score'

    def __call__(self,record,target):
        from inspect_robots.scorer import Score
        info=record.steps[-1].result.info
        return Score(value=info['score'],explanation='All six cones must meet every final-state condition.',
                     metadata=info)


def check(embodiment):
    from PIL import Image
    scene=Scene(id='place-traffic-cones',instruction=INSTRUCTION,init_seed=0)
    obs=embodiment.reset(scene,seed=0)
    for name,pixels in obs.images.items():
        assert pixels.shape==(360,640,3) and pixels.std()>10
        Image.fromarray(pixels).save(embodiment.out/f'{name}.png')
    q=obs.state['joint_pos'].copy()
    command=q.copy(); command[0]+=.12
    for _ in range(CONTROL_HZ):
        result=embodiment.step(Action(data=command))
    difference=float(np.abs(result.observation.images['left_cam'].astype(float)-obs.images['left_cam']).mean())
    assert abs(result.observation.state['joint_pos'][0]-q[0])>.05 and difference>1
    assert embodiment.runtime.ticks==240 and result.observation.state_time==1.
    assert result.info['score']==0.
    # Explicit test fixture only: never executed during Dream or the open-loop demo.
    from config import LENGTH, curve, BOARD_SIZE
    tensor=lambda values:embodiment._torch.tensor(np.asarray(values)[None],dtype=embodiment._torch.float32)
    support=np.zeros(6)
    for cone,xy in zip(embodiment.runtime.cones,curve(np.linspace(0,LENGTH,6)),strict=True):
        cone.write_root_pose_to_sim(tensor([*xy,embodiment.runtime.table_top+BOARD_SIZE[2]+.001,0,0,0,1]))
        cone.write_root_velocity_to_sim(tensor(np.zeros(6)))
        cone.write_joint_state_to_sim(tensor(np.zeros(4)),tensor(np.zeros(4)))
    for _ in range(CONTROL_HZ):
        result=embodiment.step(Action(data=command))
        support=np.maximum(support,embodiment.runtime.support_force)
    assert result.info['score']==1. and (support>.05).all(), result.info
    write_json(embodiment.out/'check.json',dict(passed=True,api_calls=0,physics_ticks=480,
        positive_fixture_score=result.info['score'],support_force_N=support.tolist(),
        moving_camera_rgb_mae=difference,input_state_keys=['joint_pos'],camera_size=[640,360],
        model_input_size=CONFIG['observation']['model_input_size_wh']))


def main():
    assert sys.argv[1:] in ([],['--check']), 'Usage: python dream_eval.py [--check]'
    check_only=bool(sys.argv[1:])
    if not check_only:
        from pathlib import Path
        assert (Path.home()/'.dreamlabs/credentials').is_file(), 'Configure the Dream client credentials'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=HERE/'runs'/f'dream-{ "check" if check_only else "eval"}-{stamp}'
    out.mkdir(parents=True)
    print('RUN_DIRECTORY',out,flush=True)
    from archive import snapshot
    snapshot(out)
    write_json(out/'run.json',dict(config=CONFIG,instruction=INSTRUCTION,check_only=check_only))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8').app
    embodiment=ConeEmbodiment(out)
    policy=None
    exit_code=0
    started=time.perf_counter()
    try:
        if check_only:
            check(embodiment)
        else:
            task=Task(name='place-traffic-cones',scenes=[Scene(id='place-traffic-cones',
                instruction=INSTRUCTION,init_seed=0)],scorer=FinalScore(),max_steps=MAX_STEPS)
            policy=DreamPolicy(model=CONFIG['model'],session=SavedDreamSession(out,lambda:embodiment._base.sim._physics_step_count))
            logs=evaluate(task,policy=policy,embodiment=embodiment,log_dir=str(out/'eval'),
                seed=0,fail_on_error=True,store_frames=True,store_actions=True)
            assert len(logs)==1 and logs[0].status=='success', [log.error for log in logs]
    except BaseException:
        exit_code=1
        traceback.print_exc()
    finally:
        write_json(out/'metrics.json',dict(exit_code=exit_code,wall_time_s=time.perf_counter()-started,
            **embodiment.last_info))
        if policy is not None:
            write_json(out/'transcript.json',policy.transcript())
            policy.close()
        embodiment.close()
        app.close(exit_code=exit_code)


if __name__=='__main__':
    main()
