"""One archived Dream-1 trial in the same physical environment as the open-loop demo."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import sys
import traceback

import numpy as np
from inspect_robots import Action, Scene, Task, StepResult, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from circuit import CONFIG, HERE, HZ, MAX_STEPS, INSTRUCTION
from runtime import CAMERAS, Experiment, write_json


def append_json(file, data):
    file.write(json.dumps(data, allow_nan=False)+"\n")
    file.flush()
    os.fsync(file.fileno())


class SavedSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out, self.tick = out/"dream", 0
        self.out.mkdir()
        for name in CAMERAS:
            (self.out/name).mkdir()
        self.observations = (self.out/"observations.jsonl").open("x")
        self.replies = (self.out/"replies.jsonl").open("x")

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames)==set(CAMERAS) and prompt==INSTRUCTION
        paths = {}
        for name,jpeg in frames.items():
            path = self.out/name/f"{self.tick:06d}.jpg"
            path.write_bytes(base64.b64decode(jpeg))
            paths[name] = str(path.relative_to(self.out))
        append_json(self.observations,dict(tick=self.tick,time_s=timestamp,frames=paths,
                                         joint_pos=state.tolist(),prompt=prompt))
        self.tick += 1
        super().observe(frames,state,prompt,timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies,dict(observation_tick=self.tick-1,
            commanded=None if commanded is None else commanded.tolist(),**chunk.to_json()))
        print("DREAM_REPLY", self.tick-1, len(chunk.rows), chunk.rtt_s, flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class RheostatEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device=CONFIG["simulation"]["device"],cam_width=640,cam_height=360,
                         control_hz=HZ,scene="station",object_jitter=0)
        self.info = replace(self.info,name="yam_rheostat", supported_target_kinds=frozenset(),
            environment_id="yam-rheostat-v1",docs="\n\n".join((JOINT_GUIDE,ROOM_DOCS["station"],INSTRUCTION)))
        self.out, self.experiment = out, None

    def reset(self, scene, *, seed=None):
        assert self.experiment is None, "One episode per archive"
        self.experiment = Experiment(self.out)
        self._env, self._base = self.experiment.env,self.experiment.base
        self._torch, self._instruction, self._t = self.experiment.torch,scene.instruction,0.
        self._ids = {name:(self._base.scene[name].find_joints(ARM_JOINTS,preserve_order=True)[0],
                          self._base.scene[name].find_joints(FINGER_JOINTS,preserve_order=True)[0]) for name in ARMS}
        return self._observation()

    def step(self, action):
        self.experiment.step(action.data,capture=True)
        self._t = self.experiment.steps/HZ
        info = self.experiment.score.result()
        if self.experiment.steps % (5*HZ)==0:
            self.experiment.save()
            print("PROGRESS",info,flush=True)
        return StepResult(observation=self._observation(),terminated=info["success"],
            termination_reason="success" if info["success"] else None,
            truncated=self.experiment.steps>=MAX_STEPS and not info["success"],info=info)

    def close(self):
        if self.experiment is not None:
            self.experiment.save()
            self.experiment.env.close()


class FinalScore:
    name = "safe_bulb_voltage"

    def __call__(self, record, target):
        from inspect_robots.scorer import Score
        info = record.steps[-1].result.info
        return Score(value=info["score"],explanation="Final voltage 4.8–5.2 V for 1 s; never above 7 V.",metadata=info)


def main():
    assert sys.argv[1:] in ([],["--check"]), "Usage: python dream_eval.py [--check]"
    check = bool(sys.argv[1:])
    out = HERE/"runs"/(("dream-check-" if check else "dream-")+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    (out/"logs").mkdir(parents=True)
    print("RUN_DIRECTORY="+str(out),flush=True)
    write_json(out/"config.json",CONFIG)
    write_json(out/"run.json",dict(instruction=INSTRUCTION,model=CONFIG["model"],seed=0,
        camera_size=[640,360],model_input_size=[320,176],api_wait_advances_physics=False))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True,kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
        f"--/log/file={out}/logs/kit.log").app
    embodiment, policy, code = RheostatEmbodiment(out),None,0
    try:
        scene = Scene(id="rheostat-experiment",instruction=INSTRUCTION,init_seed=0)
        if check:
            obs = embodiment.reset(scene,seed=0)
            assert set(obs.images)==set(CAMERAS) and set(obs.state)=={"joint_pos"} and not obs.extra
            assert obs.state["joint_pos"].shape==(14,)
            result = embodiment.step(Action(data=obs.state["joint_pos"]))
            assert result.info["score"]==0 and result.observation.state_time==.05
            assert all(a.shape==(360,640,3) for a in result.observation.images.values())
            write_json(out/"check.json",dict(passed=True,api_calls=0))
        else:
            policy = DreamPolicy(model=CONFIG["model"],session=SavedSession(out))
            task = Task(name="rheostat-experiment",scenes=[scene],scorer=FinalScore(),max_steps=MAX_STEPS)
            logs = evaluate(task,policy=policy,embodiment=embodiment,log_dir=str(out/"eval"),
                seed=0,fail_on_error=True,store_frames=True,store_actions=True)
            assert len(logs)==1 and logs[0].status=="success"
        print("RESULT",embodiment.experiment.score.result(),flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        if policy is not None:
            write_json(out/"transcript.json",policy.transcript())
            policy.close()
        embodiment.close()
        write_json(out/"completion.json",dict(exit_code=code))
        app.close(exit_code=code)


if __name__=="__main__":
    main()
