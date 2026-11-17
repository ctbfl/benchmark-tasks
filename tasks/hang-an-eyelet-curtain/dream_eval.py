"""Dream evaluation through the same environment and action interface as the demo."""
import base64
from dataclasses import replace
import json
import os

import numpy as np
from inspect_robots import Scene, StepResult, Task, eval as evaluate
from inspect_robots.scorer import Score
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from config import CONFIG, HZ, INSTRUCTION, MAX_STEPS
from runtime import CAMERAS, write_json


def append_json(file, value):
    file.write(json.dumps(value, allow_nan=False)+'\n')
    file.flush()
    os.fsync(file.fileno())


class SavedDreamSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out = out/'dream'
        self.out.mkdir()
        self.tick = 0
        self.observations = (self.out/'observations.jsonl').open('x')
        self.replies = (self.out/'replies.jsonl').open('x')
        for name in CAMERAS:
            (self.out/name).mkdir()

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        paths = {}
        for name, jpeg in frames.items():
            path = self.out/name/f'{self.tick:06d}.jpg'
            path.write_bytes(base64.b64decode(jpeg))
            paths[name] = str(path.relative_to(self.out))
        append_json(self.observations, dict(tick=self.tick, time_s=timestamp,
            frames=paths, joint_pos=state.tolist(), prompt=prompt))
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies, dict(observation_tick=self.tick-1,
            commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print(f'Dream reply: {len(chunk.rows)} actions, {chunk.rtt_s:.2f}s round trip', flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class CurtainEmbodiment(YamArenaEmbodiment):
    def __init__(self, experiment):
        width, height = CONFIG['observation']['render_size_wh']
        super().__init__(device=CONFIG['simulation']['device'], cam_width=width,
                         cam_height=height, control_hz=HZ, scene='station', object_jitter=0)
        self.experiment = experiment
        self._base = experiment.base
        self._ids = {name: (self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                           self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0])
                     for name in ARMS}
        self.info = replace(self.info, name='yam_eyelet_curtain',
            supported_target_kinds=frozenset(), environment_id='yam-eyelet-curtain-v1',
            docs='\n\n'.join((JOINT_GUIDE, ROOM_DOCS['station'], INSTRUCTION)))

    def reset(self, scene, *, seed=None):
        assert self.experiment.steps == 0 and scene.init_seed in (None, CONFIG['episode']['seed'])
        self._instruction, self._t = scene.instruction, 0.
        return self.capture()

    def capture(self):
        self.experiment.capture()
        return self._observation()

    def step(self, action):
        self.experiment.step(action.data)
        self._t = self.experiment.steps/HZ
        info = self.experiment.result()
        if self.experiment.steps % (5*HZ) == 0:
            write_json(self.experiment.out/'progress.json', info)
            print(f'Dream {self._t:.1f}s: {info["rings_on_rod"]}/{len(info["ring_on_rod"])} rings, score={info["score"]}', flush=True)
        return StepResult(observation=self.capture(), terminated=False,
                          truncated=self.experiment.steps >= MAX_STEPS, info=info)


class FinalScore:
    name = 'curtain_final_state'

    def __call__(self, record, target):
        info = record.steps[-1].result.info
        return Score(value=info['score'], explanation='0.5 per ring on the rod between its supports; 0 unless the rod rests in both cradles.',
                     metadata={'rings_on_rod': info['rings_on_rod'], 'rod_on_stand': info['rod_on_stand']})


def run(experiment):
    from pathlib import Path
    assert (Path.home()/'.dreamlabs/credentials').is_file(), 'Configure ~/.dreamlabs/credentials'
    out = experiment.out
    write_json(out/'evaluation.json', dict(model=CONFIG['model'], instruction=INSTRUCTION,
        config=CONFIG, model_input_size_wh=CONFIG['observation']['model_input_size_wh']))
    task = Task(name='hang-an-eyelet-curtain',
        scenes=[Scene(id='hang-an-eyelet-curtain', instruction=INSTRUCTION, init_seed=CONFIG['episode']['seed'])],
        scorer=FinalScore(), max_steps=MAX_STEPS)
    embodiment = CurtainEmbodiment(experiment)
    with DreamPolicy(model=CONFIG['model'], session=SavedDreamSession(out)) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out/'eval'),
                seed=CONFIG['episode']['seed'], fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == 'success'
        finally:
            write_json(out/'dream-transcript.json', policy.transcript())
