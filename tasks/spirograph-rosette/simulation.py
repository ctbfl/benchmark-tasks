"""Native YAM actions, rigid-body contact evidence and standard observations."""

from dataclasses import replace
import json

import numpy as np
import torch

from inspect_robots import StepResult
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from scene import ASSET_ROOT, CAMERAS, CONFIG, DT, EPISODE, HZ, SIM, SUBSTEPS, make_env
from scoring import RosetteScore


class DrawingStation(YamArenaEmbodiment):
    def __init__(self, out, cameras):
        super().__init__(device=SIM["device"], cam_width=640, cam_height=360, control_hz=HZ,
                         scene=CONFIG["station"]["scene"], top_cam=CONFIG["station"]["top_camera"],
                         object_jitter=EPISODE["object_jitter_m"])
        self.info = replace(self.info, name="yam_spirograph",
            supported_target_kinds=frozenset(), environment_id="yam-spirograph",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"])))
        self.out, self.cameras = out, cameras
        self.trace = (out / "physics.f32").open("wb")
        self.actions = (out / "executed-actions.jsonl").open("w", buffering=1)
        self.measurements = (out / "measurements.jsonl").open("w", buffering=1)
        self.scorer = RosetteScore()
        self.control_steps = 0
        self.ink = None

    def reset(self, scene, *, seed=None):
        import carb
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id

        assert self._env is None, "Use a new trial directory for each reset."
        self._instruction = scene.instruction
        self._env = make_env(self.cameras, self.out)
        self._base = self._env.unwrapped
        self._env.reset(seed=seed)
        self.robots = [self._base.scene[name] for name in ARMS]
        self._ids = {name: (asset.find_joints(ARM_JOINTS, preserve_order=True)[0],
                            asset.find_joints(FINGER_JOINTS, preserve_order=True)[0])
                     for name, asset in zip(ARMS, self.robots)}
        assert abs(self._base.physics_dt-DT) < 1e-10 and self._base.cfg.decimation == SUBSTEPS
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        self.objects = [self.physics.create_rigid_body_view(f"{ASSET_ROOT}/{name}") for name in ("wheel", "pen")]
        pairs = (("pen", "board"), ("wheel", "ring"), ("pen", "wheel"))
        self.contacts = [self.physics.create_rigid_contact_view(f"{ASSET_ROOT}/{a}", [f"{ASSET_ROOT}/{b}"])
                         for a, b in pairs]
        self.fixtures = [self.physics.create_rigid_body_view(f"{ASSET_ROOT}/{name}") for name in ("board", "ring")]
        print("Fixture transforms:", [view.get_transforms().tolist() for view in self.fixtures], flush=True)
        if self.cameras:
            from media import Ink
            self.ink = Ink(self._base.sim.stage)
        ready = np.array([*READY, 1., *READY, 1.])
        self.advance(ready, record=False, substeps=SUBSTEPS*10)
        initial = self.sample()
        self.trace.write(initial.tobytes())
        self.trace.flush()
        print("Settled wheel, pen and forces:", initial[16:].tolist(), flush=True)
        if self.cameras:
            for _ in range(4):
                observation = self.capture()
            return observation
        return None

    def poses(self):
        return [view.get_transforms()[0].copy() for view in self.objects]

    def sample(self):
        joints = [r.data.joint_pos.torch[0].cpu().numpy() for r in self.robots]
        forces = [np.linalg.norm(view.get_contact_force_matrix(DT)[0, 0]) for view in self.contacts]
        row = np.concatenate((*joints, *self.poses(), forces)).astype("<f4")
        assert np.isfinite(row).all(), "Non-finite physics state"
        return row

    def advance(self, command, record=True, substeps=SUBSTEPS):
        command = np.asarray(command, dtype=np.float32)
        assert command.shape == (14,) and np.isfinite(command).all()
        target = np.clip(command, self._low, self._high)
        if record:
            assert self.control_steps < EPISODE["max_control_steps"]
            self.actions.write(json.dumps(dict(step=self.control_steps, requested=command.tolist(),
                                              applied=target.tolist())) + "\n")
        base = self._base
        base.action_manager.process_action(torch.as_tensor(target[None], dtype=torch.float32))
        for _ in range(substeps):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            if record:
                raw = self.sample()
                self.trace.write(raw.tobytes())
                self.scorer.update(raw[16:23], raw[23:30], raw[30:33])
        if not record:
            return
        self.control_steps += 1
        self._t = self.control_steps / HZ
        self.last_info = self.scorer.result()
        self.measurements.write(json.dumps(self.last_info) + "\n")
        if self.control_steps % HZ == 0:
            self.trace.flush()

    def capture(self):
        from media import render_now

        self.ink.update(self.scorer.stroke)
        render_now(self._base, CAMERAS, 1/HZ)
        return self._observation()

    def step(self, action):
        self.advance(action.data)
        return StepResult(observation=self.capture(), info=self.last_info,
            truncated=self.control_steps >= EPISODE["max_control_steps"])

    def close(self):
        self.trace.close()
        self.actions.close()
        self.measurements.close()
        if self._env is not None:
            self._env.close()
