"""Run an Isaac Sim trial with scripted actions or Dream-1 through Inspect Robots."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from isaaclab.app import AppLauncher

MODE = sys.argv[1]
assert MODE in {"probe", "demo", "dream"}, "Choose probe, demo or dream"
app = AppLauncher(headless=True, enable_cameras=MODE != "probe").app

import carb
import omni.physics.tensors as tensors
import torch
from inspect_robots import Scene, StepResult
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
from isaaclab.sim.utils import get_current_stage_id
from yam_arena import station
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, YamArenaEmbodiment
from yam_arena.ik import tcp_pose
from yam_arena.urdf import quat_xyzw_to_matrix

from cord import Cord, NODES
from geometry import ANCHOR, BOARD_SIZE, FLAT, HANDLE_NAMES, RADIUS, ROLL_NAMES, WIDTH
from scene import CONFIG, HERE, INSTRUCTION, make_env
from scoring import score, section_scale

SETTINGS = CONFIG["simulation"]
HZ = SETTINGS["control_hz"]
PHYSICS = SETTINGS["physics_hz"]
DT = 1/PHYSICS
LIMIT = CONFIG["episode"]["max_control_steps"]
TASK = "cinnamon-roll-floss"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


class Trial(YamArenaEmbodiment):
    def __init__(self, out, mode):
        w, h = CONFIG["observation"]["render_size_wh"]
        super().__init__(device=SETTINGS["device"], cam_width=w, cam_height=h, control_hz=HZ,
                         scene=CONFIG["station"]["scene"], top_cam=CONFIG["station"]["top_camera"],
                         object_jitter=CONFIG["episode"]["object_jitter_m"])
        self.info = replace(self.info, name=TASK, supported_target_kinds=frozenset(),
                            environment_id=TASK, docs=JOINT_GUIDE)
        self.out, self.mode = out, mode
        self.actions = (out/"actions.jsonl").open("x", buffering=1)
        self.log = (out/"scorer.jsonl").open("x", buffering=1)
        self.trajectory = (out/"physics-f32.bin").open("xb")
        self.table = station.TABLE_TOP
        self.board = self.table+BOARD_SIZE[2]
        self.tick = self.held = 0
        self._env = make_env(out, cameras=mode != "probe")
        self._base = self._env.unwrapped
        self._torch = torch
        self._ids = {name: (self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                            self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0])
                     for name in ARMS}
        self.recorder = None

    def reset(self, scene, *, seed=None):
        self._instruction = scene.instruction
        self._env.reset(seed=seed)
        base = self._base
        assert abs(base.physics_dt-DT) < 1e-10
        assert base.cfg.decimation == PHYSICS//HZ
        stage = base.sim.stage
        root = "/World/envs/env_0/"
        names = HANDLE_NAMES+ROLL_NAMES
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        self.bodies = [self.physics.create_rigid_body_view(root+n) for n in names]
        assert all(v.count == 1 for v in self.bodies)
        self.indices = np.array([0], dtype=np.int32)
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        shapes = [str(prim.GetPath()) for side in ("Left", "Right")
                  for prim in Usd.PrimRange(stage.GetPrimAtPath(root+side+"Arm"))
                  if prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert shapes, "Missing robot collision shapes"
        self.contacts = self.physics.create_rigid_contact_view(
            [root+n for n in ROLL_NAMES], filter_patterns=[shapes, shapes], max_contact_data_count=192)
        self.poses = np.concatenate([v.get_transforms() for v in self.bodies]).astype(float)
        self.cord = Cord(self.ends())
        curve = UsdGeom.BasisCurves.Define(stage, root+"Floss")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([NODES])
        curve.CreateWidthsAttr([.00235])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([Gf.Vec3f(.96, .96, .79)])
        self.curve = curve.GetPointsAttr()
        self.scales = [UsdGeom.Xformable(stage.GetPrimAtPath(root+n+"/Surface")).AddScaleOp()
                       for n in ROLL_NAMES]
        ready = torch.tensor([[*READY, 1., *READY, 1.]], dtype=torch.float32)
        base.action_manager.process_action(ready)
        for _ in range(PHYSICS//2):
            self.substep()
        self.tick, self._t, self.held = 0, 0., 0
        self.last_info = self.measure()
        assert self.last_info["score"] == 0.
        self.save()
        if self.mode == "probe":
            return None
        from video import Recorder
        self.recorder = Recorder(self)
        self.visualize()
        for _ in range(5):
            self.recorder.refresh(force=True)
        self.recorder.screenshot()
        self.recorder.capture(0)
        return self.capture()

    def ends(self):
        return self.poses[:2, :3] + np.array([
            quat_xyzw_to_matrix(pose[3:]) @ ANCHOR for pose in self.poses[:2]])

    def substep(self):
        base = self._base
        ends = self.ends()
        velocities = np.concatenate([body.get_velocities() for body in self.bodies])
        left, right = self.poses[2:, :3]
        arm_force = self.contacts.get_contact_force_matrix(DT)
        contact = float(np.linalg.norm(arm_force, axis=-1).sum())
        pull, reaction = self.cord.step(ends, self.poses[2:, :3], self.board, self.table, DT, contact)
        bond = np.zeros(3)
        torques = np.zeros((4, 3))
        if not self.cord.released:
            bond = 930.*(right-left-[WIDTH, 0., 0.])+7.*(velocities[3,:3]-velocities[2,:3])
            a, b = self.poses[2:, 3:]
            rot = np.cross(a[:3], b[:3])+a[3]*b[:3]-b[3]*a[:3]
            twist = 1.1*rot*np.sign(np.dot(a, b))+.005*(velocities[3,3:]-velocities[2,3:])
            torques[2:] = (twist, -twist)
        torques[:2] = np.cross(ends-self.poses[:2, :3], pull)
        # The silicone mat damps slip while the tacky dough is within its contact patch.
        height = self.poses[2:, 2] - self.board - RADIUS + FLAT
        supported = np.clip(1.-np.maximum(height, 0.)/.006, 0., 1.)
        reaction[:, :2] -= supported[:, None]*np.clip(13.*velocities[2:, :2], -.85, .85)
        reaction[:, 2] -= supported*np.clip(height*580., 0., .87)
        forces = np.vstack((pull, reaction+[bond, -bond]))
        for body, f, tau in zip(self.bodies, forces, torques):
            body.apply_forces_and_torques_at_position(f[None].astype(np.float32),
                tau[None].astype(np.float32), None, self.indices, True)
        base.action_manager.apply_action()
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(DT)
        self.poses = np.concatenate([v.get_transforms() for v in self.bodies]).astype(float)
        self.speeds = np.array([np.linalg.norm(v.get_velocities()[0, :3]) for v in self.bodies])
        self.spin = np.array([np.linalg.norm(v.get_velocities()[0, 3:]) for v in self.bodies])
        self.last_contact = contact
        self.tick += 1

    def measure(self):
        tips, openings = [], []
        for name, location in zip(ARMS, station.BASES.values()):
            joint_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0].cpu().numpy()
            tips.append(tcp_pose(q[joint_ids], location)[0])
            openings.append(float(np.mean(q[finger_ids])/FINGER_OPEN))
        return score(self.cord, self.poses, self.speeds, self.spin, self.board, tips, openings)


    def save(self):
        row = np.r_[self.tick, self.poses.ravel(), self.cord.points.ravel(),
                    self.cord.height, self.cord.damage, self.last_contact]
        assert np.isfinite(row).all(), "Non-finite physics trajectory"
        self.trajectory.write(row.astype("<f4").tobytes())

    def visualize(self):
        self.curve.Set(Vt.Vec3fArray.FromNumpy(self.cord.points.astype(np.float32)))
        height, width = section_scale(self.cord.damage)
        for op in self.scales:
            op.Set(Gf.Vec3f(1., width, height))

    def capture(self):
        self.visualize()
        self.recorder.refresh()
        return self._observation()

    def summary(self):
        return dict(t=round(self.tick*DT, 3),
                    handles=self.poses[:2, :3].round(4).tolist(),
                    rolls=self.poses[2:, :3].round(4).tolist(),
                    **self.cord.last)

    def step(self, action):
        if self.tick >= LIMIT*PHYSICS//HZ:
            raise RuntimeError("Trial exceeded its action limit")
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        self.actions.write(json.dumps(dict(tick=self.tick, requested=requested.tolist(),
                                           applied=target.tolist()))+"\n")
        self._base.action_manager.process_action(torch.tensor(target[None], dtype=torch.float32))
        for _ in range(PHYSICS//HZ):
            self.substep()
            self.save()
            if self.recorder and self.tick%(PHYSICS//30) == 0:
                self.visualize()
                self.recorder.capture(self.tick)
        self._t = self.tick*DT
        self.last_info = self.measure()
        self.held = self.held+1 if self.last_info["score"] == 1. else 0
        if self.held < HZ//2:
            self.last_info["score"] = min(self.last_info["score"], .75)
        self.log.write(json.dumps(self.summary() | self.last_info)+"\n")
        return StepResult(observation=self.capture() if self.mode == "dream" else None,
                          terminated=self.held >= HZ//2, truncated=self.tick >= LIMIT*PHYSICS//HZ,
                          info=self.last_info)

    def close(self):
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        self.actions.close()
        self.log.close()
        self.trajectory.close()
        super().close()


def main():
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    hash_files = sorted([*HERE.glob("*.py"), HERE/"dream-defaults.json"])
    (out/"source").mkdir()
    for path in hash_files:
        shutil.copy2(path, out/"source"/path.name)
    run_id = os.environ["PRODUCTION_GPU_RUN_ID"]
    write_json(out/"run.json", dict(mode=MODE, gpu_run_id=run_id, instruction=INSTRUCTION, config=CONFIG,
        utc=datetime.now(timezone.utc).isoformat(),
        physics_record=dict(dtype="<f4", tick=[0,1], rigid_poses=[1,29], cord_points=[29,29+3*NODES],
                            last_columns=["cut_front_height_m", "damage", "robot_dough_contact_N"],
                            pose_order=[*HANDLE_NAMES, *ROLL_NAMES], quaternion_order="xyzw"),
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in hash_files}))
    sim = Trial(out, MODE)
    try:
        scene = Scene(id=TASK, instruction=INSTRUCTION, init_seed=CONFIG["episode"]["seed"])
        if MODE == "dream":
            from evaluate import dream_trial
            dream_trial(sim, scene)
        else:
            from demo import script
            sim.reset(scene, seed=CONFIG["episode"]["seed"])
            script(sim)
        print(json.dumps(sim.last_info), flush=True)
    finally:
        sim.close()
    if MODE != "probe":
        write_json(out/"result.json", dict(kind=MODE, status="success", score=sim.last_info["score"],
            max_score=1, gpu_run_id=run_id, scorer_log=str(out/"scorer.jsonl"),
            actions=str(out/"actions.jsonl"), physics=str(out/"physics-f32.bin"),
            video=str(out/"videos"/"original.mp4"), measurements=sim.last_info, final_tick=sim.tick))


try:
    main()
except BaseException:
    traceback.print_exc()
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=1)
    raise
else:
    app.close()
