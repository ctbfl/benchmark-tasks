"""Native joint action stepping with synchronized cord and film measurements."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from film import Film
from geometry import BODIES, HERE, LINKS, ROOT, TIP
from scene import CAMERAS, CONFIG, CONTROL_HZ, LIMIT, PHYSICS_HZ, SEED, make_env
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.ik import LOWER, UPPER

CONTROL_DT = 1 / CONTROL_HZ
DT = 1 / PHYSICS_HZ
LOW = np.r_[LOWER, 0., LOWER, 0.]
HIGH = np.r_[UPPER, 1., UPPER, 1.]
READY_ACTION = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
DECIMATION = PHYSICS_HZ // CONTROL_HZ
VIDEO_STRIDE = PHYSICS_HZ // 30


def run_directory(kind, args):
    if args:
        return Path(args[0]).resolve()
    return Path.cwd() / "runs" / (kind + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_manifest():
    paths = [*HERE.glob("*.py"), HERE / "dream-defaults.json", *HERE.glob("assets/*")]
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def make_body_view():
    import omni.physics.tensors as tensors
    from isaaclab.sim.utils import get_current_stage_id
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    view = physics.create_rigid_body_view([ROOT + "/wand_*", ROOT + "/cord_*"])
    paths = list(view.prim_paths)
    order = [paths.index(ROOT + "/" + name) for name in BODIES]
    assert view.count == len(order)
    return physics, view, order


def strands(poses):
    tips = poses[:2,:3] + Rotation.from_quat(poses[:2,3:]).apply(TIP)
    mids = poses[2:,:3]
    count = len(LINKS[0])
    return [np.vstack((tips[0], links, tips[1])) for links in (mids[:count], mids[count:])]


def update_film(film, poses):
    upper, lower = strands(poses)
    axes = Rotation.from_quat(poses[:2,3:]).as_matrix()[:,:,0]
    wand_points = poses[:2,None,:3] + axes[:,None,:] * np.linspace(-.05,.19,13)[None,:,None]
    film.update(upper, lower, poses[2:,:3], wand_points.reshape(-1,3))


def render_views(base, names=CAMERAS):
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    base.sim.render()
    ensure_isaac_rtx_render_update(force=True)
    images = {}
    for name in names:
        sensor = base.scene[name]
        sensor.update(CONTROL_DT, force_recompute=True)
        images[name] = sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy().copy()
    return images


class Simulation:
    def __init__(self, out, cameras):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "source-manifest.json", source_manifest())
        self.env = make_env(out / "isaac", cameras=cameras)
        self.base = self.env.unwrapped
        self.env.reset(seed=SEED)
        assert abs(self.base.physics_dt-DT) < 1e-10 and abs(self.base.step_dt-CONTROL_DT) < 1e-10
        self.robots = [self.base.scene[f"{side}_robot"] for side in ("left", "right")]
        self.ids = [(r.find_joints(ARM_JOINTS, preserve_order=True)[0],
                     r.find_joints(FINGER_JOINTS, preserve_order=True)[0]) for r in self.robots]
        self.physics, self.view, self.order = make_body_view()
        self.ticks = self.steps = 0
        self.actions, self.states, self.visuals, self.measurements, self.raw_poses = [], [], [], [], []
        self.phase = "settled"
        for _ in range(10):
            self.step_native(READY_ACTION, record=False)
        self.film = Film(DT)
        self.observe_physics()
        if cameras:
            from visual import FilmMesh
            self.mesh = FilmMesh(self.base.sim.stage)
            self.mesh.update(self.poses, self.visuals[-1])

    def step_native(self, target, record):
        import torch
        self.base.action_manager.process_action(torch.from_numpy(target[None]))
        for _ in range(DECIMATION):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            if record:
                self.ticks += 1
                self.observe_physics()

    def measured_action(self):
        parts = []
        for robot, (arm, fingers) in zip(self.robots, self.ids):
            q = robot.data.joint_pos.torch[0].cpu().numpy()
            parts.extend([*q[arm], np.clip(q[fingers].mean()/FINGER_OPEN, 0., 1.)])
        return np.array(parts)

    def observe_physics(self):
        self.poses = self.view.get_transforms()[self.order].copy()
        assert np.isfinite(self.poses).all()
        self.raw_poses.append(self.poses)
        update_film(self.film, self.poses)
        if self.ticks % VIDEO_STRIDE == 0:
            joint = np.stack([r.data.joint_pos.torch[0].cpu().numpy() for r in self.robots])
            self.states.append((joint.copy(), self.poses.copy()))
            self.visuals.append(self.film.visual())

    def step(self, command):
        assert self.steps < LIMIT
        cmd = np.asarray(command, dtype=np.float32)
        assert cmd.shape == (14,) and np.isfinite(cmd).all()
        target = np.clip(cmd, LOW, HIGH).astype(np.float32)
        self.actions.append(target)
        self.step_native(target, record=True)
        self.steps += 1
        self.measurements.append({**self.film.sample, "phase": self.phase, "wet": self.film.wet_until > self.film.time,
                                  "volume_m3": self.film.volume, "score": self.film.score,
                                  "joints": self.measured_action().tolist(), "wand_pose": self.poses[:2].tolist()})

    def capture(self):
        self.mesh.update(self.poses, self.film.visual())
        return render_views(self.base)

    def save(self, kind, error):
        write_json(self.out / "measurements.json", self.measurements)
        np.savez_compressed(self.out / "trajectory.npz", joints=np.stack([q for q, _ in self.states]),
                            objects=np.stack([p for _, p in self.states]), film=np.asarray(self.visuals),
                            actions=np.asarray(self.actions), physics_poses=np.stack(self.raw_poses))
        result = {"kind": kind, "status": "success" if error is None else "failed",
                  "error": error, "max_score": 1,
                  "gpu_run_id": os.environ.get("PRODUCTION_GPU_RUN_ID"),
                  "control_steps": self.steps, "physics_ticks": self.ticks,
                  "simulation_duration_s": self.ticks*DT, "raw_scorer": "measurements.json",
                  "trajectory": "trajectory.npz", "execution_config": CONFIG, **self.film.result()}
        write_json(self.out / "result.json", result)
        print("SCORE", json.dumps(result), flush=True)

    def close(self):
        self.env.close()


def start_app(cameras=True):
    from isaaclab.app import AppLauncher
    return AppLauncher(headless=True, device=CONFIG["simulation"]["device"], enable_cameras=cameras,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
