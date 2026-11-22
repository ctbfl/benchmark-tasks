"""Native PhysX execution and read-only recording of the rug and robot state."""
import json
import shutil
from pathlib import Path
import numpy as np

from cloth import topology, surfaces, update_visual
from scene import CONFIG, PHYSICS_HZ, CONTROL_HZ, EPISODE, RUG, make_env
from scoring import grade

HERE = Path(__file__).resolve().parent


class Experiment:
    def __init__(self, out, cameras=False):
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY, ARM_JOINTS
        from yam_arena.station import TABLE_TOP

        self.out, self.torch, self.floor = out, torch, TABLE_TOP
        out.mkdir(exist_ok=False)
        for path in [*HERE.glob("*.py"), HERE / "dream-defaults.json", HERE / "assets/rug.usda", HERE / "assets/weave.png"]:
            target = out / "source" / path.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        self.env = make_env(cameras=cameras)
        self.base = self.env.unwrapped
        self.env.reset(seed=EPISODE["seed"])
        self.physics = tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
        self.cloth = self.physics.create_surface_deformable_body_view(RUG + "/physics")
        self.uv, self.faces = topology()
        assert self.cloth.count == 1 and self.cloth.max_simulation_nodes_per_body == len(self.uv)
        assert np.array_equal(self.cloth.get_simulation_element_indices()[0].cpu().numpy(), self.faces)
        self.robots = [self.base.scene[s + "_robot"] for s in ("left", "right")]
        self.cameras = cameras
        from visuals import Visuals
        self.visuals = Visuals(self.base.sim.stage, self.robots, cameras)
        self.arm_ids = [r.find_joints(ARM_JOINTS, preserve_order=True)[0] for r in self.robots]
        self.home = np.array([*READY, 0., *READY, .1], dtype=np.float32)
        self.ticks = self.steps = 0
        self.frames, self.robot_frames, self.frame_times = [], [], []
        self.actions, self.metrics, self.requested = [], [], []
        self.observer = None
        for _ in range(CONTROL_HZ * 5):
            self.advance(self.home, record=False)
        self.ticks = 0
        self.save_frame()
        self.report("initial")

    def state(self):
        from yam_arena.constants import FINGER_OPEN
        return np.concatenate([np.r_[q, np.clip(r.data.joint_pos.torch[0, -2:].mean().item() / FINGER_OPEN, 0., 1.)]
            for q, r in zip(self.joints(), self.robots, strict=True)])

    def images(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        assert self.cameras
        self.visuals.update()
        update_visual(self.base.sim.stage, RUG, self.positions(), self.faces)
        tick = self.base.sim._physics_step_count
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in CONFIG["observation"]["camera_names"]:
            camera = self.base.scene[name]
            camera.update(1 / CONTROL_HZ, force_recompute=True)
            pixels = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            assert pixels.shape == (360, 640, 3) and pixels.std() > 10
            images[name] = np.ascontiguousarray(pixels)
        assert self.base.sim._physics_step_count == tick
        return images

    def positions(self):
        p = self.cloth.get_simulation_nodal_positions()[0].cpu().numpy().copy()
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite simulated rug nodes")
        return p

    def joints(self):
        return np.stack([r.data.joint_pos.torch[0, ids].cpu().numpy()
                         for r, ids in zip(self.robots, self.arm_ids, strict=True)])

    def save_frame(self):
        self.frames.append(self.positions())
        self.robot_frames.append(np.stack([r.data.joint_pos.torch[0].cpu().numpy().copy() for r in self.robots]))
        self.frame_times.append(self.ticks / PHYSICS_HZ)

    def advance(self, action, record=True):
        tensor = self.torch.tensor(action[None], dtype=self.torch.float32, device=self.base.device)
        self.base.action_manager.process_action(tensor)
        for _ in range(PHYSICS_HZ // CONTROL_HZ):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(1 / PHYSICS_HZ)
            self.ticks += 1
            if record and self.ticks % (PHYSICS_HZ // 30) == 0:
                self.save_frame()

    def step(self, action, requested):
        from yam_arena.ik import LOWER, UPPER
        if self.steps >= EPISODE["max_control_steps"]:
            raise RuntimeError("Episode limit reached")
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 14-dimensional robot action")
        applied = np.clip(action, np.r_[LOWER, 0., LOWER, 0.], np.r_[UPPER, 1., UPPER, 1.])
        self.actions.append(applied.copy())
        self.requested.append(requested.copy())
        self.advance(applied)
        self.steps += 1
        self.metrics.append(self.measure())

    def measure(self):
        p = self.positions()
        return {"time_s": self.ticks / PHYSICS_HZ, **grade(*surfaces(p, self.faces), self.faces, self.floor),
                "tip_world": p[0].tolist(),
                "finger_joint_position_m": [r.data.joint_pos.torch[0, -2:].cpu().tolist() for r in self.robots]}

    def report(self, label):
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        state = {"phase": label, **self.measure()}
        print(json.dumps(state), flush=True)
        if self.observer is None:
            camera = UsdGeom.Camera.Define(self.base.sim.stage, "/DemoObserver")
            camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(.62, -.70, 1.22), Gf.Vec3d(.32, -.04, .80), Gf.Vec3d(0, 0, 1)).GetInverse())
            camera.CreateFocalLengthAttr(38.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 20.))
            product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
            feed = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            feed.attach([product])
            self.observer = product, feed
        self.visuals.update()
        update_visual(self.base.sim.stage, RUG, self.positions(), self.faces)
        for _ in range(4):
            self.base.sim.render()
            ensure_isaac_rtx_render_update(force=True)
        pixels = self.observer[1].get_data()[..., :3]
        assert pixels.shape == (1080, 1920, 3) and pixels.std() > 10
        frames = self.out / "frames"
        frames.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(frames / (label + ".png"))
        (frames / (label + ".json")).write_text(json.dumps(state, indent=2) + "\n")

    def save(self):
        np.savez_compressed(self.out / "trajectory.npz", cloth=np.asarray(self.frames),
            robots=np.asarray(self.robot_frames), times=np.asarray(self.frame_times), faces=self.faces,
            material_uv=self.uv, actions=np.asarray(self.actions), requested=np.asarray(self.requested))
        (self.out / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        (self.out / "result.json").write_text(json.dumps(self.measure(), indent=2) + "\n")
        (self.out / "schema.json").write_text(json.dumps({"physics_hz": PHYSICS_HZ,
            "control_hz": CONTROL_HZ, "trajectory_fps": 30, "frames": len(self.frames),
            "steps": self.steps, "duration_s": self.ticks / PHYSICS_HZ,
            "robot_joint_names": [r.joint_names for r in self.robots],
            "elapsed_physics_steps": self.ticks,
            "configuration": CONFIG}, indent=2) + "\n")

    def close(self):
        if self.observer is not None:
            product, feed = self.observer
            feed.detach()
            product.destroy()
        self.env.close()
