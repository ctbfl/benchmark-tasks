"""Native actions and a threshold-activated approximation of internal rotation."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from yam_arena.actions import bimanual_joint_pos_14
from yam_arena.constants import READY
from yam_arena.station import TABLE_TOP

from assets import BOARD_Z, INERTIA, SHORT
from scene import CONFIG, CAMERAS, make_env

PHYSICS_HZ = CONFIG["simulation"]["physics_hz"]
CONTROL_HZ = CONFIG["simulation"]["control_hz"]
DT = 1 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
RAW_FRACTION = .20
SPIN_UP_SPEED = 4 * np.pi
SLIP_RELAXATION_TIME = .03
ARM_NAMES = ("LeftArm", "RightArm")


def array(value):
    return value.torch[0].cpu().numpy().copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def viscous_torque(inner_speed, shell_speed, liquid_inertia):
    slip = inner_speed - shell_speed
    return RAW_FRACTION * liquid_inertia * slip / SLIP_RELAXATION_TIME


class Simulation:
    def __init__(self, out, *, seed, pair=False, cameras=True, screenshot=False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env, self.raw, links = make_env(seed, pair=pair, cameras=cameras, screenshot=screenshot,
                                             log_dir=self.out / "sim")
        self.base = self.env.unwrapped
        self.robots = [self.base.scene[f"{side}_robot"] for side in ("left", "right")]
        self.eggs = [self.base.scene[f"egg_{i}"] for i in range(len(self.raw))]
        self.link_paths = links
        self.tip_geometry = self._tip_geometry() if pair else []
        self.frame_stride = PHYSICS_HZ//30
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.frames = []
        self.reset(seed, self.raw)

    def _tip_geometry(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.base.sim.stage
        cache = UsdGeom.XformCache()
        geometry = []
        for side, robot in zip(ARM_NAMES, self.robots, strict=True):
            tips = []
            for name in ("tip_left", "tip_right"):
                link = stage.GetPrimAtPath(f"/World/envs/env_0/{side}" + self.link_paths[name])
                parts = []
                for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    mesh = UsdGeom.Mesh(prim)
                    if mesh:
                        points = np.asarray(mesh.GetPointsAttr().Get())
                        matrix, _ = cache.ComputeRelativeTransform(prim, link)
                        matrix = np.asarray(matrix)
                        parts.append(points @ matrix[:3, :3] + matrix[3, :3])
                if not parts:
                    raise RuntimeError(f"No collision mesh under {link.GetPath()}")
                tips.append((robot.body_names.index(name), np.concatenate(parts)))
            geometry.append(tips)
        return geometry

    def _tool_trace(self, side):
        robot = self.robots[side]
        poses = array(robot.data.body_link_pose_w)
        bounds = []
        for index, points in self.tip_geometry[side]:
            position, quat = poses[index, :3], poses[index, 3:]
            world = Rotation.from_quat(quat).apply(points) + position
            bounds.append([world.min(axis=0).tolist(), world.max(axis=0).tolist()])
        gripper = poses[robot.body_names.index("gripper")]
        tcp = gripper[:3] + Rotation.from_quat(gripper[3:]).apply([0., 0., .134])
        return {"tcp_xyz_m": tcp.tolist(), "tip_bounds_world_m": bounds,
                "board_z_m": TABLE_TOP + BOARD_Z, "opening_command": float(self.action[7*side+6])}

    def reset(self, seed, raw):
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils.stage import get_current_stage_id

        self.env.reset(seed=seed)
        classes = np.array(raw, dtype=bool)
        self.raw = np.zeros_like(classes)
        self.omega_inner = np.zeros(len(self.eggs))
        self.rotor_active = np.zeros(len(self.eggs), dtype=bool)
        self.liquid_inertia = np.zeros(len(self.eggs))
        self.torques = np.zeros(len(self.eggs))
        self.rows, self.actions, self.frames = [], [], []
        self.phase = "initial"
        self.ticks = 0
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.overloaded = np.zeros(len(self.eggs), dtype=bool)
        self.views = []
        # Contact handles depend on this simulation view for their lifetime.
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        root = "/World/envs/env_0"
        filters = [f"{root}/{arm}{link}" for arm in ARM_NAMES
                   for name, link in self.link_paths.items() if name != "base"]
        self.jaw_filter_paths = filters
        self.hand_pads = [[i for i, path in enumerate(filters) if f"/{side}/" in path and "/tip_" in path]
                          for side in ARM_NAMES]
        self.pad_body_ids = [[robot.body_names.index(filters[i].rsplit("/", 1)[-1]) for i in pads]
                             for robot, pads in zip(self.robots, self.hand_pads, strict=True)]
        self.previous_pad_poses = None
        filters = [*filters, f"{root}/board/GlassTop"]
        bodies = []
        for egg in self.eggs:
            prim = egg.root_view.prim_paths[0]
            body = self.physics.create_rigid_body_view(prim)
            body.set_inertias(np.diag(INERTIA).reshape(1, 9).astype(np.float32), np.array([0], np.int32))
            bodies.append(body)
            self.views.append(self.physics.create_rigid_contact_view(prim, filters, max_contact_data_count=128))
        self.bodies = bodies
        for _ in range(12):
            self.step(self.action, record=False)
        # Place both classes at the same settled pose before exposing the first observation.
        for egg in self.eggs:
            egg.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=self.base.device))
        self.raw = classes
        self.sample()

    def pose(self, i):
        return array(self.eggs[i].data.root_pose_w)

    def velocity(self, i):
        return array(self.eggs[i].data.root_vel_w)

    def shell_angle(self, i):
        axis = Rotation.from_quat(self.pose(i)[3:]).apply([1, 0, 0])
        return float(np.arctan2(axis[1], axis[0]))

    def joints(self):
        return bimanual_joint_pos_14(self.base)[0].cpu().numpy().copy()

    def contact(self, i, center):
        view = self.views[i]
        impulses, points, normals, _, counts, starts = view.get_contact_data(1.)
        friction, fpoints, fcounts, fstarts = view.get_friction_data(1.)
        wrench = np.zeros((counts.shape[1], 4))
        pad_points = {}
        for j in np.flatnonzero(counts[0]):
            sl = slice(starts[0, j], starts[0, j] + counts[0, j])
            imp = impulses[sl].reshape(-1, 1) * normals[sl]
            wrench[j, 0] = np.abs(impulses[sl]).sum()
            wrench[j, 1:] = np.cross(points[sl]-center, imp).sum(axis=0)
            if wrench[j, 0] and any(j in pads for pads in self.hand_pads):
                pad_points[j] = np.average(points[sl], weights=np.abs(impulses[sl]).ravel(), axis=0)
        for j in np.flatnonzero(fcounts[0]):
            sl = slice(fstarts[0, j], fstarts[0, j] + fcounts[0, j])
            wrench[j, 1:] += np.cross(fpoints[sl]-center, friction[sl]).sum(axis=0)
        total = np.linalg.norm(view.get_net_contact_forces(1.)[0])
        return wrench, total, pad_points

    def pad_sweep(self, center, pad_points, wrench, poses):
        """Tangential travel of each hand's contact patch in the actual pad frame."""
        if self.previous_pad_poses is None:
            return [0., 0.]
        sweep = []
        for pads, before, after in zip(self.hand_pads, self.previous_pad_poses, poses, strict=True):
            signed_travel = normal = 0.
            for index, old, new in zip(pads, before, after, strict=True):
                if index not in pad_points:
                    continue
                point = pad_points[index]
                local = Rotation.from_quat(new[3:]).inv().apply(point-new[:3])
                delta = point - (old[:3] + Rotation.from_quat(old[3:]).apply(local))
                radius = point[:2]-center[:2]
                tangent = np.array([-radius[1], radius[0]]) / max(np.linalg.norm(radius), 1e-6)
                weight = wrench[index, 0]
                signed_travel += float((delta[:2] @ tangent)*weight)
                normal += weight
            sweep.append(signed_travel/normal if normal else 0.)
        return sweep

    def sample(self):
        eggs = []
        pad_poses = [array(robot.data.body_link_pose_w)[ids]
                     for robot, ids in zip(self.robots, self.pad_body_ids, strict=True)]
        for i in range(len(self.eggs)):
            pose, vel = self.pose(i), self.velocity(i)
            wrench, total, pad_points = self.contact(i, pose[:3])
            normal = wrench[:-1, 0]
            # Diagnostic overload marker; shell fracture is not simulated.
            if normal.max() / DT > 24 or total > .057:
                self.overloaded[i] = True
            eggs.append(dict(pose=pose.tolist(), velocity=vel.tolist(), yaw=self.shell_angle(i),
                normal_impulse_ns=normal.tolist(), contact_angular_impulse_nms=wrench[:-1, 1:].tolist(),
                pad_normal_impulse_ns=[float(wrench[pads, 0].sum()) for pads in self.hand_pads],
                pad_yaw_impulse_nms=[float(wrench[pads, 3].sum()) for pads in self.hand_pads],
                pad_sweep_m=self.pad_sweep(pose[:3], pad_points, wrench, pad_poses),
                board_normal_impulse_ns=float(wrench[-1, 0]), board_angular_impulse_nms=wrench[-1, 1:].tolist(),
                net_normal_impulse_ns=float(total), inner_rad_s=float(self.omega_inner[i]),
                rotor_active=bool(self.rotor_active[i]), liquid_inertia_kgm2=float(self.liquid_inertia[i]),
                coupling_nm=float(self.torques[i]), overloaded=bool(self.overloaded[i]),
                shell_broken=False))  # The rigid shell model has no fracture dynamics.
        row = dict(tick=self.ticks, phase=self.phase, eggs=eggs)
        if self.tip_geometry:
            row["tools"] = [self._tool_trace(side) for side in range(len(self.robots))]
        self.rows.append(row)
        self.previous_pad_poses = pad_poses

    def step(self, action, *, record=True, frame_hook=None):
        if record:
            assert len(self.actions) < CONFIG["episode"]["max_control_steps"]
            self.actions.append(np.asarray(action).tolist())
        self.action = np.asarray(action, dtype=np.float32)
        self.base.action_manager.process_action(torch.tensor(self.action[None]))
        for _ in range(SUBSTEPS):
            for i in np.flatnonzero(self.raw):
                egg = self.eggs[i]
                velocity = self.velocity(i)
                shell = velocity[5]
                if not self.rotor_active[i]:
                    if abs(shell) < SPIN_UP_SPEED:
                        continue
                    rotation = Rotation.from_quat(self.pose(i)[3:]).as_matrix()
                    inertia = rotation @ np.diag(INERTIA) @ rotation.T
                    self.liquid_inertia[i] = (1-RAW_FRACTION) * inertia[2, 2]
                    # Transfer existing vertical momentum into the internal mode.
                    self.omega_inner[i] = (1-RAW_FRACTION) * (inertia @ velocity[3:])[2] / self.liquid_inertia[i]
                    self.bodies[i].set_inertias(np.diag(INERTIA*RAW_FRACTION).reshape(1, 9).astype(np.float32),
                                                np.array([0], np.int32))
                    self.rotor_active[i] = True
                torque = viscous_torque(self.omega_inner[i], shell, self.liquid_inertia[i])
                self.omega_inner[i] -= torque * DT / self.liquid_inertia[i]
                self.torques[i] = torque
                egg.instantaneous_wrench_composer.add_forces_and_torques_index(
                    torques=torch.tensor([[[0., 0., torque]]], dtype=torch.float32), is_global=True)
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            if record:
                self.ticks += 1
                self.sample()
                if frame_hook is not None and self.ticks % self.frame_stride == 0:
                    frame_hook()

    def capture(self, names=CAMERAS):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in names:
            camera = self.base.scene[name]
            camera.update(DT, force_recompute=True)
            images[name] = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
        return images

    def close(self):
        self.env.close()
