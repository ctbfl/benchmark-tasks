"""Place the two assets on the existing YAM station and save RTX screenshots."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/yam-scene"
TAPE_CENTER = (.30, .105)
TAPE_HALF_SIZE = .060
TAPE_WIDTH = .008


def add_tape_frame(stage, table_height):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = "/World/envs/env_0/RedPaperTape"
    UsdGeom.Xform.Define(stage, root)
    material = UsdShade.Material.Define(stage, root + "/Material")
    shader = UsdShade.Shader.Define(stage, root + "/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.65, .008, .012))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for axis in range(2):
        for sign in (-1, 1):
            position = [*TAPE_CENTER, table_height + .000075 + axis * .00003]
            position[axis] += sign * (TAPE_HALF_SIZE + TAPE_WIDTH / 2)
            size = [2 * (TAPE_HALF_SIZE + TAPE_WIDTH)] * 2 + [.00015]
            size[axis] = TAPE_WIDTH
            strip = UsdGeom.Cube.Define(stage, f"{root}/strip_{axis}_{'a' if sign < 0 else 'b'}")
            strip.CreateSizeAttr(1)
            strip.AddTranslateOp().Set(Gf.Vec3d(*position))
            strip.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdShade.MaterialBindingAPI.Apply(strip.GetPrim()).Bind(material)


def spawn_passive_asset(prim_path, cfg, translation=None, orientation=None):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd, UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    # Contact activation disables sleep. Keep reports, restore the asset's native sleep behavior.
    for body in Usd.PrimRange(prim):
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            body.RemoveProperty("physxRigidBody:sleepThreshold")
            assert body.GetAttribute("physxRigidBody:sleepThreshold").Get() > 0
    return prim


def make_env(control_hz=10, enable_cameras=True, log_dir=OUT / "logs", use_fabric=False,
             device="cuda:0", physics_dt=1 / 240):
    """Build the calibrated dual-arm station with a passive holder and free phone."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

    embodiment = BimanualYamEmbodiment(enable_cameras=enable_cameras, rig=station.rig(),
                                     cam_width=640, cam_height=360, robot_contact_offset=0.0005)
    renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=0.0,
        carb_settings={"/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 2}))
    mounts = {"top_cam": embodiment.rig.top_cam,
              **{side + "_cam": embodiment.rig.wrist_cams[side] for side in ("left", "right")}}
    for name, mount in mounts.items():
        cfg = getattr(embodiment.camera_config, name)
        cfg.renderer_cfg = renderer
        lens = mount.lens.resized(640, 360)
        cfg.spawn.horizontal_aperture = lens.width * lens.focal_mm / lens.fx
        cfg.spawn.vertical_aperture = lens.height * lens.focal_mm / lens.fy
        cfg.spawn.horizontal_aperture_offset = (lens.width / 2 - lens.cx) * lens.focal_mm / lens.fx
        cfg.spawn.vertical_aperture_offset = (lens.cy - lens.height / 2) * lens.focal_mm / lens.fy
    rotation = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
    assets = [
        Object(name="phoneholder", object_type=ObjectType.ARTICULATION,
               usd_path=str(ROOT / "assets/phoneholder.usda"),
               spawn_cfg_addon={"func": spawn_passive_asset},
               # PhysX chooses turntable as the root link, 4.1 mm above the base origin.
               initial_pose=Pose((0.30, 0.105, station.TABLE_TOP + 0.0051), rotation)),
        Object(name="phone", object_type=ObjectType.RIGID,
               usd_path=str(ROOT / "assets/iphone17pro/iphone17pro.usda"),
               spawn_cfg_addon={"func": spawn_passive_asset},
               initial_pose=Pose((0.30, -0.105, station.TABLE_TOP + 0.012), rotation)),
    ]

    def configure(cfg):
        cfg = env_cfg_callback(control_hz)(cfg)
        cfg.sim.dt = physics_dt
        cfg.decimation = round(1 / (control_hz * physics_dt))
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.log_dir = str(log_dir)
        return cfg

    description = IsaacLabArenaEnvironment(
        name="yam_use_phoneholder", embodiment=embodiment,
        scene=Scene([*task_room("station", embodiment.rig.top_cam), *assets]),
        env_cfg_callback=configure,
    )
    env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(
        num_envs=1, solve_relations=False, disable_fabric=not use_fabric, device=device,
    )).make_registered()
    import omni.usd
    add_tape_frame(omni.usd.get_context().get_stage(), station.TABLE_TOP)
    return env


def run():
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom
    from yam_arena import station
    from yam_arena.constants import READY

    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    base = env.unwrapped
    try:
        env.reset()
        action = torch.tensor([[*READY, 1.0, *READY, 1.0]], device=base.device)
        for _ in range(30):
            env.step(action)

        holder, phone = base.scene["phoneholder"], base.scene["phone"]
        assert holder.num_bodies == 4 and holder.num_joints == 3
        assert not holder.is_fixed_base
        positions = {}
        for name, asset, xy in (("phoneholder", holder, (0.30, 0.105)),
                                ("phone", phone, (0.30, -0.105))):
            pose = asset.data.root_link_pose_w.torch[0].cpu().numpy()
            assert np.linalg.norm(pose[:2] - xy) < 0.005, (name, pose)
            assert station.TABLE_TOP < pose[2] < station.TABLE_TOP + 0.025, (name, pose)
            positions[name] = pose.tolist()
        joint_deg = np.rad2deg(holder.data.joint_pos.torch[0].cpu().numpy())
        assert np.max(np.abs(joint_deg)) < 3.0, joint_deg

        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, "/PreviewCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        transform = camera.AddTransformOp()
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
        product = rep.create.render_product(str(camera.GetPath()), (1600, 1100))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        shots = (
            ("overview.png", (1.25, -1.25, 1.65), (0.18, 0.0, 0.92), 40),
            ("assets-closeup.png", (0.77, -0.45, 1.27), (0.30, 0.0, 0.76), 60),
        )
        for filename, eye, target, focal in shots:
            transform.Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
            camera.GetFocalLengthAttr().Set(focal)
            for _ in range(16):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pixels = rgb.get_data()
            assert pixels.shape == (1100, 1600, 4), pixels.shape
            assert pixels[..., :3].std() > 5, "Camera did not render scene geometry"
            Image.fromarray(np.ascontiguousarray(pixels[..., :3])).save(OUT / filename)
        rgb.detach()
        product.destroy()
        stage.GetRootLayer().Export(str(OUT / "scene.usda"))
        report = {"table_height_m": station.TABLE_TOP, "settle_seconds": 3.0,
                  "root_link_poses_xyzw": positions,
                  "holder_joint_names": holder.joint_names,
                  "holder_joint_angles_deg": joint_deg.tolist(),
                  "robot_action": action[0].cpu().tolist(),
                  "screenshots": [shot[0] for shot in shots]}
        (OUT / "placement.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
