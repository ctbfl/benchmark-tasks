"""One environment and configuration for development, replay and Dream."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "dream-defaults.json").read_text())
BOARD_XY = (.17, .18)
RULER_XY = (.17, .02)
CALCULATOR_XY = (.23, -.14)


def spawn_asset(prim_path, cfg, translation=None, orientation=None):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd, UsdPhysics
    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveProperty("physxRigidBody:sleepThreshold")
    return prim


def make_env(out, cameras=False):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from model import ASSETS, RULER_SIZE

    sim = CONFIG["simulation"]
    assert sim["backend"] == "physx" and sim["physics_hz"] % sim["control_hz"] == 0
    rig = station.rig(CONFIG["station"]["top_camera"])
    w, h = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig, cam_width=w, cam_height=h)
    renderer = CONFIG["renderer"]
    for name, mount in {"top_cam": rig.top_cam, **{side+"_cam": rig.wrist_cams[side] for side in ("left", "right")}}.items():
        cfg = getattr(robot.camera_config, name)
        lens = mount.lens.resized(w, h)
        cfg.spawn.horizontal_aperture = w*lens.focal_mm/lens.fx
        cfg.spawn.vertical_aperture = h*lens.focal_mm/lens.fy
        cfg.spawn.horizontal_aperture_offset = (w/2-lens.cx)*lens.focal_mm/lens.fx
        cfg.spawn.vertical_aperture_offset = (lens.cy-h/2)*lens.focal_mm/lens.fy
        cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=renderer["ambient_light_intensity"],
            carb_settings={"/rtx/rendermode": renderer["mode"], "/rtx/post/aa/op": renderer["antialiasing_op"]}))
    rotation = (0., 0., -math.sqrt(.5), math.sqrt(.5))
    props = []
    for name, kind, position, orientation in (
        ("board", ObjectType.BASE, (*BOARD_XY, station.TABLE_TOP), rotation),
        ("ruler", ObjectType.RIGID, (*RULER_XY, station.TABLE_TOP+RULER_SIZE[2]/2+.0005), (0., 0., 0., 1.)),
        ("calculator", ObjectType.ARTICULATION, (*CALCULATOR_XY, station.TABLE_TOP+.0005), rotation),
    ):
        props.append(Object(name=name, object_type=kind, usd_path=str(ASSETS/(name+".usda")),
                            spawn_cfg_addon={"func": spawn_asset}, initial_pose=Pose(position, orientation)))

    def configure(cfg):
        cfg = env_cfg_callback(sim["control_hz"])(cfg)
        cfg.seed = CONFIG["episode"]["seed"]
        cfg.sim.dt = 1/sim["physics_hz"]
        cfg.decimation = sim["physics_hz"]//sim["control_hz"]
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.sim.log_dir = str(out/"logs")
        cfg.log_dir = str(out/"logs")
        return cfg

    description = IsaacLabArenaEnvironment(name="yam_measure_triangle", embodiment=robot,
        scene=Scene([*task_room(CONFIG["station"]["scene"], rig.top_cam), *props]), env_cfg_callback=configure)
    return ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=sim["num_envs"],
        solve_relations=False, disable_fabric=not sim["use_fabric"], device=sim["device"])).make_registered()
