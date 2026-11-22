"""Calibrated YAM station with one rug on the level work surface."""
import json
from pathlib import Path

from scoring import CENTER_X, FRAME_MIN, FRAME_MAX, TAPE_WIDTH, TAPE_THICKNESS

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
RUG = ROOT + "/rug"
SIM = CONFIG["simulation"]
EPISODE = CONFIG["episode"]
PHYSICS_HZ, CONTROL_HZ = SIM["physics_hz"], SIM["control_hz"]
INSTRUCTION = ("Retrieve the rug corner tucked underneath and lay the whole rug flat, "
               "with every part inside the red tape frame.")


def make_env(cameras=True, screenshot=False):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena.embodiment import BimanualYamEmbodiment, _pinhole
    from yam_arena.scene import env_cfg_callback, task_room
    from yam_arena.station import TABLE_TOP, rig
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, cam_width=width, cam_height=height,
                                  contact_sensors=False, robot_contact_offset=.0005,
                                  rig=rig(CONFIG["station"]["top_camera"]))
    if cameras:
        render = CONFIG["renderer"]
        renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=render["ambient_light_intensity"],
            carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))
        for name in CONFIG["observation"]["camera_names"]:
            getattr(robot.camera_config, name).renderer_cfg = renderer
        if screenshot:
            top = robot.camera_config.top_cam
            top.width, top.height = 1920, 1080
            top.spawn = _pinhole(robot.rig.top_cam, 1080, 1920, near=.05)

    def configure(cfg):
        cfg = env_cfg_callback(CONTROL_HZ)(cfg)
        cfg.sim.dt = 1 / PHYSICS_HZ
        cfg.decimation = PHYSICS_HZ // CONTROL_HZ
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = SIM["device"]
        cfg.sim.gravity = tuple(SIM["gravity_m_s2"])
        cfg.sim.use_fabric = SIM["use_fabric"]
        cfg.sim.physics.solver_type = SIM["solver_type"]
        cfg.episode_length_s = EPISODE["max_control_steps"] / CONTROL_HZ
        return cfg

    rug = Object(name="rug", usd_path=str(HERE / "assets/rug.usda"), object_type=ObjectType.BASE,
                 initial_pose=Pose(position_xyz=(CENTER_X, 0., TABLE_TOP), rotation_xyzw=(0., 0., 0., 1.)))
    arena = IsaacLabArenaEnvironment(name="retrieve_tucked_rug_corner",
        scene=Scene([*task_room(CONFIG["station"]["scene"], robot.rig.top_cam), rug]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=SIM["num_envs"], solve_relations=False,
        device=SIM["device"], disable_fabric=not SIM["use_fabric"], seed=EPISODE["seed"])
    env = ArenaEnvBuilder(arena, build).make_registered()
    stage = env.unwrapped.sim.stage
    frame_path = ROOT + "/tape_frame"
    frame = UsdGeom.Xform.Define(stage, frame_path)
    frame.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    material = UsdShade.Material.Define(stage, frame_path + "/RedPaperTape")
    shader = UsdShade.Shader.Define(stage, frame_path + "/RedPaperTape/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.72, .025, .018))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    center, size = (FRAME_MIN + FRAME_MAX) / 2, FRAME_MAX - FRAME_MIN
    strips = [
        ("near", (center[0], FRAME_MIN[1] - TAPE_WIDTH / 2), (size[0] + 2 * TAPE_WIDTH, TAPE_WIDTH)),
        ("far", (center[0], FRAME_MAX[1] + TAPE_WIDTH / 2), (size[0] + 2 * TAPE_WIDTH, TAPE_WIDTH)),
        ("left", (FRAME_MIN[0] - TAPE_WIDTH / 2, center[1]), (TAPE_WIDTH, size[1])),
        ("right", (FRAME_MAX[0] + TAPE_WIDTH / 2, center[1]), (TAPE_WIDTH, size[1])),
    ]
    for name, xy, wh in strips:
        strip = UsdGeom.Cube.Define(stage, frame_path + "/" + name)
        strip.CreateSizeAttr(1.)
        strip.AddTranslateOp().Set(Gf.Vec3d(float(xy[0]), float(xy[1]), TABLE_TOP + TAPE_THICKNESS / 2))
        strip.AddScaleOp().Set(Gf.Vec3f(float(wh[0]), float(wh[1]), TAPE_THICKNESS))
        UsdShade.MaterialBindingAPI.Apply(strip.GetPrim()).Bind(material)
    return env
