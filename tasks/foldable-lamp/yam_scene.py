"""Load the folded lamp and a closed LIBERO book onto the calibrated YAM station."""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/scene"
BOOK = ROOT / "assets/book/closed_book.usda"
BOOK_XY = (.34, .0)
BOOK_SIZE = (.18, .11, .02)  # Small closed novel: length, width, thickness (m).
BOOK_MASS = .20
LAMP_XY = (.50, .18)  # Robot faces +X; its left is +Y and the book's far side is +X.
ENVIRONMENT_LIGHT_SCALE = .5
TARGET_XY = (.43, .16)  # Behind and to the left of the book, seen from the robots.
TARGET_YAW = math.degrees(math.atan2(BOOK_XY[0]-TARGET_XY[0], TARGET_XY[1]-BOOK_XY[1]))
TARGET_MARGIN = .01
BASE_HALF_SIZE = (.038, .070)
LAMP_MIN_HEIGHT = .20  # Lowest point of the shade/diffuser above the tabletop.
LAMP_MAX_TILT = 15.  # Emitting-face normal relative to downward vertical, degrees.
TASK_SECONDS = 120


def add_target_tape(stage, table_height):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = "/World/envs/env_0/LampTarget"
    frame = UsdGeom.Xform.Define(stage, root)
    frame.AddTranslateOp().Set(Gf.Vec3d(*TARGET_XY, table_height + .0001))
    frame.AddRotateZOp().Set(TARGET_YAW)
    mat = UsdShade.Material.Define(stage, root + "/RedPaperTape")
    shader = UsdShade.Shader.Define(stage, root + "/RedPaperTape/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((.65, .015, .01))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.95)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    hx, hy = (v + TARGET_MARGIN for v in BASE_HALF_SIZE)
    width = .004
    for axis in (0, 1):
        for sign in (-1, 1):
            pos, size = [0., 0., 0.], [2*hx, 2*hy, .0001]
            pos[axis] = sign * ((hx, hy)[axis] + width/2)
            size[axis] = width
            size[1-axis] += 2*width
            strip = UsdGeom.Cube.Define(stage, root + f"/strip_{axis}_{sign+1}")
            strip.CreateSizeAttr(1.)
            strip.AddTranslateOp().Set(Gf.Vec3d(*pos))
            strip.AddScaleOp().Set(Gf.Vec3d(*size))
            UsdShade.MaterialBindingAPI.Apply(strip.GetPrim()).Bind(mat)


def make_env(control_hz=10, enable_cameras=True, log_dir=OUT / "logs"):
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
    from model import ASSET

    embodiment = BimanualYamEmbodiment(enable_cameras=enable_cameras, rig=station.rig(),
        cam_width=640, cam_height=360, robot_contact_offset=.0005)
    if enable_cameras:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        for name in ("left_cam", "top_cam", "right_cam"):
            cfg = getattr(embodiment.camera_config, name)
            camera, lens = cfg.spawn, cfg.spawn.distortion
            camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
            camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
            camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
            camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
            cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=0., carb_settings={
                    "/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 2,
                }))
    objects = [
        Object(name="foldable_lamp", object_type=ObjectType.ARTICULATION, usd_path=str(ASSET),
            initial_pose=Pose((*LAMP_XY, station.TABLE_TOP+.001), (0., 0., -math.sqrt(.5), math.sqrt(.5)))),
        Object(name="book", object_type=ObjectType.RIGID, usd_path=str(BOOK),
            initial_pose=Pose((*BOOK_XY, station.TABLE_TOP+.001))),
    ]

    def configure(cfg):
        cfg = env_cfg_callback(control_hz)(cfg)
        cfg.sim.dt = 1/480
        cfg.decimation = cfg.sim.render_interval = round(480 / control_hz)
        cfg.sim.physics.solver_type = 0
        cfg.sim.physics.enable_external_forces_every_iteration = False
        cfg.sim.log_dir = str(log_dir)
        return cfg

    room = task_room("station", embodiment.rig.top_cam)
    lights = [obj for obj in room if obj.name in ("light", "lamp")]
    assert len(lights) == 2
    for light in lights:
        light.object_cfg.spawn.intensity *= ENVIRONMENT_LIGHT_SCALE
    description = IsaacLabArenaEnvironment(name="yam_foldable_lamp", embodiment=embodiment,
        scene=Scene([*room, *objects]), env_cfg_callback=configure)
    env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(
        num_envs=1, solve_relations=False, disable_fabric=True, device="cpu")).make_registered()
    add_target_tape(env.unwrapped.sim.stage, station.TABLE_TOP)
    return env
