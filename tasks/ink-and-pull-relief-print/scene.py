"""One calibrated YAM station and reset used by both policies."""
import json
from pathlib import Path

from geometry import AXLE_HOME, BLOCK, PAD_TOP, PALETTE, SHEET_HOME

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
PHYSICS_HZ = CONFIG["simulation"]["physics_hz"]
HZ = CONFIG["simulation"]["control_hz"]
DT = 1 / PHYSICS_HZ
MAX_STEPS = CONFIG["episode"]["max_control_steps"]
SEED = CONFIG["episode"]["seed"]
CAMERAS = tuple(CONFIG["observation"]["camera_names"])
INSTRUCTION = (
    "Roll the brayer back and forth through the ink on the pale slab until it is evenly coated. "
    "Roll it over the raised design on the block and set it down. Lay the clean paper on the block "
    "and press it evenly. Lift it by its corner tab without dragging it sideways to pull a clear print."
)
OBJECTS = (
    ("palette", PALETTE, "BASE"),
    ("block", BLOCK, "BASE"),
    ("pad", (*SHEET_HOME[:2], PAD_TOP), "BASE"),
    ("brayer", AXLE_HOME, "ARTICULATION"),
    ("sheet", (SHEET_HOME[0], SHEET_HOME[1], SHEET_HOME[2]+.001), "ARTICULATION"),
)


def make_env(*, cameras=True, screenshot=False):
    import carb.settings
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

    sim = CONFIG["simulation"]
    assert sim["backend"] == "physx" and PHYSICS_HZ % HZ == 0
    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, cam_width=width, cam_height=height,
                                 rig=rig(CONFIG["station"]["top_camera"]))
    if cameras:
        render = CONFIG["renderer"]
        renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=render["ambient_light_intensity"],
            carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))
        for name in CAMERAS:
            getattr(robot.camera_config, name).renderer_cfg = renderer
        if screenshot:
            top = robot.camera_config.top_cam
            top.width, top.height = 1920, 1080
            top.spawn = _pinhole(robot.rig.top_cam, 1080, 1920, near=.05)

    def configure(cfg):
        cfg = env_cfg_callback(HZ)(cfg)
        cfg.sim.dt, cfg.decimation = DT, PHYSICS_HZ // HZ
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.episode_length_s = MAX_STEPS / HZ
        return cfg

    objects = [Object(name=name, usd_path=str(HERE / "assets" / (name + ".usda")),
                      object_type=ObjectType[kind],
                      initial_pose=Pose(position_xyz=(x, y, TABLE_TOP+z), rotation_xyzw=(0, 0, 0, 1)))
               for name, (x, y, z), kind in OBJECTS]
    arena = IsaacLabArenaEnvironment(name="relief_print",
        scene=Scene([*task_room(CONFIG["station"]["scene"], robot.rig.top_cam), *objects]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=SEED)
    env = ArenaEnvBuilder(arena, build).make_registered()
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    return env
