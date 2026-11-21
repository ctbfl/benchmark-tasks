"""Build the shared calibrated YAM station and colliding task asset."""
import json

from geometry import HERE

CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
PHYSICS_HZ = CONFIG["simulation"]["physics_hz"]
CONTROL_HZ = CONFIG["simulation"]["control_hz"]
LIMIT = CONFIG["episode"]["max_control_steps"]
SEED = CONFIG["episode"]["seed"]
CAMERAS = tuple(CONFIG["observation"]["camera_names"])


def make_env(log_dir, cameras=True):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from yam_arena.station import rig

    sim = CONFIG["simulation"]
    assert sim["backend"] == "physx" and PHYSICS_HZ % CONTROL_HZ == 0
    width, height = CONFIG["observation"]["render_size_wh"]
    station = CONFIG["station"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, cam_width=width, cam_height=height,
                                 rig=rig(station["top_camera"]))
    render = CONFIG["renderer"]
    if cameras:
        renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=render["ambient_light_intensity"],
            carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))
        for name in CAMERAS:
            getattr(robot.camera_config, name).renderer_cfg = renderer

    def configure(cfg):
        cfg = env_cfg_callback(CONTROL_HZ)(cfg)
        cfg.sim.dt = 1 / PHYSICS_HZ
        cfg.decimation = PHYSICS_HZ // CONTROL_HZ
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.episode_length_s = LIMIT / CONTROL_HZ
        cfg.sim.log_dir = str(log_dir)
        return cfg

    kit = Object(name="bubble_kit", usd_path=str(HERE / "assets" / "wand-kit.usda"), object_type=ObjectType.BASE)
    arena = IsaacLabArenaEnvironment(name="two_stick_soap_bubble",
        scene=Scene([*task_room(station["scene"], robot.rig.top_cam), kit]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=SEED)
    return ArenaEnvBuilder(arena, build).make_registered()
