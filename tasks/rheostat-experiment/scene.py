"""One shared calibrated workstation for development, replay and Dream."""
from circuit import CONFIG, HERE, HZ, PHYSICS_HZ


def make_env(out, cameras=True, screenshot=False):
    from isaaclab.sensors import CameraCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from assets import RHEOSTAT_X

    sim = CONFIG["simulation"]
    assert sim["backend"] == "physx" and PHYSICS_HZ % HZ == 0
    assert sim["num_envs"] == 1 and not CONFIG["episode"]["advance_physics_during_api_wait"]
    rig = station.rig(CONFIG["station"]["top_camera"])
    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig, contact_sensors=False,
                                  cam_width=width, cam_height=height)
    if screenshot:
        cfg = robot._camera_cfg(1080, 1920, False).top_cam
        cfg.prim_path = "{ENV_REGEX_NS}/ScreenshotCamera"
        robot.camera_config = make_configclass("ScreenshotCameras",
            [("screenshot", CameraCfg, cfg)], bases=(type(robot.camera_config),))()
    for name in robot.camera_config.camera_names():
        cfg = getattr(robot.camera_config, name)
        lens, camera = cfg.spawn.distortion, cfg.spawn
        camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
        camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
        camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
        camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
        cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=CONFIG["renderer"]["ambient_light_intensity"],
            carb_settings={"/rtx/rendermode": CONFIG["renderer"]["mode"],
                           "/rtx/post/aa/op": CONFIG["renderer"]["antialiasing_op"]}))
    circuit = Object(name="circuit", object_type=ObjectType.BASE,
        usd_path=str(HERE/"assets/circuit.usda"), initial_pose=Pose((0,0,station.TABLE_TOP), (0,0,0,1)))
    rheostat = Object(name="rheostat", object_type=ObjectType.ARTICULATION,
        usd_path=str(HERE/"assets/rheostat.usda"),
        initial_pose=Pose((RHEOSTAT_X,0,station.TABLE_TOP+.001), (0,0,0,1)))
    rheostat.object_cfg.init_state.joint_pos = {"slide": 0.}

    def configure(cfg):
        cfg = env_cfg_callback(HZ)(cfg)
        cfg.sim.dt = 1 / PHYSICS_HZ
        cfg.decimation = PHYSICS_HZ // HZ
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.log_dir = cfg.sim.log_dir = str(out/"logs")
        return cfg

    spec = IsaacLabArenaEnvironment(name="rheostat_experiment", embodiment=robot,
        scene=Scene([*task_room(CONFIG["station"]["scene"], rig.top_cam), circuit, rheostat]),
        env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False,
        disable_fabric=not sim["use_fabric"], device=sim["device"])).make_registered()
