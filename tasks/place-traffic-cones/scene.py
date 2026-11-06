"""One shared calibrated workstation for development, open-loop replay and Dream."""
from config import (HERE, CONFIG, DT, CONTROL_HZ, SUBSTEPS, STACK_XY, STACK_PITCH)


def make_env(*, cameras=False, log_dir=None):
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

    log_dir = HERE / 'runs/scene/logs' if log_dir is None else log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    rig = station.rig(CONFIG['station']['top_camera'])
    width, height = CONFIG['observation']['render_size_wh']
    robot = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig,
        cam_width=width, cam_height=height, contact_sensors=False)
    for name in robot.camera_config.camera_names():
        camera = getattr(robot.camera_config, name)
        lens, spawn = camera.spawn.distortion, camera.spawn
        spawn.horizontal_aperture = camera.width * spawn.focal_length / lens.fx
        spawn.vertical_aperture = camera.height * spawn.focal_length / lens.fy
        spawn.horizontal_aperture_offset = (camera.width/2-lens.cx)*spawn.focal_length/lens.fx
        spawn.vertical_aperture_offset = (lens.cy-camera.height/2)*spawn.focal_length/lens.fy
        camera.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=CONFIG['renderer']['ambient_light_intensity'],
            carb_settings={'/rtx/rendermode': CONFIG['renderer']['mode'],
                           '/rtx/post/aa/op': CONFIG['renderer']['antialiasing_op']}))
    assets = [*task_room('station', rig.top_cam),
        Object(name='board', object_type=ObjectType.BASE, usd_path=str(HERE/'assets/board.usda'),
               initial_pose=Pose((0., 0., station.TABLE_TOP)))]
    for i in range(6):
        item = Object(name=f'cone_{i}', object_type=ObjectType.ARTICULATION,
            usd_path=str(HERE/'assets/cone.usda'),
            initial_pose=Pose((*STACK_XY, station.TABLE_TOP + .0005 + i*STACK_PITCH)))
        item.object_cfg.init_state.joint_pos = {'.*': 0.}
        assets.append(item)

    def configure(cfg):
        cfg = env_cfg_callback(CONTROL_HZ)(cfg)
        sim = CONFIG['simulation']
        cfg.sim.dt = DT
        cfg.decimation = SUBSTEPS
        cfg.sim.render_interval = SUBSTEPS
        cfg.sim.use_fabric = sim['use_fabric']
        cfg.sim.device = sim['device']
        cfg.sim.gravity = tuple(sim['gravity_m_s2'])
        cfg.sim.physics.solver_type = sim['solver_type']
        cfg.sim.log_dir = str(log_dir)
        cfg.log_dir = str(log_dir)
        return cfg

    spec = IsaacLabArenaEnvironment(name='place_traffic_cones', embodiment=robot,
        scene=Scene(assets), env_cfg_callback=configure)
    env = ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=CONFIG['simulation']['num_envs'],
        solve_relations=False, disable_fabric=not CONFIG['simulation']['use_fabric'],
        device=CONFIG['simulation']['device'])).make_registered()
    assert abs(env.unwrapped.physics_dt-DT) < 1e-12
    for side in ('left', 'right'):
        cfg = env.unwrapped.scene[side+'_robot'].cfg
        for group in ('shoulder', 'wrist', 'gripper'):
            actuator = cfg.actuators[group]
            reference = CONFIG['robot_reference'][group]
            assert actuator.stiffness == reference['stiffness']
            assert actuator.damping == reference['damping']
            assert actuator.joint_effort_limit == reference['effort']
            assert actuator.joint_velocity_limit == reference['velocity']
        assert cfg.spawn.articulation_props.solver_position_iteration_count == 16
        assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 4
        assert not cfg.spawn.articulation_props.enabled_self_collisions
    # Arena contact activation disables sleeping; retain reports with native asset sleeping.
    from pxr import PhysxSchema
    for i in range(6):
        item = env.unwrapped.scene[f'cone_{i}']
        assert not item.actuators and not item.is_fixed_base
        for name in item.body_names:
            prim = env.unwrapped.sim.stage.GetPrimAtPath(f'/World/envs/env_0/cone_{i}/{name}')
            prim.RemoveProperty('physxRigidBody:sleepThreshold')
    return env
