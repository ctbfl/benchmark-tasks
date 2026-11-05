"""Passive headphones and stand on the calibrated YAM dual-arm, three-camera station."""

import math
from pathlib import Path

START_ANGLES = {"left_swivel": -90, "right_swivel": 90, "left_fold": -10, "right_fold": 83.9}

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/scene"
# The articulation root is at the crown, offset from the folded footprint centre.
CROWN_XY = (.49355, -.12418)
STAND_XY = (.37355, .10582)
FLAT_XYZW = (.5, .5, .5, .5)  # Rz(+90) Rx(90): cups toward the arms, crown away.
YAW_XYZW = (0., 0., -math.sqrt(.5), math.sqrt(.5))


def reset_headphones(env, env_ids):
    """Set a folded initial state once on reset; leave every joint passive afterwards."""
    import torch
    from yam_arena import station

    asset = env.scene["headphones"]
    q = asset.data.default_joint_pos.torch.clone()
    asset.write_joint_state_to_sim(q, torch.zeros_like(q))
    pose = asset.data.default_root_state.torch[:, :7].clone()
    pose[:, 2] = station.TABLE_TOP + .0181
    asset.write_root_pose_to_sim(pose)
    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


def make_env(enable_cameras=True, control_hz=10, log_dir=OUT / "logs"):
    from isaaclab.managers import EventTermCfg
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

    rig = station.rig()
    embodiment = BimanualYamEmbodiment(enable_cameras=enable_cameras, rig=rig, contact_sensors=False,
                                     robot_contact_offset=.0005)
    # Preserve each camera's calibrated intrinsics while rendering clean static views.
    for name in embodiment.camera_config.camera_names():
        cfg = getattr(embodiment.camera_config, name)
        lens, camera = cfg.spawn.distortion, cfg.spawn
        camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
        camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
        camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
        camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
        cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=0., carb_settings={
                "/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 4,
            }))

    headphones = Object(name="headphones", object_type=ObjectType.ARTICULATION,
        usd_path=str(HERE / "assets/headphones.usda"),
        # Spawn clear of the tabletop; the reset event initializes the folded pose.
        initial_pose=Pose((*CROWN_XY, station.TABLE_TOP+.075), FLAT_XYZW))
    headphones.object_cfg.init_state.joint_pos = {"bend_.*": 0., **{
        name:math.radians(angle) for name, angle in START_ANGLES.items()}}
    stand = Object(name="headphone_stand", object_type=ObjectType.RIGID,
        usd_path=str(HERE / "assets/stand.usda"),
        initial_pose=Pose((*STAND_XY, station.TABLE_TOP+.001), YAW_XYZW))

    def configure(cfg):
        cfg = env_cfg_callback(control_hz)(cfg)
        cfg.sim.use_fabric = False
        cfg.sim.log_dir = str(log_dir)
        cfg.log_dir = str(log_dir)
        cfg.events.reset_headphones = EventTermCfg(func=reset_headphones, mode="reset")
        return cfg

    spec = IsaacLabArenaEnvironment(name="yam_elastic_headphones", embodiment=embodiment,
        scene=Scene([*task_room("station", rig.top_cam), headphones, stand]),
        env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(
        num_envs=1, solve_relations=False, disable_fabric=True, device="cpu")).make_registered()
