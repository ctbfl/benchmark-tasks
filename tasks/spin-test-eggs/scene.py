"""The shared calibrated YAM station and egg reset distribution."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from assets import ASSETS, BOARD_Z, TRAYS

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
BOARD_XY = (.365, 0.)
START_YAW = np.pi/2
PAIR_TRIALS = ((.335, 0., START_YAW), (.339, .004, START_YAW+.10), (.331, -.004, START_YAW-.11))
CAMERAS = tuple(CONFIG["observation"]["camera_names"])
INSTRUCTION = (
    "Determine whether the egg is raw or hard-boiled by spinning it and stopping it abruptly. "
    "Place it in the correctly labeled basket."
)


def placements(seed, pair=False):
    if pair:
        return [PAIR_TRIALS[seed]], [False]
    rng = np.random.default_rng(seed)
    x, y = np.array([.335, 0.]) + rng.uniform(-.008, .008, 2)
    return [(x, y, START_YAW+rng.uniform(-.12, .12))], [bool(rng.integers(2))]


def reset_egg_pose(env, env_ids, asset_cfg, pose, velocity=None):
    """Write an XYZW transform and velocity using PhysX's float32 interface."""
    import torch

    assert velocity is None
    state = torch.tensor((*pose.position_xyz, *pose.rotation_xyzw),
                         dtype=torch.float32, device=env.device).repeat(len(env_ids), 1)
    state[:, :3] += env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=env.device)
    egg = env.scene[asset_cfg.name]
    egg.write_root_pose_to_sim(state, env_ids=env_ids)
    egg.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def make_env(seed, *, pair=False, screenshot=False, cameras=True, log_dir):
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
    hz = sim["control_hz"]
    assert sim["backend"] == "physx" and sim["physics_hz"] % hz == 0
    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, cam_width=width, cam_height=height,
                                 rig=rig(CONFIG["station"]["top_camera"]), robot_contact_offset=.001)
    render = CONFIG["renderer"]
    renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=render["ambient_light_intensity"],
        carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))
    if cameras:
        for name in CAMERAS:
            getattr(robot.camera_config, name).renderer_cfg = renderer
        if screenshot:
            top = robot.camera_config.top_cam
            top.width, top.height = 1920, 1080
            top.spawn = _pinhole(robot.rig.top_cam, 1080, 1920, near=.05)

    def configure(cfg):
        cfg = env_cfg_callback(hz)(cfg)
        cfg.sim.dt = 1 / sim["physics_hz"]
        cfg.decimation = sim["physics_hz"] // hz
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.episode_length_s = CONFIG["episode"]["max_control_steps"] / hz
        cfg.sim.log_dir = str(log_dir)
        for i in range(len(spawn)):
            getattr(cfg.events, f"egg_{i}").func = reset_egg_pose
        return cfg

    def obj(name, asset, kind, xy, z=0., yaw=0.):
        return Object(name=name, usd_path=str(ASSETS / f"{asset}.usda"), object_type=kind,
            initial_pose=Pose(position_xyz=(*map(float, xy), TABLE_TOP+z),
                              rotation_xyzw=tuple(Rotation.from_euler("z", yaw).as_quat())))

    spawn, raw = placements(seed, pair)
    # Set the shell just above its measured native side-rest height.
    eggs = [obj(f"egg_{i}", "egg", ObjectType.RIGID, (x, y), BOARD_Z+.0192, yaw)
            for i, (x, y, yaw) in enumerate(spawn)]
    board = obj("board", "board", ObjectType.BASE, BOARD_XY)
    trays = [obj(kind, kind, ObjectType.BASE, xy) for kind, xy in TRAYS.items()]
    arena = IsaacLabArenaEnvironment(name="spin_test_eggs",
        scene=Scene([*task_room(CONFIG["station"]["scene"], robot.rig.top_cam), board, *trays, *eggs]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=seed)
    return ArenaEnvBuilder(arena, build).make_registered(), raw, robot.link_paths
