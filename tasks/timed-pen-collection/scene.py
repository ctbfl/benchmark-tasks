"""One calibrated YAM scene and configuration for development and evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
CUP_XY = (.46, -.325)
TIMER_XY = (.39, .325)
PEN_REGION = (.22, .61, -.215, .215)


def pen_placements(seed):
    """Seeded, fully contained, separated line segments, with a finite retry cap."""
    from build_assets import PEN_LENGTH
    rng = np.random.default_rng(seed)
    placed = []
    for _ in range(5000):
        x, y = rng.uniform((.24, -.19), (.59, .19))
        angle = rng.uniform(-np.pi, np.pi)
        axis = np.array([np.cos(angle), np.sin(angle)])
        endpoints = np.array([x, y]) + np.array([[-1], [1]]) * axis * PEN_LENGTH/2
        if not (np.all(endpoints > [.23, -.205]) and np.all(endpoints < [.60, .205])):
            continue
        # A capsule-distance check prevents overlaps while preserving random headings.
        segment = np.linspace(*endpoints, 41)
        if any(np.min(np.linalg.norm(segment[:, None] - p[3][None], axis=2)) < .025 for p in placed):
            continue
        placed.append((x, y, angle, segment))
        if len(placed) == 5:
            return [(f"pen_{i}", "pen.usda", "RIGID", (x, y, .009), (0., 90., np.degrees(a)))
                    for i, (x, y, a, _) in enumerate(placed)]
    raise RuntimeError("Could not place five non-overlapping pens inside the tape")


PLACEMENTS = [
    ("timer", "timer.usda", "ARTICULATION", (*TIMER_XY, .001), (0, 0, -90)),
    ("cup", "cup.usda", "RIGID", (*CUP_XY, .001), (0, 0, 0)),
    *pen_placements(CONFIG["episode"]["seed"]),
]


def make_env(*, review=False, cameras=True):
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.events import reset_articulation_joints
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena.embodiment import BimanualYamEmbodiment, _pinhole
    from yam_arena.scene import env_cfg_callback, task_room
    from yam_arena.station import TABLE_TOP, rig

    sim = CONFIG["simulation"]
    assert sim["backend"] == "physx" and sim["physics_hz"] % sim["control_hz"] == 0
    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=cameras, cam_width=width,
        cam_height=height, rig=rig(CONFIG["station"]["top_camera"]))
    render = CONFIG["renderer"]
    renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=render["ambient_light_intensity"],
        carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))
    if cameras:
        for name in CONFIG["observation"]["camera_names"]:
            getattr(robot.camera_config, name).renderer_cfg = renderer
        if review:
            top = robot.camera_config.top_cam
            top.width, top.height = 1920, 1080
            top.spawn = _pinhole(robot.rig.top_cam, 1080, 1920, near=.05)

    def configure(cfg):
        cfg = env_cfg_callback(sim["control_hz"])(cfg)
        cfg.sim.dt = 1 / sim["physics_hz"]
        cfg.decimation = sim["physics_hz"] // sim["control_hz"]
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.episode_length_s = CONFIG["episode"]["max_control_steps"] / sim["control_hz"]
        cfg.events.reset_timer_button = EventTermCfg(func=reset_articulation_joints, mode="reset",
            params={"asset_cfg": SceneEntityCfg("timer")})
        cfg.sim.log_dir = str(HERE / "runs/logs")
        return cfg

    objects = []
    for name, filename, kind, (x, y, z), angles in PLACEMENTS:
        path = HERE / "assets" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rotation = tuple(float(v) for v in Rotation.from_euler("xyz", angles, degrees=True).as_quat())
        objects.append(Object(name=name, usd_path=str(path), object_type=ObjectType[kind],
            initial_pose=Pose(position_xyz=(float(x), float(y), float(TABLE_TOP+z)), rotation_xyzw=rotation)))
    objects.append(Object(name="tape", usd_path=str(HERE / "assets/tape.usda"), object_type=ObjectType.BASE,
        initial_pose=Pose(position_xyz=(0., 0., TABLE_TOP), rotation_xyzw=(0, 0, 0, 1))))
    arena = IsaacLabArenaEnvironment(name="timed_pen_collection",
        scene=Scene([*task_room("station", robot.rig.top_cam), *objects]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=CONFIG["episode"]["seed"])
    return ArenaEnvBuilder(arena, build).make_registered()
