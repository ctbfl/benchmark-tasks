"""One calibrated YAM scene and configuration shared by the demo and the model evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_assets import COINS, JAR_FLOOR, JAR_INNER

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
JAR_XY = (.54, 0.)
COINS_PER_KIND = 5
COIN_NAMES = [f"penny_{i}" for i in range(COINS_PER_KIND)] + [f"quarter_{i}" for i in range(COINS_PER_KIND)]


def coin_placements(seed):
    """Two coins per layer on opposite sides of the jar axis, alternating kinds, then free fall."""
    rng = np.random.default_rng(seed)
    order = [name for pair in zip(COIN_NAMES[:COINS_PER_KIND], COIN_NAMES[COINS_PER_KIND:]) for name in pair]
    rng.shuffle(order)
    placements = []
    offset = .0145
    for layer in range(COINS_PER_KIND):
        heading = rng.uniform(0, 2*np.pi)
        z = JAR_FLOOR + .004 + .0055*layer
        for k, name in enumerate(order[2*layer:2*layer+2]):
            kind = name.split("_")[0]
            assert offset + COINS[kind]["radius"] < JAR_INNER - .0005
            angle = heading + np.pi*k
            x, y = JAR_XY[0] + offset*np.cos(angle), JAR_XY[1] + offset*np.sin(angle)
            tilt = rng.uniform(-8, 8, 2)
            placements.append((name, kind + ".usda", "RIGID", (x, y, z), (*tilt, rng.uniform(0, 360))))
    return sorted(placements, key=lambda item: COIN_NAMES.index(item[0]))


PLACEMENTS = [
    ("jar", "jar.usda", "RIGID", (*JAR_XY, .0005), (0., 0., 0.)),
    *coin_placements(CONFIG["episode"]["seed"]),
]


def spawn_with_native_sleep(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Arena enables contact reporting on rigid objects, which writes sleepThreshold=0.

    Coins and the jar keep contact reporting but use PhysX's native sleep threshold, so a
    resting pile does not keep exchanging tiny solver impulses.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd, UsdPhysics
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for body in Usd.PrimRange(prim):
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            body.RemoveProperty("physxRigidBody:sleepThreshold")
            assert body.GetAttribute("physxRigidBody:sleepThreshold").Get() > 0
    return prim


def make_env(*, review=False, cameras=True):
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
        cfg.sim.physics.enable_ccd = sim["enable_ccd"]
        cfg.episode_length_s = CONFIG["episode"]["max_control_steps"] / sim["control_hz"]
        cfg.sim.log_dir = str(HERE / "runs/logs")
        return cfg

    objects = []
    for name, filename, kind, (x, y, z), angles in PLACEMENTS:
        path = HERE / "assets" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rotation = tuple(float(v) for v in Rotation.from_euler("xyz", angles, degrees=True).as_quat())
        objects.append(Object(name=name, usd_path=str(path), object_type=ObjectType[kind],
            spawn_cfg_addon={"func": spawn_with_native_sleep},
            initial_pose=Pose(position_xyz=(float(x), float(y), float(TABLE_TOP+z)), rotation_xyzw=rotation)))
    objects.append(Object(name="tape", usd_path=str(HERE / "assets/tape.usda"), object_type=ObjectType.BASE,
        initial_pose=Pose(position_xyz=(0., 0., TABLE_TOP), rotation_xyzw=(0, 0, 0, 1))))
    arena = IsaacLabArenaEnvironment(name="split_coins",
        scene=Scene([*task_room("station", robot.rig.top_cam), *objects]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=CONFIG["episode"]["seed"])
    return ArenaEnvBuilder(arena, build).make_registered()
