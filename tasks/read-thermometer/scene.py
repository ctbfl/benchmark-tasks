"""One calibrated YAM environment shared by modeling, demo, and evaluation."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
# +Y is left in the calibrated camera; the table's long edge follows Y.
CUP_Y = .16
PLACEMENTS = (
    ("stand", "stand_clamp/stand.usda", "RIGID", (.43, .19, 0), 0),
    ("beaker", "lab_glass/beaker.usda", "RIGID", (.43, CUP_Y, .014), 0),
    ("clamp", "stand_clamp/clamp.usda", "ARTICULATION", (.43, .28, .345), 0),
    ("thermometer", "lab_glass/thermometer.usda", "RIGID", (.43, CUP_Y, .040), 0),
    ("timer", "electronics/timer.usda", "ARTICULATION", (.46, -.235, .001), 0),
    ("calculator", "electronics/calculator.usda", "ARTICULATION", (.23, -.235, .001), -90),
)


def make_env(*, review=False, test_probe=False):
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
    assert sim["backend"] == "physx"
    assert sim["physics_hz"] % sim["control_hz"] == 0
    width, height = CONFIG["observation"]["render_size_wh"]
    robot = BimanualYamEmbodiment(enable_cameras=True, cam_width=width,
        cam_height=height, rig=rig(CONFIG["station"]["top_camera"]))
    render = CONFIG["renderer"]
    renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=render["ambient_light_intensity"],
        carb_settings={"/rtx/rendermode": render["mode"],
                       "/rtx/post/aa/op": render["antialiasing_op"],
                       "/rtx/rtpt/maxBounces": render["max_bounces"],
                       "/rtx/rtpt/maxSpecularAndTransmissionBounces": render["max_specular_transmission_bounces"]}))
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
        cfg.sim.log_dir = str(HERE / "runs/logs")
        return cfg

    objects = []
    for name, filename, kind, (x, y, z), yaw in PLACEMENTS:
        path = HERE / "assets" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        angle = math.radians(yaw) / 2
        rotation = (0, 0, math.sin(angle), math.cos(angle))
        objects.append(Object(name=name, usd_path=str(path), object_type=ObjectType[kind],
            initial_pose=Pose(position_xyz=(x, y, TABLE_TOP + z), rotation_xyzw=rotation)))
    if test_probe:
        import isaaclab.sim as sim_utils
        from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg, PhysxCollisionPropertiesCfg
        objects.append(Object(name="test_probe", object_type=ObjectType.RIGID,
            spawner_cfg=sim_utils.SphereCfg(radius=.006,
                rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=.0003, rest_offset=0),
                mass_props=sim_utils.MassPropertiesCfg(mass=.01)),
            initial_pose=Pose(position_xyz=(.8, -.5, 1.5), rotation_xyzw=(0, 0, 0, 1))))
        objects[-1].disable_reset_pose()
    arena = IsaacLabArenaEnvironment(name="read_thermometer",
        scene=Scene([*task_room("station", robot.rig.top_cam), *objects]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=CONFIG["episode"]["seed"])
    return ArenaEnvBuilder(arena, build).make_registered()
