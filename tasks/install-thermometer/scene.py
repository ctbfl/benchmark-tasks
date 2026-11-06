"""One calibrated YAM environment shared by modeling, demo, and evaluation."""
import json
from scipy.spatial.transform import Rotation
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
ROOT = "/World/envs/env_0"
# +Y is left in the calibrated camera; the table's long edge follows Y.
CUP_Y = .16
PLACEMENTS = (
    ("stand", "stand_clamp/stand.usda", "RIGID", (.43, .19, 0), (0, 0, 0)),
    ("beaker", "lab_glass/beaker.usda", "RIGID", (.43, CUP_Y, .014), (0, 0, 0)),
    ("clamp", "stand_clamp/clamp.usda", "ARTICULATION", (.27, -.16, .016), (0, 90, 0)),
    ("holder", "holder.usda", "RIGID", (.47, -.23, 0), (0, 0, 0)),
    ("thermometer", "lab_glass/thermometer.usda", "RIGID", (.47, -.23, .006), (0, 0, -90)),
)



def make_env(*, review=False):
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
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab_arena.terms.events import reset_articulation_joints
        cfg.events.reset_clamp_joints = EventTermCfg(func=reset_articulation_joints, mode="reset",
            params={"asset_cfg": SceneEntityCfg("clamp")})
        cfg.sim.log_dir = str(HERE / "runs/logs")
        return cfg

    objects = []
    for name, filename, kind, (x, y, z), angles in PLACEMENTS:
        path = HERE / "assets" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rotation = tuple(float(v) for v in Rotation.from_euler("xyz", angles, degrees=True).as_quat())
        objects.append(Object(name=name, usd_path=str(path), object_type=ObjectType[kind],
            initial_pose=Pose(position_xyz=(x, y, TABLE_TOP + z), rotation_xyzw=rotation)))
        if name == "clamp":
            objects[-1].object_cfg.init_state.joint_pos = {
                "jaw_hinge": 0.0, "screw_turn": 2 * 3.141592653589793, "screw_travel": .002}
    arena = IsaacLabArenaEnvironment(name="install_thermometer",
        scene=Scene([*task_room("station", robot.rig.top_cam), *objects]),
        embodiment=robot, env_cfg_callback=configure)
    build = ArenaEnvBuilderCfg(num_envs=sim["num_envs"], solve_relations=False,
        device=sim["device"], disable_fabric=not sim["use_fabric"], seed=CONFIG["episode"]["seed"])
    return ArenaEnvBuilder(arena, build).make_registered()
