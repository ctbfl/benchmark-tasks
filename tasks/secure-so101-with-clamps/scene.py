"""Bimanual YAM fixture and task objects."""

import json
from pathlib import Path

from geometry import EDGE_X, GAP_CENTER_Z, PARK_ANGLE, PARK_PIVOT_X, RACK_CLAMP_Y, START_OFFSET, TABLE_Z

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
HZ = CONFIG["simulation"]["control_hz"]
DT = 1 / CONFIG["simulation"]["physics_hz"]
MAX_STEPS = CONFIG["episode"]["max_control_steps"]
ARMS = ("left_robot", "right_robot")
CLAMPS = ("clamp_a", "clamp_b")
CAMERAS = tuple(CONFIG["observation"]["camera_names"])
ROOT = "/World/envs/env_0"
INSTRUCTION = (
    "Use your left hand to raise and steady the folded SO-101 from its forward-resting position. "
    "Keep supporting it while your right hand fits both spring clamps directly on the two base feet "
    "and under the small table edge. Squeeze each clamp's handles to open its jaws, position it, "
    "then relax the handles. Withdraw both hands and leave the arm upright and stable."
)


def make_env(out, cameras=False):
    import carb
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from scipy.spatial.transform import Rotation
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room

    cfg_sim = CONFIG["simulation"]
    renderer = CONFIG["renderer"]
    rig = station.rig(CONFIG["station"]["top_camera"])
    width, height = CONFIG["observation"]["render_size_wh"]
    embodiment = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig,
                                      cam_width=width, cam_height=height, robot_contact_offset=.001)
    if cameras:
        for name in CAMERAS:
            cfg = getattr(embodiment.camera_config, name)
            lens, cam = cfg.spawn.distortion, cfg.spawn
            cam.horizontal_aperture = cfg.width * cam.focal_length / lens.fx
            cam.vertical_aperture = cfg.height * cam.focal_length / lens.fy
            cam.horizontal_aperture_offset = (cfg.width/2 - lens.cx) * cam.focal_length / lens.fx
            cam.vertical_aperture_offset = (lens.cy - cfg.height/2) * cam.focal_length / lens.fy
            cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=renderer["ambient_light_intensity"], carb_settings={
                    "/rtx/rendermode": renderer["mode"],
                    "/rtx/post/aa/op": renderer["antialiasing_op"]}))

    def configure(env_cfg):
        env_cfg = env_cfg_callback(HZ)(env_cfg)
        env_cfg.sim.dt = DT
        env_cfg.decimation = round(1 / (HZ * DT))
        assert env_cfg.decimation * DT == 1 / HZ
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.device = cfg_sim["device"]
        env_cfg.sim.gravity = tuple(cfg_sim["gravity_m_s2"])
        env_cfg.sim.use_fabric = cfg_sim["use_fabric"]
        env_cfg.sim.physics.solver_type = cfg_sim["solver_type"]
        env_cfg.sim.log_dir = str(out)
        env_cfg.log_dir = str(out)
        return env_cfg

    fixtures = Object(name="fixture", object_type=ObjectType.BASE, usd_path=str(ASSETS / "table.usdc"))
    so101 = Object(name="so101", object_type=ObjectType.RIGID, usd_path=str(ASSETS / "so101.usdc"),
                   initial_pose=Pose((0.3879883289337158, -7.465807721018791e-05, 0.9463586807250977), (-0.0006314784404821694, 0.13699279725551605, 0.0011553093791007996, 0.9905712604522705)))
    rotation = Rotation.from_euler("y", -PARK_ANGLE / 2).as_quat()
    clamps = [Object(name=name, object_type=ObjectType.ARTICULATION, usd_path=str(ASSETS / "spring-clamp.usdc"),
                     initial_pose=Pose((PARK_PIVOT_X, y, GAP_CENTER_Z), tuple(map(float, rotation))))
              for name, y in zip(CLAMPS, RACK_CLAMP_Y)]
    for clamp in clamps:
        clamp.object_cfg.init_state.joint_pos = {"hinge": PARK_ANGLE}
    desc = IsaacLabArenaEnvironment(
        name="yam_secure_so101", embodiment=embodiment,
        scene=Scene([*task_room(CONFIG["station"]["scene"], rig.top_cam), fixtures, so101, *clamps]),
        env_cfg_callback=configure)
    assert cfg_sim["backend"] == "physx" and CONFIG["action"]["size"] == 14
    env = ArenaEnvBuilder(desc, ArenaEnvBuilderCfg(num_envs=cfg_sim["num_envs"], solve_relations=False,
                          disable_fabric=not cfg_sim["use_fabric"], device=cfg_sim["device"])).make_registered()
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    return env
