"""The calibrated YAM station with a procedural baking board and cutting tool."""
import json
from pathlib import Path

from geometry import BOARD_SIZE, BOARD_X, CUT_X, FLAT, HANDLE_NAMES, HANDLE_SIZE, HANDLE_X, HANDLE_Y
from geometry import MARK, RADIUS, ROLL_NAMES, TEAL, WIDTH, roll_asset

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
INSTRUCTION = (
    "Cut the cinnamon-roll log in two at its dark center mark without squashing the spirals. "
    "Pick up one blue floss grip in each hand. Keep the floss low and slide it under the log. "
    "Bring the ends up both sides, cross them above the log and pull them apart. "
    "Keep both loaded strands between the two guide marks, and draw the loop upward through the log. "
    "Leave both rolls on the board and put down the floss."
)


def make_env(out, cameras):
    import carb
    import isaaclab.sim as sim_utils
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

    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    sim = CONFIG["simulation"]
    render = CONFIG["renderer"]
    rig = station.rig(CONFIG["station"]["top_camera"])
    table = station.TABLE_TOP
    board = table + BOARD_SIZE[2]
    w, h = CONFIG["observation"]["render_size_wh"]
    yam = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig, cam_width=w,
                                 cam_height=h, contact_sensors=False)
    for name in yam.camera_config.camera_names():
        cfg = getattr(yam.camera_config, name)
        lens, camera = cfg.spawn.distortion, cfg.spawn
        camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
        camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
        camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
        camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
        cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=render["ambient_light_intensity"], carb_settings={
                "/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"],
            }))

    def box(name, size, pos, color, *, friction=.63, mass=None, collide=True):
        return Object(name=name, object_type=ObjectType.BASE if mass is None else ObjectType.RIGID,
                      initial_pose=Pose(pos, (0., 0., 0., 1.)), spawner_cfg=sim_utils.CuboidCfg(
            size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=.71),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0008, rest_offset=0.)
            if collide else None,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction,
                                                            dynamic_friction=friction*.84) if collide else None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=.8) if mass else None,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass else None))

    objects = [box("board", BOARD_SIZE, (BOARD_X, 0., table+BOARD_SIZE[2]/2), (.16, .22, .20), friction=.92)]
    for side in (-1, 1):
        for i, dx in enumerate((-.010, 0., .010)):
            objects.append(box(f"cut_mark{side+1}_{i}", (.0015, .035 if i == 1 else .022, .0002),
                               (CUT_X+dx, side*.062, board+.0001), MARK, collide=False))
    for name, side in zip(HANDLE_NAMES, (1, -1)):
        objects.append(box(name, HANDLE_SIZE, (HANDLE_X, side*HANDLE_Y, board+HANDLE_SIZE[2]/2),
                           TEAL, friction=.92, mass=.021))
    for name, side in zip(ROLL_NAMES, (-1, 1)):
        path = out / f"{name}.usda"
        roll_asset(path, -side)
        objects.append(Object(name=name, object_type=ObjectType.RIGID, usd_path=str(path),
                              initial_pose=Pose((CUT_X+side*WIDTH/2, 0., board+RADIUS-FLAT+.0003),
                                                (0., 0., 0., 1.))))

    def configure(cfg):
        assert sim["backend"] == "physx"
        cfg = env_cfg_callback(sim["control_hz"])(cfg)
        cfg.sim.dt = 1/sim["physics_hz"]
        cfg.decimation = sim["physics_hz"]//sim["control_hz"]
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = sim["device"]
        cfg.sim.gravity = tuple(sim["gravity_m_s2"])
        cfg.sim.use_fabric = sim["use_fabric"]
        cfg.sim.physics.solver_type = sim["solver_type"]
        cfg.episode_length_s = CONFIG["episode"]["max_control_steps"] / sim["control_hz"]
        cfg.sim.log_dir = str(out / "logs")
        cfg.log_dir = str(out / "logs")
        return cfg

    spec = IsaacLabArenaEnvironment(name="yam_cinnamon_roll_floss", embodiment=yam,
        scene=Scene([*task_room(CONFIG["station"]["scene"], rig.top_cam), *objects]),
        env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(
        num_envs=sim["num_envs"], solve_relations=False, disable_fabric=not sim["use_fabric"],
        device=sim["device"])).make_registered()
