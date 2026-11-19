"""Calibrated YAM workstation and the passive drawing kit."""

import json
from pathlib import Path

from geometry import CENTER, ORBIT, PAPER, PARK

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
SIM = CONFIG["simulation"]
EPISODE = CONFIG["episode"]
HZ = SIM["control_hz"]
DT = 1 / SIM["physics_hz"]
SUBSTEPS = SIM["physics_hz"] // HZ
CAMERAS = tuple(CONFIG["observation"]["camera_names"])
INSTRUCTION = (
    "Pick up the orange pen. Set its tip on the paper through the hole in the green wheel. "
    "Guide the pen around to draw one continuous closed rosette while the wheel stays meshed "
    "with the blue ring. It takes two circuits of the wheel center around the ring. Lift the "
    "pen when the line meets its starting point."
)
ASSET_ROOT = "/World/envs/env_0"


def make_env(cameras, log_dir, top_size=None):
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

    assert SIM["backend"] == "physx" and SUBSTEPS * HZ == SIM["physics_hz"]
    assert EPISODE["object_jitter_m"] == 0 and not EPISODE["advance_physics_during_api_wait"]
    rig = station.rig(CONFIG["station"]["top_camera"])
    embodiment = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig, contact_sensors=False)
    if cameras:
        for name in CAMERAS:
            cfg = getattr(embodiment.camera_config, name)
            assert [cfg.width, cfg.height] == CONFIG["observation"]["render_size_wh"]
            lens, camera = cfg.spawn.distortion, cfg.spawn
            camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
            camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
            camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
            camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
            if name == "top_cam" and top_size:
                sx, sy = top_size[0]/cfg.width, top_size[1]/cfg.height
                cfg.width, cfg.height = top_size
                camera.distortion = lens.replace(fx=lens.fx*sx, fy=lens.fy*sy,
                    cx=lens.cx*sx, cy=lens.cy*sy, image_size=top_size)
            render = CONFIG["renderer"]
            cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=render["ambient_light_intensity"],
                carb_settings={"/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"]}))

    z = station.TABLE_TOP + .0002
    objects = []
    for name, offset, dz in (
        ("board", (0., 0.), 0.),
        ("ring", (0., 0.), PAPER),
        ("wheel", (ORBIT, 0.), PAPER+.00008),
        ("pen", PARK, .01008),
    ):
        objects.append(Object(name=name, object_type=ObjectType.RIGID,
            usd_path=str(HERE / "assets" / f"{name}.usda"),
            initial_pose=Pose((CENTER[0]+offset[0], CENTER[1]+offset[1], z+dz), (0., 0., 0., 1.))))

    def configure(cfg):
        cfg = env_cfg_callback(HZ)(cfg)
        cfg.sim.dt = DT
        cfg.decimation = cfg.sim.render_interval = SUBSTEPS
        cfg.sim.device = SIM["device"]
        cfg.sim.gravity = tuple(SIM["gravity_m_s2"])
        cfg.sim.use_fabric = SIM["use_fabric"]
        cfg.sim.physics.solver_type = SIM["solver_type"]
        cfg.episode_length_s = EPISODE["max_control_steps"]/HZ
        cfg.sim.log_dir = cfg.log_dir = str(log_dir)
        return cfg

    spec = IsaacLabArenaEnvironment(name="yam_spirograph", embodiment=embodiment,
        scene=Scene([*task_room(CONFIG["station"]["scene"], rig.top_cam), *objects]), env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=SIM["num_envs"],
        solve_relations=False, disable_fabric=not SIM["use_fabric"], device=SIM["device"])).make_registered()
