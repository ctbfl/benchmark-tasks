"""Calibrated YAM station: a flat folded tripod and a precisely sized tape goal."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/yam-scene"
if __name__=="__main__":
    from isaaclab.app import AppLauncher
    (OUT/"logs").mkdir(parents=True,exist_ok=True)
    app = AppLauncher(headless=True,enable_cameras=True,kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6 "
        f"--/log/file={OUT}/logs/kit.log").app

import numpy as np
from scipy.spatial.transform import Rotation

GOAL_CENTER = np.array([.36, 0.])
START_HUB_XY = np.array([.30, .19])
GOAL_YAW_DEG = 15.  # Equal X/Y extents of the three-leg footprint.
CLEARANCE = .010
TAPE_WIDTH = .012
DT = 1 / 480


def asset_points(path):
    """Visible mesh vertices in the articulation root link (hub) frame."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    hub = stage.GetPrimAtPath("/MiniTripod/hub")
    origin = np.array(cache.GetLocalToWorldTransform(hub).ExtractTranslation())
    points = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            matrix = np.array(cache.GetLocalToWorldTransform(prim))
            points.append(mesh @ matrix[:3,:3] + matrix[3,:3] - origin)
    assert points
    return np.concatenate(points)


def layout():
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics
    from yam_arena import station
    from model import ASSETS, material, mesh_prim

    opened = asset_points(ASSETS / "mini_tripod_open.usda")
    yaw = Rotation.from_euler("z", GOAL_YAW_DEG, degrees=True)
    rotated = yaw.apply(opened)
    low, high = rotated[:,:2].min(axis=0), rotated[:,:2].max(axis=0)
    footprint = high-low
    assert abs(footprint[0]-footprint[1]) < .0001
    side = float(footprint.max()+2*CLEARANCE)
    goal_low, goal_high = GOAL_CENTER-side/2, GOAL_CENTER+side/2
    # Center the footprint's bounding square, which differs from the hub center.
    goal_hub = np.r_[GOAL_CENTER-(low+high)/2, station.TABLE_TOP-rotated[:,2].min()]
    placed = rotated+goal_hub
    margins = np.r_[placed[:,:2].min(axis=0)-goal_low, goal_high-placed[:,:2].max(axis=0)]
    assert np.allclose(margins,CLEARANCE,atol=.0001), margins

    flat = Rotation.from_euler("y",90,degrees=True)*Rotation.from_euler("z",30,degrees=True)
    folded = flat.apply(asset_points(ASSETS / "mini_tripod.usda"))
    start = np.r_[START_HUB_XY,station.TABLE_TOP-folded[:,2].min()+.001]
    assert (folded+start)[:,1].min() > goal_high[1]+.015

    path = ASSETS / "yam_goal.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,"Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
    root = UsdGeom.Xform.Define(stage,"/Goal").GetPrim()
    red = material(stage,"/Goal/Looks/RedTape",(.55,.008,.006),roughness=.85)
    w, s = TAPE_WIDTH, side
    for i,(size,center) in enumerate((
        ((s+2*w,w,.0002),(0,-(s+w)/2,.0001)),
        ((s+2*w,w,.0002),(0, (s+w)/2,.0001)),
        ((w,s,.0002),(-(s+w)/2,0,.0001)),
        ((w,s,.0002),( (s+w)/2,0,.0001)),
    )):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(center)
        mesh_prim(stage,f"/Goal/edge_{i}",mesh,red,collision=False)
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    return {"goal_inner_side_m":side,"tape_width_m":w,"goal_center_xy_m":GOAL_CENTER.tolist(),
        "goal_low_xy_m":goal_low.tolist(),"goal_high_xy_m":goal_high.tolist(),
        "fully_open_footprint_xy_m":footprint.tolist(),"goal_yaw_degrees":GOAL_YAW_DEG,
        "goal_hub_position_m":goal_hub.tolist(),"goal_hub_quaternion_xyzw":yaw.as_quat().tolist(),
        "clearances_xlow_ylow_xhigh_yhigh_m":margins.tolist(),
        "initial_hub_position_m":start.tolist(),"initial_quaternion_xyzw":flat.as_quat().tolist()}


def spawn_tripod(prim_path,cfg,translation=None,orientation=None):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd,UsdPhysics

    prim=spawn_from_usd(prim_path,cfg,translation,orientation)
    # Keep contact reports without disabling the asset's normal sleep behavior.
    for body in Usd.PrimRange(prim):
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            body.RemoveProperty("physxRigidBody:sleepThreshold")
    return prim


def make_env(enable_cameras=True,control_rate=10):
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
    from model import ASSETS

    info = layout()
    rig = station.rig()
    embodiment = BimanualYamEmbodiment(enable_cameras=enable_cameras,rig=rig,cam_width=640,cam_height=360,
        robot_contact_offset=.0005)
    for name in embodiment.camera_config.camera_names():
        cfg = getattr(embodiment.camera_config,name)
        lens, camera = cfg.spawn.distortion, cfg.spawn
        camera.horizontal_aperture = cfg.width*camera.focal_length/lens.fx
        camera.vertical_aperture = cfg.height*camera.focal_length/lens.fy
        camera.horizontal_aperture_offset = (cfg.width/2-lens.cx)*camera.focal_length/lens.fx
        camera.vertical_aperture_offset = (lens.cy-cfg.height/2)*camera.focal_length/lens.fy
        cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=0.,carb_settings={"/rtx/rendermode":"RealTimePathTracing",
                "/rtx/post/aa/op":2}))
    tripod = Object(name="mini_tripod",object_type=ObjectType.ARTICULATION,
        usd_path=str(ASSETS/"mini_tripod.usda"),
        spawn_cfg_addon={"func":spawn_tripod},
        initial_pose=Pose(tuple(info["initial_hub_position_m"]),tuple(info["initial_quaternion_xyzw"])))
    goal = Object(name="goal",object_type=ObjectType.BASE,usd_path=str(ASSETS/"yam_goal.usda"),
        initial_pose=Pose((*GOAL_CENTER,station.TABLE_TOP),(0,0,0,1)))

    def configure(cfg):
        cfg = env_cfg_callback(control_rate)(cfg)
        cfg.sim.dt = DT
        cfg.decimation = round(1/DT/control_rate)
        assert np.isclose(cfg.decimation*DT*control_rate,1)
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.physics.solver_type = 1
        cfg.sim.physics.enable_external_forces_every_iteration = True
        cfg.sim.use_fabric = False
        cfg.sim.log_dir = str(OUT/"logs")
        cfg.log_dir = str(OUT/"logs")
        return cfg

    description = IsaacLabArenaEnvironment(name="yam_mini_tripod",embodiment=embodiment,
        scene=Scene([*task_room("station",rig.top_cam),tripod,goal]),env_cfg_callback=configure)
    env = ArenaEnvBuilder(description,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,
        disable_fabric=True,device="cpu")).make_registered()
    return env, info


def run():
    import omni.physics.tensors as tensors
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena import station
    from yam_arena.constants import READY
    from model import ASSETS

    env, info = make_env()
    base = env.unwrapped
    try:
        env.reset()
        action = torch.tensor([[*READY,1.,*READY,1.]],dtype=torch.float32,device=base.device)
        for _ in range(30):
            env.step(action)
        tripod = base.scene["mini_tripod"]
        assert tripod.num_bodies==4 and tripod.num_joints==3 and not tripod.is_fixed_base
        pose = tripod.data.root_link_pose_w.torch[0].cpu().numpy()
        q = tripod.data.joint_pos.torch[0].cpu().numpy()
        velocity = tripod.data.root_link_vel_w.torch[0].cpu().numpy()
        hinges = [i for i,n in enumerate(tripod.joint_names) if n.startswith("fold")]
        assert len(hinges)==3 and np.abs(q[hinges]).max()<np.deg2rad(3)
        rotation = Rotation.from_quat(pose[3:])
        axis = rotation.apply([0,0,1])
        assert abs(axis[2])<.1, ("Tripod is not lying flat",axis)
        assert np.linalg.norm(velocity[:3])<.005 and np.linalg.norm(velocity[3:])<.05

        physics = tensors.create_simulation_view("numpy",stage_id=get_current_stage_id())
        view = physics.create_articulation_view("/World/envs/env_0/mini_tripod")
        points = dict(np.load(ASSETS/"collision_vertices.npz"))
        poses = view.get_link_transforms()[0]
        cloud = np.concatenate([Rotation.from_quat(p[3:]).apply(points[n])+p[:3]
            for n,p in zip(view.shared_metatype.link_names,poses)])
        assert station.TABLE_TOP-.001<cloud[:,2].min()<station.TABLE_TOP+.001
        assert cloud[:,2].max()-station.TABLE_TOP<.04
        assert np.all(view.get_dof_stiffnesses()==0) and np.all(view.get_dof_dampings()==0)
        before = pose.copy()
        for _ in range(10):
            env.step(action)
        after = tripod.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        drift = float(np.linalg.norm(after[:3]-before[:3]))
        assert drift<.001

        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage,"/OverviewCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition",Sdf.ValueTypeNames.Token).Set("env_0")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.35,-1.55,1.95),Gf.Vec3d(.20,0,1.02),Gf.Vec3d(0,0,1)).GetInverse())
        camera.CreateFocalLengthAttr(36)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20))
        product = rep.create.render_product(str(camera.GetPath()),(1600,1100))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
        rgb.attach([product])
        for _ in range(12):
            base.sim.render()
            ensure_isaac_rtx_render_update(force=True)
        pixels = rgb.get_data()[...,:3]
        assert pixels.shape==(1100,1600,3) and pixels.std()>5
        Image.fromarray(pixels).save(OUT/"overview.png")
        cameras = {}
        for name in ("top_cam","left_cam","right_cam"):
            sensor = base.scene[name]
            sensor.update(base.step_dt,force_recompute=True)
            pixels = sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy()
            assert pixels.shape==(360,640,3) and pixels.std()>5
            Image.fromarray(pixels).save(OUT/(name+".png"))
            cameras[name] = {"intrinsics":sensor.data.intrinsic_matrices.torch[0].cpu().tolist(),
                "position_world_m":sensor.data.pos_w.torch[0].cpu().tolist(),
                "resolution":[640,360],"prim_path":sensor.cfg.prim_path}
        rgb.detach()
        product.destroy()
        stage.Export(str(OUT/"scene.usdc"))
        info.update({"station":"station-calibrated","physics":"CPU TGS 480 Hz",
            "table_top_m":station.TABLE_TOP,"settled_root_pose_xyzw":after.tolist(),
            "joint_names":tripod.joint_names,"joint_positions":q.tolist(),
            "height_above_table_m":float(cloud[:,2].max()-station.TABLE_TOP),
            "lowest_surface_above_table_m":float(cloud[:,2].min()-station.TABLE_TOP),
            "drift_last_1s_m":drift,"cameras":cameras,"robot_action":action[0].cpu().tolist(),
            "screenshots":["overview.png","top_cam.png","left_cam.png","right_cam.png"]})
        (OUT/"setup.json").write_text(json.dumps(info,indent=2)+"\n")
        print(json.dumps(info,indent=2),flush=True)
    finally:
        env.close()


if __name__=="__main__":
    import traceback
    try:
        run()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
    else:
        app.close()
