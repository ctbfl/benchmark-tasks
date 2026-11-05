"""Load the tea dispenser assembly into the existing YAM station and take screenshots."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/yam-station-large-display"
PLACEMENT_XY = (0.46, 0.0)
PLACEMENT_YAW = -math.pi / 2  # The faucet faces the arms, along world -X.
TOP_RESOLUTIONS = ((1920, 1080),)
STILL_SAMPLES = 64


def pathtraced_camera(cfg):
    """Set render mode before product creation and express the calibration in USD too."""
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

    lens, camera = cfg.spawn.distortion, cfg.spawn
    camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
    camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
    camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
    camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
    cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=0., carb_settings={
            "/rtx/rendermode": "PathTracing", "/rtx/post/aa/op": 2,
            "/rtx/pathtracing/spp": 32, "/rtx/pathtracing/totalSpp": 32,
        }))
    return cfg


def run():
    import numpy as np
    import omni.physics.tensors as tensors
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab.sensors import CameraCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena import station
    from yam_arena.constants import READY
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from model import build_scale, display_reading

    OUT.mkdir(parents=True, exist_ok=True)
    scale_stage = Usd.Stage.CreateNew(str(HERE/"assets/scale.usda"))
    UsdGeom.SetStageMetersPerUnit(scale_stage, 1.0)
    UsdGeom.SetStageUpAxis(scale_stage, "Z")
    scale_stage.SetDefaultPrim(build_scale(scale_stage))
    scale_stage.GetRootLayer().Save()
    assembly_path = HERE / "assets/water-task.usda"
    assembly = Usd.Stage.CreateNew(str(assembly_path))
    UsdGeom.SetStageMetersPerUnit(assembly, 1.0)
    UsdGeom.SetStageUpAxis(assembly, "Z")
    assembly.SetDefaultPrim(UsdGeom.Xform.Define(assembly, "/WaterTask").GetPrim())
    for name in ("Dispenser", "Scale", "Cup", "Stand"):
        assembly.DefinePrim("/WaterTask/"+name, "Xform").GetReferences().AddReference(name.lower()+".usda")
    assembly.GetRootLayer().Save()

    rig = station.rig()
    embodiment = BimanualYamEmbodiment(enable_cameras=True, contact_sensors=False, rig=rig)
    pathtraced_camera(embodiment.camera_config.top_cam)
    extra_cameras = []
    for width, height in TOP_RESOLUTIONS:
        cfg = pathtraced_camera(embodiment._camera_cfg(height, width, overview=False).top_cam)
        cfg.prim_path = f"{{ENV_REGEX_NS}}/TopCam_{height}p"
        extra_cameras.append((f"top_cam_{height}p", CameraCfg, cfg))
    embodiment.camera_config = make_configclass(
        "TopResolutionComparisonCfg", extra_cameras, bases=(type(embodiment.camera_config),))()
    position = (*PLACEMENT_XY, station.TABLE_TOP)
    tea = Object(
        name="water_task", object_type=ObjectType.BASE, usd_path=str(assembly_path),
        initial_pose=Pose(position_xyz=position,
                          rotation_xyzw=(0, 0, math.sin(PLACEMENT_YAW/2), math.cos(PLACEMENT_YAW/2))),
    )

    def configure(cfg):
        cfg = env_cfg_callback()(cfg)
        cfg.sim.use_fabric = False
        cfg.sim.log_dir = str(OUT/"logs")
        cfg.log_dir = str(OUT/"logs")
        return cfg

    spec = IsaacLabArenaEnvironment(
        name="yam_water_dispensing_preview", embodiment=embodiment,
        scene=Scene([*task_room("station", rig.top_cam), tea]), env_cfg_callback=configure,
    )
    env = ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cpu")).make_registered()
    products = []
    try:
        env.reset()
        base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        # Neutral white fill and a softer key keep the frosted jar from looking dark red.
        fill = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        fill.GetColorAttr().Set(Gf.Vec3f(1.0))
        fill.GetIntensityAttr().Set(900.0)
        key = UsdLux.SphereLight(stage.GetPrimAtPath("/World/envs/env_0/lamp"))
        key.GetColorAttr().Set(Gf.Vec3f(1.0))
        key.GetRadiusAttr().Set(0.30)
        key.GetIntensityAttr().Set(15000.0)
        action = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=base.device)
        for _ in range(10):
            env.step(action)

        root = "/World/envs/env_0/water_task"
        physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        valve = physics.create_articulation_view(root+"/Dispenser")
        assert valve.count == 1 and valve.max_dofs == 1
        assert abs(float(valve.get_dof_positions()[0, 0])) < 1e-5
        assert np.all(valve.get_dof_stiffnesses() == 0)
        body = UsdGeom.Xformable(stage.GetPrimAtPath(root+"/Dispenser/Body"))
        body_position = body.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        assert np.allclose(body_position, position, atol=1e-5), (body_position, position)
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetPrimAtPath(root)).ComputeAlignedRange()
        assert abs(bbox.GetMin()[2]-station.TABLE_TOP) < 1e-5

        # Overview cameras must also see Arena's partitioned environment geometry.
        rep.settings.carb_settings("/rtx/scenePartitioning/showAllPartitionsByDefault", True)
        for name, eye, target, focal, size in (
            ("overview", (-1.45, -0.45, 1.85), (0.25, 0, 1.08), 30.0, (1600, 1000)),
            ("detail", (-0.40, -0.16, 1.37), (0.42, 0, 1.00), 34.0, (1200, 1000)),
        ):
            camera = UsdGeom.Camera.Define(stage, "/PreviewCameras/"+name)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
            camera.CreateFocalLengthAttr(focal)
            camera.CreateHorizontalApertureAttr(36.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20))
            product = rep.create.render_product(str(camera.GetPath()), size)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach([product])
            products.append((name, product, rgb, size))
        for _ in range(12):
            base.sim.render()
            ensure_isaac_rtx_render_update(force=True)
        for name, _, rgb, size in products:
            pixels = rgb.get_data()[..., :3]
            Image.fromarray(pixels).save(OUT/(name+".png"))
            assert pixels.shape[:2] == size[::-1] and pixels.std() > 5, (name, pixels.shape, pixels.std())
        camera_resolutions = {"overview": [1600, 1000], "detail": [1200, 1000]}
        def save_native_stills(names, suffix=""):
            # Integrate fresh samples of one unchanged scene, with no prior reading history.
            sums = {name: np.zeros((base.scene[name].cfg.height, base.scene[name].cfg.width, 3),
                                  dtype=np.float32) for name in names}
            for _ in range(STILL_SAMPLES):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
                for name in names:
                    camera = base.scene[name]
                    camera.update(base.step_dt, force_recompute=True)
                    sums[name] += camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            for name, total in sums.items():
                pixels = np.rint(total/STILL_SAMPLES).astype(np.uint8)
                assert pixels.std() > 5
                Image.fromarray(pixels).save(OUT/(name+suffix+".png"))

        top_intrinsics = base.scene["top_cam"].data.intrinsic_matrices.torch[0].cpu().numpy()
        for name in embodiment.camera_config.camera_names():
            camera = base.scene[name]
            camera_resolutions[name] = [camera.cfg.width, camera.cfg.height]
            if name.startswith("top_cam_"):
                scale = camera.cfg.width / 640
                intrinsics = camera.data.intrinsic_matrices.torch[0].cpu().numpy()
                assert np.allclose(intrinsics[:2], top_intrinsics[:2]*scale, atol=1e-3)
                assert np.allclose(camera.data.pos_w.torch, base.scene["top_cam"].data.pos_w.torch)
        save_native_stills(embodiment.camera_config.camera_names())

        # A varied display pattern makes thin segments easier to judge than all zeros.
        # These are display calibration images, not a robot dispensing episode.
        display_reading(stage, 75.1, root+"/Scale")
        bindings = {str(i): {c: str(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(
            f"{root}/Scale/Display/Digits/D{i}/{c}")).ComputeBoundMaterial()[0].GetPath())
            for c in "abcdefg"} for i in range(4)}
        (OUT/"display-bindings.json").write_text(json.dumps(bindings, indent=2)+"\n")
        for _ in range(8):
            env.step(action)
            ensure_isaac_rtx_render_update(force=True)
        for name, _, rgb, _ in products:
            Image.fromarray(rgb.get_data()[..., :3]).save(OUT/(name+"_75.1g.png"))
        save_native_stills([f"top_cam_{height}p" for _, height in TOP_RESOLUTIONS], "_large_75.1g")
        display_reading(stage, 0.0, root+"/Scale")
        stage.Export(str(OUT/"scene.usdc"))
        info = {"station": rig.name, "table_top_z": station.TABLE_TOP,
                "assembly_position": position, "assembly_yaw_degrees": math.degrees(PLACEMENT_YAW),
                "assembly_bounds": [list(bbox.GetMin()), list(bbox.GetMax())],
                "robot_articulations": list(base.scene.articulations),
                "lighting": {"color": [1, 1, 1], "dome_intensity": 900,
                             "key_intensity": 15000, "key_radius_m": 0.30},
                "camera_resolution": camera_resolutions,
                "display_calibration_reading_g": 75.1,
                "still_samples_per_image": STILL_SAMPLES,
                "faucet_dofs": valve.max_dofs, "faucet_angle_rad": float(valve.get_dof_positions()[0, 0]),
                "scope": "Loaded assets with both arms holding READY; no dispensing policy executed."}
        (OUT/"placement.json").write_text(json.dumps(info, indent=2)+"\n")
        print(json.dumps(info, indent=2), flush=True)
    finally:
        for _, product, rgb, _ in products:
            rgb.detach()
            product.destroy()
        env.close()


if __name__ == "__main__":
    import traceback
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        run()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()
