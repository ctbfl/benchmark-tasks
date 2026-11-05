"""YAM grasps the faucet and dispenses 150 g through physical finger contact."""

import json
import math
from pathlib import Path

import numpy as np

from config import CUP_FLOOR_Z, CUP_RADIUS, DENSITY_G_M3, DT, FPS, RESERVOIR_INITIAL_G, TOLERANCE_G
from water import WaterTask, flight_time, flow_g_s

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/yam-demo-multiview"
RECORD = True
TARGET_G = 150.0
OPEN_ANGLE = math.radians(-60)
FINE_ANGLE = math.radians(-20)
CLOSE_S = 0.6
GRIP_RADIUS = 0.027
ROBOT_CAMERAS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
                 ("right_cam", "Right Wrist Camera"))


def run():
    import carb
    import omni.physics.tensors as tensors
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, READY, TCP_OFFSET
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room
    from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix
    from model import WaterVisual
    from yam_preview import pathtraced_camera

    OUT.mkdir(parents=True, exist_ok=True)
    rig = station.rig()
    embodiment = BimanualYamEmbodiment(enable_cameras=RECORD, cam_height=720, cam_width=1280,
                                      contact_sensors=False,
                                      robot_contact_offset=0.0005, rig=rig)
    if RECORD:
        for name, _ in ROBOT_CAMERAS:
            pathtraced_camera(getattr(embodiment.camera_config, name))
    tea = Object(name="water_task", object_type=ObjectType.BASE,
                 usd_path=str(HERE/"assets/water-task.usda"),
                 initial_pose=Pose(position_xyz=(0.46, 0, station.TABLE_TOP),
                                   rotation_xyzw=(0, 0, -math.sqrt(0.5), math.sqrt(0.5))))

    def configure(cfg):
        cfg = env_cfg_callback(FPS)(cfg)
        cfg.sim.use_fabric = False
        cfg.sim.log_dir = cfg.log_dir = str(OUT/"logs")
        return cfg

    spec = IsaacLabArenaEnvironment(name="yam_water_150g", embodiment=embodiment,
        scene=Scene([*task_room("station", rig.top_cam), tea]), env_cfg_callback=configure)
    env = ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cpu")).make_registered()
    products, writers = [], []
    try:
        env.reset()
        base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        fill = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        fill.GetColorAttr().Set(Gf.Vec3f(1))
        fill.GetIntensityAttr().Set(900.)
        key = UsdLux.SphereLight(stage.GetPrimAtPath("/World/envs/env_0/lamp"))
        key.GetColorAttr().Set(Gf.Vec3f(1))
        key.GetIntensityAttr().Set(15000.)
        key.GetRadiusAttr().Set(0.30)
        root = "/World/envs/env_0/water_task"
        task = WaterTask(TARGET_G)
        visual = WaterVisual(stage, root)
        visual.update(task)
        physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        valve = physics.create_articulation_view(root+"/Dispenser")
        assert valve.count == 1 and valve.max_dofs == 1
        assert np.all(valve.get_dof_stiffnesses() == 0)
        finger_pattern = "/World/envs/env_0/LeftArm" + embodiment.link_paths["gripper"] + "/tip_*"
        contacts = physics.create_rigid_contact_view(finger_pattern, [root+"/Dispenser/Lever"])
        robot = base.scene["left_robot"]
        arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        gripper_id = robot.find_bodies("gripper")[0][0]
        pivot = np.array([0.335, 0., station.TABLE_TOP + 0.257])
        grip_rot = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
        q_command = np.array(READY, dtype=float)
        grip_command = 1.0
        rows, phases = [], []
        max_ik_position = max_ik_rotation = 0.
        mass_at_closed = airborne_at_closed = None

        if not RECORD:
            rep.settings.set_render_pathtraced(samples_per_pixel=32)
        rep.settings.carb_settings("/rtx/scenePartitioning/showAllPartitionsByDefault", True)
        for name, eye, target, focal, size in (
            ("observer", (-0.75, -0.55, 1.85), (0.23, 0, 1.02), 34., (1920, 1080)),
        ):
            camera = UsdGeom.Camera.Define(stage, "/DemoCameras/"+name)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse())
            camera.CreateFocalLengthAttr(focal)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
            product = rep.create.render_product(str(camera.GetPath()), size)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach([product])
            products.append((name, product, rgb))
        if RECORD:
            import imageio_ffmpeg
            from PIL import ImageDraw, ImageFont
            (OUT/"videos").mkdir(exist_ok=True)
            for name, size in (("four-camera-full", (3840, 2880)), ("four-camera-preview", (1920, 1440))):
                writer = imageio_ffmpeg.write_frames(str(OUT/"videos"/(name+".mp4")),
                    size, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                    macro_block_size=2, output_params=["-crf", "20", "-preset", "fast",
                                                       "-threads", "4", "-movflags", "+faststart"])
                writer.send(None)
                writers.append(writer)
            fonts = {scale: ImageFont.load_default(size=32*scale) for scale in (1, 2)}
            small_font = ImageFont.load_default(size=24)

        def annotate(pixels, title, subtitle=None, scale=1):
            panel = Image.fromarray(pixels)
            draw = ImageDraw.Draw(panel)
            font = fonts[scale]
            width = max(draw.textlength(title, font=font),
                        draw.textlength(subtitle, font=small_font) if subtitle else 0)
            draw.rectangle((0, 0, 32*scale+width, (84 if subtitle else 52)*scale),
                           fill=(20, 26, 34))
            draw.text((16*scale, 10*scale), title, font=font, fill="white")
            if subtitle:
                draw.text((16, 50), subtitle, font=small_font, fill=(175, 215, 240))
            return panel
        for _ in range(8):
            base.sim.render()
            ensure_isaac_rtx_render_update(force=True)

        def capture(label, save=None):
            visual.update(task)
            base.sim.render()
            ensure_isaac_rtx_render_update(force=True)
            pictures = {name: np.ascontiguousarray(rgb.get_data()[..., :3]) for name, _, rgb in products}
            if save:
                for name, pixels in pictures.items():
                    Image.fromarray(pixels).save(OUT/f"{save}-{name}.png")
            if RECORD:
                observer = annotate(pictures["observer"], "External Observer View", "Not a robot camera")
                frame = Image.new("RGB", (3840, 2880))
                frame.paste(observer.resize((3840, 2160), Image.Resampling.LANCZOS))
                for index, (name, title) in enumerate(ROBOT_CAMERAS):
                    camera = base.scene[name]
                    camera.update(1/FPS, force_recompute=True)
                    pixels = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                    assert pixels.shape == (720, 1280, 3), (name, pixels.shape)
                    frame.paste(annotate(pixels, title, scale=2), (index*1280, 2160))
                    if save:
                        Image.fromarray(pixels).save(OUT/f"{save}-{name}-720p.png")
                preview = frame.resize((1920, 1440), Image.Resampling.LANCZOS)
                if save is None:
                    writers[0].send(np.asarray(frame))
                    writers[1].send(np.asarray(preview))
                else:
                    preview.save(OUT/f"{save}.png")

        def advance(label):
            nonlocal mass_at_closed, airborne_at_closed
            action = torch.tensor([[*q_command, grip_command, *READY, 1.]], dtype=torch.float32)
            base.action_manager.process_action(action)
            for _ in range(round(1/(FPS*DT))):
                previous_angle = float(valve.get_dof_positions()[0, 0])
                base.action_manager.apply_action()
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(DT)
                angle = float(valve.get_dof_positions()[0, 0])
                task.step((previous_angle+angle)/2, DT)
                force = np.linalg.norm(contacts.get_contact_force_matrix(DT), axis=-1).reshape(-1)
                state = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy()
                assert np.isfinite(state).all() and math.isfinite(angle)
                assert task.cup_g < TARGET_G+5, "Overfilled cup"
                if label == "Close faucet" and task.flow == 0 and mass_at_closed is None:
                    mass_at_closed, airborne_at_closed = task.cup_g, task.airborne_g
                rows.append([task.time, angle, task.flow, task.cup_g, task.airborne_g,
                             task.emitted_g, task.spilled_g, *force, *state])
            if RECORD:
                capture(label)
                if round(task.time*FPS) % FPS == 0:
                    print(f"Recording {task.time:.0f}s: {label}, {task.reading_g:.1f} g", flush=True)

        def checkpoint(label):
            angle = float(valve.get_dof_positions()[0, 0])
            pose = robot.data.body_link_pose_w.torch[0, gripper_id].cpu().numpy()
            tcp = pose[:3] + quat_xyzw_to_matrix(pose[3:]) @ np.array(TCP_OFFSET)
            record = dict(label=label, time_s=task.time, reading_g=task.reading_g,
                          faucet_deg=math.degrees(angle), tcp=tcp.tolist())
            phases.append(record)
            print(json.dumps(record), flush=True)
            capture(label, f"{len(phases):02d}-{label.lower().replace(' ', '-')}")

        def solve_pose(pos, rot):
            nonlocal q_command, max_ik_position, max_ik_rotation
            q_command, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_command,
                                         base_pos=station.BASES["left"], iters=80)
            max_ik_position, max_ik_rotation = max(max_ik_position, pe), max(max_ik_rotation, re)
            assert pe < .003 and re < .03, ("Unreachable TCP", pos, pe, re)

        def move_pose(pos, rot, duration, label):
            start, r0 = ik.tcp_pose(q_command, station.BASES["left"])
            for i in range(round(duration*FPS)):
                s = ik.smoothstep((i+1)/(duration*FPS))
                solve_pose(start+s*(pos-start), ik.slerp_matrix(r0, rot, s))
                advance(label)
            checkpoint(label)

        def faucet_pose(angle):
            rotation = axis_angle_matrix(np.array([0.,0,1]), angle) @ grip_rot
            point = pivot + np.array([-GRIP_RADIUS*math.cos(angle), -GRIP_RADIUS*math.sin(angle), .005])
            return point, rotation

        def turn(start, end, duration, label):
            for i in range(round(duration*FPS)):
                angle = start+(end-start)*ik.smoothstep((i+1)/(duration*FPS))
                solve_pose(*faucet_pose(angle))
                advance(label)
            checkpoint(label)

        def hold(seconds, label):
            for _ in range(round(seconds*FPS)):
                advance(label)
            checkpoint(label)

        hold(.4, "Ready")
        grasp_pos, rot = faucet_pose(0)
        approach = grasp_pos + np.array([-.075, 0., 0.])
        move_pose(approach, rot, 2., "Approach faucet")
        move_pose(grasp_pos, rot, 1.2, "Align fingers")
        for i in range(round(.8*FPS)):
            grip_command = 1-ik.smoothstep((i+1)/(.8*FPS))
            advance("Grasp handle")
        checkpoint("Grasp handle")
        turn(0, OPEN_ANGLE, 1.6, "Open faucet")
        assert float(valve.get_dof_positions()[0,0]) < math.radians(-40), "Fingers did not turn the valve"
        while task.reading_g < 128:
            assert task.time < 30, "No sustained flow"
            advance("Fill cup")
        checkpoint("Fill cup")
        turn(OPEN_ANGLE, FINE_ANGLE, .8, "Reduce flow")
        while True:
            angle = float(valve.get_dof_positions()[0,0])
            surface = CUP_FLOOR_Z+task.reading_g/(DENSITY_G_M3*math.pi*CUP_RADIUS**2)
            closing_mass = sum(flow_g_s(angle*(1-ik.smoothstep((i+.5)*DT/CLOSE_S)))*DT
                               for i in range(round(CLOSE_S/DT)))
            pending = flow_g_s(angle)*(flight_time(surface)+.10) + closing_mass
            if task.reading_g+pending >= TARGET_G:
                break
            assert task.time < 35, "Fine fill timed out"
            advance("Fine fill")
        checkpoint("Fine fill")
        turn(FINE_ANGLE, 0, CLOSE_S, "Close faucet")
        hold(.6, "Wait for tail")
        for i in range(round(.6*FPS)):
            grip_command = ik.smoothstep((i+1)/(.6*FPS))
            advance("Release handle")
        checkpoint("Release handle")
        move_pose(approach, rot, 1.2, "Withdraw hand")
        move_pose(*ik.tcp_pose(np.array(READY), station.BASES["left"]), 2., "Return to ready")
        hold(1., "Settled")
        data = np.asarray(rows)
        error = np.abs(data[:,5]-data[:,3]-data[:,4]-data[:,6]).max()
        checks = {"task_success": task.success, "mass_conserved": bool(error < 1e-7),
                  "fingers_contacted_faucet": bool(data[:,7:9].max() > .1),
                  "valve_closed": abs(float(valve.get_dof_positions()[0,0])) < math.radians(10),
                  "airborne_empty": task.airborne_g == 0,
                  "tail_landed_after_closing": mass_at_closed is not None and airborne_at_closed > 0
                      and abs(task.cup_g-mass_at_closed-airborne_at_closed) < 1e-7,
                  "reservoir_conserved": abs(task.reservoir_g+task.emitted_g-RESERVOIR_INITIAL_G) < 1e-7}
        info = dict(target_g=TARGET_G, tolerance_g=TOLERANCE_G, final_g=task.cup_g, reading_g=task.reading_g,
                    error_g=task.cup_g-TARGET_G, checks=checks, phases=phases,
                    max_ik_position_mm=max_ik_position*1000, max_ik_rotation_deg=math.degrees(max_ik_rotation),
                    mass_at_closed_g=mass_at_closed, airborne_at_closed_g=airborne_at_closed,
                    peak_finger_contact_N=data[:,7:9].max(axis=0).tolist(),
                    max_mass_error_g=float(error), duration_s=task.time,
                    physics_dt_s=DT, control_hz=FPS, video_fps=FPS,
                    camera_resolutions={"observer": [1920, 1080], **{name: [1280, 720] for name, _ in ROBOT_CAMERAS}},
                    video_resolutions={"full": [3840, 2880], "preview": [1920, 1440]},
                    observer_is_robot_camera=False,
                    control="Scripted IK + physical finger contact; simulated scale feedback, no image OCR; no faucet actuation.")
        (OUT/"metrics.json").write_text(json.dumps(info, indent=2)+"\n")
        np.savetxt(OUT/"trajectory.csv", data, delimiter=",", comments="",
                   header="time_s,angle_rad,flow_g_s,cup_g,airborne_g,emitted_g,spilled_g,left_finger_N,right_finger_N,j1,j2,j3,j4,j5,j6")
        print(json.dumps(info, indent=2), flush=True)
        assert all(checks.values()), checks
    finally:
        for writer in writers:
            writer.close()
        for _, product, rgb in products:
            rgb.detach()
            product.destroy()
        env.close()


if __name__ == "__main__":
    import traceback
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=12 --/omni/tbb/globalcontrol/maxThreadCount=12").app
    try:
        run()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()
