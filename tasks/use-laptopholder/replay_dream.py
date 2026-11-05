"""Render the archived Dream trajectory; no inference and no new physics steps."""

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zlib

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FPS = 30


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render(run, check_only):
    schema = json.loads((run / "replay-schema.json").read_text())
    manifest = json.loads((run / "run.json").read_text())
    for name in ("model.py", "runtime.py", "benchmark.py", "task.py", "camera_ready.py"):
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == manifest["source_sha256"][name], name
    columns = schema["columns"]
    states = np.memmap(run / "physics-f64.bin", mode="r", dtype=schema["dtype"]).reshape(-1, len(columns))
    assert np.isfinite(states).all()
    assert np.allclose(np.diff(states[:, 0]), schema["physics_dt"], atol=1e-12, rtol=0)
    stride = round(1 / (FPS * schema["physics_dt"]))
    assert abs(stride * schema["physics_dt"] - 1 / FPS) < 1e-12
    marks = {round(s / schema["physics_dt"]) for s in (0, 24, 60, 90, 120)}
    assert max(marks) == len(states)-1
    indices = sorted(marks) if check_only else range(0, len(states), stride)
    out = run / ("replay-check" if check_only else "camera-ready")
    out.mkdir()
    shutil.copy2(__file__, out / "replay_dream_source.py")
    video = run / "videos/dream-four-camera.mp4"
    if not check_only:
        assert not video.exists(), video
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args="--/plugins/carb.tasking.plugin/threadCount=8 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    camera, writer, env = None, None, None
    exit_code = 0
    try:
        import imageio_ffmpeg
        import omni.physx
        import omni.timeline
        import torch
        from benchmark import LaptopHolderTask
        from camera_ready import Recorder

        env = LaptopHolderTask(cameras=False)
        camera = Recorder(env, out, record=False)
        assert camera.cameras == schema["cameras"], "Camera calibration changed"
        assert np.allclose(camera.observer_pose, schema["observer_fixed_world_transform"], atol=1e-12)
        assert env.holder.names == schema["holder_body_order"]
        robot_columns = [([columns.index(f"{side}.pos.{name}") for name in robot.joint_names],
                          [columns.index(f"{side}.vel.{name}") for name in robot.joint_names])
                         for side, robot in zip(("left", "right"), env.robots)]
        root = [i for i, name in enumerate(columns) if name.startswith("holder.root.")]
        root_vel = [i for i, name in enumerate(columns) if name.startswith("holder.root_vel.")]
        dof = [i for i, name in enumerate(columns) if name.startswith("holder.dof_pos.")]
        dof_vel = [i for i, name in enumerate(columns) if name.startswith("holder.dof_vel.")]
        body_columns = [[columns.index(f"holder.{name}.{k}") for k in
                         ("x", "y", "z", "qx", "qy", "qz", "qw")] for name in env.holder.names]
        original_frames = next((run / "eval/frames").iterdir())
        ids = np.array([0], dtype=np.uint32)
        physics_count = env.sim._physics_step_count
        comparisons, errors = [], []
        if not check_only:
            writer = imageio_ffmpeg.write_frames(str(video), (1920, 1440), fps=FPS,
                codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
                output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                               "-threads", "4", "-movflags", "+faststart"])
            writer.send(None)
        started = time.perf_counter()
        for frame, index in enumerate(indices):
            row = states[index]
            omni.timeline.get_timeline_interface().set_current_time(float(row[0]))
            for robot, (pos, vel) in zip(env.robots, robot_columns):
                robot.write_joint_state_to_sim_index(
                    position=torch.tensor(row[pos][None], dtype=torch.float32),
                    velocity=torch.tensor(row[vel][None], dtype=torch.float32))
            env.art.set_root_transforms(row[root][None].astype(np.float32), ids)
            env.art.set_root_velocities(row[root_vel][None].astype(np.float32), ids)
            env.art.set_dof_positions(row[dof][None].astype(np.float32), ids)
            env.art.set_dof_velocities(row[dof_vel][None].astype(np.float32), ids)
            env.holder.view.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            actual = env.holder.bodies.get_transforms()
            target = row[body_columns]
            position_error = float(np.linalg.norm(actual[:, :3]-target[:, :3], axis=1).max())
            # q and -q represent the same rotation.
            rotation_error = float(np.minimum(np.linalg.norm(actual[:, 3:]-target[:, 3:], axis=1),
                                             np.linalg.norm(actual[:, 3:]+target[:, 3:], axis=1)).max())
            assert position_error < .0001 and rotation_error < .001, (index, position_error, rotation_error)
            errors.append((position_error, rotation_error))
            for robot in env.robots:
                robot.update(1/FPS)
            for _ in range(3 if frame == 0 else 1):
                images = camera.read_views()
            assert env.sim._physics_step_count == physics_count, "Replay advanced physics"
            composite = camera.compose(images)
            if writer is not None:
                writer.send(np.asarray(composite))
            camera.ticks = index
            camera.frames.append(dict(index=frame, physics_tick=index, time_s=float(row[0]),
                crc32={name: zlib.crc32(rgb) for name, rgb in images.items()}))
            if index in marks:
                label = f"{row[0]:07.3f}s"
                folder = out / "raw_frames" / label
                folder.mkdir()
                for name, rgb in images.items():
                    Image.fromarray(rgb).save(folder / (name + ".png"))
                composite.save(out / "screenshots" / (label + ".png"))
                tick = round(row[0] * manifest["control_hz"])
                for name, archived in (("left_wrist", "left_cam"), ("top", "top_cam"), ("right_wrist", "right_cam")):
                    original = np.load(next(original_frames.glob(f"*_{archived}_{tick:06d}.npy")))
                    mae = float(np.abs(images[name].astype(float)-original.astype(float)).mean())
                    comparisons.append(dict(time_s=float(row[0]), camera=name, native_rgb_mae=mae))
            if frame % 150 == 0 or frame == len(indices)-1:
                progress = dict(frames=frame+1, total_frames=len(indices), time_s=float(row[0]),
                                wall_s=time.perf_counter()-started)
                write_json(out / "progress.json", progress)
                print(f"Replay {frame+1}/{len(indices)} frames; {row[0]:.1f}s; "
                      f"wall {progress['wall_s']:.1f}s", flush=True)
        write_json(out / "render.json", dict(passed=True, api_calls=0, replay_physics_steps=0,
            trajectory_sha256=hashlib.sha256((run / "physics-f64.bin").read_bytes()).hexdigest(),
            fps=FPS, frames=len(indices), recorded_duration_s=float(states[-1, 0]),
            video=None if check_only else str(video), renderer="RealTimePathTracing",
            max_holder_position_error_m=max(e[0] for e in errors),
            max_holder_quaternion_error=max(e[1] for e in errors), native_frame_comparisons=comparisons,
            wall_render_s=time.perf_counter()-started,
            note="Replay of measured states; original model inputs were 320x176 at 20 Hz."))
        print("REPLAY CHECK PASSED" if check_only else "REPLAY RENDER COMPLETE", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        if camera is not None:
            camera.close()
        if env is not None:
            env.sim.stop()
        app.close(exit_code=exit_code)
    return exit_code


def verify(run):
    out = run / "camera-ready"
    recording = json.loads((out / "recording.json").read_text())
    render_info = json.loads((out / "render.json").read_text())
    schema = json.loads((run / "replay-schema.json").read_text())
    frames = json.loads((out / "frame_times.json").read_text())
    assert render_info["passed"] and render_info["api_calls"] == render_info["replay_physics_steps"] == 0
    assert hashlib.sha256((run / "physics-f64.bin").read_bytes()).hexdigest() == render_info["trajectory_sha256"]
    assert len(frames) == recording["frames"] == render_info["frames"] == 3601
    assert frames[0]["time_s"] == 0 and frames[-1]["time_s"] == 120
    assert np.all(np.diff([f["physics_tick"] for f in frames]) == 32)
    assert np.allclose(np.diff([f["time_s"] for f in frames]), 1/FPS, atol=1e-12, rtol=0)
    assert recording["labels"] == {"title_px": 32, "subtitle_px": 24, "outer_margin": 0}
    assert recording["cameras"] == schema["cameras"]
    assert recording["observer_fixed_world_transform"] == schema["observer_fixed_world_transform"]
    assert recording["renderer"] == "RealTimePathTracing"
    for name, camera in recording["cameras"].items():
        assert camera["resolution"] == ([1920, 1080] if name == "observer" else [640, 360])
        assert len({f["crc32"][name] for f in frames}) > 30
        for folder in (out / "raw_frames").iterdir():
            with Image.open(folder / (name + ".png")) as picture:
                assert list(picture.size) == camera["resolution"]
    video = Path(render_info["video"])
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                       "-show_format", "-of", "json", str(video)], text=True))
    assert len(probe["streams"]) == 1
    stream = probe["streams"][0]
    assert [stream["width"], stream["height"]] == [1920, 1440]
    assert stream["codec_name"] == "h264" and stream["profile"] == "High" and stream["pix_fmt"] == "yuv420p"
    assert stream["avg_frame_rate"] == "30/1" and int(stream["nb_frames"]) == len(frames)
    assert abs(float(probe["format"]["duration"]) - len(frames)/FPS) < .002
    atoms = []
    with video.open("rb") as file:
        while file.tell() < video.stat().st_size:
            size, kind = struct.unpack(">I4s", file.read(8))
            assert size >= 8
            atoms.append(kind.decode("ascii"))
            file.seek(size-8, 1)
    assert atoms.index("moov") < atoms.index("mdat")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", "4", "-i", str(video),
                    "-an", "-f", "null", "-"], check=True)
    review = out / "review"
    review.mkdir(exist_ok=True)
    comparisons = []
    for seconds in (0, 60, 120):
        path = review / f"{seconds:03d}s.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "4", "-ss", str(seconds),
                        "-i", str(video), "-frames:v", "1", "-update", "1", str(path)], check=True)
        decoded = np.asarray(Image.open(path)).astype(float)
        original = np.asarray(Image.open(out / "screenshots" / f"{seconds:07.3f}s.png")).astype(float)
        mae = float(np.abs(decoded-original).mean())
        assert mae < 4, (seconds, mae)
        comparisons.append(dict(time_s=seconds, decoded_vs_composite_rgb_mae=mae))
    report = dict(passed=True, video=str(video), bytes=video.stat().st_size, MB=video.stat().st_size/1e6,
        resolution=[1920, 1440], fps=FPS, frames=len(frames), duration_s=len(frames)/FPS,
        full_decode_passed=True, faststart=True, calibrated_cameras_preserved=True,
        native_sizes_verified=True, synchronized_frame_clock=True, comparisons=comparisons,
        max_holder_position_error_m=render_info["max_holder_position_error_m"],
        wall_render_s=render_info["wall_render_s"])
    write_json(out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) in (2, 3) and (len(sys.argv) == 2 or sys.argv[2] in ("--check", "--verify")), \
        "Usage: python replay_dream.py RUN_DIR [--check|--verify]"
    run = Path(sys.argv[1]).resolve()
    if sys.argv[-1] == "--verify":
        verify(run)
    else:
        sys.exit(render(run, len(sys.argv) == 3))
