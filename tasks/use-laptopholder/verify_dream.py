"""Check a completed Dream archive without simulation or inference."""

import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from inspect_robots_dream._station import jpeg
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(run):
    manifest, metrics, schema = (read(run / name) for name in
                                ("run.json", "metrics.json", "replay-schema.json"))
    assert metrics["exit_code"] == 0 and not manifest["check_only"]
    n, hz = metrics["control_steps"], manifest["control_hz"]
    assert n > 32 and (metrics["success"] or n == manifest["max_steps"])
    assert metrics["time_s"] == n / hz
    observations = lines(run / "dream/observations.jsonl")
    replies = lines(run / "dream/replies.jsonl")
    actions = lines(run / "executed-actions.jsonl")
    measurements = lines(run / "measurements.jsonl")
    assert len(observations) == len(actions) == len(measurements) == n
    stride = round(1 / (hz * schema["physics_dt"]))
    assert abs(stride * schema["physics_dt"] - 1 / hz) < 1e-12
    states = np.memmap(run / "physics-f64.bin", mode="r", dtype=schema["dtype"]).reshape(-1, len(schema["columns"]))
    assert len(states) == metrics["physics_rows"] == n * stride + 1
    assert np.isfinite(states).all()
    assert np.allclose(np.diff(states[:, 0]), schema["physics_dt"], atol=1e-12, rtol=0)
    assert abs(states[-1, 0] - n/hz) < 1e-12
    for i, (obs, action, measurement) in enumerate(zip(observations, actions, measurements)):
        assert obs["tick"] == action["step"] == i
        assert obs["time_s"] == action["time_s"] == i/hz
        assert measurement["time_s"] == (i+1)/hz
        assert obs["prompt"] == manifest["instruction"]
        assert set(obs["frames"]) == set(manifest["input_cameras"])
    next_tick = 32
    for reply in replies:
        assert reply["observation_tick"] == next_tick
        rows = np.asarray(reply["rows"])
        assert rows.shape == (48, 14) and np.isfinite(rows).all()
        next_tick += len(rows)
    assert n <= next_tick < n + 48
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == digest
    frame_dir = next((run / "eval/frames").iterdir())
    compared = []
    for camera in manifest["input_cameras"]:
        frames = sorted(frame_dir.glob(f"*_{camera}_*.npy"))
        jpegs = sorted((run / "dream" / camera).glob("*.jpg"))
        assert len(frames) == n+1 and len(jpegs) == n
        for tick in (0, n//2, n-1):
            rgb = np.load(frames[tick])
            assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8
            saved = run / "dream" / observations[tick]["frames"][camera]
            assert base64.b64decode(jpeg(rgb)) == saved.read_bytes()
            assert Image.open(saved).size == (320, 176)
            compared.append(dict(camera=camera, tick=tick, exact_jpeg_match=True))
    columns = schema["columns"]
    joint_state = []
    for side in ("left", "right"):
        arm = [columns.index(f"{side}.pos.{name}") for name in ARM_JOINTS]
        fingers = [columns.index(f"{side}.pos.{name}") for name in FINGER_JOINTS]
        joint_state.extend(states[::stride, arm].T)
        joint_state.append(np.clip(states[::stride, fingers].mean(axis=1)/FINGER_OPEN, 0, 1))
    joint_state = np.asarray(joint_state).T
    assert np.allclose(joint_state[:-1], [o["joint_pos"] for o in observations], atol=1e-7)
    log = read(next((run / "eval").glob("*.json")))
    assert log["status"] == "success" and log["stats"]["total_steps"] == n
    report = dict(passed=True, control_steps=n, physics_rows=len(states),
        input_jpegs_per_camera=n, native_frames_per_camera=n+1, reply_chunks=len(replies),
        returned_action_rows=sum(len(r["rows"]) for r in replies),
        unexecuted_model_rows=next_tick-n, exact_jpeg_checks=compared,
        source_hashes_verified=True, observed_joints_match_physics=True)
    (run / "archive-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python verify_dream.py <run-directory>"
    verify(Path(sys.argv[1]).resolve())
