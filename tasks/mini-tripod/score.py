"""Rescore saved Dream measurements or a demo trajectory without API calls."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def demo_rows(run):
    from yam_arena import YAM_URDF, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.urdf import forward_kinematics

    trace = np.load(run / "trajectory.npz")
    layout = json.loads((ROOT / "assets/layout.json").read_text())
    vertices = dict(np.load(ROOT / "assets/collision_vertices.npz"))
    rows = []
    for i in range(max(0, len(trace["time"]) - 150), len(trace["time"])):
        pose, q = trace["hub_pose"][i], trace["tripod_q"][i]
        root = Rotation.from_quat(pose[3:])
        clouds = {"hub": root.apply(vertices["hub"]) + pose[:3]}
        for leg in range(3):
            phi = np.deg2rad(-90 + 120 * leg)
            pivot = .0078 * np.array([np.cos(phi), np.sin(phi), 0])
            frame = Rotation.from_euler("z", phi) * Rotation.from_euler("x", np.pi)
            frame = frame * Rotation.from_euler("y", q[leg])
            clouds[f"leg_{leg}"] = root.apply(frame.apply(vertices[f"leg_{leg}"]) + pivot) + pose[:3]
        cloud = np.concatenate(list(clouds.values()))
        lo, hi = cloud[:, :2].min(0), cloud[:, :2].max(0)
        hands = []
        for side, robot_q in zip(("left", "right"), trace["robot_q"][i]):
            frames = forward_kinematics(YAM_URDF, dict(zip((*ARM_JOINTS, *FINGER_JOINTS), robot_q, strict=True)))
            hands.extend(frames[n][:3, 3] + station.BASES[side] for n in ("gripper", "tip_left", "tip_right"))
        rows.append(dict(time_s=float(trace["time"][i]), hub_pose=pose.tolist(), hinges_deg=np.rad2deg(q).tolist(),
            tilt_deg=float(np.rad2deg(np.arccos(np.clip(root.as_matrix()[2, 2], -1, 1)))),
            tape_clearances_m=np.r_[lo - layout["goal_low_xy_m"], layout["goal_high_xy_m"] - hi].tolist(),
            center_error_m=float(np.linalg.norm((lo + hi) / 2 - layout["goal_center_xy_m"])),
            foot_heights_m=[float(clouds[f"leg_{leg}"][:, 2].min() - station.TABLE_TOP) for leg in range(3)],
            foot_contact_N=np.linalg.norm(trace["foot_contact"][i], axis=-1).reshape(3).tolist(),
            hand_contact_N=float(np.linalg.norm(trace["finger_contact"][i], axis=-1).max()),
            gripper_clearance_m=float(np.linalg.norm(cloud[:, None] - np.array(hands)[None], axis=-1).min())))
    return rows, 30


def score(rows, hz):
    count = 5 * hz
    window = rows[-count:]
    assert window, "No recorded states"
    angles = np.array([r["hinges_deg"] for r in window])
    poses = np.array([r["hub_pose"][:3] for r in window])
    drift = float(np.linalg.norm(np.ptp(poses, axis=0)))
    support = np.max([r["foot_contact_N"] for r in window], axis=0)
    checks = dict(
        within_120_seconds=0 < window[-1]["time_s"] <= 120,
        five_seconds_recorded=len(window) == count and np.allclose(np.diff([r["time_s"] for r in window]), 1 / hz),
        all_three_legs_fully_open=bool(np.max(np.abs(angles - 32)) < 1),
        upright=all(r["tilt_deg"] < 5 for r in window),
        inside_red_square=all(min(r["tape_clearances_m"]) > 0 for r in window),
        centered=all(r["center_error_m"] < .005 for r in window),
        feet_on_table=all(max(abs(x) for x in r["foot_heights_m"]) < .0015 for r in window),
        hands_clear=all(r["hand_contact_N"] < .01 and r["gripper_clearance_m"] > .02 for r in window),
        supported=bool(np.all(support > .03)), stable=drift < .002)
    return dict(score=int(all(checks.values())), success=bool(all(checks.values())),
        time_s=window[-1]["time_s"], final_hinges_deg=window[-1]["hinges_deg"],
        drift_last_5s_m=drift, foot_support_peak_N=support.tolist(), checks=checks)


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] in ("dream", "demo"), "Usage: python score.py dream|demo RUN_DIRECTORY"
    kind, run = sys.argv[1], Path(sys.argv[2])
    if kind == "dream":
        rows = [json.loads(line) for line in (run / "measurements.jsonl").read_text().splitlines()]
        hz = 20
    else:
        rows, hz = demo_rows(run)
    print(json.dumps(score(rows, hz), indent=2, allow_nan=False))
