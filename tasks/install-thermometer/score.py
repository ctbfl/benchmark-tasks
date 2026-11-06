"""Score the four installation outcomes from an independently measured final window."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from scene import CONFIG
from yam_arena.station import TABLE_TOP

STABLE_SECONDS = 1.0


def local(pose, point):
    pose = np.asarray(pose)
    return Rotation.from_quat(pose[3:]).inv().apply(np.asarray(point) - pose[:3])


def world(pose, point):
    pose = np.asarray(pose)
    return pose[:3] + Rotation.from_quat(pose[3:]).apply(point)


def stable(samples, name):
    poses = np.array([s[name] for s in samples])
    rotation = Rotation.from_quat(poses[:, 3:])
    return bool(np.linalg.norm(poses[:, :3] - poses[-1, :3], axis=1).max() <= .002
                and (rotation[-1].inv() * rotation).magnitude().max() <= np.deg2rad(2))


def score(samples):
    result = {"clamp_fixed": False, "thermometer_fixed_vertical": False,
              "tip_immersed": False, "scale_clear": False}
    if not samples or samples[-1]["time_s"] - samples[0]["time_s"] < STABLE_SECONDS - 1e-7:
        return {"score": 0., "criteria": result}
    last = samples[-1]
    limit = CONFIG["episode"]["max_control_steps"] / CONFIG["simulation"]["control_hz"]
    if last["time_s"] > limit + 1e-7:
        return {"score": 0., "criteria": result}
    clamp, stand, thermometer, beaker = (last[name] for name in
                                        ("clamp", "stand", "thermometer", "beaker"))
    rod = local(clamp, world(stand, (0, .09, local(stand, clamp[:3])[2])))
    clamp_height = local(stand, clamp[:3])[2]
    released_clamp = all(s["clamp_robot_force_N"] < .01 for s in samples)
    stand_supported = all(abs(world(stand, (x, y, 0))[2] - TABLE_TOP) < .003
                          for x in (-.055, .055) for y in (-.105, .105))
    stand_released = all(s["stand_robot_force_N"] < .01 for s in samples)
    mounted = (abs(rod[0]) < .0015 and abs(rod[1]) < .003 and
               .024 < clamp_height < .43 and last["screw_travel"] < .0003)
    # The two physical mating surfaces must remain together after the robot leaves.
    result["clamp_fixed"] = bool(mounted and released_clamp and stand_supported and stand_released and stable(samples, "clamp")
                                 and stable(samples, "stand"))
    jaw = local(thermometer, world(clamp, (0, -.12, 0)))
    vertical = Rotation.from_quat(thermometer[3:]).apply([0, 0, 1])[2]
    held = np.linalg.norm(jaw[:2]) < .006 and .036 < jaw[2] < .312 and last["jaw_angle"] > -.12
    released_stem = all(s["thermometer_robot_force_N"] < .01 for s in samples)
    result["thermometer_fixed_vertical"] = bool(result["clamp_fixed"] and held and
        released_stem and stable(samples, "thermometer") and vertical >= np.cos(np.deg2rad(20)))
    tip = local(beaker, world(thermometer, (0, 0, .004)))
    result["tip_immersed"] = bool(result["thermometer_fixed_vertical"] and
        np.linalg.norm(tip[:2]) < .0375 and .005 < tip[2] < .078)
    scale_low, scale_high = (.065 + (t + 20) / 130 * .230 for t in (20, 60))
    # All clamp parts lie within 18 mm of its plane, including the round knob.
    stem_axis = Rotation.from_quat(thermometer[3:]).apply([0, 0, 1])
    clamp_axis = Rotation.from_quat(clamp[3:]).apply([0, 0, 1])
    alignment = float(np.dot(stem_axis, clamp_axis))
    extent = .018 + .14 * np.sqrt(max(0., 1 - alignment * alignment))
    result["scale_clear"] = bool(result["thermometer_fixed_vertical"] and
        (jaw[2] + extent < scale_low or jaw[2] - extent > scale_high))
    return {"score": sum(result.values()) / 4, "criteria": result,
            "measurements": {"socket_offset_m": rod.tolist(), "grip_height_m": float(jaw[2]),
                "tilt_deg": float(np.rad2deg(np.arccos(np.clip(vertical, -1, 1)))),
                "tip_in_beaker_m": tip.tolist()}}


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    out = Path(sys.argv[1])
    samples = json.loads((out / "measurements.json").read_text())
    final = samples[-1]["time_s"]
    result = score([s for s in samples if s["time_s"] >= final - STABLE_SECONDS - 1e-7])
    (out / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
