"""Replay a matched material comparison using the robot probe's measured motion."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from honey import Honey
from world import DT


def compare(root):
    comparison = json.loads((root/"comparison.json").read_text())
    cases = comparison["conditions"]
    trajectories = {label: np.load(root/label/"trajectory.npy") for label in cases}
    ref = trajectories["spin-still"]
    phase = cases["spin-still"]
    start = round(phase["transport_start"]/DT)
    arrival = phase["dwell_start"]
    finish = phase["dwell_end"]
    assert all(np.array_equal(rows[:, 0], ref[:, 0]) for rows in trajectories.values())
    poses = ref[:, 17:24]
    axes_ref = Rotation.from_quat(poses[:, 3:]).as_matrix()[:, :, 0]

    def actual_roll(rows):
        axes = Rotation.from_quat(rows[:, 20:24]).as_matrix()[:, :, 0]
        return np.einsum("ij,ij->i", axes, rows[:, 27:30])

    measured = {label: actual_roll(rows) for label, rows in trajectories.items()}
    midpoint = (phase["transport_start"] + arrival)/2
    between = (ref[:, 0] > midpoint) & (ref[:, 0] < arrival)
    base_spin = measured["spin-still"]
    # Reuse the same carry for the two target conditions, including the
    # exact film distribution on arrival. Then change only the target roll.
    before_arrival = ref[:, 0] <= arrival
    measured["spin-spin"] = np.where(before_arrival, base_spin, measured["spin-spin"])
    measured["still-spin"] = np.where(before_arrival, measured["still-still"], measured["still-spin"])
    measured["midway-stop"] = np.where(between, measured["still-still"], base_spin)
    result = {}
    for label, spin in measured.items():
        h = Honey()
        angular = ref[:, 27:30] + (spin-base_spin)[:, None]*axes_ref
        samples = []
        for i in range(1, len(ref)):
            h.update(DT, poses[i, :3], axes_ref[i], ref[i, 24:27],
                     ref[i, 27:30] if i <= start else angular[i])
            samples.append((ref[i, 0], h.rot, h.retained_g, *poses[i, :3]))
        samples = np.asarray(samples)
        case = root/label
        case.mkdir(exist_ok=True)
        np.save(case/"matched-material-trace.npy", samples)
        metrics = h.measurements()
        lands = metrics["landings"]
        during = [d for d in lands if phase["transport_start"] <= d["release_time_s"] < arrival]
        at_toast = [d for d in lands if arrival <= d["release_time_s"] < finish and d["surface"] == "toast"]
        result[label] = {
            "initial_on_tool_g": samples[start-1, 2],
            "transport_mean_abs_roll_rad_s": float(np.abs(samples[start-1:round(arrival/DT)-1, 1]).mean()),
            "second_half_transport_mean_abs_roll_rad_s": float(np.abs(samples[(samples[:, 0] >= midpoint)
                & (samples[:, 0] < arrival), 1]).mean()),
            "dwell_mean_abs_roll_rad_s": float(np.abs(samples[round(arrival/DT)-1:round(finish/DT)-1, 1]).mean()),
            "retained_at_arrival_g": samples[round(arrival/DT)-1, 2],
            "retained_final_g": samples[-1, 2],
            "transport_off_target_g": sum(d["mass_g"] for d in during if d["surface"] in ("plate", "table")),
            "dwell_toast_g": sum(d["mass_g"] for d in at_toast),
            **metrics,
        }
    spin = result["spin-still"]
    still = result["still-still"]
    continue_spin = result["spin-spin"]
    assert len({r["initial_on_tool_g"] for r in result.values()}) == 1
    assert still["transport_off_target_g"] - spin["transport_off_target_g"] >= .15
    assert result["midway-stop"]["transport_off_target_g"] - spin["transport_off_target_g"] >= .15
    assert spin["dwell_toast_g"] - continue_spin["dwell_toast_g"] >= .30
    path_error = {label: float(np.linalg.norm(rows[start:, 17:20]-ref[start:, 17:20], axis=1).max())
                  for label, rows in trajectories.items()}
    evidence = dict(matched_position_reference="spin-still/trajectory.npy", source_gpu_run_id=comparison["gpu_run_id"],
                    actual_robot_max_position_difference_m=path_error,
                    transport_start_s=phase["transport_start"], dwell_start_s=arrival,
                    dwell_end_s=finish, midway_stop_s=midpoint, passed=True, conditions=result)
    (root/"matched-comparison.json").write_text(json.dumps(evidence, indent=2)+"\n")
    return result


if __name__ == "__main__":
    for label, row in compare(Path(sys.argv[1])).items():
        print(label, {key: value for key, value in row.items() if key.endswith(("_g", "_rad_s"))})
