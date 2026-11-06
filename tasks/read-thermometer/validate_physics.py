"""Physical probe tests. This fixture is absent from the task environment."""
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent


def main():
    (HERE / "validation").mkdir(exist_ok=True)
    import numpy as np
    import torch
    import omni.physics.tensors as tensors
    from isaaclab.sim.utils import get_current_stage_id
    from yam_arena.constants import READY
    from yam_arena.urdf import quat_xyzw_to_matrix
    from scene import make_env, ROOT, CONFIG
    from task_runtime import TaskRuntime

    env = make_env(test_probe=True)
    base = env.unwrapped
    env.reset(seed=0)
    ready = torch.tensor([[*READY, 1., *READY, 1.]], dtype=torch.float32)
    for _ in range(40):
        env.step(ready)
    runtime = TaskRuntime(env)
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    physics.set_subspace_roots("/")
    probe = physics.create_rigid_body_view(ROOT + "/test_probe")
    assert probe.count == 1
    meta = json.loads((HERE / "assets/electronics/metadata.json").read_text())
    report = {"method": "6 mm kinematic sphere physically pressing spring keys; no state injection", "presses": []}
    clamp_view = physics.create_articulation_view(ROOT + "/clamp/boss")
    report["clamp_drive_targets_rad_m"] = clamp_view.get_dof_position_targets().tolist()
    report["clamp_drive_stiffness"] = clamp_view.get_dof_stiffnesses().tolist()
    print("PASSIVE_DRIVES", report["clamp_drive_targets_rad_m"], report["clamp_drive_stiffness"], flush=True)

    def advance(steps):
        for _ in range(steps):
            runtime.step(ready)

    initial_thermometer = base.scene["thermometer"].data.root_pose_w.torch[0].cpu().numpy().copy()
    advance(300)
    assert runtime.timer.display == "00:00" and abs(runtime.temperature - 45) < 1e-6
    final_thermometer = base.scene["thermometer"].data.root_pose_w.torch[0].cpu().numpy().copy()
    report["thermal"] = runtime.state()
    report["thermometer_drift_m"] = float(np.linalg.norm(final_thermometer[:3] - initial_thermometer[:3]))
    report["thermometer_initial_pose"] = initial_thermometer.tolist()
    report["thermometer_final_pose"] = final_thermometer.tolist()
    report["clamp_joints"] = dict(zip(base.scene["clamp"].joint_names,
        base.scene["clamp"].data.joint_pos.torch[0].cpu().numpy().tolist()))
    (HERE / "validation/physical_checks.json").write_text(json.dumps(report, indent=2))
    print("STABILITY", json.dumps(report), flush=True)

    def move(point, steps=1):
        target = np.array([[*point, 0, 0, 0, 1]], dtype=np.float32)
        for _ in range(steps):
            probe.set_kinematic_targets(target, np.array([0], dtype=np.uint32))
            runtime.step(ready)

    def press(device, key):
        obj = base.scene[device]
        key_info = meta[device]["buttons"][key]
        body_index = obj.body_names.index(key_info["body"])
        pose = obj.data.body_link_pose_w.torch[0, body_index].cpu().numpy()
        center = pose[:3].copy()
        normal = quat_xyzw_to_matrix(pose[3:]) @ np.array([0, 0, 1])
        start = center + normal * .021
        half_height = .0015 if device == "calculator" else .0025
        end = center + normal * (.006 + half_height - .92 * key_info["travel_m"])
        move(start, 3)
        lowest = 0.
        for fraction in np.linspace(0, 1, 16):
            move(start * (1 - fraction) + end * fraction)
            q = obj.data.joint_pos.torch[0, runtime.key_ids[device][key]].item()
            lowest = min(lowest, q)
        move(end, 3)
        for fraction in np.linspace(0, 1, 12):
            move(end * (1 - fraction) + start * fraction)
        move((.8, -.5, 1.5), 8)
        released = obj.data.joint_pos.torch[0, runtime.key_ids[device][key]].item()
        report["presses"].append({"device": device, "key": key, "minimum_q_m": lowest,
                                  "released_q_m": released, "state": runtime.state()})
        (HERE / "validation/physical_checks.json").write_text(json.dumps(report, indent=2))
        print("PRESS", json.dumps(report["presses"][-1]), flush=True)
        assert lowest < key_info["press_threshold_m"], (device, key, lowest)
        assert released > key_info["release_threshold_m"], (device, key, released)

    for key in ("4", "5", "="):
        press("calculator", key)
    assert runtime.score == 1 and runtime.calculator.display == "45"
    press("timer", "M")
    assert runtime.timer.display == "01:00"
    press("timer", "S")
    assert runtime.timer.display == "01:01"
    press("timer", "START")
    assert runtime.timer.running
    press("timer", "START")
    assert not runtime.timer.running
    paused = runtime.timer.remaining_seconds
    advance(20)
    assert runtime.timer.remaining_seconds == paused
    report["electronics_passed"] = True
    report["stability_passed"] = report["thermometer_drift_m"] < .002
    (HERE / "validation/physical_checks.json").write_text(json.dumps(report, indent=2))
    print("PHYSICAL_CHECKS", json.dumps(report), flush=True)
    assert report["stability_passed"], "Thermometer slipped during the countdown"
    env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    try:
        main()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
