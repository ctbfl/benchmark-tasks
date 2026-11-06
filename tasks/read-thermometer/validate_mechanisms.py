"""Bench torques test the passive clamp; these loads are never used by a policy."""
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
    from scene import make_env, ROOT
    env = make_env()
    base = env.unwrapped
    env.reset(seed=0)
    ready = torch.tensor([[*READY, 1., *READY, 1.]], dtype=torch.float32)
    for _ in range(40):
        env.step(ready)
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    physics.set_subspace_roots("/")
    view = physics.create_articulation_view(ROOT + "/clamp/boss")
    obj = base.scene["clamp"]
    indices = {name: obj.joint_names.index(name) for name in obj.joint_names}
    jaw, knob, slider = (indices[name] for name in ("jaw_hinge", "screw_turn", "screw_travel"))
    forces = np.zeros((1, 3), dtype=np.float32)
    rows = []
    base.action_manager.process_action(ready)

    def step(jaw_torque=0, knob_torque=0):
        base.action_manager.apply_action()
        base.scene.write_data_to_sim()
        forces[0, jaw], forces[0, knob] = jaw_torque, knob_torque
        view.set_dof_actuation_forces(forces, np.array([0], dtype=np.uint32))
        base.sim.step(render=False)
        base.scene.update(1 / 240)
        q = obj.data.joint_pos.torch[0].cpu().numpy().copy()
        rows.append(q.tolist())
        return q

    for i in range(120):
        q = step(jaw_torque=-.17 * min(1, i / 60))
    open_angle = float(q[jaw])
    for _ in range(240):
        q = step()
    released_angle = float(q[jaw])
    assert open_angle < -.20, open_angle
    assert abs(released_angle - np.deg2rad(4)) < .02, released_angle
    initial_height = obj.data.root_pose_w.torch[0, 2].item()
    for i in range(960):
        q = step(knob_torque=.12 * min(1, i / 96))
    for _ in range(120):
        q = step()
    final_height = obj.data.root_pose_w.torch[0, 2].item()
    expected = float(q[knob] * .002 / (2 * np.pi))
    error = abs(float(q[slider]) - expected)
    report = {"fixture": "External joint torques applied only in this bench test",
              "jaw_opening_torque_Nm": -.17, "knob_turning_torque_Nm": .12,
              "jaw_open_deg": float(np.rad2deg(open_angle)),
              "jaw_after_release_deg": float(np.rad2deg(released_angle)),
              "knob_angle_deg": float(np.rad2deg(q[knob])),
              "screw_travel_m": float(q[slider]), "coupling_error_m": error,
              "released_clamp_drop_m": initial_height - final_height,
              "jaw_return_passed": True, "screw_release_passed": bool(
                  error < .0001 and q[slider] > .003 and initial_height - final_height > .01)}
    (HERE / "validation/mechanism_checks.json").write_text(json.dumps(report, indent=2))
    (HERE / "validation/mechanism_joint_trace.json").write_text(json.dumps(rows))
    print("MECHANISMS", json.dumps(report), flush=True)
    assert report["screw_release_passed"], report
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
