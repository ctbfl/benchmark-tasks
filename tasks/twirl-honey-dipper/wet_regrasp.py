"""Execute a native-action regression: dry carry, then switch to the wet head."""
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yam_arena.constants import READY
from yam_arena.ik import LOWER, UPPER, smoothstep, tcp_pose
from yam_arena.station import BASES

from world import DT, HZ, SUBSTEPS, World
from scoring import rescore


def regrasp(world):
    before = world.grade()
    assert before["score"] == .75 and not before["parked"], before
    pose, _ = world.tool()
    head = pose[:3]+.115*Rotation.from_quat(pose[3:]).as_matrix()[:, 0]
    direction = np.array([.6, -.3, -.742])
    direction /= np.linalg.norm(direction)
    closing = np.cross(direction, [1, 0, 0])
    closing /= np.linalg.norm(closing)
    rot = np.column_stack((closing, np.cross(direction, closing), direction))
    right = world.actions[-1][7:].copy()

    def solve(target, seed):
        def err(q):
            p, r = tcp_pose(q, BASES["left"])
            return np.r_[p-target, .2*(r-rot).ravel()]
        fit = least_squares(err, seed, bounds=(LOWER+1e-7, UPPER-1e-7), max_nfev=80)
        assert np.linalg.norm(fit.fun) < .001, (target, fit.fun)
        return fit.x

    def move(start, end, seconds, opening):
        n = round(seconds*HZ)
        for i in range(1, n+1):
            q = start + (end-start)*smoothstep(i/n)
            world.step(np.r_[q, opening, right])

    start = solve(head-.072*direction, [-.57, 1.88, 1.58, -.55, -.07, -.67])
    grip = solve(head, start)
    raise_head = solve(head+[0, 0, .035], grip)
    move(np.asarray(READY), start, 1.4, .46)
    move(start, grip, .85, .46)
    move(grip, grip, .65, 0.)
    after_close = world.tool()[0].copy()
    right[-1] = 1.
    move(grip, grip, .4, 0.)
    move(grip, raise_head, 1.1, 0.)
    after = world.grade()
    measured_lift = float(world.tool()[0][2]-after_close[2])
    assert measured_lift > .020, measured_lift
    assert after["wet_contact_first_s"] is not None and after["score"] == 0., after
    return before, after, measured_lift


def seed_contact_fixture(world, prefix):
    """Resume at the measured end of the drizzle for this contact test."""
    import omni.physics.tensors as tensors
    import omni.physx
    from pxr import UsdUtils

    world.reset(0)
    phase = json.loads((prefix/"demo-phases.json").read_text())
    prefix_steps = round(phase["return_start"]*HZ)
    rows = np.load(prefix/"trajectory.npy")[:prefix_steps*SUBSTEPS+1]
    actions = np.load(prefix/"actions.npy")[:prefix_steps]
    measured = json.loads((prefix/"measurements.json").read_text())
    axes = Rotation.from_quat(rows[:, 20:24]).as_matrix()[:, :, 0]
    for row, axis in zip(rows, axes, strict=True):
        world.scorer.advance(row)
        if row[0]:
            world.honey.update(DT, row[17:20], axis, row[24:27], row[27:30])
    assert world.honey.measurements()["landings"] == [
        drop for drop in measured["landings"] if drop["time_s"] <= rows[-1, 0]+1e-9]

    last = rows[-1]
    tensor = lambda data: world.torch.tensor(data[None], dtype=world.torch.float32)
    for i, (robot, ids) in enumerate(zip(world.robots, world.joint_ids, strict=True)):
        robot.write_joint_state_to_sim(tensor(last[1+i*8:9+i*8]),
                                        tensor(np.zeros(8)), joint_ids=ids)
    world.dipper.write_root_pose_to_sim(tensor(last[17:24]))
    world.dipper.write_root_velocity_to_sim(tensor(np.zeros(6)))
    stage = world.base.sim.stage
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    tensors.create_simulation_view("numpy", stage_id=stage_id).update_articulations_kinematic()
    omni.physx.get_physx_interface().update_transformations(False, True, True)
    world.scene.update(DT)
    assert np.allclose(world.tool()[0], last[17:24], atol=.0001)
    assert np.allclose(world.robot_joints(), last[1:17], atol=.0001)
    world.previous_pose = world.tool()[0]
    world.trajectory = list(rows)
    world.actions = list(actions)
    world.steps = len(actions)
    world.tick = len(rows)-1
    return json.loads((prefix/"result.json").read_text())["gpu_run_id"]


def main():
    from isaaclab.app import AppLauncher

    prefix, out = (Path(arg).resolve() for arg in sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(headless=True, enable_cameras=False).app
    exit_code = 0
    world = None
    try:
        world = World(out)
        prefix_run_id = seed_contact_fixture(world, prefix)
        prefix_steps = world.steps
        before, after, lift = regrasp(world)
        world.save()
        assert rescore(out) == after
        result = dict(gpu_run_id=os.environ.get("PRODUCTION_GPU_RUN_ID"), prefix_run_id=prefix_run_id,
                      fixture="measured endpoint reset, native actions after reset", before=before,
                      after=after, wet_grip_lift_m=lift, native_suffix_steps=world.steps-prefix_steps)
        (out/"negative-result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        if world is not None and hasattr(world, "trajectory"):
            world.save()
    finally:
        if world is not None:
            world.env.close()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
