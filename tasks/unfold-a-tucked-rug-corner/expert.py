"""Retrieve the tucked corner through native right-hand grasping, lifting and setdown."""
from pathlib import Path
import sys
import traceback
import numpy as np
from scene import SIM, CONTROL_HZ
from isaaclab.app import AppLauncher

assert len(sys.argv) == 2, "Usage: python expert.py RUN_DIRECTORY"
app = AppLauncher(headless=True, device=SIM["device"], enable_cameras=True,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app


def run():
    from runtime import Experiment
    from kinematics import solve, attitude, pad_pose, tool_floor_clearance
    from yam_arena.ik import smoothstep
    from yam_arena.station import TABLE_TOP
    from collision import hulls, clear

    experiment = Experiment(Path(sys.argv[1]).resolve())
    action = experiment.home.copy().astype(float)
    bias = np.zeros(6)
    rotation = attitude(90., 60., 180.)

    def send():
        command = action.copy()
        command[7:13] += bias
        valid, pair = clear(hulls(command[:6], "left", command[6]),
                            hulls(command[7:13], "right", command[13]), margin=.002)
        if not valid:
            raise RuntimeError(f"Inter-arm collision in commanded path: {pair}")
        experiment.step(command, action)
        bias[:] = np.clip(bias + .08 * (action[7:13] - experiment.joints()[1]), -.08, .08)

    def wait(seconds):
        for _ in range(round(seconds * CONTROL_HZ)):
            send()

    def joint_move(goal, seconds):
        start = action.copy()
        for t in np.linspace(0., 1., round(seconds * CONTROL_HZ) + 1)[1:]:
            action[:] = start + smoothstep(t) * (goal - start)
            send()

    def move(point, seconds):
        first, _ = pad_pose(action[7:13], "right")
        goal = np.asarray(point) + [0., 0., TABLE_TOP]
        for t in np.linspace(0., 1., round(seconds * CONTROL_HZ) + 1)[1:]:
            target = first + smoothstep(t) * (goal - first)
            q, error = solve(target, rotation, "right", action[7:13])
            if error[0] > .001 or error[1] > .025:
                raise RuntimeError(f"Unreachable right pad pose: {target}, {error}")
            if tool_floor_clearance(q, "right", action[13]) < -.0002:
                raise RuntimeError("Nominal right tool penetrates the support")
            action[7:13] = q
            send()

    try:
        goal = action.copy()
        goal[7:13], error = solve([.285, -.245, TABLE_TOP + .15], rotation, "right", action[7:13])
        assert error[0] < .001 and error[1] < .025, error
        goal[13] = .4
        joint_move(goal, 2.5)
        move([.285, -.235, .0175], 2.)
        move([.285, -.150, .0175], 3.9)
        experiment.report("inserted")
        goal = action.copy()
        goal[13] = 0.
        joint_move(goal, 1.)
        wait(.4)
        move([.285, -.150, .0675], 2.)
        wait(2.)
        experiment.report("lifted")
        move([.285, -.150, .0175], 2.)
        goal = action.copy()
        goal[13] = .4
        joint_move(goal, .7)
        move([.285, -.235, .0175], 2.)
        move([.285, -.235, .15], 1.5)
        wait(5.)
        experiment.report("released")
        assert experiment.measure()["score"] == 1., "Expert did not complete the task"
    finally:
        experiment.save()
        experiment.close()


try:
    run()
except BaseException:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()
