"""Mount the SO-101 with standard YAM joint position actions and physical contact."""

from pathlib import Path
import sys
import traceback


def main():
    from isaaclab.app import AppLauncher

    mode, out = sys.argv[1:]
    assert mode in ("demo", "unsupported", "unclamped", "support")
    app = AppLauncher(headless=True, enable_cameras=False, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app
    try:
        from trial import Trial, write_json

        trial = Trial(Path(out))
        try:
            trial.reset()
            expert = Expert(trial)
            if mode == "support":
                expert.support(diagnose=True)
                trial.result("support")
            else:
                expert.mount(mode)
            if mode == "demo":
                result = trial.result("demo")
                print("DEMO SCORE", result["score"], "steps", result["steps"], flush=True)
                assert result["score"] == 1., result["final"]
            elif mode != "support":
                print("CONTROL", mode, trial.last, flush=True)
                write_json(trial.out / "control-result.json", dict(mode=mode, final=trial.last))
                assert trial.last["score"] == 0 and not trial.last["aligned"], trial.last
        except Exception:
            trial.result(mode, status="failed")
            raise
        finally:
            trial.close()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


class Expert:
    def __init__(self, trial):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from trial import READY_ACTION

        self.trial = trial
        self.q = READY_ACTION.copy()
        self.bias = np.zeros((2, 6))
        self.rotation = Rotation.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        self.left_rotation = Rotation.from_matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.orientations = [self.left_rotation, self.rotation]
        self.tip_offset = .140
        self.goal = [None, None]

    def mark(self, phase):
        from trial import array

        self.trial.phase = phase
        m = self.trial.last
        print(self.trial.steps, phase,
              "plate", [round(v, 4) for v in m["plate_pose"][:3]], "tilt", round(m["tilt_deg"],2),
              "left up N", round(m["left_net_upward_N"],2), "seated", m["seated_sites"],
              "supported", m["held_during_attachment"], "streaks", m["stable_steps"],
              "hinges", [array(c.data.joint_pos).round(3).tolist() for c in self.trial.clamps],
              "tips", [self.hand(i)[0].round(4).tolist() for i in range(2)], flush=True)

    def hand(self, side):
        from scipy.spatial.transform import Rotation
        from trial import array

        pose = array(self.trial.robots[side].data.body_link_pose_w)[self.trial.hand_ids[side]]
        r = Rotation.from_quat(pose[3:])
        return pose[:3] + r.apply([0, 0, self.tip_offset]), r

    def step(self):
        import numpy as np
        from trial import HZ, array

        command = self.q.copy()
        for i in range(2):
            offset = 7*i
            command[offset:offset+6] += self.bias[i]
        self.trial.perform(command)
        for i, robot in enumerate(self.trial.robots):
            offset = 7*i
            actual = array(robot.data.joint_pos)[self.trial._ids[("left_robot", "right_robot")[i]][0]]
            self.bias[i] = np.clip(self.bias[i] + (1.7 / HZ) * (self.q[offset:offset+6] - actual), -.14, .14)

    def wait(self, duration):
        from trial import HZ

        for _ in range(round(duration * HZ)):
            self.step()

    def move(self, side, point, sec=1., approach=False, rotation=None):
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        from yam_arena.ik import solve, smoothstep
        from yam_arena import station
        from trial import HZ

        offset = 7*side
        point = np.array(point)
        initial_rot = self.orientations[side]
        final_rot = initial_rot if rotation is None else rotation
        rotate = Slerp([0, 1], Rotation.concatenate([initial_rot, final_rot]))
        initial = self.goal[side]
        joints = self.q[offset:offset+6].copy()
        if approach:
            q0 = joints.copy()
            joints[:] = [0, 1.55, .6, 1.1, 0, -1.57]
        base_pos = station.BASES[("left", "right")[side]]
        for k in range(1, round(sec*HZ) + 1):
            v = smoothstep(k / round(sec*HZ))
            p = point if approach else (1-v)*initial + v*point
            rot = rotate(v)
            tcp = p + rot.apply([0, 0, .134 - self.tip_offset])
            joints, distance, angle = solve(tcp, rot.as_quat(), joints, base_pos,
                                             damping=.015, iters=90, tol=1e-5)
            assert distance < .0025 and angle < .025, (side, p.tolist(), distance, angle)
            self.q[offset:offset+6] = (1-v)*q0 + v*joints if approach else joints
            self.step()
        self.goal[side] = point
        self.orientations[side] = final_rot

    def grip(self, side, command, sec=.6):
        self.q[side*7+6] = command
        self.wait(sec)

    def support(self, diagnose=False):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from geometry import BASE_ROOT_X, TABLE_Z
        from trial import array, write_json

        pose = array(self.trial.arm.data.root_pose_w)
        object_rotation = Rotation.from_quat(pose[3:])
        local_grasp = np.array([0., .0, .137])
        grasp = pose[:3] + object_rotation.apply(local_grasp)
        orientation = object_rotation * self.left_rotation
        self.mark("approach folded upper links")
        self.grip(0, 1., .2)
        self.move(0, grasp + [0., .12, .032], 2.4, approach=True, rotation=orientation)
        self.move(0, grasp + [0., .06, .005], 1.25, rotation=orientation)
        self.move(0, grasp, 1.35, rotation=orientation)
        self.grip(0, 0., .85)
        self.mark("grasp folded links")
        self.move(0, self.goal[0] + [0., 0., .025], .9, rotation=orientation)
        target = np.array([BASE_ROOT_X, 0., TABLE_Z+.00000026])

        def align(goal, attempts):
            for _ in range(attempts):
                pose = array(self.trial.arm.data.root_pose_w)
                error = goal-pose[:3]
                correction = Rotation.from_quat(pose[3:]).inv()
                if np.linalg.norm(error) < .002 and correction.magnitude() < .025:
                    break
                point = self.goal[0] + np.clip(.85*error, -.012, .012)
                self.move(0, point, .65, rotation=correction*self.orientations[0])
                self.wait(.25)

        align(target + [0., 0., .025], 10)
        self.mark("upright in the supporting hand")
        self.move(0, self.goal[0] - [0., 0., .025], 1.)
        self.wait(.4)
        align(target, 8)
        self.mark("left hand supports upright base")
        if diagnose:
            write_json(self.trial.out / "support-check.json", self.trial.last)
        assert self.trial.last["aligned"] and self.trial.last["left_supporting"], self.trial.last

    def clamp(self, index, keep_open=False):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from geometry import CLAMP_Y, EDGE_X, MOUNT_GAP_Z
        from trial import array

        clamp = self.trial.clamps[index]
        links = array(clamp.data.body_link_pose_w)[[clamp.body_names.index("upper"),
                                                   clamp.body_names.index("lower")]]
        # Midpoint of the two actual outer handle pads.
        handles = [p[:3] + Rotation.from_quat(p[3:]).apply([-.055, 0, sign*.041])
                   for p,sign in zip(links,(1,-1))]
        x,y,z = np.mean(handles, axis=0)
        self.mark("approach the parked clamp")
        if self.goal[1] is None:
            self.move(1, (x-.029, y, z+.04), 2.3, approach=True)
        else:
            self.move(1, (x-.029, y, z+.035), 1.45, rotation=self.rotation)
        self.move(1, (x-.018, y, z), .75)
        self.move(1, (x+.003, y, z), .85)
        self.wait(.45)
        self.grip(1, .105, .85)
        self.mark("squeeze handles to spread the jaws")
        x,y,z = self.goal[1]
        self.move(1, (x-.036, y, z+.006), 1.35)
        self.wait(.55)
        self.mark("lift the clamp off its rail")
        pivot = array(clamp.data.root_pose_w)[:3]
        offset = pivot - self.hand(1)[0]
        dest = np.array([EDGE_X-.010, CLAMP_Y[index], MOUNT_GAP_Z]) - offset
        self.move(1, (dest[0]-.065, dest[1], dest[2]+.005), 2.8)
        self.wait(.55)
        pad_midpoint = np.mean(self.trial.last["clamp_pad_centers_m"][index], axis=0)
        destination_pad = np.array([EDGE_X+.025, CLAMP_Y[index], MOUNT_GAP_Z])
        dest = self.goal[1] + (destination_pad - pad_midpoint)
        self.move(1, dest, 1.5)
        self.wait(.65)
        self.mark("open jaws around the edge")
        if keep_open:
            return
        for value in (.24, .34, .43, .54, .70, 1.):
            self.grip(1, value, .20)
        self.mark("release spring force onto the plate and table")
        self.move(1, dest - [.060, 0, 0], .85)
        self.wait(.3)
        self.mark("clamp hand clear")

    def release(self):
        import numpy as np
        self.grip(0, 1., .8)
        self.move(0, self.goal[0] + np.array([0., .105, .005]), 1.2)
        self.wait(2.4)
        self.mark("both hands clear")

    def mount(self, mode):
        self.support()
        if mode != "unsupported":
            self.clamp(0, keep_open=mode == "unclamped")
        if mode == "demo":
            self.clamp(1)
        self.release()


if __name__ == "__main__":
    main()
