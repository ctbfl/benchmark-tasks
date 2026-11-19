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
        self.orientations = [self.rotation, self.rotation]
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
        from geometry import EDGE_X, PLATE_THICKNESS, PLATE_X, START_OFFSET, SUPPORT_Y
        from trial import array

        pose = array(self.trial.arm.data.root_pose_w)
        z = float(pose[2]) + PLATE_THICKNESS / 2
        yaw = Rotation.from_quat(pose[3:]).as_euler("zyx")[0]
        c, s = np.cos(yaw), np.sin(yaw)
        contact_y = (SUPPORT_Y-pose[1]-s*PLATE_X[0])/c
        entry = pose[0] + c*PLATE_X[0] - s*contact_y + .005
        self.mark("approach the mounting plate")
        measurements = []

        def sample_approach():
            from scene import DT

            current = array(self.trial.arm.data.root_pose_w)
            force = np.linalg.norm(self.trial.plate_contact.get_contact_force_matrix(DT)[0], axis=-1)
            measurements.append({"tick": self.trial.ticks, "plate_pose": current.tolist(),
                                 "contact_N_by_link": force.tolist()})

        if diagnose:
            self.trial.approach_probe = sample_approach
        self.grip(0, .32, .65)
        self.move(0, (.245, .217, 1.035), 2.6, approach=True)
        self.move(0, (.247, SUPPORT_Y, 1.015), 1.6)
        self.move(0, (.289, SUPPORT_Y, z), 1.45)
        self.wait(.5)
        self.move(0, (entry, SUPPORT_Y, z), 1.35)
        self.wait(.65)
        self.trial.approach_probe = None
        if diagnose:
            from trial import write_json

            poses = np.array([r["plate_pose"] for r in measurements])
            forces = np.array([r["contact_N_by_link"] for r in measurements])
            motion = float(np.max(np.linalg.norm(poses[:,:3] - pose[:3], axis=1)))
            turns = Rotation.from_quat(poses[:, 3:]) * Rotation.from_quat(pose[3:]).inv()
            angle = float(np.degrees(turns.magnitude()).max())
            before_grasp = dict(samples=measurements, target_aperture=.32, max_drift_m=motion,
                                max_rotation_deg=angle, max_contact_N=float(forces.max()),
                                robot_link_paths=self.trial.robot_paths)
            write_json(self.trial.out / "approach.json", before_grasp)
            assert motion < .001 and angle < .5 and forces.max() < .5, (
                "The unclosed gripper disturbed the assembly", motion, angle, float(forces.max()))
            print("APPROACH CLEAR", motion, angle, float(forces.max()), flush=True)
        self.grip(0, 0., .85)
        self.mark("grasp the mounting plate")
        self.move(0, (entry, SUPPORT_Y, z+.016), .9)
        yaw_correction = Rotation.from_euler("z", -6., degrees=True) * self.rotation
        self.move(0, (entry-START_OFFSET-.004, SUPPORT_Y-.006, z+.016), 2.3,
                  rotation=yaw_correction)
        self.move(0, (entry-START_OFFSET-.004, SUPPORT_Y-.006, z+.0005), .9)
        self.wait(.8)
        for _ in range(3):
            x, y = array(self.trial.arm.data.root_pose_w)[:2]
            if self.trial.last["aligned"] and abs(x - EDGE_X) < .004 and abs(y) < .004:
                break
            self.move(0, self.goal[0] + np.clip([EDGE_X - x, -y, 0], -.012, .012), .85)
            self.wait(.65)
        self.mark("support at the table edge")
        if diagnose:
            import os
            from trial import write_json

            readings = []
            for _ in range(20):
                self.step()
                readings.append(self.trial.last)
            forces = [r["left_net_upward_N"] for r in readings]
            support = self.trial.last
            write_json(self.trial.out / "support-check.json", dict(
                gpu_run_id=os.environ["PRODUCTION_GPU_RUN_ID"], approach="approach.json",
                min_upward_support_N=min(forces), supported_samples=readings, final=support))
            assert min(forces) > .4 and all(r["aligned"] and r["seated_sites"] == [] for r in readings), (
                "The supported pose drifted or the grip lost the overhanging load", min(forces), support)
            print("SUPPORTED EDGE POSE", min(forces), support["alignment_error_m"], flush=True)

    def clamp(self, index, keep_open=False):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from geometry import CLAMP_Y, EDGE_X, GAP_CENTER_Z
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
            self.move(1, (x-.029, y, z+.035), 1.45)
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
        dest = np.array([EDGE_X-.010, CLAMP_Y[index], GAP_CENTER_Z]) - offset
        self.move(1, (dest[0]-.034, dest[1], dest[2]+.005), 2.8)
        self.wait(.55)
        pad_midpoint = np.mean(self.trial.last["clamp_pad_centers_m"][index], axis=0)
        destination_pad = np.array([EDGE_X+.025, CLAMP_Y[index], GAP_CENTER_Z])
        dest = self.goal[1] + (destination_pad - pad_midpoint)
        self.move(1, dest, 1.5)
        self.wait(.65)
        self.mark("open jaws around the edge")
        if keep_open:
            return
        if index == 1:
            # Aim the pad's inboard edge downward while clearing the lower lever.
            mouth = np.mean(self.trial.last["clamp_pad_centers_m"][index], axis=0)
            pitch = Rotation.from_euler("y", 6., degrees=True)
            dest = mouth + pitch.apply(dest - mouth) + [0., 0., -.0025]
            self.move(1, dest, .85, rotation=pitch * self.orientations[1])
            self.wait(.25)
        for value in (.24, .34, .43, .54, .70, 1.):
            self.grip(1, value, .20)
        self.mark("release spring force onto the plate and table")
        self.move(1, dest - [.045, 0, 0], .85)
        self.wait(.3)
        self.mark("clamp hand clear")

    def release(self):
        import numpy as np

        self.grip(0, .30, .55)
        self.move(0, self.goal[0] - np.array([.041, 0, 0]), 1.)
        self.grip(0, 1., .3)
        self.wait(2.4)
        self.mark("released the SO-101")

    def mount(self, mode):
        self.support()
        if mode != "unsupported":
            self.clamp(0, keep_open=mode == "unclamped")
        if mode == "demo":
            self.clamp(1)
        self.release()


if __name__ == "__main__":
    main()
