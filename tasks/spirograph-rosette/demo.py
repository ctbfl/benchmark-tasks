"""Scripted pen grasp, seating, and physical drawing with measured pose feedback."""

import math

import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import station
from yam_arena.constants import DOWN_XYZW, READY
from yam_arena.ik import solve, smoothstep, tcp_pose

from geometry import CENTER, GRASP, HOLE, ORBIT, THICKNESS
from scene import DT, HZ
from scoring import BOARD_TOP, STRAY_LIMIT, local_axes, wrapped


class Expert:
    def __init__(self, sim):
        self.sim = sim
        self.action = np.array([*READY, 1., *READY, 1.])
        self.arm = sim.robots[1]
        self.base = station.BASES["right"]
        self.rotation = Rotation.from_euler("z", 180, degrees=True)*Rotation.from_quat(DOWN_XYZW)
        self.bias = np.zeros(3)

    def tcp(self):
        ids = self.sim._ids["right_robot"][0]
        return tcp_pose(self.arm.data.joint_pos.torch[0, ids].cpu().numpy(), self.base)[0]

    def step(self, point, rotation=None, integral=True):
        rotation = self.rotation if rotation is None else rotation
        q, pos_error, rot_error = solve(point+self.bias, rotation.as_quat(), self.action[7:13],
                                         self.base, iters=60, damping=.018, tol=1e-5)
        assert pos_error < .00045 and rot_error < .008, (point, pos_error, rot_error)
        self.action[7:13] = q
        self.sim.advance(self.action)
        if integral:
            self.bias = np.clip(self.bias+.12*(point-self.tcp()), -.016, .016)

    def move(self, target, seconds=1., start_rotation=None, integral=True):
        start = self.tcp()
        relative = (self.rotation*start_rotation.inv()).as_rotvec() if start_rotation else None
        for s in np.linspace(0., 1., round(seconds*HZ)):
            a = smoothstep(s)
            rotation = Rotation.from_rotvec(a*relative)*start_rotation if start_rotation else self.rotation
            self.step(start + a*(target-start), rotation, integral)
        for _ in range(8):
            self.step(target, integral=integral)

    def report(self, label):
        wheel, pen = self.sim.poses()
        print(label, round(self.sim.control_steps/HZ, 2), "wheel", wheel.round(6).tolist(),
              "pen", pen.round(6).tolist(), "tcp", self.tcp().round(6).tolist(),
              "force", self.sim.sample()[30:].round(4).tolist(), self.sim.scorer.result(), flush=True)


def draw(sim, short_arc=False):
    expert = Expert(sim)
    wheel, pen = sim.poses()
    start = pen[:3] + [0., 0., GRASP]
    _, rotation = tcp_pose(np.array(READY), expert.base)
    expert.action[13] = .50
    expert.move(start + [0., 0., .072], 2., Rotation.from_matrix(rotation))
    expert.move(start, 1.6)
    expert.report("aligned")
    expert.action[13] = .036
    for _ in range(HZ):
        expert.step(start)
    expert.report("gripped")
    expert.move(start + [0., 0., .055], 1.2)
    expert.report("lifted")
    _, pen = sim.poses()
    tip_offset = pen[:3] - expert.tcp()
    assert pen[2] > BOARD_TOP+.035, "The pen did not lift from its socket."
    bore = wheel[:2] + [HOLE, 0.]
    expert.move(np.r_[bore, BOARD_TOP+.040] - tip_offset, 1.5)
    expert.move(np.r_[bore, BOARD_TOP+.0035] - tip_offset, 1.3)
    expert.bias[:] = 0.
    _, pen = sim.poses()
    height = BOARD_TOP-.0042
    expert.move(expert.tcp() + [0., 0., height-pen[2]], 1.4, integral=False)
    expert.report("seated")
    lead = tangential_lead = .0032
    speed = progress = extra_preload = 0.
    angle = None
    for k in range(1000 if short_arc else 1650):
        wheel, pen = sim.poses()
        wx, wz = local_axes(wheel)
        _, pz = local_axes(pen)
        assert pz[2] > math.cos(math.radians(8)), "The pen is rotating in the grasp."
        assert abs(wheel[2]-BOARD_TOP) < .001 and wz[2] > .998, "The wheel lifted off the paper."
        radial = wheel[:2]-CENTER
        radius = np.linalg.norm(radial)
        radial /= radius
        tangent = np.array([-radial[1], radial[0]])
        orbit = math.atan2(radial[1], radial[0])
        if angle is not None:
            movement = wrapped(orbit-angle)
            progress += movement
            speed = .65*speed + .35*HZ*movement
        angle = orbit
        target_speed = .30 if progress < 11.5 else .22
        lead = np.clip(lead + .00035*(target_speed-speed), -.0020, .0080)
        goal_lead = np.clip(lead - .0045*(speed-target_speed), -.0040, .0090)
        tangential_lead += np.clip(goal_lead-tangential_lead, -.00040, .00040)
        center = wheel[:3] + HOLE*wx + THICKNESS/2*wz
        # Lift neither the nib nor the wheel when the two loaded flanks unload.
        contact_age = DT*(sim.scorer.tick - min(sim.scorer.last_mesh, sim.scorer.last_bore))
        if math.isfinite(contact_age):
            extra_preload = np.clip(extra_preload + (.00075 if contact_age > .040 else -.00012),
                                    0., .0030)
        radial_lead = .0026 + extra_preload + np.clip(3.6*(ORBIT+.00033-radius), -.00055, .004)
        shank = center[:2] + radial_lead*radial + tangential_lead*tangent
        tip = np.r_[shank - pz[:2]*(center[2]-height)/pz[2], height]
        expert.step(tip + expert.tcp()-pen[:3], integral=False)
        if k % 20 == 0:
            expert.report(f"drawing lead={tangential_lead*1000:.1f}mm preload={radial_lead*1000:.1f}mm speed={speed:.2f}rad/s")
        if (sim.scorer.closed or sim.scorer.stray > STRAY_LIMIT
                or (short_arc and progress >= 8.1)):
            break
    if not short_arc:
        expert.move(expert.tcp() + [0., 0., .055], 1.7)
        expert.move(expert.tcp() + [0., -.144, .023], 1.8)
        expert.report("pen clear")
    else:
        expert.report("arc end")
