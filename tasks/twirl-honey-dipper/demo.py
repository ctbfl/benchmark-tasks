"""Execute the scripted grasp and transfer through the native YAM action interface."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from layout import GRIP_X, TOAST
from world import HZ

TRANSPORT_S = 2.6
DWELL_S = 3.3
ROLL_RATE = 3.1
ROLL_RANGE = 1.76


def plan(world, carry_spin=True, dwell_spin=False, dwell_s=DWELL_S, settle_s=.30,
         return_to_rest=False, drain_s=0.):
    from yam_arena.constants import READY
    from yam_arena.ik import LOWER, UPPER, smoothstep, tcp_pose
    from yam_arena.station import BASES

    def send(right, opening):
        world.step(np.r_[READY, 1., right, opening])

    pose, _ = world.tool()
    axis = Rotation.from_quat(pose[3:]).as_matrix()[:, 0]
    assert axis[0] > .98, pose
    rest_root = pose[:3].copy()
    start = rest_root + GRIP_X*axis
    tip_z = .885
    y0 = TOAST[1] - .026
    y1 = TOAST[1] + .026
    lifted = start + [0, 0, .041]
    at_toast = np.array([start[0], y0, tip_z])
    approach = start + [-.040, 0, 0]
    raised = approach + [0, 0, .052]
    q = np.array([.2, 1.5, .2, 1.3, .2])
    phase = {}

    def send_cartesian(p, roll, opening):
        nonlocal q
        def residual(joints):
            xyz, frame = tcp_pose(np.r_[joints, 0.], BASES["right"])
            return np.r_[xyz-p, .25*(frame[:, 2]-[1., 0., 0.])]
        sol = least_squares(residual, q, bounds=(LOWER[:5]+1e-8, UPPER[:5]-1e-8),
                            gtol=1e-8, max_nfev=60)
        assert np.linalg.norm(sol.fun) < .001, (p, sol.fun)
        q = sol.x
        send(np.r_[q, roll], opening)

    def segment(p0, p1, duration, roll, opening):
        n = round(duration*HZ)
        for i in range(1, n+1):
            u = smoothstep(i/n)
            send_cartesian(p0+(p1-p0)*u, roll(i/HZ), opening)

    def roll_angle(t):
        return ROLL_RANGE-abs((ROLL_RATE*t+ROLL_RANGE) % (4*ROLL_RANGE)-2*ROLL_RANGE)

    # Seed Cartesian IK without advancing the actual trajectory.
    def reach(p):
        def residual(joints):
            xyz, frame = tcp_pose(np.r_[joints, 0.], BASES["right"])
            return np.r_[xyz-p, .25*(frame[:, 2]-[1., 0., 0.])]
        return least_squares(residual, q, bounds=(LOWER[:5]+1e-8, UPPER[:5]-1e-8)).x

    q = reach(raised)
    for i in range(1, 25):
        s = smoothstep(i/24)
        send((1-s)*np.asarray(READY) + s*np.r_[q, 0.], .44)
    segment(raised, approach, .85, lambda _: 0., .44)
    segment(approach, start, .90, lambda _: 0., .44)
    segment(start, start, .65, lambda _: 0., 0.)
    segment(start, lifted, .95, lambda _: 0., 0.)
    pose, _ = world.tool()
    print("Lifted tool", pose[:3].round(5), "initial", start.round(5), flush=True)
    assert pose[2] - start[2] > .027, pose
    if drain_s:
        phase["drain_start"] = world.steps/HZ
        segment(lifted, lifted, drain_s, lambda _: 0., 0.)
    phase["transport_start"] = world.steps/HZ
    phase["initial_g"] = world.honey.retained_g
    segment(lifted, at_toast, TRANSPORT_S, roll_angle if carry_spin else lambda _: 0., 0.)
    phase["dwell_start"] = world.steps/HZ
    arrived_roll = roll_angle(TRANSPORT_S) if carry_spin else 0.
    end = at_toast.copy()
    end[1] = y1
    def dwell_roll(t):
        # Both moving variants continue the same angular schedule upon arrival.
        return roll_angle(t + (TRANSPORT_S if carry_spin else 0.)) if dwell_spin else arrived_roll
    segment(at_toast, end, dwell_s, dwell_roll, 0.)
    phase["dwell_end"] = world.steps/HZ
    segment(end, end, settle_s, lambda _: dwell_roll(dwell_s), 0.)
    if return_to_rest:
        phase["return_start"] = world.steps/HZ
        segment(end, lifted, TRANSPORT_S, lambda t: roll_angle(TRANSPORT_S-t), 0.)
        segment(lifted, lifted, .35, lambda _: 0., 0.)
        pose, _ = world.tool()
        correction = rest_root + [0., 0., .0015] - pose[:3]
        above_rest = lifted + [*correction[:2], 0.]
        seat = lifted + correction
        segment(lifted, above_rest, .60, lambda _: 0., 0.)
        segment(above_rest, seat, 1.15, lambda _: 0., 0.)
        phase["release_start"] = world.steps/HZ
        segment(seat, seat, .45, lambda _: 0., .60)
        clear = seat + [-.049, 0., .045]
        segment(seat, clear, .90, lambda _: 0., .60)
        phase["clear_start"] = world.steps/HZ
        segment(clear, clear, 1.30, lambda _: 0., .60)
    return phase
