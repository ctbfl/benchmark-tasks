"""Position-controlled parallel-jaw motion. Class decisions use only shell motion."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.constants import READY
from yam_arena.ik import smoothstep, solve, tcp_pose
from yam_arena.station import BASES

from simulation import CONTROL_HZ, DT
from scoring import ARRESTED_SPEED


def down(yaw):
    # Fingers close normal to the egg's long axis.
    return Rotation.from_euler("z", yaw) * Rotation.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


class Arm:
    def __init__(self, sim, side, frame_hook=None):
        self.sim = sim
        self.side = side
        self.sl = slice(7*side, 7*side+6)
        self.grip = 7*side+6
        self.q = np.asarray(READY, dtype=float)
        self.opening = 1.
        self.bias = np.zeros(6)
        self.frame_hook = frame_hook

    def target(self, point, yaw, *, tilt=0.):
        name = ("left", "right")[self.side]
        orientation = Rotation.from_euler("z", yaw-np.pi) * Rotation.from_euler("x", tilt) * down(np.pi)
        q, ep, er = solve(point, orientation.as_quat(), self.q, BASES[name],
                          iters=180, damping=.018, tol=1e-5)
        assert ep < .0015 and er < .025, (name, point, yaw, ep, er)
        return q

    def step(self, q, opening, bias=True):
        action = self.sim.action.copy()
        action[self.sl] = q + self.bias
        action[self.grip] = opening
        self.sim.step(action, frame_hook=self.frame_hook)
        self.track(q, opening, bias)

    def track(self, q, opening, bias):
        if bias and np.max(np.abs(q-self.q)) < .04:
            self.bias = np.clip(self.bias + .18 * (q - self.sim.joints()[self.sl]), -.10, .10)
        self.q, self.opening = q, opening

    def hold(self, seconds, opening=None, bias=True):
        for _ in range(round(seconds * CONTROL_HZ)):
            self.step(self.q, self.opening if opening is None else opening, bias)

    def move(self, point, yaw, opening, seconds=.6, settle=.2):
        q = self.target(point, yaw)
        start, grip = self.q.copy(), self.opening
        for j in range(1, round(seconds * CONTROL_HZ)+1):
            u = smoothstep(j / round(seconds * CONTROL_HZ))
            self.step(start + u*(q-start), grip + u*(opening-grip))
        self.hold(settle)


def spin(arms, center, yaw):
    """Launch, arrest and release the egg with opposing inner finger pads."""
    sim = arms[0].sim
    record = arms[0].frame_hook
    end_offset, tilt = .022, 35.
    opening=np.full(2,.70)
    center = np.asarray(center).copy()
    yaw -= np.pi/2
    heading = Rotation.from_euler("z", yaw)
    axis=Rotation.from_quat(sim.pose(0)[3:]).apply([1.,0.,0.])
    cap_slope=axis[2]/np.linalg.norm(axis[:2])
    rotations = [heading * Rotation.from_euler("x",180-sign*tilt,degrees=True)
                 * Rotation.from_euler("z",180 if side=="left" else 0,degrees=True)
                 for side,sign in (("left",1),("right",-1))]

    def paired_step(arms, targets, opening, *, bias=True):
        values=np.broadcast_to(opening,(2,))
        action=sim.action.copy()
        for arm,target,value in zip(arms,targets,values,strict=True):
            action[arm.sl]=target+arm.bias
            action[arm.grip]=value
        sim.step(action,frame_hook=record)
        for arm,target,value in zip(arms,targets,values,strict=True):
            arm.track(target,float(value),bias)

    tracking=True

    def points(stroke=0., height=.012, distance=end_offset):
        distance = np.broadcast_to(distance, (2,))
        ends = center + heading.apply([[stroke,distance[0],distance[0]*cap_slope],[-stroke,-distance[1],-distance[1]*cap_slope]])
        return np.stack([end-rotation.apply([0,0,height/abs(rotation.as_matrix()[2,2])])
                         for end,rotation in zip(ends,rotations,strict=True)])

    def targets(p):
        result = []
        for arm, point, rotation, side in zip(arms,p,rotations,("left","right"),strict=True):
            q,ep,er = solve(point,rotation.as_quat(),arm.q,BASES[side],
                            iters=180,damping=.018,tol=1e-5)
            if ep>.0015 or er>.025:
                raise RuntimeError(f"Unreachable {side} pose: {ep}, {er}")
            result.append(q)
        return np.stack(result)

    def move(p, seconds):
        start = np.stack([tcp_pose(a.q,BASES[s])[0]
                          for a,s in zip(arms,("left","right"),strict=True)])
        steps = round(seconds*CONTROL_HZ)
        for j in range(1,steps+1):
            paired_step(arms,targets(start+smoothstep(j/steps)*(p-start)),opening,bias=tracking)

    def hold(seconds):
        for _ in range(round(seconds*CONTROL_HZ)):
            paired_step(arms,np.stack([a.q for a in arms]),opening,bias=tracking)

    def mark(phase):
        sim.phase = phase
        e=sim.rows[-1]["eggs"][0]
        print(json.dumps({"phase":phase,"time_s":sim.ticks*DT,
            "speed_rad_s":e["velocity"][5],"overloaded":e["overloaded"],
            "pad_impulses_ns":e["pad_normal_impulse_ns"],
            "center":e["pose"][:3]}),flush=True)

    mark("approach")
    goal=targets(points(height=.060))
    start=np.stack([a.q for a in arms])
    for j in range(1,41):
        paired_step(arms,start+smoothstep(j/40)*(goal-start),opening)
    hold(.3)
    move(points(),1.0)
    hold(.5)
    mark("open_fingers_straddle_egg_ends")
    hold(.3)
    mark("light_inner_pad_preparation")
    for _ in range(40):
        loads=np.asarray([row["eggs"][0]["pad_normal_impulse_ns"] for row in sim.rows[-12:]])
        touching=loads.max(0)/DT>.2
        if touching.all():
            break
        opening[~touching]-=.01
        hold(.05)
    mark("ready_inner_pads")
    hold(.05)
    tracking=False
    mark("open_gap_before_fast_stroke")
    opening=np.maximum(opening,.50)
    tracking=True
    move(points(height=.008),.20)
    hold(.10)
    tracking=False
    launch_start=len(sim.rows)
    mark("opposed_spin_stroke")
    launch_target=points(stroke=.090,height=.008)
    launch_target[:,1]+=.016
    assert CONTROL_HZ==20
    launch_start_tcp=np.stack([tcp_pose(a.q,BASES[side])[0]
                               for a,side in zip(arms,("left","right"),strict=True)])
    paired_step(arms,targets(.5*(launch_start_tcp+launch_target)),opening,bias=False)
    launch_target[:,0]-=.012
    launch_target[:,1]+=.012
    paired_step(arms,targets(launch_target),opening,bias=False)
    hold(.05)
    mark("lift_clear_after_spin")
    move(points(stroke=.090,height=.050),.15)
    mark("open_and_return_to_opposed_cage")
    opening=np.full(2,.80)
    center=sim.pose(0)[:3].copy()
    move(points(height=.050),.15)
    mark("free_revolution_before_braking")
    full_turn=False
    for _ in range(60):
        contacts=[i for i in range(launch_start,len(sim.rows))
                  if any(sim.rows[i]["eggs"][0]["normal_impulse_ns"])
                  or np.linalg.norm(sim.rows[i]["eggs"][0]["contact_angular_impulse_nms"])>1e-10]
        first_clear=contacts[-1]+1 if contacts else launch_start
        free=sim.rows[first_clear:]
        yaw_values=np.unwrap([r["eggs"][0]["yaw"] for r in free])
        free_turn=float(yaw_values[-1]-yaw_values[0])
        if abs(free_turn)>=2*np.pi:
            full_turn=True
            break
        hold(.05)
    assert full_turn, "A complete free revolution is required before the brake"
    stopped=False
    mark("settle_open_above_egg")
    opening=np.full(2,1.0)
    for arm in arms:
        arm.bias[:]=0.
    tracking=True
    hold(.40)
    center=sim.pose(0)[:3].copy()
    center[:2]+=sim.velocity(0)[:2]*.80
    cap_slope=0.
    mark("lower_open_opposed_fingers")
    move(points(height=.012,distance=.014),.60)
    mark("settle_open_opposed_fingers")
    hold(.30)
    tracking=False
    mark("close_inner_pads_to_arrest")
    opening=np.full(2,.44)
    had_contact=False
    for _ in range(8):
        hold(.05)
        recent=[r["eggs"][0] for r in sim.rows[-3:]]
        if all(any(e["pad_normal_impulse_ns"]) and abs(e["velocity"][5])<=ARRESTED_SPEED
               for e in recent):
            stopped=True
            break
        loads=np.asarray([r["eggs"][0]["pad_normal_impulse_ns"] for r in sim.rows[-12:]])/DT
        had_contact |= bool(loads.max()>.05)
        if had_contact:
            loose=loads[-3:].max(0)<.05
            opening[loose]-=.008
    mark("immediate_open_after_arrest" if stopped else "open_after_unqualified_arrest")
    opening=np.full(2,.95)
    hold(.05)
    mark("lift_clear_for_restart_observation")
    move(points(height=.050,distance=.014),.10)
    mark("hands_clear_restart_observation")
    hold(1.0)
