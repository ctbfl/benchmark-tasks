"""Feedback demonstration using the measured valve stroke and remaining gravy."""

import math

import numpy as np
from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP, READY

from geometry import DAMPING, PADDLE, PADDLE_TOP, SPRING, XY
from liquid import DT, HZ, LIMIT, SEAT, STROKE


def run(env):
    q = np.array(READY)
    arm_base = station.BASES["right"]
    x, y = XY[0]+PADDLE[0], XY[1]+PADDLE[1]
    tilt = .44
    orientation = (math.cos(tilt/2), 0., math.sin(tilt/2), 0.)
    x -= math.sin(tilt) * PAD_TIP_PAST_TCP
    z = station.TABLE_TOP + PADDLE_TOP + math.cos(tilt) * PAD_TIP_PAST_TCP
    open_z = z - STROKE - .007
    waypoint = {}
    for name, height in (("above", z+.038), ("near", z+.010), ("press", open_z)):
        point = (x, y, height)
        q, error, angle = ik.solve(point, orientation, q_init=q, base_pos=arm_base)
        assert error < .002 and angle < .03, (name, error, angle)
        waypoint[name] = q

    q = np.asarray(READY)
    grip = 1.

    def move(goal, next_grip, seconds):
        nonlocal q, grip
        steps = round(seconds * HZ)
        for i in range(1, steps + 1):
            pos = ik.smoothstep(i / steps)
            env.advance([*READY, 1., *(q+(goal-q)*pos), grip+(next_grip-grip)*pos])
        q, grip = goal, next_grip

    def hold():
        env.advance([*READY, 1., *q, grip])
        assert env.drain.ticks * DT * HZ < LIMIT, env.metrics()

    move(waypoint["above"], 0., 1.60)
    move(waypoint["near"], 0., .60)
    move(waypoint["press"], 0., .55)
    for _ in range(8):
        hold()
    print("Held open", env.metrics(), flush=True)
    assert env.peak_stroke > STROKE * .70 and env.peak_contact > .5, env.metrics()

    # Estimate the flow that continues while the pads retreat and the spring returns.
    while True:
        drain = env.drain
        prediction = drain.flow * (.13 + DAMPING / SPRING)
        if drain.gravy <= 5.2 + prediction:
            break
        hold()
    print("Releasing", env.metrics(), flush=True)
    move(waypoint["near"], 0., .22)
    move(waypoint["above"], 0., .40)
    move(np.asarray(READY), 1., 1.10)
    hold()
    print("Settled", env.metrics(), flush=True)
    assert env.drain.stroke <= SEAT
