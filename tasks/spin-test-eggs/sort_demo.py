"""Scripted sorting with class decisions derived from measured shell rotation."""
import numpy as np
from yam_arena.constants import READY
from yam_arena.station import TABLE_TOP

from assets import SHORT, TRAYS, TRAY_FLOOR
from expert import Arm, spin
from scoring import SpinTest, check_spin, shell_span
from simulation import SUBSTEPS, write_json

def near_center_grasp(yaw, side):
    center = -.55 if side == 0 else .55
    return np.pi + center + (yaw-center+np.pi/2) % np.pi - np.pi/2


def park(arm):
    arm.sim.phase = "park"
    start = arm.q.copy()
    for phase in np.linspace(0, 1, 21)[1:]:
        t = phase*phase*(3-2*phase)
        arm.step((1-t)*start + t*np.asarray(READY), 1., bias=False)
    arm.hold(.20, bias=False)


def carry(arm, egg_index, xy, floor):
    """Move a shell by the pads, preserving the measured grasp-to-shell offset."""
    sim = arm.sim
    pose = sim.pose(egg_index)
    yaw = sim.shell_angle(egg_index)
    approach_yaw = near_center_grasp(yaw, arm.side)
    pick = pose[:3] + [0., 0., .007]
    up = np.array([0., 0., .060])
    sim.phase = "approach_pick"
    arm.move(pick+up, approach_yaw, .88, seconds=1.0)
    arm.move(pick, approach_yaw, .68, seconds=.55, settle=.20)
    sim.phase = "grasp"
    arm.hold(.20, .426)
    for target in np.linspace(.426, .395, 6)[1:]:
        arm.step(arm.q, target)
    arm.hold(.15)
    sim.phase = "carry"
    arm.move(pick+up, approach_yaw, .395, seconds=.55, settle=.18)
    assert sim.pose(egg_index)[2] >= pose[2]+.04, "Egg did not lift with the jaws"
    hover = np.r_[xy, floor + SHORT + .009] + up
    arm.move(hover, approach_yaw, .395, seconds=.8, settle=.15)
    shell = sim.pose(egg_index)
    radius = shell_span(shell)[2]
    # Set the opening height from the lifted shell instead of pressing it into the surface.
    place = hover - [0., 0., shell[2] - floor - radius - .004]
    sim.phase = "place"
    arm.move(place, approach_yaw, .395, seconds=.65, settle=.18)
    arm.hold(.20, .86)
    arm.move(hover, approach_yaw, .86, seconds=.45, settle=.25)


def read_motion(rows, index, test):
    release = test["brake_release_tick"]
    end = test["observation_end_tick"]
    yaw = np.unwrap([row["eggs"][index]["yaw"] for row in rows
                     if release <= row["tick"] <= end and (row["tick"]-release) % SUBSTEPS == 0])
    assert len(yaw) >= 5, "Incomplete hands-off observation"
    direction = np.sign(test["spin_speed_rad_s"])
    returned = float(max(direction*(yaw-yaw[0])))
    return returned >= np.deg2rad(5), {
        "release_tick": release, "post_release_turn_deg": float(np.rad2deg(returned)),
        "frame_yaw_deg": np.rad2deg(yaw-yaw[0]).tolist(),
    }


def sort(sim, frame_hook):
    arms = [Arm(sim, side, frame_hook) for side in (0, 1)]
    pose = sim.pose(0)
    for _ in range(2):
        start = len(sim.rows)
        spin(arms, sim.pose(0)[:3], sim.shell_angle(0))
        test = SpinTest()
        for row in sim.rows[start:]:
            egg = row["eggs"][0]
            observed_pose = np.asarray(egg["pose"])
            check_spin(test, row["tick"], egg, observed_pose, shell_span(observed_pose))
        if test.evidence is not None:
            break
    else:
        raise RuntimeError("Two attempts did not complete the spin and abrupt-stop observation")
    raw, measures = read_motion(sim.rows[start:], 0, test.evidence)
    destination = "raw" if raw else "boiled"
    write_json(sim.out/"demo-motion-decisions.json", {
        "egg_position_at_test_m": pose[:3].tolist(), "chosen_basket": destination, **measures})
    side = 0 if raw else 1
    park(arms[1-side])
    carry(arms[side], 0, TRAYS[destination], TABLE_TOP+TRAY_FLOOR)
    park(arms[side])
    sim.phase = "settle_final"
    arms[0].hold(.70, bias=False)
