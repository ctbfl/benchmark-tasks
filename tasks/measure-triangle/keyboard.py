"""Physical calculator key validation using only the native right-arm interface."""

import numpy as np
from scipy.spatial.transform import Rotation

from expert import Expert


def press_keys(expert, keys):
    sim = expert.sim
    down = Rotation.from_euler("z", -90, degrees=True)*Rotation.from_euler("x", 180, degrees=True)
    expert.grip(1, 0.)
    for number, key in enumerate(keys):
        sim.phase = "press_"+key
        index = sim.calculator.body_names.index("key_"+key)
        point = sim.calculator.data.body_link_pose_w.torch[0, index, :3].cpu().numpy().copy()
        if number == 0:
            expert.move(1, point+[0, 0, .10], down, 3.)
        expert.move(1, point+[0, 0, .040], down, 1.2)
        expert.move(1, point+[0, 0, .0068], down, 1.2)
        expert.wait(.3)
        print("KEY", key, sim.instruments.events, flush=True)
        assert [event["key"] for event in sim.instruments.events] == list(keys[:number+1])
        expert.move(1, point+[0, 0, .040], down, .8)
    p, r = expert.hand(1)
    expert.move(1, p+[0, 0, .08], r, 1.5)
    expert.wait(1.)
    if keys[-1] == "equals":
        sim.phase = "inspect_result"
        pose = sim.calculator.data.root_link_pose_w.torch[0].cpu().numpy()
        screen = pose[:3]+Rotation.from_quat(pose[3:]).apply([0, .047, .014])
        expert.observe(1, screen, 270)


def run(out):
    from simulation import Simulation, archive_source
    archive_source(out)
    sim = Simulation(out)
    try:
        press_keys(Expert(sim), ("1", "0", "0", "C", "5", "0", "C", "1", "plus", "2", "equals"))
        assert sim.instruments.calculator.submitted == "3"
        positions = sim.calculator.data.joint_pos.torch[0].cpu().numpy()
        assert np.all(positions > -.00025), "A released calculator key is still depressed"
        sim.save("keyboard_check_passed")
    except Exception as error:
        sim.save("error", str(error))
        raise
    finally:
        sim.close()
