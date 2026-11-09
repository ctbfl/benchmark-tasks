"""Physical button state, simulation-clock display, and task measurements."""
import numpy as np
from scipy.spatial.transform import Rotation

from build_assets import CUP_FLOOR, CUP_HEIGHT, CUP_INNER, PEN_LENGTH, PEN_RADIUS, SEGMENTS
from scene import CONFIG, ROOT


class Timer:
    def __init__(self):
        self.running = False
        self.armed = True
        self.elapsed = 0.
        self.started_at = None
        self.stopped_at = None
        self.events = []

    def update(self, time, displacement):
        event = None
        if displacement > -.001:
            self.armed = True
        if self.armed and displacement <= -.003:
            self.armed = False
            self.running = not self.running
            if self.running:
                self.started_at = time
                self.elapsed = 0.
                self.stopped_at = None
                event = "start"
            else:
                self.stopped_at = time
                self.elapsed = time - self.started_at
                event = "stop"
            self.events.append({"event": event, "time_s": time})
        if self.running:
            self.elapsed = time - self.started_at
        return event

    @property
    def display(self):
        centiseconds = int(self.elapsed*100+1e-7)
        return f"{centiseconds//6000:02}:{centiseconds//100%60:02}:{centiseconds%100:02}"


def in_holder(pen, cup):
    """The pen's lower end and its passage through the rim lie in the cavity."""
    inverse = Rotation.from_quat(cup[3:]).inv()
    ends_world = np.asarray(pen[:3]) + Rotation.from_quat(pen[3:]).apply([[0, 0, -PEN_LENGTH/2], [0, 0, PEN_LENGTH/2]])
    ends = inverse.apply(ends_world - np.asarray(cup[:3]))
    ends = ends[np.argsort(ends[:, 2])]
    low, high = ends
    if not (CUP_FLOOR-.002 <= low[2] < CUP_HEIGHT and high[2] > CUP_FLOOR):
        return False
    if Rotation.from_quat(cup[3:]).apply([0, 0, 1])[2] <= 0:
        return False
    rim = low + (high-low) * np.clip((CUP_HEIGHT-low[2]) / max(high[2]-low[2], 1e-9), 0, 1)
    # The physical wall contacts constrain the pen radius; test its axis here.
    # A constant maximum radius would reject a tapered tip resting in a corner.
    half = np.asarray(CUP_INNER)/2
    radius = .010
    for point in (low, rim):
        q = np.abs(point[:2]) - (half-radius)
        if np.linalg.norm(np.maximum(q, 0)) + min(max(q), 0) > radius:
            return False
    return True


class TaskRuntime:
    def __init__(self, env):
        import carb
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.robot import asset_link_paths

        self.base = env.unwrapped
        self.timer = Timer()
        self.ticks = 0
        self.time = 0.
        self.measurements = []
        self.start_valid = False
        self.stop_valid = False
        self.early_pickup = False
        self.result = {"score": 0., "contained": [], "released": []}
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        paths = asset_link_paths()
        robot_paths = [ROOT + "/" + side + paths[body] for side in ("LeftArm", "RightArm")
                       for body in self.base.scene["left_robot"].body_names]
        self.contacts = [self.physics.create_rigid_contact_view(ROOT+f"/pen_{i}/body", robot_paths) for i in range(5)]
        supports = [ROOT+"/cup/body", *[ROOT+f"/pen_{i}/body" for i in range(5)]]
        self.support_contacts = [self.physics.create_rigid_contact_view(ROOT+f"/pen_{i}/body", supports) for i in range(5)]
        self.initial_positions = [self.pose(f"pen_{i}")[:3].copy() for i in range(5)]
        self.displayed = None
        self.measure()
        self.update_display()

    def pose(self, name):
        return self.base.scene[name].data.root_pose_w.torch[0].cpu().numpy().copy()

    def measure(self, event=None):
        cup = self.pose("cup")
        pens = [self.pose(f"pen_{i}") for i in range(5)]
        forces = [float(np.linalg.norm(contact.get_contact_force_matrix(1/CONFIG["simulation"]["physics_hz"]), axis=-1).sum())
                  for contact in self.contacts]
        contained = [bool(in_holder(pen, cup)) for pen in pens]
        released = [force < .01 for force in forces]
        support = np.array([np.linalg.norm(contact.get_contact_force_matrix(1/CONFIG["simulation"]["physics_hz"]), axis=-1).reshape(-1)
                            for contact in self.support_contacts]) > .001
        supported = set(np.flatnonzero(support[:, 0]))
        for _ in range(5):
            supported |= {i for i in range(5) if any(support[i, j+1] for j in supported)}
        if self.timer.started_at is None:
            # Lifting a pen off the tabletop before the start violates the requested order.
            self.early_pickup |= any(p[2] - initial[2] > PEN_RADIUS*2 for p, initial in zip(pens, self.initial_positions))
        if event == "start":
            self.start_valid = not self.early_pickup and not any(contained) and len(self.timer.events) == 1
            self.stop_valid = False
        if event == "stop":
            self.stop_valid = self.start_valid and all(contained) and all(released)
        # Retention is checked again at the final measured state, not latched on entry.
        success = (self.stop_valid and not self.timer.running and all(contained) and all(released) and len(supported) == 5
                   and self.time <= CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"])
        self.result = {"score": float(success), "contained": contained, "released": released,
                       "supported": sorted(int(i) for i in supported),
                       "start_valid": self.start_valid, "stop_valid": self.stop_valid}
        self.measurements.append({"time_s": self.time, "cup": cup.tolist(),
            "pens": [p.tolist() for p in pens], "robot_forces_N": forces,
            "button_m": self.button_position(), "display": self.timer.display,
            "event": event, **self.result})

    def button_position(self):
        return float(self.base.scene["timer"].data.joint_pos.torch[0, 0])

    def update_display(self):
        from pxr import UsdGeom
        display = self.timer.display
        if display == self.displayed:
            return
        self.displayed = display
        for i, digit in enumerate(display.replace(":", "")):
            for segment in "abcdefg":
                prim = self.base.sim.stage.GetPrimAtPath(ROOT+f"/timer/base/display/d{i}_{segment}")
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if segment in SEGMENTS[int(digit)] else "invisible")

    def step(self, action, on_physics_step=None):
        base = self.base
        hz = CONFIG["simulation"]["physics_hz"]
        decimation = hz // CONFIG["simulation"]["control_hz"]
        base.action_manager.process_action(action)
        for _ in range(decimation):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(1/hz)
            self.ticks += 1
            self.time = self.ticks/hz
            event = self.timer.update(self.time, self.button_position())
            if event or self.ticks % decimation == 0:
                self.measure(event)
            self.update_display()
            if on_physics_step is not None:
                on_physics_step()

    def state(self):
        return {"time_s": self.time, "timer_running": self.timer.running,
                "elapsed_s": self.timer.elapsed, "display": self.timer.display,
                "events": self.timer.events.copy(), **self.result}
