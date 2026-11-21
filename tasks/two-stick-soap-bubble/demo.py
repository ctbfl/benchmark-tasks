"""Pick up both sticks and form a bubble through the native position action."""
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from film import in_solution
from geometry import (CORD_RADIUS, GRIP_X, GRIP_Z, HALF_SPAN, RIM, ROOT, SURFACE,
                      TABLE_TOP, TIP, TRAY_X, TRAY_Y, WANDS)
from simulation import CONTROL_HZ, DT, READY_ACTION, Simulation, run_directory, start_app, strands, write_json
from yam_arena.constants import DOWN_XYZW, TCP_OFFSET
from yam_arena.ik import smoothstep, solve
from yam_arena.station import BASES

DOWN = Rotation.from_quat(DOWN_XYZW) * Rotation.from_euler("z", 180., degrees=True)
GRIP_COMMAND = .20
APPROACH_SEED = np.array([-1.14, 1.58, 1.85, -1.51, -.54, -1.08])
SIDES = ("left", "right")


class Expert:
    def __init__(self, sim, probe=False):
        self.sim = sim
        self.desired = READY_ACTION.copy()
        self.previous = self.desired.copy()
        self.bias = np.zeros(14)
        self.pose = None
        self.probe = probe
        self.pickup_rows = []
        self.dip_rows = []
        self.finger_paths = []
        if probe:
            self.setup_contacts()

    def setup_contacts(self):
        import carb
        from yam_arena.robot import asset_link_paths

        paths = asset_link_paths()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.contacts = []
        for side, wand in zip(("Left", "Right"), WANDS):
            pads = [f"/World/envs/env_0/{side}Arm" + paths[f"tip_{finger}"] for finger in ("left", "right")]
            self.finger_paths.append(pads)
            supports = [ROOT + f"/Cradle/post_{i}" for i in range(4)]
            walls = [ROOT + f"/Basin/{edge}_{i}" for edge in ("end","side") for i in range(2)]
            self.contacts.append(self.sim.physics.create_rigid_contact_view(ROOT + "/" + wand, pads + supports + walls))

    def pickup_state(self):
        forces, tcps, orientations, links, native = [], [], [], [], []
        for j, (robot, contacts) in enumerate(zip(self.sim.robots, self.contacts)):
            pos = robot.data.body_link_pose_w.torch[0].cpu().numpy()
            grip = pos[robot.body_names.index("gripper")]
            tcps.append(grip[:3] + Rotation.from_quat(grip[3:]).apply(TCP_OFFSET))
            orientations.append(grip[3:].tolist())
            links.append([pos[robot.body_names.index(f"tip_{finger}")].tolist() for finger in ("left", "right")])
            forces.append(np.linalg.norm(contacts.get_contact_force_matrix(DT)[0], axis=-1))
            indices = self.sim.ids[j][0]
            drive = robot.actuators.target_command
            gravity = robot.data.gravity_compensation_forces.torch[0, -robot.num_joints:]
            term = self.sim.base.action_manager.get_term(SIDES[j] + "_arm")
            native.append({"requested_rad": (self.desired+self.bias)[7*j:7*j+6].tolist(),
                           "processed_rad": term.processed_actions[0].cpu().tolist(),
                           "target_rad": drive.position.torch[0, indices].cpu().tolist(),
                           "measured_rad": robot.data.joint_pos.torch[0, indices].cpu().tolist(),
                           "effort_target_Nm": drive.effort.torch[0, indices].cpu().tolist(),
                           "gravity_Nm": gravity[indices].cpu().tolist(),
                           "grip_effort_estimate_N": robot.actuators["gripper"].computed_effort[0].cpu().tolist()})
        return {"phase": self.sim.phase, "time_s": self.sim.steps/CONTROL_HZ,
                "tcp": np.asarray(tcps).tolist(), "tcp_quat_xyzw": orientations, "pad_link_pose": links,
                "wand": self.sim.poses[:2].tolist(), "force_N": np.asarray(forces).tolist(),
                "native_drive": native}


    def dip_sample(self):
        poses = self.sim.poses
        nodes = poses[2:,:3]
        upper, lower = strands(poses)
        ties = upper[[0,-1]]
        axes = Rotation.from_quat(poses[:2,3:]).apply([1.,0.,0.])
        rod_centers = (poses[:2,None,:3] + axes[:,None,:]*np.linspace(-.053,.191,81)[None,:,None]).reshape(-1,3)
        sleeve_centers = (poses[:2,None,:3] + axes[:,None,:]*np.linspace(-.042,.042,29)[None,:,None]).reshape(-1,3)
        x0,x1 = TRAY_X
        y0,y1 = TRAY_Y
        w = .008
        walls = [([x0-w,y0-w,TABLE_TOP], [x0,y1+w,RIM]),
                 ([x1,y0-w,TABLE_TOP], [x1+w,y1+w,RIM]),
                 ([x0,y0-w,TABLE_TOP], [x1,y0,RIM]),
                 ([x0,y1,TABLE_TOP], [x1,y1+w,RIM])]

        def clearance(points, radius):
            return min(np.linalg.norm(points-np.clip(points, lo, hi), axis=1).min()
                       for lo, hi in walls)-radius

        return {"time_s": self.sim.steps/CONTROL_HZ, "phase": self.sim.phase,
                "cord_nodes_xyz_m": nodes.tolist(), "ties_xyz_m": ties.tolist(),
                "submerged_nodes": int(in_solution(nodes).sum()), "submerged_ties": int(in_solution(ties).sum()),
                "lowest_cord_m": float(nodes[:,2].min()), "wet": self.sim.film.wet_until > self.sim.film.time,
                "wet_remaining_s": max(0., self.sim.film.wet_until-self.sim.film.time),
                "wand_to_tray_wall_m": float(min(clearance(rod_centers, .006), clearance(sleeve_centers, .012))),
                "cord_to_tray_wall_m": float(clearance(nodes, CORD_RADIUS)),
                "grip_contact_N": [f[:2] for f in self.pickup_rows[-1]["force_N"]],
                "wand_to_tray_contact_N": [f[6:] for f in self.pickup_rows[-1]["force_N"]]}

    def step(self):
        self.sim.step(self.desired+self.bias)
        if self.probe:
            self.pickup_rows.append(self.pickup_state())
            if self.sim.phase in ("raise_loop_clear", "over_basin", "immerse", "withdraw"):
                self.dip_rows.append(self.dip_sample())
        q = self.sim.measured_action()
        for j in (0, 7):
            arm = slice(j, j+6)
            if np.max(np.abs(self.desired[arm]-self.previous[arm])) < .028:
                self.bias[arm] = np.clip(self.bias[arm]+.16*(self.desired[arm]-q[arm]), -.13, .13)
        self.previous = self.desired.copy()

    def wait(self, duration):
        for _ in range(round(duration*CONTROL_HZ)):
            self.step()

    def move(self, name, x, y, z, duration, *, pitch=-35., toe=0., grip=None, steady=False):
        self.sim.phase = name
        target = np.column_stack((np.broadcast_to(x, 2), [y, -y] if np.isscalar(y) else y,
                                  np.broadcast_to(z, 2), np.broadcast_to(pitch, 2), [toe, -toe]))
        start_q = self.desired.copy()
        initial = self.pose is None
        start = target if initial else self.pose
        for i in range(1, round(duration*CONTROL_HZ)+1):
            u = i/round(duration*CONTROL_HZ)
            u = u if steady else smoothstep(u)
            for j, sign in enumerate((1, -1)):
                px, py, pz, pitch0, toe0 = start[j] + u*(target[j]-start[j])
                index = 7*j
                arm = slice(index,index+6)
                rotation = Rotation.from_euler("ZY", [-toe0, pitch0], degrees=True) * DOWN
                seed = APPROACH_SEED * [sign, 1, 1, 1, sign, sign] if initial else self.desired[arm]
                q, ep, er = solve((px, py, pz), rotation.as_quat(), seed,
                                  BASES[SIDES[j]], iters=90, damping=.015, tol=5e-5)
                assert ep < .002 and er < .025, (name, SIDES[j], ep, er, (px, py, pz, pitch0, toe0))
                self.desired[arm] = start_q[arm] + u*(q-start_q[arm]) if initial else q
                if grip is not None:
                    self.desired[index+6] = start_q[index+6] + u*(grip-start_q[index+6])
            self.step()
        self.pose = target
        print(name, "t", self.sim.steps/CONTROL_HZ, self.sim.film.sample, "wand grip",
              np.round(self.sim.poses[:2,:3], 4).tolist(), "score", self.sim.film.score, flush=True)

    def pickup(self):
        # Read the settled grip sleeves, then approach from above and from the rear.
        rest = self.sim.poses[:2].copy()
        axes = Rotation.from_quat(rest[:,3:]).apply([1.,0.,0.])
        nominal = np.array([[GRIP_X, HALF_SPAN, GRIP_Z], [GRIP_X, -HALF_SPAN, GRIP_Z]])
        assert np.max(np.linalg.norm(rest[:,:3]-nominal, axis=1)) < .025, rest
        assert np.min(axes[:,0]) > .96, axes
        picked = rest[:,:3] + .014*axes
        x, y, z = picked.T
        self.move("approach", x-.020, y, z+.053, 2.4, grip=.53, pitch=-46.)
        self.wait(.65)
        self.move("bracket_grips", x, y, z+.002, 1.15, pitch=-46.)
        self.wait(.45)
        self.move("pinch", x, y, z+.002, .55, grip=GRIP_COMMAND, pitch=-46.)
        self.wait(.40)
        self.move("short_lift", x, y, z+.054, 1., pitch=-46.)
        self.wait(.50)
        rise = self.sim.poses[:2,2]-rest[:,2]
        if self.probe:
            final = self.pickup_rows[-6:]
            forces = np.array([r["force_N"] for r in final])
            grip_contact = forces[:,:,:2].mean(axis=0)
            supports = forces[:,:,2:6].max()
            slip = np.linalg.norm((np.array(final[-1]["tcp"])-self.sim.poses[:2,:3]) -
                                  (np.array(final[0]["tcp"])-np.array(final[0]["wand"])[:,:3]), axis=1)
            data = {"nominal_sleeve_center_m": nominal.tolist(), "settled_sleeve_pose": rest.tolist(),
                    "sleeve_axes": axes.tolist(), "final_rise_m": rise.tolist(),
                    "mean_bilateral_contact_N": grip_contact.tolist(), "max_cradle_force_N": float(supports),
                    "relative_slip_m": slip.tolist(), "finger_paths": self.finger_paths,
                    "samples": self.pickup_rows}
            write_json(self.sim.out / "pickup-evidence.json", data)
            print("PICKUP EVIDENCE", {k:v for k,v in data.items() if k!="samples"}, flush=True)
            assert grip_contact.min() > .25 and supports < .1 and slip.max() < .007
        assert rise.min() > .035, ("Both sticks must rise clear of the cradle", rise)

    def dip(self):
        self.move("raise_loop_clear", .250, HALF_SPAN, 1.048, 1.10, pitch=-73.)
        self.move("over_basin", .403, HALF_SPAN, 1.046, 1.20, pitch=-73.)
        self.move("immerse", .405, HALF_SPAN, SURFACE-.018+(TIP[0]-.014)*.5, 1.85, pitch=-16.)
        self.wait(.75)
        self.move("withdraw", .406, HALF_SPAN, 1.044, 1.85, pitch=-75.)
        self.wait(.70)
        if self.probe:
            holds = [r for r in self.dip_rows if r["submerged_ties"] == 2]
            fully_dipped = [r for r in holds if r["submerged_nodes"] == len(self.sim.poses)-2]
            after = self.dip_rows[-1]
            data = {"liquid_height_m": SURFACE, "rim_height_m": RIM,
                    "fully_immersed_samples": len(fully_dipped),
                    "wet_after_lift": bool(after["wet"]), "wet_remaining_s": after["wet_remaining_s"],
                    "final_lowest_cord_m": after["lowest_cord_m"],
                    "minimum_wand_clearance_m": min(r["wand_to_tray_wall_m"] for r in self.dip_rows),
                    "max_wand_to_tray_contact_N": float(np.max([r["wand_to_tray_contact_N"] for r in self.dip_rows])),
                    "minimum_cord_clearance_m": min(r["cord_to_tray_wall_m"] for r in self.dip_rows),
                    "min_pad_normal_contact_N": float(np.min([r["grip_contact_N"] for r in self.dip_rows])),
                    "max_pad_normal_contact_N": float(np.max([r["grip_contact_N"] for r in self.dip_rows])),
                    "samples": self.dip_rows}
            write_json(self.sim.out / "immersion-evidence.json", data)
            print("IMMERSION EVIDENCE", {k:v for k,v in data.items() if k!="samples"}, flush=True)
            assert fully_dipped and data["min_pad_normal_contact_N"] > .25, data
            assert data["minimum_wand_clearance_m"] > .002 and data["max_wand_to_tray_contact_N"] < .2, data
        assert self.sim.film.wet_until > self.sim.film.time, self.sim.film.result()
        assert self.sim.poses[2:,2].min() > SURFACE+.007, self.sim.film.sample

    def run(self):
        self.pickup()
        self.dip()
        self.move("gather_wet_loop", .432, .103, 1.049, .65, pitch=-74., toe=9.)
        self.move("spread", .432, .118, 1.049, .60, pitch=-74.)
        self.move("inflate", .229, .118, 1.049, .78, pitch=-73., steady=True)
        self.wait(.65)
        self.move("close_tips", .229, .101, 1.049, .72, pitch=-73., toe=31.)
        self.sim.phase = "free_flight"
        self.wait(1.40)
        self.save_air_motion()

    def save_air_motion(self):
        stages = ("gather_wet_loop", "spread", "inflate", "close_tips", "free_flight")
        rows = [r for r in self.pickup_rows if r["phase"] in stages]
        track = [r for r in self.sim.measurements if r["phase"] in stages]
        forces = np.array([r["force_N"] for r in rows])[:,:,:2]
        offsets = np.array([Rotation.from_quat(r["tcp_quat_xyzw"]).inv().apply(
            np.asarray(r["wand"])[:,:3]-np.asarray(r["tcp"])) for r in rows])
        drive = np.abs(np.array([[arm["grip_effort_estimate_N"] for arm in row["native_drive"]] for row in rows]))
        sustained = max(forces[i:i+5].mean(axis=0).max() for i in range(len(forces)-4))
        result = {"min_pad_contact_N": float(forces.min()), "max_pad_contact_N": float(forces.max()),
                  "median_pad_contact_N": float(np.median(forces)), "peak_quarter_second_contact_mean_N": float(sustained),
                  "grip_drive_effort_estimate_N": dict(median=float(np.median(drive)), peak=float(drive.max())),
                  "maximum_wand_translation_relative_to_tcp_m": np.linalg.norm(offsets-offsets[0],axis=-1).max(axis=0).tolist(),
                  "peak_air_speed_m_s": max(r["air_m_s"] for r in track),
                  "peak_mouth_area_m2": max(r["area_m2"] for r in track),
                  "minimum_tip_span_m": min(r["span_m"] for r in track),
                  "film_events": self.sim.film.events, "motion": rows}
        write_json(self.sim.out / "air-motion-evidence.json", result)
        print("AIR MOTION EVIDENCE", {k:v for k,v in result.items() if k!="motion"}, flush=True)
        assert result["min_pad_contact_N"] > .25, "Both pads must keep holding each wand"


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in ("--pickup", "--dip") else None
    assert len(args) <= 1, "Usage: python demo.py [--pickup | --dip] [RUN_DIRECTORY]"
    kind = {None: "demo", "--pickup": "pickup", "--dip": "immersion"}[mode]
    out = run_directory(kind, args)
    app = start_app(cameras=False)
    sim = None
    error = None
    try:
        sim = Simulation(out, cameras=False)
        expert = Expert(sim, probe=True)
        if mode is not None:
            expert.pickup()
            if mode == "--dip":
                expert.dip()
        else:
            expert.run()
            assert sim.film.score == 1., sim.film.result()
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        if sim is not None:
            sim.save(kind, error)
            sim.close()
        app.close(exit_code=int(error is not None))


if __name__ == "__main__":
    main()
