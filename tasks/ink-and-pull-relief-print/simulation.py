"""Native action stepping, contact sampling and neutral replay records."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import carb.settings
import numpy as np
import omni.physics.tensors as tensors
import torch
from isaaclab.sim.utils import get_current_stage_id
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
from pxr import Vt
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.ik import LOWER, UPPER
from yam_arena.station import TABLE_TOP

from contact_rule import PrintContact
from assets import CORAL, PALETTE_COLOR, PAPER, RUBBER
from geometry import CUT, PANELS
from ink import Film
from scene import CAMERAS, CONFIG, DT, HERE, HZ, MAX_STEPS, PHYSICS_HZ, ROOT, SEED, make_env
from scoring import grade, paper_over_relief

INK_COLOR = np.array([.028, .032, .088], dtype=np.float32)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def archive_source(out):
    source = out / "source"
    source.mkdir()
    for path in [*HERE.glob("*.py"), HERE / "dream-defaults.json"]:
        shutil.copy2(path, source / path.name)
    shutil.copytree(HERE / "assets", source / "assets")
    write_json(out / "source_manifest.json", {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*") if p.is_file()})


def as_array(value):
    return value.torch[0].cpu().numpy().copy()


class FilmView:
    def __init__(self, stage):
        paths = [("palette/film", PALETTE_COLOR), ("brayer/drum/film", RUBBER), ("block/film", CORAL)]
        paths.extend((f"sheet/panel_{i}/film", PAPER) for i in range(PANELS))
        self.surfaces = [(stage.GetPrimAtPath(f"{ROOT}/{path}").GetAttribute("primvars:displayColor"),
                          np.array(rgb, dtype=np.float32)) for path, rgb in paths]
        assert all(primvar for primvar, _ in self.surfaces)
        self.last = [None] * len(paths)

    def paint(self, values):
        for index, ((attr, rgb), ink) in enumerate(zip(self.surfaces, values, strict=True)):
            if np.array_equal(ink, self.last[index]):
                continue
            a = ink.reshape(-1, 1).astype(np.float32) / 255
            colors = rgb + a * (INK_COLOR - rgb)
            attr.Set(Vt.Vec3fArray.FromNumpy(colors))
            self.last[index] = ink.copy()


def camera_refresh(base):
    import carb.eventdispatcher
    carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
    carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
    base.sim.render()
    ensure_isaac_rtx_render_update(force=True)


class Simulation:
    def __init__(self, out, *, cameras=False, screenshot=False):
        self.out = Path(out)
        self.env = make_env(cameras=cameras, screenshot=screenshot)
        self.base = self.env.unwrapped
        self.env.reset(seed=SEED)
        assert abs(self.base.physics_dt - DT) < 1e-10
        assert abs(self.base.step_dt - 1/HZ) < 1e-10
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.robots = [self.base.scene[s + "_robot"] for s in ("left", "right")]
        self.ids = [(robot.find_joints(ARM_JOINTS, preserve_order=True)[0],
                     robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]) for robot in self.robots]
        self.low, self.high = np.r_[LOWER, 0., LOWER, 0.], np.r_[UPPER, 1., UPPER, 1.]
        self.props = [self.base.scene[name] for name in ("brayer", "sheet")]
        self.drum_id = self.props[0].body_names.index("drum")
        self.panel_ids = [self.props[1].body_names.index(f"panel_{i}") for i in range(PANELS)]
        self.view = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        filters = [ROOT+"/palette/slab", *[ROOT + f"/block/ridge_{i}" for i in range(len(CUT))]]
        self.drum_contact = self.view.create_rigid_contact_view(ROOT+"/brayer/drum", filters)
        self.sheet_contacts = [self.view.create_rigid_contact_view(ROOT + f"/sheet/panel_{i}", filters)
                               for i in range(PANELS)]
        self.print_contact = PrintContact()
        wet_surfaces = [ROOT+"/palette/slab", ROOT+"/brayer/drum/rubber"]
        self.wet_contacts = [self.view.create_rigid_contact_view(ROOT + f"/sheet/panel_{i}", wet_surfaces,
                                                                max_contact_data_count=32)
                             for i in range(PANELS)]
        self.film = Film(TABLE_TOP, DT)
        self.visual = FilmView(self.base.sim.stage) if cameras else None
        self.steps = self.ticks = self.clear_ticks = 0
        self.ever_inked = False
        self.records, self.commands, self.frames, self.measurements, self.forces = [], [], [], [], []
        self.phase = "Initial"
        for _ in range(PHYSICS_HZ // 2):
            self._advance(self.action if _ == 0 else None)
        assert abs(as_array(self.props[1].data.root_pose_w)[2] - (TABLE_TOP+.040)) < .002
        self._record()
        self._measure()

    @property
    def time(self):
        return self.ticks * DT

    def _advance(self, action=None):
        base = self.base
        if action is not None:
            base.action_manager.process_action(torch.tensor(action[None], dtype=torch.float32))
        base.action_manager.apply_action()
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(DT)

    def measures(self):
        drum = as_array(self.props[0].data.body_link_pose_w)[self.drum_id]
        panels = as_array(self.props[1].data.body_link_pose_w)[self.panel_ids]
        roller_force = np.maximum(0., self.drum_contact.get_contact_force_matrix(DT)[0, :, 2])
        sheet_force = np.maximum(0., np.stack(
            [v.get_contact_force_matrix(DT)[0, :, 2] for v in self.sheet_contacts]))
        return drum, roller_force, panels, sheet_force

    def sampled_ink(self):
        film = self.film
        to_byte = lambda x: np.round(np.sqrt(np.clip(x, 0, 1)) * 255).astype(np.uint8)
        from geometry import BLOCK_MASK
        return tuple(to_byte(x) for x in (film.palette, film.roller, film.block[BLOCK_MASK], *film.paper))

    def _record(self):
        row = {"time_s": self.time}
        for name, obj in zip(("left", "right", "brayer", "sheet"), [*self.robots, *self.props], strict=True):
            row[name+"_q"] = as_array(obj.data.joint_pos)
            row[name+"_pose"] = as_array(obj.data.root_pose_w)
        self.records.append(row)
        self.frames.append(self.sampled_ink())

    def _measure(self):
        panels = as_array(self.props[1].data.body_link_pose_w)[self.panel_ids]
        if paper_over_relief(panels, TABLE_TOP):
            self.clear_ticks = 0
        else:
            self.clear_ticks += PHYSICS_HZ // HZ
        self.info = grade(self.film.block, self.film.paper, self.clear_ticks >= PHYSICS_HZ*.4,
                          self.ever_inked, print_eligible=self.print_contact.first is None)
        self.ever_inked |= self.info["block_coverage"] >= .60
        self.info.update(time_s=self.time, phase=self.phase,
                         rollout_mm=self.film.roll_mm.tolist(), skid_mm=self.film.skid_mm.tolist(),
                         paper_slip_mm=self.film.paper_slide_mm,
                         paper_transfer=self.film.paper_transfer.tolist(),
                         relief_transfer=self.film.paper_relief.sum(axis=(1, 2)).tolist(),
                         slab_transfer=(self.film.paper - self.film.paper_relief).sum(axis=(1, 2)).tolist(),
                         ink_mass_error=self.film.mass()-self.film.initial_mass,
                         first_forbidden_contact=self.print_contact.first)
        self.measurements.append(self.info.copy())

    def step(self, action):
        assert self.steps < MAX_STEPS
        assert action.shape == (14,) and np.isfinite(action).all()
        self.action = np.clip(action, self.low, self.high).astype(np.float32)
        self.commands.append(self.action.copy())
        for i in range(PHYSICS_HZ // HZ):
            self._advance(self.action if i == 0 else None)
            drm, rf, panels, pf = self.measures()
            self.film.step(drm, rf, panels, pf)
            self.ticks += 1
            self.print_contact.update_loaded(self.film)
            for index, (view, panel) in enumerate(zip(self.wet_contacts, panels, strict=True)):
                self.print_contact.observe(view, panel, index, DT, self.time)
            self.forces.append((self.time, *drm, *rf, *panels.ravel(), *pf.ravel()))
            if self.ticks % (PHYSICS_HZ // 30) == 0:
                self._record()
        self.steps += 1
        self._measure()

    def capture(self, names=CAMERAS):
        self.visual.paint(self.sampled_ink())
        camera_refresh(self.base)
        pictures = {}
        for name in names:
            sensor = self.base.scene[name]
            sensor.update(1/HZ, force_recompute=True)
            pictures[name] = as_array(sensor.data.output["rgb"])[..., :3]
        return pictures

    def save(self, kind, status, error=None):
        archive_source(self.out)
        write_json(self.out / "control-measurements.json", self.measurements)
        np.savez_compressed(self.out / "trajectory.npz",
            **{key: np.asarray([row[key] for row in self.records]) for key in self.records[0]},
            actions=self.commands, contacts=self.forces, palette=self.film.palette,
            block=self.film.block, brayer=self.film.roller, paper=self.film.paper,
            paper_relief=self.film.paper_relief, paper_slab=self.film.paper-self.film.paper_relief,
            **{("ink_"+str(i)): np.asarray(seq) for i, seq in enumerate(zip(*self.frames))})
        result = {"kind":kind, "status":status, "max_score":1.,
                  "gpu_run_id":os.environ["PRODUCTION_GPU_RUN_ID"],
                  "raw_measurements":str(self.out / "control-measurements.json"),
                  "trajectory":str(self.out / "trajectory.npz"),
                  "control_steps":self.steps, "physics_ticks":self.ticks,
                  "error":error, **self.info}
        write_json(self.out / "result.json", result)
        print("RESULT", json.dumps(result), flush=True)

    def close(self):
        self.env.close()
