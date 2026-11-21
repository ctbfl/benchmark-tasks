"""Recompute film provenance and scoring from a recorded native YAM execution."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from yam_arena.station import TABLE_TOP

from contact_rule import CONTACT_MARGIN, archived_clearance
from geometry import BLOCK_MASK, CUT, PANELS
from ink import Film
from scene import DT, HERE, HZ, PHYSICS_HZ
from scoring import grade, paper_over_relief


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def evaluate_archive(original, out):
    old = json.loads((original / "result.json").read_text())
    manifest = json.loads((original / "source_manifest.json").read_text())
    assert old["status"] == "success" and old["kind"] in {"demo", "dream"}
    for name in ("geometry.py", "scene.py", "dream-defaults.json", "assets.py", "expert.py", "dream_eval.py",
                 *(n for n in manifest if n.startswith("assets/"))):
        assert digest(HERE / name) == manifest[name], name
    trajectory = original / "trajectory.npz"
    decimation = PHYSICS_HZ // HZ
    stride = PHYSICS_HZ // 30
    channels = len(CUT)+1
    shot_file = original / "control-measurements.json"
    recorded = json.loads(shot_file.read_text())
    film = Film(TABLE_TOP, DT)
    history = []
    clear_ticks = decimation
    ever_inked = False

    def record(time, step):
        nonlocal ever_inked
        row = grade(film.block, film.paper, clear_ticks >= PHYSICS_HZ*.4, ever_inked, print_eligible=True)
        ever_inked |= row["block_coverage"] >= .60
        before = recorded[step]
        assert row["sheet_clear"] == before["sheet_clear"]
        assert row["block_coverage"] == before["block_coverage"]
        row.update(time_s=time, phase=before["phase"],
                   relief_transfer=film.paper_relief.sum(axis=(1,2)).tolist(),
                   slab_transfer=(film.paper-film.paper_relief).sum(axis=(1,2)).tolist(),
                   ink_mass_error=film.mass()-film.initial_mass)
        history.append(row)

    def check_frame(arrays, index):
        for current, rendered in zip((film.palette, film.roller, film.block[BLOCK_MASK], *film.paper),
                                     arrays, strict=True):
            pixels = np.round(np.sqrt(np.clip(current, 0, 1))*255).astype(np.uint8)
            np.testing.assert_array_equal(pixels, rendered[index])

    out.mkdir(parents=True)
    with np.load(trajectory) as src:
        contacts = src["contacts"]
        assert len(contacts) == old["control_steps"]*decimation == old["physics_ticks"]
        assert len(src["actions"]) == old["control_steps"] and len(recorded) == old["control_steps"]+1
        np.testing.assert_allclose(contacts[:,0], np.arange(1,len(contacts)+1)*DT, atol=1e-10)
        panels_at_tick = contacts[:,8+channels:29+channels].reshape(-1,PANELS,7)
        contact_evidence = archived_clearance(panels_at_tick, contacts[:,1:8], TABLE_TOP)
        assert all(e["separating_axis_m"] > CONTACT_MARGIN for e in contact_evidence.values()), contact_evidence
        slab_normal = float(contacts[:,29+channels:].reshape(-1,PANELS,channels)[:,:,0].max())
        assert slab_normal == 0.
        video_ink = [src[f"ink_{i}"] for i in range(3+PANELS)]
        assert len(video_ink[0]) == len(contacts)//stride+1 == len(src["time_s"])
        record(0., 0)
        check_frame(video_ink, 0)
        for tick, row in enumerate(contacts, 1):
            t = float(row[0])
            data = row[1:].astype(np.float32)
            panel_start = 7+channels
            force_start = panel_start+7*PANELS
            panels = data[panel_start:force_start].reshape(PANELS,7)
            film.step(data[:7], data[7:panel_start], panels, data[force_start:].reshape(PANELS,channels))
            if tick % stride == 0:
                check_frame(video_ink, tick//stride)
            if tick % decimation == 0:
                clear_ticks = 0 if paper_over_relief(panels, TABLE_TOP) else clear_ticks+decimation
                record(t, tick//decimation)
        for name, reconstructed in (("paper",film.paper),("brayer",film.roller),
                                    ("block",film.block),("palette",film.palette)):
            np.testing.assert_array_equal(reconstructed, src[name])
    slab = film.paper-film.paper_relief
    assert slab.min() >= -1e-12 and abs(film.mass()-film.initial_mass)<1e-10
    contact_audit = {"source":"Measured 240 Hz poses and slab loads",
                     "slab_normal_force_max_N":slab_normal,
                     "expanded_envelope_margin_m":CONTACT_MARGIN,
                     "minimum_swept_clearance":contact_evidence}
    save_json(out/"print-face-contact-audit.json", contact_audit)
    save_json(out/"control-measurements.json", history)
    np.savez_compressed(out/"final-ink-ledgers.npz", total=film.paper, relief=film.paper_relief, slab=slab)
    return {"kind":old["kind"], "status":"success", "max_score":1.,
            "source_gpu_run_id":old["gpu_run_id"], "source_result":str(original/"result.json"),
            "source_measurements":str(shot_file), "trajectory":str(trajectory),
            "trajectory_sha256":digest(trajectory), "rendered_film_frames_verified":len(video_ink[0]),
            "raw_measurements":str(out/"control-measurements.json"),
            "source_scores_sha256":digest(shot_file),
            "ink_ledgers":str(out/"final-ink-ledgers.npz"),
            "contact_audit":str(out/"print-face-contact-audit.json"),
            "contact_rule_sha256":digest(HERE/"contact_rule.py"),
            "scorer_sha256":digest(HERE/"scoring.py"), "ink_code_sha256":digest(HERE/"ink.py"),
            "control_steps":old["control_steps"], "physics_ticks":old["physics_ticks"], **history[-1]}


if __name__ == "__main__":
    original, out = map(Path, sys.argv[1:])
    gpu_run_id = os.environ["PRODUCTION_GPU_RUN_ID"]
    result = evaluate_archive(original, out)
    result["gpu_run_id"] = gpu_run_id
    save_json(out/"result.json", result)
    print("RESULT", json.dumps(result, allow_nan=False), flush=True)
