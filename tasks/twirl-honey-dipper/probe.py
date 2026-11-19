"""Measure crossed roll conditions or a drain-then-carry negative control."""
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=False).app

from demo import plan
from scoring import MIN_DOSE_G, rescore
from world import World


def run_comparison(world):
    out = world.out
    results = {}
    for carry in (True, False):
        for dwell in (True, False):
            label = f"{'spin' if carry else 'still'}-{'spin' if dwell else 'still'}"
            world.out = out/label
            world.out.mkdir()
            world.reset(0)
            print("Reset tool", world.tool(), flush=True)
            phase = plan(world, carry, dwell)
            m = world.save()
            a = np.asarray(world.trajectory)
            during = a[(a[:, 0] >= phase["transport_start"]) & (a[:, 0] < phase["dwell_start"])]
            target = a[(a[:, 0] >= phase["dwell_start"]) & (a[:, 0] < phase["dwell_end"])]
            results[label] = {**phase, "transport_mean_abs_roll_rad_s": float(np.abs(during[:, -2]).mean()),
                "dwell_mean_abs_roll_rad_s": float(np.abs(target[:, -2]).mean()),
                "retained_at_arrival_g": float(target[0, -1]), "retained_final_g": m["retained_g"],
                "transport_off_target_g": sum(d["mass_g"] for d in m["landings"]
                    if phase["transport_start"] <= d["release_time_s"] < phase["dwell_start"]
                    and d["surface"] in ("plate", "table")),
                "dwell_toast_g": sum(d["mass_g"] for d in m["landings"]
                    if phase["dwell_start"] <= d["release_time_s"] < phase["dwell_end"]
                    and d["surface"] == "toast"), "landed_g": m["landed_g"]}
            print(label, results[label], flush=True)
    (out/"comparison.json").write_text(json.dumps({"gpu_run_id": os.environ.get("PRODUCTION_GPU_RUN_ID"),
        "conditions": results}, indent=2)+"\n")


out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
world = World(out)
try:
    if sys.argv[2:] == ["drain"]:
        world.reset(0)
        phase = plan(world, dwell_s=5., settle_s=3., return_to_rest=True, drain_s=32.)
        measured = world.save()
        grade = world.grade()
        assert rescore(out) == grade
        assert grade["carry_pass"] and grade["toast_g"] < MIN_DOSE_G and grade["score"] == 0., grade
        number = dict(gpu_run_id=os.environ.get("PRODUCTION_GPU_RUN_ID"), phases=phase,
                      control_steps=world.steps, grade=grade, total_landed_g=measured["landed_g"],
                      drain_to_jar_g=sum(d["mass_g"] for d in measured["landings"]
                          if d["surface"] == "jar" and phase["drain_start"] <= d["time_s"] < phase["transport_start"]))
        (out/"drain-result.json").write_text(json.dumps(number, indent=2)+"\n")
        print(json.dumps(number, indent=2), flush=True)
    else:
        assert len(sys.argv) == 2, "Usage: python probe.py RUN_DIRECTORY [drain]"
        run_comparison(world)
except BaseException:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
finally:
    world.env.close()
    app.close()
