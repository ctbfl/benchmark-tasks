"""Mechanical acceptance tests use forces and live PhysX states, never pose playback."""

import json
import numpy as np
from pxr import Gf, UsdGeom

import model
from runtime import PHYSICS_HZ, Mechanism, simulation, step


def seated(m, gear):
    return all(s["gear"] == gear and abs(s["seat_error"][0]) < .0029
               and abs(s["seat_error"][1]) < .0018 and abs(s["seat_error"][2]) < .0005
               for s in m["sides"].values())


def run(hz=PHYSICS_HZ):
    sim = simulation(1/hz)
    for g in range(8):
        model.build(sim.stage, f"/World/H{g}", model.CLOSED_WIDTH if g==0 else model.OPEN_WIDTH,
                    g, origin=(.4*(g%4), .4*(g//4), .002))
    asset = UsdGeom.Xform.Define(sim.stage,"/World/Falling")
    asset.GetPrim().GetReferences().AddReference(str(model.ROOT/"assets/laptop_holder.usda"))
    asset.AddTranslateOp().Set(Gf.Vec3d(0,1,.5))
    sim.reset()
    holders = [Mechanism(sim,f"/World/H{g}") for g in range(8)]
    falling = Mechanism(sim,"/World/Falling")
    report = {"physics_hz": hz, "mass_kg": sum(falling.mass.values()), "checks": [], "stages": {}}

    def check(name, passed, value):
        report["checks"].append({"name":name,"passed":bool(passed),"value":value})
        print(name, bool(passed), value, flush=True)

    p0 = falling.measure()
    step(sim,.1)
    p1 = falling.measure()
    actual = np.asarray(p1["com"])
    expected = np.asarray(p0["com"]) + np.asarray(p0["com_velocity"])*.1 + np.array([0,0,-.5*9.81*.1**2])
    error = float(np.linalg.norm(actual-expected))
    check("free fall COM", error < .0008, error)
    check("mass", abs(report["mass_kg"]-.24)<1e-6, report["mass_kg"])
    step(sim,2.9)
    report["stages"]["settled"] = [h.measure() for h in holders]
    for g,h in enumerate(holders):
        m=h.measure()
        check(f"gear {g} static stability", m["max_speed"]<.003 and m["max_spin"]<.05 and m["joint_error"]<.0003,
              {k:m[k] for k in ("max_speed","max_spin","joint_error")})
        check(f"gear {g} floor clearance", m["low"][2]>-.0005, m["low"][2])
        if g:
            check(f"gear {g} seats", seated(m,g), m["sides"])
        else:
            check("folded width", abs(m["width"]-model.CLOSED_WIDTH)<.001, m["width"])

    # A distributed 2 kg equivalent downward load, one half on each deck COM.
    def load(_):
        for h in holders[1:]:
            h.wrench({"deck_L":[0,0,-9.81],"deck_R":[0,0,-9.81]})
    step(sim,2,load)
    report["stages"]["loaded_2kg"] = [h.measure() for h in holders[1:]]
    report["loaded_velocities"] = {n: v.tolist() for n,(_,_,v) in holders[4].state().items()}
    for g,h in enumerate(holders[1:],1):
        m=h.measure()
        check(f"gear {g} 2kg load", seated(m,g) and m["joint_error"]<.0005 and m["max_speed"]<.01,
              {"sides":m["sides"],"joint_error":m["joint_error"],"speed":m["max_speed"]})
    step(sim,1)
    report["stages"]["unloaded"] = [h.measure() for h in holders[1:]]

    # Pull the two decks apart by physical horizontal forces; release and inspect.
    step(sim,2,lambda _: holders[0].wrench({"deck_L":[0,-2,0],"deck_R":[0,2,0]}))
    step(sim,1)
    m=holders[0].measure()
    report["stages"]["spread"] = m
    check("force-driven width expansion", m["width"]>.180 and m["joint_error"]<.0005, m)

    path=model.ROOT/f"outputs/validation/physics_{hz}.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    sim.stop()
    assert all(c["passed"] for c in report["checks"]), f"Failed checks: {path}"
    return report
