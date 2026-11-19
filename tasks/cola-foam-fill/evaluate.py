"""Run the expert or Dream using the same task environment and process scorer."""
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from inspect_robots import Scene
from yam_arena import ik, station
from yam_arena.constants import READY

from process import next_press


def expert(env, scene):
    from geometry import BUTTON
    from simulation import HZ, LIMIT, write_json

    from recording import save_native

    first = env.reset(scene)
    if env.still:
        save_native(env.out, 0, first)
    pose = np.array(BUTTON) + np.array([0, 0, station.TABLE_TOP])
    rotation = station.aim((1,0,0))
    base = station.BASES["right"]

    def waypoint(dx, prior):
        q, err, ang = ik.solve(pose + [dx,0,0], rotation, q_init=prior, base_pos=base, iters=600)
        assert err < .001 and ang < .02, (err, ang)
        return q

    staging = waypoint(-.073, READY)
    free = waypoint(-.030, staging)
    pressed = waypoint(-.005, free)
    log = []
    for i in range(LIMIT):
        if i < HZ:
            q = np.asarray(READY) + (staging - READY) * ik.smoothstep((i+1)/HZ)
        elif i < HZ * 2:
            q = staging + (free - staging) * ik.smoothstep((i+1-HZ)/HZ)
        else:
            q = pressed if next_press(env.pour) else free
        opened = env.pour.open
        env.drive(np.array([*READY, 1., *q, .0]))
        if env.still:
            save_native(env.out, i + 1, env.capture())
        if opened != env.pour.open:
            log.append(env.metrics())
            print("VALVE", log[-1], flush=True)
        if env.pour.score == 1.:
            break
    write_json(env.out / "valve-transitions.json", log)
    if env.pour.score != 1.:
        raise RuntimeError(f"Expert did not reach full score: {env.metrics()}")


def dream(env, scene):
    from inspect_robots import Task, eval as evaluate
    from inspect_robots.scorer import Score
    from inspect_robots_dream.policy import DreamPolicy

    from simulation import CFG, LIMIT, write_json

    class ColaScore:
        name = "cola_score"

        def __call__(self, record, target):
            value = float(record.steps[-1].result.info["score"]) if record.steps else 0.
            return Score(value=value)

    task = Task(name="cola-foam-fill", scenes=[scene], scorer=ColaScore(), max_steps=LIMIT)
    with DreamPolicy(model=CFG["model"]) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=env, log_dir=str(env.out/"inspect"),
                            seed=scene.init_seed, fail_on_error=True, store_frames=True, store_actions=True)
        finally:
            write_json(env.out/"dream-transcript.json", policy.transcript())
    assert len(logs) == 1 and logs[0].status == "success", "Inspect evaluation failed"
    assert logs[0].results.metrics["cola_score"] == env.pour.score
    write_json(env.out/"inspect-result.json", logs[0].to_dict())


def probe(env, scene):
    """Compare equal valve tapes driven through three measured collision paths."""
    from geometry import CUP
    from process import CONDITIONS, Pour
    from simulation import DT, HZ, write_json

    env.reset(scene)
    target = np.array([CUP[0]-.009, CUP[1], station.TABLE_TOP+CUP[2]+.130])
    rotation = station.aim((1.,0.,0.))
    prior = np.array(READY)
    results = {}
    reference_jet = ((0., .027),)
    for name, offset in (("clear", None), ("intercept", 0.), ("side_pass", .027)):
        if offset is not None:
            desired, err, angle = ik.solve(target+[0.,offset,0.], rotation, q_init=prior,
                                             base_pos=station.BASES["left"], iters=600)
            assert err < .001 and angle < .02, (err, angle)
            for i in range(2*HZ):
                q = prior+(desired-prior)*ik.smoothstep((i+1)/(2*HZ))
                env.drive(np.array([*q, 0., *READY, 1.]))
            prior = desired
        p = Pour(CONDITIONS[0])
        paths = []
        for step in range(3*HZ):
            q = prior.copy()
            if name == "side_pass":
                q[0] += .011 * np.sin(step/HZ*3.)
            env.drive(np.array([*q, 0. if offset is not None else 1., *READY, 1.]))
            hits = env.jet_hits()
            paths.append(dict(step=step, hits=hits, adjacent_hits=env.jet_hits(reference_jet)))
            for tick in range(12):
                p.step(.006 if step < 2*HZ else 0., DT, hits)
        balance = p.emitted-p.liquid-p.pending-p.intercepted-p.spill_liquid
        assert abs(balance) < 1e-8
        results[name] = p.measure() | dict(paths=paths, mass_balance_error_ml=balance)
    write_json(env.out/"jet-probe.json", dict(open_s=2., duration_s=3., results=results))
    for name,p in results.items():
        print(name, "emitted", p["emitted_ml"], "cup", p["liquid_ml"], "intercepted", p["intercepted_ml"],
              "hit_ticks", sum(bool(t["hits"]) for t in p["paths"]),
              "adjacent_hit_ticks", sum(bool(t["adjacent_hits"]) for t in p["paths"]), flush=True)
    assert results["clear"]["intercepted_ml"] == results["side_pass"]["intercepted_ml"] == 0.
    assert any(t["adjacent_hits"] for t in results["side_pass"]["paths"])
    assert results["intercept"]["intercepted_ml"] > .5*results["intercept"]["emitted_ml"]
    assert results["clear"]["liquid_ml"] == results["side_pass"]["liquid_ml"]
    assert 0. < results["clear"]["score"] == results["side_pass"]["score"] < 1.
    assert results["intercept"]["score"] == 0.


def main():
    kind, path = sys.argv[1:]
    assert kind in ("demo", "demo-alt", "dream", "dream-alt", "probe")
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=False)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True,
                      kit_args="--/carb/tasking/plugin/threadCount=12 "
                               "--/omni/tbb.globalcontrol/maxThreadCount=12").app
    from simulation import CFG, Fountain, HERE, INSTRUCTION, write_json

    seed = 1 if kind.endswith("-alt") else CFG["episode"]["seed"]
    write_json(out/"run.json", dict(kind=kind, seed=seed, instruction=INSTRUCTION, config=CFG))
    source = out / "source"
    source.mkdir()
    for path in [*HERE.glob("*.py"), HERE/"dream-defaults.json"]:
        shutil.copyfile(path, source/path.name)
    env = Fountain(out, film=kind != "probe", still=kind.startswith("demo"))
    scene = Scene(id="cola-foam-fill", instruction=INSTRUCTION, init_seed=seed)
    exit_code = 0
    try:
        if kind == "probe":
            probe(env, scene)
            return
        (dream if kind.startswith("dream") else expert)(env, scene)
        result = env.metrics()
        write_json(out/"result.json", dict(
            kind="dream" if kind.startswith("dream") else "demo", status="success",
            gpu_run_id=os.environ.get("PRODUCTION_GPU_RUN_ID"), max_score=1,
            process_csv=os.path.relpath(out/"process.csv"),
            state_log=os.path.relpath(out/"poses-f64.bin"),
            actions_log=os.path.relpath(out/"actions.jsonl"),
            video=os.path.relpath(out/"video.mp4") if env.film else None, **result))
        print("RESULT", result, flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        if hasattr(env, "pour"):
            write_json(out/"failure-metrics.json", env.metrics())
    finally:
        env.close()
        app.close(exit_code=exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
