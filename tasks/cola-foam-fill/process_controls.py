"""Run the inexpensive process counterfactuals at the production physics rate."""
import csv
import json
import sys
from math import comb
from pathlib import Path

from process import CAPACITY, CONDITIONS, HEIGHT, TAIL_ML, TARGET, Pour, next_press

DT = 1 / 240
STEPS = 40 * 240
TRAVEL = .006


def run(name, condition, command, out):
    pour = Pour(condition)
    trace, tape = [], []
    for i in range(STEPS):
        press = command(pour, i)
        tape.append(press)
        pour.step(TRAVEL if press else 0., DT)
        trace.append(pour.measure())
        assert abs(pour.emitted - pour.liquid - pour.pending - pour.spill_liquid - pour.intercepted) < 1e-7
    with (out / f"{name}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=trace[0])
        writer.writeheader()
        writer.writerows(trace)
    closes = [i for i in range(1, len(tape)) if tape[i-1] and not tape[i]]
    transitions = [(round((i+1)*DT, 4), v) for i, v in enumerate(tape)
                   if v != (tape[i-1] if i else False)]
    tail = trace[closes[0]:closes[0]+120]
    current = trace[closes[0]-1]
    summary = trace[-1] | {
        "first_closure_after_land_ml": sum(t["landed_ml"] for t in tail),
        "first_closure_after_nozzle_ml": sum(t["nozzle_ml"] for t in tail),
        "first_closure_foam_growth_ml": max(t["foam_ml"] for t in tail) - current["foam_ml"],
        "max_crest_mm": max(t["crest_before_spill_mm"] for t in trace),
        "valve_transitions_s": transitions,
    }
    return tape, trace, summary


def fixed_timing(opens, waits):
    starts = [0]
    for duration, wait in zip(opens, waits):
        starts.append(starts[-1] + round((duration+wait)/DT))
    windows = [(start, start+round(duration/DT)) for start,duration in zip(starts,opens)]
    scores = []
    for condition in CONDITIONS:
        p = Pour(condition)
        crest = 0.
        for tick in range(STEPS):
            pressed = any(a <= tick < b for a,b in windows)
            p.step(TRAVEL if pressed else 0., DT)
            crest = max(crest, p.crest)
        scores.append(p.measure() | {"max_crest_mm": 1000*crest*HEIGHT/CAPACITY})
    return dict(open_s=opens, gaps_s=waits, conditions=scores)


def search():
    """Bound the full timing grid, and integrate representative schedules with foam."""
    a,b = (c.flow_ml_s for c in CONDITIONS)
    bounds, examples = [], [fixed_timing([5.,1.25], [10.])]
    for n in (1,2,3):
        # Every full score needs zero rim loss. After landing, A can retain at
        # most a*T+n*tail; B must retain at least b*T if there is no diversion.
        minimum, maximum = (TARGET-n*TAIL_ML)/a, CAPACITY/b
        assert minimum > maximum
        grid, possible = 0, 0
        for total in range(n, 800-5-(n-1)+1):
            count = comb(total-1, n-1)*comb(800-5-total, n-1)
            grid += count
            if minimum <= total/20 <= maximum:
                possible += count
        bounds.append(dict(pours=n, open_min_slow_s=minimum, open_max_fast_s=maximum,
                           schedules_20hz=grid, eligible_in_both=possible))
        for total in (4.8, 5.2, 5.45, 6.15, 6.35, 6.6):
            if n == 1:
                examples.append(fixed_timing([total], []))
                continue
            for tail in (.25, round(total/n,2)):
                opens = [round(total-tail*(n-1),2), *([tail]*(n-1))]
                for wait in (.25, 3., 10., round((39.-total)/(n-1),2)):
                    examples.append(fixed_timing(opens, [wait]*(n-1)))
    assert not any(all(c["score"] == 1. for c in case["conditions"]) for case in examples)
    return dict(bounds=bounds, integrated_examples=examples)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    feedback = lambda p, i: next_press(p)
    tape, nominal, normal = run("feedback_nominal", CONDITIONS[0], feedback, out)
    first_segment = next(i for i, v in enumerate(tape) if not v)
    _, _, early = run("early_close", CONDITIONS[0], lambda p, i: i < first_segment, out)
    direct = lambda p, i: p.liquid < TARGET
    _, _, continuous = run("continuous", CONDITIONS[0], direct, out)
    _, _, replay = run("same_timing_alt", CONDITIONS[1], lambda p, i: tape[i], out)
    _, _, adaptive = run("feedback_alt", CONDITIONS[1], feedback, out)
    _, _, alt_continuous = run("continuous_alt", CONDITIONS[1], direct, out)
    report = dict(target_ml=TARGET, rim_volume_ml=CAPACITY,
                  feedback_nominal=normal, early_close=early,
                  continuous=continuous, same_timing_alt=replay, feedback_alt=adaptive,
                  continuous_alt=alt_continuous)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    first_reopen = next(i for i in range(first_segment, len(tape)) if tape[i])
    assert continuous["liquid_ml"] >= TARGET and continuous["openings"] == 1
    assert continuous["max_crest_mm"] > HEIGHT * 1000 and continuous["score"] == 0
    assert early["first_closure_after_nozzle_ml"] > 0.
    assert early["first_closure_after_land_ml"] > 0.
    assert early["first_closure_foam_growth_ml"] > 0.
    assert nominal[first_reopen]["foam_ml"] < .5 * nominal[first_segment]["foam_ml"]
    assert normal["score"] == adaptive["score"] == 1.
    assert normal["spill_foam_ml"] == adaptive["spill_foam_ml"] == 0.
    assert normal["openings"] <= 3 and adaptive["openings"] <= 3
    assert replay["valve_transitions_s"] == normal["valve_transitions_s"]
    assert replay["max_crest_mm"] > HEIGHT * 1000 and replay["score"] == 0.
    assert alt_continuous["liquid_ml"] >= TARGET and alt_continuous["openings"] == 1
    assert alt_continuous["spill_foam_ml"] > 0. and alt_continuous["score"] == 0.
    (out / "schedule-search.json").write_text(json.dumps(search(), indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
