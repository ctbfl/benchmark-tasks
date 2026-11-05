"""Counterexamples based on measured, successful force-fixture states."""

import copy
import json
from pathlib import Path

from task import Score

ROOT=Path(__file__).resolve().parent


def main():
    data=json.loads((ROOT/"outputs/demo/mechanics.json").read_text())
    assert data["passed"]
    valid=data["phases"][-1]["state"]
    rows=[]

    def evaluate(name,state,contact=0.0):
        scorer=Score(gear=4,low=(-.06,.19),high=(.27,.41))
        for _ in range(20):
            result=scorer.update(state,contact,.05)
        rows.append({"case":name,"success":result["success"],"score":result["score"]})
        return result

    assert evaluate("released target state",valid)["success"]
    for phase in data["phases"][:-1]:
        if phase["name"]=="11 Lower to target":
            continue  # Physically already grounded; the fixture is still holding it.
        assert not evaluate(phase["name"],phase["state"])["success"]
    assert not evaluate("robot still supports stand",valid,.1)["success"]
    for name,change in (
        ("one pin out",lambda m:m["sides"]["L"]["seat_error"].__setitem__(2,.005)),
        ("wrong left slot",lambda m:m["sides"]["L"].__setitem__("gear",3)),
        ("still translating",lambda m:m.__setitem__("max_speed",.05)),
        ("still rotating",lambda m:m.__setitem__("max_spin",.5)),
        ("base tilted",lambda m:m["sides"]["R"].__setitem__("base_tilt_deg",5)),
        ("overhang outside zone",lambda m:m["high"].__setitem__(1,.42)),
        ("hovering",lambda m:m["low"].__setitem__(2,.01)),
        ("constraint broken",lambda m:m.__setitem__("joint_error",.01)),
    ):
        changed=copy.deepcopy(valid)
        change(changed)
        assert not evaluate(name,changed)["success"]
    scorer=Score(gear=4,low=(-.06,.19),high=(.27,.41))
    for _ in range(10):
        assert not scorer.update(valid,0,.05)["success"]
    assert not scorer.update(valid,1,.05)["success"]
    for _ in range(19):
        assert not scorer.update(valid,0,.05)["success"]
    assert scorer.update(valid,0,.05)["success"]
    rows.append({"case":"hold timer resets after contact","passed":True})
    (ROOT/"outputs/validation/scoring.json").write_text(json.dumps({"passed":True,"cases":rows},indent=2)+"\n")
    print(f"Passed {len(rows)} scoring cases")


if __name__=="__main__":
    main()
