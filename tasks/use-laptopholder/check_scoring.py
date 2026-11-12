"""Check the four-part rubric using a measured seated stand and counterexamples."""

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

    def evaluate(name,state,expected):
        scorer=Score(gear=4,low=(-.06,.19),high=(.27,.41))
        result=scorer.update(state)
        assert result["score"]==expected, (name,result)
        assert result["success"]==(expected==1.0), (name,result)
        rows.append({"case":name,"success":result["success"],"score":result["score"]})
        return result

    evaluate("measured folded stand",data["phases"][0]["state"],0.0)
    evaluate("measured spread stand",data["phases"][1]["state"],0.25)
    # The first three items earn credit independently; containment only earns the last quarter.
    for spread,left,right,inside,expected in (
        (False,False,False,True,0.0),
        (True,False,False,True,0.25),
        (False,True,False,True,0.25),
        (False,False,True,True,0.25),
        (True,True,False,True,0.5),
        (True,False,True,True,0.5),
        (False,True,True,True,0.5),
        (True,True,True,False,0.75),
        (True,True,True,True,1.0),
    ):
        changed=copy.deepcopy(valid)
        changed["width"]=.185 if spread else .15
        for side,seated in (("L",left),("R",right)):
            changed["sides"][side]["gear"]=4 if seated else 3
        if not inside:
            changed["high"][1]=.42
        evaluate(f"spread={spread}, left={left}, right={right}, inside={inside}",changed,expected)

    for name,change,expected in (
        ("pin above opening",lambda m:m["sides"]["L"]["seat_error"].__setitem__(2,.005),.5),
        ("pin past slot end",lambda m:m["sides"]["R"]["seat_error"].__setitem__(0,.006),.5),
        ("pin beside opening",lambda m:m["sides"]["R"]["seat_error"].__setitem__(1,.006),.5),
        ("pin below slot floor",lambda m:m["sides"]["L"]["seat_error"].__setitem__(2,-.01),.5),
        ("visible seating with offset",lambda m:m["sides"]["L"].__setitem__("seat_error",[.003,.002,.001]),1.0),
        ("outline on inner edge",lambda m:m["high"].__setitem__(1,.41),1.0),
        ("outline past inner edge",lambda m:m["high"].__setitem__(1,.411),.75),
        ("base height is not scored",lambda m:m["sides"]["R"].__setitem__("base_z",.002),1.0),
        ("velocity is not scored",lambda m:m.__setitem__("max_speed",.05),1.0),
        ("joint residual is not scored",lambda m:m.__setitem__("joint_error",.001),1.0),
    ):
        changed=copy.deepcopy(valid)
        change(changed)
        evaluate(name,changed,expected)
    out=ROOT/"outputs/validation/scoring.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"passed":True,"cases":rows},indent=2)+"\n")
    print(f"Passed {len(rows)} scoring cases")


if __name__=="__main__":
    main()
