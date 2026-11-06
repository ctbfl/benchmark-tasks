"""Recompute the final score from recorded physical states and contact measurements."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from config import DT,HZ,PIVOT_Y,PIVOT_Z
from scoring import Score


def score_run(out):
    states=np.load(out/'states.npy',allow_pickle=False)
    schema=json.loads((out/'schema.json').read_text())
    measures=json.loads((out/'measurements.json').read_text())
    names=schema['binder_joint_names']
    ids=[names.index(n) for n in ('left_hinge','right_hinge','press')]
    scorer=Score()
    for i,m in enumerate(measures):
        row=states[round((i+1)/HZ/DT)]
        root=row[33:40]
        rr=Rotation.from_quat(root[3:])
        q=row[46:49][ids]
        rail=np.r_[root[:3]+rr.apply([0,-PIVOT_Y,PIVOT_Z]),
                   (rr*Rotation.from_euler('x',q[1])).as_quat()]
        result=scorer.update(float(row[0]),q,rail,row[52:59],root,m['card_robot_force_N'])
        assert result['score']==m['score'],(i,result,m)
    return scorer.result()


if __name__=='__main__':
    assert len(sys.argv)==2,'Usage: python score.py RUN_DIRECTORY'
    result=score_run(Path(sys.argv[1]).resolve())
    print(json.dumps(result,indent=2))
