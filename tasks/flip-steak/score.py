"""Recompute every event and the final score from measured physics states."""
import json
from pathlib import Path
import sys
import numpy as np
from config import DT
from scoring import Scorer


def score(out):
    out=Path(out)
    states=np.load(out/'trajectory.npz')['states']
    schema=json.loads((out/'schema.json').read_text())
    def value(row,key): return row[slice(*schema[key])]
    def body(row,name): return np.r_[value(row,name+'_pose'),value(row,name+'_velocity')]
    assert abs(states[0,0])<1e-9 and np.allclose(np.diff(states[:,0]),DT)
    scorer=Scorer(body(states[0],'steak'),body(states[0],'pan'))
    for row in states[1:]:
        contacts=value(row,'steak_contacts').reshape(-1,3)
        scorer.update(body(row,'steak'),body(row,'pan'),bool(np.linalg.norm(contacts[1:],axis=1).max()>.05))
    result=scorer.result()
    expected=json.loads((out/'result.json').read_text())
    for key in result: assert result[key]==expected[key],(key,result[key],expected[key])
    (out/'score.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    assert len(sys.argv)==2,'Usage: python score.py RUN_DIRECTORY'
    score(sys.argv[1])
