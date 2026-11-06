"""Recompute the final score from the complete measured physical trajectory."""
from pathlib import Path
import sys

import numpy as np

from circuit import PHYSICS_HZ, Score, intact
from runtime import write_json


def score_run(out):
    states = np.load(out/"states.npy",allow_pickle=False)
    assert states.ndim==2 and states.shape[1]==48 and np.isfinite(states).all()
    assert len(states)>1 and abs(states[0,0])<1e-10
    assert np.allclose(np.diff(states[:,0]),1/PHYSICS_HZ)
    score = Score()
    for row in states[1:]:
        score.update(float(row[46]),intact(row[33:40],states[0,33:40]))
    result = score.result()
    write_json(out/"rescored.json",result)
    return result


if __name__=="__main__":
    assert len(sys.argv)==2, "Usage: python score.py RUN_DIRECTORY"
    print(score_run(Path(sys.argv[1])))
