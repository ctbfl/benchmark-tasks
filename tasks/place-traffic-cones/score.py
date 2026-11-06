"""Recompute the final score from an archived physical trial, without simulator or API."""
import json
from pathlib import Path
import sys
import numpy as np
from yam_arena.station import TABLE_TOP
from scoring import measure


def score(out):
    records=json.loads((out/'measurements.json').read_text())
    assert records
    scores=[measure(row['roots'],row['robot_contact_N'],TABLE_TOP,row['time_s']) for row in records]
    expected=json.loads((out/'score.json').read_text())
    assert scores[-1]['score']==expected['score']
    np.testing.assert_allclose(scores[-1]['curve_distance_m'],expected['curve_distance_m'],atol=1e-12)
    result=scores[-1]
    (out/'rescored.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    assert len(sys.argv)==2, 'Usage: python score.py RUN_DIRECTORY'
    score(Path(sys.argv[1]))
