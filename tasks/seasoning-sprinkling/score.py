"""Recompute the final score from archived physical state without a model call."""
import json
from pathlib import Path
import sys
import numpy as np
from config import HZ, MAX_STEPS
from scoring import score

assert len(sys.argv)==2, 'Usage: python score.py RUN_DIRECTORY'
out=Path(sys.argv[1])
final=np.load(out/'final-state.npz',allow_pickle=False)
actions=np.load(out/'actions.npy',allow_pickle=False)
assert 0<=len(actions)<=MAX_STEPS and actions.shape[1:]==(14,)
result=score(final['grains'],final['steak'],final['bottle'])
result.update(control_steps=len(actions),time_s=len(actions)/HZ)
archived=json.loads((out/'metrics.json').read_text())
assert result==archived,(result,archived)
print(json.dumps(result,indent=2))
