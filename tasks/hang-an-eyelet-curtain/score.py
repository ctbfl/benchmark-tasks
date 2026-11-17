"""Recompute the final score from a measured trajectory without running a policy."""
import json
from pathlib import Path
import sys

import numpy as np
from config import HZ, MAX_STEPS, RINGS
from scoring import evaluate


def score(out):
    schema = json.loads((out/'schema.json').read_text())
    states = np.load(out/'states.npy', allow_pickle=False, mmap_mode='r')
    actions = np.load(out/'actions.npy', allow_pickle=False, mmap_mode='r')
    assert actions.shape[1:] == (14,) and len(actions) <= MAX_STEPS
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    assert np.allclose(np.diff(states[:, 0]), 1/schema['physics_hz'])
    assert abs(states[-1, 0]-len(actions)/HZ) < 1e-8
    fields = schema['fields']
    row = states[-1].copy()
    links = row[slice(*fields['rings'])].reshape(-1, 7)
    ring_ids = [schema['rings'].index(f'ring_{i}') for i in range(RINGS)]
    result = evaluate(links[ring_ids], row[slice(*fields['rod'])], row[slice(*fields['stand'])])
    result.update(control_steps=len(actions), time_s=len(actions)/HZ)
    saved = json.loads((out/'metrics.json').read_text())
    for key in ('score', 'ring_on_rod', 'rod_on_stand', 'control_steps'):
        assert result[key] == saved[key], (key, result[key], saved[key])
    return result


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python score.py RUN_DIRECTORY'
    print(json.dumps(score(Path(sys.argv[1])), indent=2))
