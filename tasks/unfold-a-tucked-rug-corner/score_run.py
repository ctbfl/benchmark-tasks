"""Regrade the final measured rug state without advancing physics."""
import json
import sys
from pathlib import Path
import numpy as np
from cloth import surfaces
from scoring import grade
from yam_arena.station import TABLE_TOP
assert len(sys.argv) == 2, "Usage: python score_run.py RUN_DIRECTORY"
data = np.load(Path(sys.argv[1]) / "trajectory.npz")
print(json.dumps(grade(*surfaces(data["cloth"][-1], data["faces"]), data["faces"], TABLE_TOP), indent=2))
