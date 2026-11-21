"""Recompute the live scorer from the 240 Hz wand and cord measurements."""
import json
from pathlib import Path
import sys

import numpy as np

from film import Film
from simulation import DT, VIDEO_STRIDE, source_manifest, update_film


def score(out):
    film = Film(DT)
    actual = json.loads((out / "result.json").read_text())
    assert actual["status"] == "success", actual["status"]
    with np.load(out / "trajectory.npz") as record:
        assert len(record["physics_poses"]) == actual["physics_ticks"] + 1
        frame = 0
        for tick, pose in enumerate(record["physics_poses"]):
            update_film(film, pose)
            if tick % VIDEO_STRIDE == 0:
                # Scoring changes must not alter the recorded film or bubble motion.
                np.testing.assert_array_equal(film.visual(), record["film"][frame])
                frame += 1
        assert frame == len(record["film"])
    return film.result()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    out = Path(sys.argv[1])
    result = score(out)
    (out / "score.json").write_text(json.dumps({**result, "scoring_sources": source_manifest()},
                                              indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
