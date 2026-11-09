"""Recompute the score from the final measured state, without simulation."""
import json
from pathlib import Path
import sys
import numpy as np
from task import score


def rescore(folder):
    report = json.loads((folder/"result.json").read_text())
    states = np.load(folder/"trajectory.npz", allow_pickle=False)["states"]
    row, values, start = states[-1], {}, 0
    for name, width in report["record_columns"].items():
        values[name] = row[start:start+width]
        start += width
    assert start == len(row)
    result = score([values[f"tile_{i}_pose"] for i in range(1, 9)],
                   values["robot_contact_n"], values["time"][0])
    assert result["score"] == report["score"]
    return result


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    print(json.dumps(rescore(Path(sys.argv[1])), indent=2))
