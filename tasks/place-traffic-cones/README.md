# Place traffic cones along a curved road

Separate six nested traffic cones, gripping only their top 5 mm, and place them
upright along the dashed centreline of a curved road. Put one at each endpoint
and four evenly between, release all cones and move the grippers clear within
120 seconds. This tabletop analogue represents temporary traffic guidance at events.

The cones start stacked to the right of a white board. The task uses dual YAM arms
and three calibrated robot cameras. Each hollow cone has a passive compressible
tip. See `task.json` for dimensions and layout, and `ASSETS.md` for asset licensing
and the physical approximations.

## Scoring

Score the final state: **1.0 only if all six cones meet every condition**, otherwise 0:

- Upright within 10 degrees, directly supported and released.
- Base centres within 2 cm of the continuous road centreline.
- Nearest centreline points, sorted by arc length, start and end within 2 cm of
  the respective curve endpoints.
- The largest of the five adjacent arc-length gaps differs from total curve
  length / 5 by at most 5 cm.

Support uses a 1 mm base-to-surface height tolerance; release requires measured
robot contact below 0.001 N per cone. These checks are implemented in `scoring.py`.
`score.py` recomputes scores from saved states and contacts without model calls.
Both policies report `score.json` and `metrics.json` in separate `runs/` directories.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env` and install
`requirements.txt` inside that environment. Tested with Python 3.12, Isaac Sim
6.1.0.0, Isaac Lab 3 (package 17.0.2, commit
`91a0906f93802eb859192dcb0bca879f891e92d6`) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
The two task USDs are self-contained; robot assets and camera calibration come
from yam-arena. The station furniture is procedural.

Run all commands below from this directory after activating the environment:

```bash
export OPENBLAS_NUM_THREADS=1
python -m unittest test_scoring
```

## Evaluation

Both evaluations use the fixed initial layout, seed 0 and CPU PhysX at 240 Hz.
Both use native YAM actuator settings: gripper stiffness 5000 N/m, damping
100 N s/m and effort limit 50 N. Objects move through physical contact without
grasp attachments. Reset includes two seconds of settling outside the episode.

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot
joint/gripper positions. Native 640x360 images are resized to 320x176 JPEGs at the
model interface. Control runs at 20 Hz for 2,400 steps (120 simulation seconds),
including the 32-step observation preroll; API waits pause physics.
The task prompt is `config.INSTRUCTION`, and the
installed client reads `~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds** (385 seconds wall
time on RTX 5090). It did not complete upright placement of all six cones along
the curve.

```bash
python dream_eval.py --check       # Offline scene, camera, action and scoring checks
python dream_eval.py               # One Dream-1 trial; requires a configured API key
python score.py runs/dream-eval-<timestamp>
```

### Demo policy

The demo uses a fixed open-loop action script: 2,162 prerecorded joint-position
and gripper commands from `demo_actions.npy`, sent at 20 Hz. During execution,
the policy uses no privileged simulator information. Its commands and timing
are fixed. `runtime.py` reads simulator states only for independent scoring and
recording; those measurements do not feed back into the policy.

The independent replay scored **1.0/1.0 in 108.1 simulation seconds**, including
three seconds of final settling with both arms clear (128 seconds wall time).
This command recording targets the specified fixed initial layout.

```bash
python demo.py
python score.py runs/demo-<timestamp>
```

### Recordings

Render saved physical states without rerunning a trial:

```bash
python render.py runs/demo-<timestamp>
python compress.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Videos are saved in each run's
`videos/`, sampled images in `frames/`. The fixed 1080p observer and three native
640x360 robot views form a 1920x1440, 30 FPS video. The observer is for human
review only. The offline `--check` includes a synthetic six-cone scoring positive
control; evaluations always start with the nested stack.
