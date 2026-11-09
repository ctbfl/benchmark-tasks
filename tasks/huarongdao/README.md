# Solve a sliding number puzzle

Slide adjacent tiles into the empty cell of the fixed tray until the rows read
`1 2 3 / 4 5 6 / 7 8 blank`. Keep the tiles flat and release them when finished.
The fixed initial layout is `1 2 3 / 5 blank 6 / 4 7 8`; a shortest solution moves
5, 4, 7 and 8. The recessed perimeter of each tile gives the gripper access.
The task uses dual YAM arms and three calibrated robot cameras. See `task.json`
for dimensions and `ASSETS.md` for asset sources and licenses.

## Scoring

Score the final state: **1.0** for the completed arrangement, **0.0** otherwise.
Seven tile centers must be within 3 mm of their target centers; at most one may
be within 10 mm. Tiles must be seated (base height within 1 mm of the tray floor),
correctly oriented (rotation error at most 10 degrees) and released (robot contact
below 0.05 N per tile). The limit is 120 seconds. No intermediate or partial credit.
`task.py` implements this rule; `score.py` recomputes it from the final recorded
physical state without simulation or model calls.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env` and install
`requirements.txt` inside that environment. Tested with Python 3.12, Isaac Sim
6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
At this Arena revision, `ARENA_NUCLEUS_DIR` must point to the production
`ISAACLAB_NUCLEUS_DIR`; `arena-assets.patch` supplies that resource-URL correction
for a fresh installation (`git -C <ARENA_DIR> apply <TASK_DIR>/arena-assets.patch`).
Robot assets and camera calibration come from yam-arena; task assets are bundled.
Optional asset regeneration uses DejaVu Sans Bold (`fonts-dejavu-core`).

Run commands from this directory in the activated environment, using a new output
directory for each trial. Both policies share `dream-defaults.json`: seed 0,
fixed layout, CPU PhysX TGS at 240 Hz, native YAM actuators and 14-dimensional
joint-position/gripper commands at 20 Hz. Reset settles the scene for two seconds
before starting the task clock. Tiles move through physical contact without drives
or grasp attachments.

```bash
export OPENBLAS_NUM_THREADS=1
python -m unittest -v test_task
python run.py check runs/check
python run.py adapter-check runs/adapter-check
```

## Evaluation

### Dream-1 model

Dream-1 beta receives only three calibrated robot camera images and 14 measured
joint/gripper positions. Native 640x360 images become 320x176 JPEGs at the model
interface. The limit is 2,400 control steps, including the 32-step observation
preroll; API waits pause physics. The full prompt is `dream_eval.INSTRUCTION`.
The client reads its configured `~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds**
(approximately 268 seconds wall time on an RTX 5090).

```bash
python run.py dream runs/dream
python score.py runs/dream
```

### Demo policy

The demo uses a fixed open-loop action script: 1,292 prerecorded joint-position
and gripper commands from `demo_actions.npz`, sent at 20 Hz. During execution, the
policy uses no privileged simulator information. Its commands and timing are fixed.
`simulation.py` reads simulator states only for independent scoring and recording;
those measurements do not feed back into the policy.

The independent replay scored **1.0/1.0 in 64.6 simulation seconds**, including
three seconds after retracting the arms. It took approximately 42 seconds wall
time on an RTX 5090. The maximum final tile-center error was 1.79 mm.

```bash
python run.py replay runs/demo
python score.py runs/demo
```

### Recordings

Each trial saves `result.json`, source hashes, executed actions and measured
240 Hz states in `trajectory.npz`. Dream also archives the exact input JPEGs and
complete action replies. Render saved states without repeating a trial:

```bash
python run.py video runs/demo
python pack_video.py runs/demo/videos
python run.py video runs/dream
python pack_video.py runs/dream/videos
```

Videos go in each run's `videos/`; sampled images go in `frames/`. A fixed 1080p
observer and three 640x360 robot views form a synchronized 1920x1440, 30 FPS video.
Compression preserves all frames and produces a file under 10 MB. The observer
is for human review only. `python run.py preview runs/preview` saves the initial
calibrated Top Camera image at 1920x1080 to `submission/uploads/screenshot.png`.
