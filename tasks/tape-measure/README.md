# Measure a wooden bar and enter its length

Measure the bar's longest edge using the retractable tape, enter the length as
integer millimetres on the calculator, and press `=`. The calculator starts on
the robot's left, the upright tape on its right, and the bar behind both.
See `task.json` for object dimensions, the fixed layout and simulation fidelity.

## Scoring

Score 1 if the first `=` press submits the correct integer length within 120
simulation seconds; otherwise score 0. Tape handling and final placement add
no scoring conditions. Physical key travel triggers the calculator events.
`score.py` recomputes the score from a saved submission and its time.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its setup instructions, then activate `source <ARENA_DIR>/yam.env` and
install `requirements.txt` inside that environment. Tested with Python 3.12,
Isaac Sim 6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
At this Arena revision, `ARENA_NUCLEUS_DIR` must point to the production
`ISAACLAB_NUCLEUS_DIR`; `arena-assets.patch` supplies that resource-URL correction
for a fresh installation (`git -C <ARENA_DIR> apply <TASK_DIR>/arena-assets.patch`).
Robot assets and camera calibration come from yam-arena. Task assets are
self-contained; notices are in `ASSETS.md`.

Run commands from this directory after activating the environment. Use a fresh
output directory for each execution. Both evaluations use `dream-defaults.json`:
fixed layout, seed 0, CPU PhysX TGS at 240 Hz, native YAM actuators and the same
14-dimensional joint-position/gripper interface at 20 Hz.

```bash
python run.py check runs/check
python run.py adapter-check runs/adapter-check
```

Both checks run without model calls. The limit is 2,400 control steps, or 120
simulation seconds, including Dream's observation preroll. API waits do not
advance physics.

## Evaluation

The recorded Dream-1 beta trial scored **0.0/1.0 in 120 simulation seconds**
(276 seconds wall time on RTX 5090); it made no calculator submission.
Dream receives only three calibrated robot camera views and 14 measured
joint/gripper positions. Native 640x360 images become 320x176 JPEGs at the model
interface. The installed client reads `~/.dreamlabs/credentials`.

```bash
python run.py dream runs/dream
python score.py runs/dream
```

The demo scored **1.0/1.0 in 113.5 simulation seconds**, submitting `173` at
109.883 seconds (about 72 seconds wall time on RTX 5090, excluding app startup).
It measures the bar, unhooks and releases the tape for passive
return, places the case on the table, and presses `1`, `7`, `3`, `=`.
Replay sends 2,270 fixed commands from `demo_actions.npz`; policy execution
uses no simulator observations. Independent replay reproduced every recorded
physical state from the development execution.

```bash
python run.py replay runs/demo
python score.py runs/demo
```

Each execution saves states, actions, calculator events, its result and a
source/configuration/asset snapshot. Dream also saves the exact input JPEGs and
complete returned action chunks. Render saved states without repeating inference:

```bash
python run.py video runs/demo
python pack_video.py runs/demo/videos
python run.py video runs/dream
python pack_video.py runs/dream/videos
```

The fixed observer and three native robot views form a synchronized 1920x1440,
30 FPS video. `videos/original.mp4` preserves the original encoding;
`videos/video.mp4` is the full-length compressed version, limited to 10 MB.
The observer is for human review and is never supplied to Dream.
