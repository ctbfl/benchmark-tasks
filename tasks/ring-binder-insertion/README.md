# Insert a tabbed divider into a six-ring binder

Open the six rings, insert the red divider, release it onto the notebook's right
side, and squeeze the rings closed within 120 seconds. The right hand steadies the
notebook while the left presses the near lever. After a horizontal side grasp,
the right gripper and divider tilt down-left to align with the exposed ring ends.

The task uses dual YAM arms and three calibrated robot cameras. See `task.json`
for dimensions, materials, the fixed layout and reset details.

## Scoring

Score up to 1.0:

- **0.2:** Pressed the button to open all six rings.
- **0.6:** Release the divider on the right with all six holes threaded.
- **0.2:** Close the rings after insertion.

Opening credit is retained. Insertion and closure use the final state; an initially
closed binder earns no closure credit. `scoring.py` intersects the measured ring
centerlines with the six hole planes, accounts for wire radius, and checks that
robot contact with the divider is below 0.1 N. Closure requires the gap between
ring tips to be smaller than the 1.6 mm divider thickness. `score.py` recomputes
scores from saved physical states and contacts without model calls.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env` and install
`requirements.txt` inside that environment. Tested with Python 3.12, Isaac Sim
6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
The three task USDs are self-contained; robot assets and camera calibration come
from yam-arena. Original task geometry and materials are MIT licensed.

Run all commands below from this directory after activating the environment.

## Evaluation

Both evaluations use the fixed initial layout, seed 0, CPU PhysX at 240 Hz and
14 absolute joint/gripper targets at 20 Hz. Both use native YAM actuator settings:
gripper stiffness 5000 N/m, damping 100 N s/m and effort limit 50 N. Objects move
through physical contact without grasp attachments. `dream-defaults.json` is the
shared execution configuration.

The binder approximates a snap mechanism with contact-triggered spring rest angles.
Pressing its lever at least 2.5 mm with over 3 N opens the rails. Squeezing both
rails inward with over 2 N per side past the latch travel closes them. The same
mechanism runs during the demo and model evaluation; it never teleports objects.

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot
joint/gripper positions. Native 640x360 images are resized to 320x176 JPEGs at the
model interface. API waits pause physics. The full prompt is `config.INSTRUCTION`,
and the installed client reads `~/.dreamlabs/credentials`.

```bash
python dream_eval.py --check       # Offline adapter check; no model calls
python dream_eval.py               # One trial, at most 120 simulation seconds
python score.py runs/dream-<timestamp>
```

The recorded trial scored **0.0/1.0 in 120 simulation seconds** (243 seconds wall time
on RTX 5090). Each run saves `metrics.json`,
executed actions, physics states and source hashes. Model runs additionally archive
input JPEGs and complete action replies.

### Demo policy

The demo initially braces the outer part of the right paper stack.
It replays 1,704 fixed joint-position/gripper commands from `demo_actions.npy`
at 20 Hz without task-state feedback. Independent replay scored **1.0/1.0 in
85.2 simulation seconds**, including three seconds of final settling.

```bash
python run.py check                # Initial 1080p screenshot and camera/action checks
python test_scoring.py             # Scoring positive and negative controls
python run.py demo
python score.py runs/demo-<timestamp>
```

### Recordings

Render saved physical states without rerunning a trial:

```bash
python render.py runs/demo-<timestamp>
python encode_video.py runs/demo-<timestamp>/videos/original.mp4
```

The same commands accept a Dream run directory. Videos are saved in each run's
`videos/`, sampled images in `frames/`. The fixed 1080p observer and three native
640x360 robot views form a 1920x1440, 30 FPS video. The observer is for human review
only. `delivery.mp4` preserves every frame and is limited to 10 MB.
