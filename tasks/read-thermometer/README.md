# Read and record a timed thermometer measurement

Wait for the desk timer to reach zero, read the thermometer with the left wrist
camera, then use the closed right gripper to enter the temperature on the
calculator and press `=`. The thermometer hangs in a beaker on a laboratory
stand. The timer and calculator sit on the right of the table.
See `task.json` for dimensions, the fixed layout and simulation fidelity.

## Scoring

Score **1** if the calculator's latest `=` submission equals **45 degrees Celsius**
within 120 simulation seconds; otherwise score **0**. Physical key travel triggers
input. There is no partial credit or additional scored camera-position condition.
`score.py` independently reconstructs submissions from recorded button positions.

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
Robot assets and camera calibration come from yam-arena. Task assets are
self-contained; licenses and notices are in `ASSETS.md`.

Run commands from this directory after activating the environment. Both policies
use `dream-defaults.json`: fixed layout, seed 0, CPU PhysX TGS at 240 Hz, native
YAM actuators and the same 14-dimensional joint-position/gripper interface at
20 Hz. Reset settles mechanics for two seconds before starting the task clock.

```bash
python -m unittest -v test_electronics.py
python dream_eval.py --check
```

Both checks run without model calls. Each trial is limited to 2,400 control steps,
including Dream's 32-step observation preroll. API waits pause physics.

## Evaluation

The recorded Dream-1 beta trial scored **0.0/1.0 in 120 simulation seconds**
(285.5 seconds wall time on RTX 5090); it made no calculator submission.

Dream receives only three calibrated robot camera views and 14 measured joint/gripper
positions. Native 640x360 images become 320x176 JPEGs at the model interface. The
full prompt is `dream_eval.INSTRUCTION`; the installed client reads
`~/.dreamlabs/credentials`.

```bash
python dream_eval.py
python score.py runs/dream-eval-<timestamp>
```

The demo scored **1.0/1.0 in 40.2 simulation seconds**, submitting `45` after
33.783 seconds. It observes the timer, reads the thermometer, and presses
`4`, `5`, `=`. Replay sends 804 fixed commands from `demo_actions.npy`.
Policy execution uses no simulator observations; independent measurements only
record and score the trial.

```bash
python demo.py
python score.py runs/demo-<timestamp>
```

Each execution saves measured states, applied actions, results and a source snapshot.
Dream also archives the exact input JPEGs and full returned action chunks.
Render and compress saved states without repeating inference:

```bash
python render.py runs/demo-<timestamp>
python pack_video.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Videos go in each run's `videos/`;
sampled images go in `frames/`. A fixed 1920x1080 external observer and three native
640x360 robot views form a synchronized 1920x1440, 30 FPS video. The observer is
for human review only. `python inspect_scene.py` captures the initial calibrated
Top Camera at 1920x1080 in `submission/uploads/screenshot.png`.
