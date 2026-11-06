# Measure a right triangle

Use the right arm to place a wooden ruler along both perpendicular sides of the
triangle on the fixed whiteboard. Check zero alignment and read each side with
the wrist cameras, then enter the area in square centimetres on the calculator
and press `=`. Leave the ruler at the second side and finish within 120 seconds.

The whiteboard starts on the left, the ruler beside it and the calculator farther
right. The task uses dual YAM arms and three calibrated robot cameras. The ruler
and calculator are free bodies; calculator keys have passive spring return.
See [task.json](task.json) for the layout and [ASSETS.md](ASSETS.md) for asset sources.

## Scoring

The first physical `=` press submits and freezes the displayed result:

- **1.0:** Submit a numeric value of 50 cm² within 120 seconds.
- **0.0:** Any other result, including no submission.

Measurement steps receive no separate credit and are not scoring conditions.
`task.py` implements the rule; `score.py` recomputes it from the archived
`result.json` without simulation or model calls.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
and its YAM assets, then activate `source /path/to/IsaacLab-Arena/yam.env` and run
`python -m pip install -r requirements.txt` from this task directory. Tested with
Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`. Task assets are bundled; optional
regeneration requires DejaVu Sans Bold (`fonts-dejavu-core` on Debian/Ubuntu).

Run all commands below from this directory in the activated environment, using
a new output directory for each execution.

```bash
python -m unittest -v test_task
python run.py check runs/check
```

## Evaluation

Both evaluations share `dream-defaults.json`: fixed layout, seed 0, 240 Hz CPU
PhysX and 20 Hz native 14-dimensional joint-position/gripper commands. Native
YAM actuator settings are unchanged; objects move through physical contact.

### Dream-1 model

Dream-1 beta receives three calibrated 640×360 robot camera images, resized to
320×176 JPEG, plus measured robot joint/gripper positions. The limit is 2,400
control steps, including the 32-step observation preroll; API waits pause physics.
The full prompt is `dream_eval.INSTRUCTION`. The client reads its configured
`~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds**, with no calculator
submission (265 seconds wall time on an RTX 5090).

```bash
python run.py adapter-check runs/adapter-check  # Offline camera and action checks
python run.py dream runs/dream                 # One Dream-1 trial
python score.py runs/dream
```

### Demo policy

The demo uses a fixed open-loop action script: 1,920 prerecorded joint-position and
gripper commands from `demo_actions.npz`, sent at 20 Hz. During execution, the policy
uses no privileged simulator information. Its commands and timing
are fixed. `simulation.py` reads simulator states only for independent scoring and
recording; those measurements do not feed back into the policy.

The demo scored **1.0/1.0**, submitting 50 at **86.683 seconds** and finishing at
**96 simulation seconds**, including result inspection. An independent replay
took 62 seconds wall time on an RTX 5090.

```bash
python run.py replay runs/demo
python score.py runs/demo
```

### Recordings

Each run saves `result.json`, source hashes, executed actions and 240 Hz measured
states in `trajectory.npz`. Dream also archives exact input JPEGs and complete
action replies. Render saved states without rerunning a trial:

```bash
python run.py video runs/demo
python pack_video.py runs/demo/videos
python run.py video runs/dream
python pack_video.py runs/dream/videos
```

Videos are saved in each run's `videos/`, sampled images in `frames/`. A fixed
1080p observer and three 640×360 robot views form a synchronized 1920×1440,
30 FPS recording. Compression preserves all frames and produces a file under
10 MB. The observer is for human review only. `python run.py preview runs/preview`
saves the calibrated 1080p initial image to `submission/uploads/screenshot.png`.
