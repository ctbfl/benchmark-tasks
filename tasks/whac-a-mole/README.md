# Whac-a-mole buttons

Play whac-a-mole on a tabletop button device: every 5 seconds one of four round
buttons lights up for 1 second. Press the lit button while it is lit; each valid
press is counted on the device's score display. Get at least 5 successful presses
within a 120-second trial.

The device sits on the table in front of the robot. The task uses dual YAM arms and
three calibrated robot cameras. The device is a free rigid body with four passive
spring-return buttons and a two-digit score display.
See `task.json` for dimensions, the light schedule and reset details, and
[ASSETS.md](ASSETS.md) for asset sources.

## Scoring

Score the number of valid presses, 0.0 or 1.0:

- **1.0:** At least 5 valid presses within 120 seconds.
- **0.0:** Fewer than 5 valid presses.

A press is valid when the lit button's travel crosses 4 mm downward while it is lit
(each light gives at most one hit; the lamp turns off when hit). Pressing an unlit
button, or holding a button down from before the light, does not count and is not
penalized. Lights start at 5, 10, ..., 115 seconds; the button sequence comes from
scene seed 0. `task.py` implements the rule; `score.py` recomputes it from the
archived `result.json` without simulation or model calls.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
and its YAM assets, then activate `source /path/to/IsaacLab-Arena/yam.env` and run
`python -m pip install -r requirements.txt` from this task directory. Tested with
Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`. The task USD is bundled; `python run.py build`
regenerates it.

Run all commands below from this directory in the activated environment, using
a new output directory for each execution.

```bash
python -m unittest -v test_task
python run.py check runs/check
```

## Evaluation

Both evaluations share `dream-defaults.json`: fixed layout, seed 0, 240 Hz CPU
PhysX and 20 Hz native 14-dimensional joint-position/gripper commands. Native
YAM actuator settings are unchanged; buttons move only through physical contact.

### Dream-1 model

Dream-1 beta receives three calibrated 640×360 robot camera images, resized to
320×176 JPEG, plus measured robot joint/gripper positions. The limit is 2,400
control steps, including the 32-step observation preroll; API waits pause physics.
The full prompt is `dream_eval.INSTRUCTION`. The client reads its configured
`~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds**, with no button
pressed (302 seconds wall time on an RTX 5090).

```bash
python run.py adapter-check runs/adapter-check  # Offline camera and action checks
python run.py dream runs/dream                 # One Dream-1 trial
python score.py runs/dream
```

### Demo policy

Because the lit button is random, the demo is not a fixed action script. It is a
scripted closed-loop policy whose only observation is the robot's Top Camera image.
`demo.py` samples a fixed 7×7 pixel region on each button and treats a region as lit
when its red and green channels are both bright. It then sends precomputed
joint-position targets (inverse kinematics of the fixed layout) to press that button.
The left arm covers the left buttons and the right arm the right buttons, both
waiting above the far buttons, which keeps every region visible. The policy reads no
simulator object or lamp state; `simulation.py` reads those only for scoring and
recording.

The demo scored **1.0/1.0**, with hits 0.15–0.31 s after each of the first five
lights, the fifth at **25.31 seconds**. The run stops 2 seconds later, at 27.35
simulation seconds (45 seconds wall time on an RTX 5090).

```bash
python run.py demo runs/demo
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
saves the calibrated 1080p initial Top Camera image to `runs/preview/screenshot.png`.
