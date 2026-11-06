# Set a bulb to its rated voltage

Light a 5 V bulb safely: inspect the wiring and digital voltmeter, steady the
rheostat base with the left hand, and move its slider with the right hand. Bring
the voltage from 1 V to 5 V and hold it there for one second within a 120-second trial.

The 15 V source is on the left. The bulb, voltmeter and rheostat lie on the table's
centerline, in that order away from the robot. The task uses dual YAM arms and three
calibrated cameras, a free rheostat base and a passive slider starting at its midpoint.
See `task.json` for dimensions, materials, wiring, simulation simplifications and reset details.

## Scoring

Score the final state: **1.0** if the bulb voltage remains 4.8–5.2 V continuously
for one second and never exceeds 7 V during the trial; **0.0** otherwise.
During the final one-second window, the base must stay within 3 cm horizontally
and 1 cm vertically of its settled reset position, with tilt at most 15 degrees.
Hands may remain in contact. Passing through the target voltage does not count.

`circuit.py` implements these checks at 240 Hz. `score.py` recomputes the score
from saved physical states without model calls. Both demo and Dream save
`metrics.json`; rescoring writes `rescored.json` in the selected run directory.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh` and YAM asset conversion instructions, then activate
`source <ARENA_DIR>/yam.env`. Tested with Python 3.12, Isaac Sim 6.1.0.0,
Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
At this Arena revision, `isaaclab_arena/assets/nucleus.py` must use
`ARENA_NUCLEUS_DIR = ISAACLAB_NUCLEUS_DIR` for production asset URLs.

The task USDs and label textures are included; robot assets and camera calibration
come from yam-arena. See `LICENSE` and `THIRD_PARTY.md` for licensing.
Run all commands below from this directory in the activated environment:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_circuit.py
python run.py check                 # Offline scene, camera and motion checks
```

## Evaluation

Demo and Dream share `dream-defaults.json`: fixed layout, seed **0**, CPU PhysX TGS
at **240 Hz**, and native 14-D YAM joint-position/gripper actions at **20 Hz**.
Both retain the native gripper stiffness of 5000 N/m, damping of 100 N s/m and
effort limit of 50 N. Objects move through physical contact without grasp attachments.
Each run prints `RUN_DIRECTORY`; use that directory in the scoring and rendering commands.

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot
joint/gripper positions. Native 640×360 images become 320×176 JPEGs at the model
interface. The limit is 2,400 control steps, including the 32-step observation
preroll; API waits pause physics. The full prompt is `circuit.INSTRUCTION`, and
the installed client reads `~/.dreamlabs/credentials`.

Both recorded trials scored **0.0/1.0 in 120 simulation seconds**.
The latest trial ended at **1.000 V**.

```bash
python dream_eval.py --check        # Offline model-adapter check; no API calls
python dream_eval.py                # One trial; requires a configured API key
python score.py runs/dream-TIMESTAMP
```

### Demo policy

The demo replays **1,013 fixed commands** from `demo_actions.npy` at 20 Hz.
Its actions and timing do not depend on simulator feedback; state measurements
are used only for independent scoring and recording.

The independently replayed demo scored **1.0/1.0 in 50.65 simulation seconds**,
ending at **4.960 V**, with a peak of **5.157 V**. The action recording targets
the specified fixed initial layout.

```bash
python run.py demo
python score.py runs/demo-TIMESTAMP
```

### Recordings

Render saved physical states without rerunning a trial:

```bash
python render.py runs/demo-TIMESTAMP
python render.py runs/dream-TIMESTAMP
```

Each run saves configuration, source hashes, actions and 240 Hz physical states.
Rendering writes `videos/original.mp4` and sampled images under `frames/`.
A fixed 1080p observer and three native 640×360 robot views form a synchronized
1920×1440, 30 FPS video. The observer is for human review only.
`run.py check` saves the initial 1920×1080 Top Camera image as `frames/screenshot.png`.
