# Install a laboratory thermometer

Attach a loose thermometer clamp to the iron stand, tighten its round screw knob,
and use its spring jaws to support the thermometer in the beaker water. Leave the
thermometer within 20 degrees of vertical with the 20–60 °C scale unobstructed,
then open both grippers and withdraw.

The stand and beaker start on the left. The thermometer stands in a square holder
on the right, with its scale facing left; the loose clamp lies in front of it.
The task uses dual YAM arms and three calibrated robot cameras. See `task.json`
for dimensions, materials, the fixed layout and simulation fidelity.

## Scoring

Score the final state, up to 1.0:

- **0.25:** The clamp remains fixed to the stand after release.
- **0.25:** The thermometer remains held by the clamp, within 20 degrees of vertical.
- **0.25:** Its lower tip is inside the beaker below the water surface.
- **0.25:** The clamp leaves the 20–60 °C scale clear.

Thermometer credit requires the clamp to be mounted; immersion and scale credit
require the thermometer to be secured. Stability is measured over the final second:
motion stays within 2 mm and 2 degrees, and robot contact stays below 0.01 N.
The stand's feet stay within 3 mm of the tabletop. Socket, jaw, tip and scale checks use the
asset geometry in `score.py`; scores cannot be earned by merely holding the parts
in place. `score.py` independently recomputes the result from saved measurements.

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
20 Hz. Reset settles the scene for two seconds before starting the task clock.
All object interactions use physical contact; no grasp attachments are added.

```bash
python -m unittest -v test_score.py
python dream_eval.py --check
```

Both checks run without model calls. Each trial is limited to 2,400 control steps,
including Dream's 32-step observation preroll. API waits pause physics.

## Evaluation

The recorded Dream-1 beta trial scored **0.0/1.0 in 120 simulation seconds**
(255.9 seconds wall time on RTX 5090). The open-loop demo scored **1.0/1.0 in
116 simulation seconds**, replaying 2,320 fixed commands. Both results were
independently recomputed from the saved measurements.

Dream receives only three calibrated robot camera views and 14 measured joint/gripper
positions. Native 640×360 images become 320×176 JPEGs at the model interface. The
full prompt is `dream_eval.INSTRUCTION`; the installed client reads
`~/.dreamlabs/credentials`.

```bash
python dream_eval.py
python score.py runs/dream-eval-<timestamp>
```

The demo replays fixed joint and gripper commands from `demo_actions.npy`.
Policy execution uses no simulator observations; independent measurements only
record and score the trial.

```bash
python demo.py
python score.py runs/demo-<timestamp>
```

Each execution saves measured states, applied actions, results and a source snapshot.
Dream also archives exact input JPEGs and full returned action chunks.
Render and compress saved states without repeating inference:

```bash
python render.py runs/demo-<timestamp>
python pack_video.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Videos go in each run's `videos/`;
sampled images go in `frames/`. A fixed 1920×1080 external observer and three native
640×360 robot views form a synchronized 1920×1440, 30 FPS video. The observer uses
the calibrated Top Camera pose and field of view, rendered at 1080p for human review.
`python inspect_scene.py` captures the initial calibrated
Top Camera at 1920×1080 in `submission/uploads/screenshot.png`.
