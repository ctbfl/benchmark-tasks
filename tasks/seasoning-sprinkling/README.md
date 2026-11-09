# Safely season a steak with a dual-opening shaker

Inspect a pepper-salt bottle before seasoning a steak: one flip-top cover hides a
wide pouring opening, while the other hides four small dispensing holes. Close
the wide side, put the bottle down, turn it around, open the perforated side and
sprinkle twice without overdosing the food.

A plate holding one steak starts in the middle of the calibrated dual-YAM
workstation. The closed bottle stands to its right, with the wide opening facing
left. All task geometry and textures are included; robot and workstation assets
come from yam-arena.

## Scoring

Score the final state, up to 1.0:

- **0.5:** Seasoning remains on the steak.
- **0.5:** The seasoning on the steak weighs more than 0 g and no more than 2 g.

No seasoning earns 0.0. A positive dose up to 2 g earns 1.0; an overdose earns 0.5.
Each simulated grain weighs 0.0125 g. The scorer sums grains resting directly on
the steak and supported grains in a pile. Airborne grains and table spills do not
count as seasoning on the steak. `score.py` recomputes the result from the archived
final physical state without model calls.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env` and install
`requirements.txt` inside that environment. Tested with Python 3.12, Isaac Sim
6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`. At that Arena revision,
apply `git -C <ARENA_DIR> apply <TASK_DIR>/arena-assets.patch` to point
Arena assets to the production resource bucket.

Run commands from this directory after activating the environment:

```bash
export OPENBLAS_NUM_THREADS=1
python -m unittest -v test_scoring.py
python run.py check
python dream_eval.py --check
```

Both policies use seed 0, a fixed initial layout, CPU PhysX at 240 Hz, native YAM
actuator parameters, grain continuous collision detection and the same
14-dimensional position/gripper action interface at 20 Hz. The limit is 2,400 control steps, or 120 simulation seconds. Start a new
process for each trial to restore the bottle and all 800 grains.

## Evaluation

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot
joint/gripper positions. Native 640×360 images become 320×176 JPEGs at the model
interface. API waits pause physics. The instruction is `config.INSTRUCTION`;
the installed client reads `~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds**, with 0 g on the
steak: neither scoring item was met. It took about 17.0 minutes
wall time on an RTX 5090 workstation. The archive contains 7,200 input JPEGs,
50 complete action replies and all 2,400 executed actions.

```bash
python dream_eval.py
python score.py runs/dream-<timestamp>
```

### Demo policy

The demo replays 2,110 fixed actions from `demo_actions.npy` at 20 Hz. It does not read
object poses, passive joint angles or scoring feedback. The runner records
physical state separately for scoring and rendering.

A fresh replay scored **1.0/1.0 in 105.5 simulation seconds**, leaving 0.325 g
on the steak. The run took about 5.1 minutes wall time on an RTX 5090 workstation.

```bash
python run.py demo
python score.py runs/demo-<timestamp>
```

### Recordings

Render archived physical states without rerunning a trial:

```bash
python render.py runs/demo-<timestamp>
python encode_video.py runs/demo-<timestamp>/videos/original.mp4
```

The fixed 1080p observer and three native 640×360 robot views form a 1920×1440,
30 FPS video. The observer is for human review only. Videos are saved in `videos/`
and sampled images in `frames/` under the selected run.
