# Unfold and hang folding headphones

Prepare folded over-ear headphones for use: pull the inward-folded earcup out from
inside the headband, turn both earcups to their wearing positions with cushions facing
inward, and hang the headband in the stand's U-shaped cradle. Release both hands and
leave the headphones hanging stably for five seconds, within a 120-second trial.

The headphones start folded on the right of a table, with the stand on the left.
The task uses dual YAM arms and three calibrated robot cameras. The headset has a
passive elastic headband and folding/swivel joints; the stand is a free rigid body.
See `task.json` for object dimensions, materials, the fixed layout and reset details.

## Scoring

Score the final state, up to 1.0:

- **0.4:** Pull the inward-folded earcup out from inside the headband (both folding joints within 15 degrees of fully unfolded).
- **0.2 each:** Turn each earcup to its wearing position, cushion facing inward, within 15 degrees.
- **0.2:** Hang the headband in the stand's U cradle, release both hands and keep it stable for 5 seconds.

Each item scores independently; an unmet item scores zero. The simulator measures each
cup's wearing angle at its swivel joint. Unfolding and cup credit require
all headphone link centres to be at least table height minus 30 mm.
For hanging, the headband crown must be inside the cradle (stand-local |x| <=24 mm,
|y| <=17 mm, z=240–265 mm), all headphone link centres at least 20 mm above the table,
and the stand upright within 5 degrees with its base at table height within 5 mm.
Robot contact must stay below 0.05 N; measured stand support must exceed 0.5 N at least once
in the five-second window. Crown/stand motion must stay within 3/2 mm, using the Euclidean
norm of the per-axis position ranges. These checks are implemented in `scoring.py`.
`score.py` recomputes scores from saved states and contacts without model calls. The demo
reports `score.json`; Dream reports `metrics.json`, both under their `runs/` directories.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its `setup.sh`, then activate `source <ARENA_DIR>/yam.env` and install `requirements.txt`
inside that environment. Tested with Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 3 (package
17.0.2) and Arena 0.3.0 at `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
At this Arena revision, `ARENA_NUCLEUS_DIR` must point to the production
`ISAACLAB_NUCLEUS_DIR`; `arena-assets.patch` supplies that resource-URL correction for a
fresh installation (`git -C <ARENA_DIR> apply <TASK_DIR>/arena-assets.patch`).
The two task USDs are self-contained; robot assets and camera calibration come from yam-arena.

Run all commands below from this directory after activating the environment:

```bash
export OPENBLAS_NUM_THREADS=1
```

## Evaluation

Both evaluations use the fixed initial layout, seed 0 and CPU PhysX at 240 Hz.
Both use native YAM actuator settings: gripper stiffness 5000 N/m, damping 100 N s/m
and effort limit 50 N. Objects move through physical contact without grasp attachments.

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot joint/gripper
positions. Native 640x360 images are resized to 320x176 JPEGs at the model interface.
Control runs at 20 Hz for at most 2,400 steps, including the 32-step observation preroll;
API waits pause physics. The full prompt is `dream_eval.INSTRUCTION`, and the installed
client reads `~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds** (252 seconds wall time
on RTX 5090).

```bash
python dream_eval.py --check       # Offline scene, camera, action and scoring checks
python dream_eval.py               # One Dream-1 trial; requires a configured API key
python score.py dream runs/dream-eval-<timestamp>
```

### Demo policy

The demo uses a fixed open-loop action script: 2,395 prerecorded joint-position and
gripper commands from `demo_actions.npy`, sent at 30 Hz. During execution, the policy
uses no privileged simulator information or observation feedback. Its commands and timing
are fixed. `record_demo.py` reads simulator states only for independent scoring and
recording; those measurements do not feed back into the policy.

The demo scored **1.0/1.0 in 79.833 simulation seconds**, including five seconds of
released hanging. This command recording targets the specified fixed initial layout.

```bash
python demo.py
python score.py demo runs/demo
```

### Recordings

Render saved physical states without rerunning a trial:

```bash
python render.py demo runs/demo
python render.py dream runs/dream-eval-<timestamp>
```

Videos are saved in each run's `videos/`, sampled images in `frames/`. The fixed 1080p
observer and three native 640x360 robot views form a 1920x1440, 30 FPS video. The observer
is for human review only. The offline `--check` uses `fixtures/hanging.json` as a scoring
positive control; evaluations start with the folded headphones on the table.
