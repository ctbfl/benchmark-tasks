# Split coins

Pick up an open jar holding five quarters and five pennies, pour the coins onto the
middle of the table and put the jar back. Then sort the coins by pushing them along
the table: pennies into the left red tape frame, quarters into the right one. The task
uses dual YAM arms and three calibrated robot cameras.
This represents sorting spare change accumulated in a household jar before counting
or using it.

## Scoring

Score the final state within 120 seconds: **0.1 per penny** wholly inside the left
frame and **0.1 per quarter** wholly inside the right frame. The score is the number
correctly sorted divided by ten; all ten earn **1.0**. Wrong-frame, out-of-frame and
still-in-jar coins earn no credit. Coins may touch or overlap. A coin counts when its
horizontal footprint, including any tilt, is inside the tape's inner edge; touching
that edge is allowed. `task_runtime.py` measures the
coin poses; `score.py` checks saved trials without model calls. The scorer excludes
coins whose centres remain inside the jar's cavity, measured in the jar's own
coordinate system so the exclusion follows the jar when it moves or tilts.

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
The task USDs are self-contained; robot assets and camera calibration come from
yam-arena. [LICENSE](LICENSE) contains the asset declarations and pinned sources:
original coin, jar and tape models are CC BY 4.0; YAM robot assets and upstream
calibration are MIT. It also identifies the runtime-generated workstation geometry.
Task code retains MIT. See [ASSETS.md](ASSETS.md) for dimensions and simulation details.

## Evaluation

Both policies use `dream-defaults.json`: seed 0, a fixed initial layout, CPU PhysX at
240 Hz and native 14-dimensional joint/gripper commands at 20 Hz. The coins start in
the jar from fixed layered poses and settle for two seconds before the episode.
The robot keeps its native actuator settings. Objects move through physical contact
without grasp attachments or table anchors. Run commands from this directory.

### Dream-1 model

Dream-1 beta receives three calibrated 640×360 camera images, resized to
320×176 JPEG, and 14 measured joint/gripper positions. API waits pause physics.
The client reads `~/.dreamlabs/credentials`. The full prompt is
`dream_eval.INSTRUCTION`. One recorded trial scored **0.0/1.0 in 120 simulation
seconds** (272 seconds wall time on RTX 5090). Recomputing the archived final state
with per-coin credit also gives **0.0**; no new model inference was run.

```bash
python dream_eval.py --check
python dream_eval.py
python score.py runs/dream-eval-<timestamp>
```

### Demo policy

The demo replays 1,963 fixed commands from `demo_actions.npy`, without simulator
state feedback to the policy. The right arm grasps the jar from the right, tilts it
toward the robot to pour the coins onto the middle of the table and returns it to its
starting position. One fingertip then stirs the pile in an outward spiral to separate
stacked coins. Coins are pushed one at a time between the open gripper pads, pennies
by the left arm and quarters by the right; a coin on the wrong side is first pushed
across a gap to its own side. An independent replay from the submitted files scored
**1.0/1.0 in 98.15 seconds**, also confirmed by rescoring its archived final state
with per-coin credit.

The command script was recorded from a scripted planner (`expert.py`) that finds the
coins with computer vision on the top camera's RGB image (`coin_vision.py`), rendered at
1920×1080 with the same calibrated pose and lens as the robot's top camera. Pixels are
classified by colour as copper pennies or silver quarters, projected onto the table plane,
and individual fully visible coins are located as peaks of a top-down distance transform.
The planner then chooses pushes whose paths and fingertip landing spots are clear of the
other detected coins, and stirs the pile when no coin can be moved. Besides the camera
image it reads only the robot's own joint and gripper state; it never reads coin or jar
poses from the simulator. The jar grasp uses the jar's fixed starting position. A
validation run of `expert.py` reproduced all 1,963 commands in `demo_actions.npy`.

```bash
python demo.py
python score.py runs/demo-<timestamp>
python -m unittest -v test_task
python expert.py          # regenerates runs/expert-<timestamp>/actions.npy
```

### Recordings

Render saved measured states without rerunning a policy:

```bash
python render.py runs/demo-<timestamp>
python pack_video.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Four synchronized views form a
1920×1440, 30 FPS video: a fixed 1080p observer, which is not a robot camera, above
the native 640×360 left wrist, top and right wrist cameras.
