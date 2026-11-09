# Timed pen collection

Press the clock's large red button, collect five scattered pens into the white
holder, and press the button again to stop timing. The clock displays minutes,
seconds and hundredths of a second. The task uses dual YAM arms and three
calibrated robot cameras, with red tape marking the scatter and holder areas.

## Scoring

Score the final state: **1.0** for starting the timer before collecting the pens,
releasing all five into the holder and then stopping the timer within 120 seconds;
otherwise **0.0**. There is no partial credit. Containment uses the pen axis inside
the holder cavity and rim, with contact support through the holder or other pens
and no robot grasp. Pens still in flight or held by the robot do not earn success.
`task_runtime.py` measures the sequence and contacts; `score.py` checks saved trials
without model calls.

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
yam-arena. See `ASSETS.md` for dimensions, mechanisms and licenses.

## Evaluation

Both policies use `dream-defaults.json`: seed 0, a reproducible random scatter,
CPU PhysX at 240 Hz and native 14-dimensional joint/gripper commands at 20 Hz.
The robot keeps its native actuator settings. Objects move through physical
contact without grasp attachments or table anchors. Run commands from this directory.

### Dream-1 model

Dream-1 beta receives three calibrated 640×360 camera images, resized to
320×176 JPEG, and 14 measured joint/gripper positions. API waits pause physics.
The client reads `~/.dreamlabs/credentials`. The full prompt is
`dream_eval.INSTRUCTION`. One recorded trial scored **0.0/1.0 in 120 simulation
seconds** (230 seconds wall time on RTX 5090).

```bash
python dream_eval.py --check
python dream_eval.py
python score.py runs/dream-eval-<timestamp>
```

### Demo policy

The demo replays 1,788 fixed commands from `demo_actions.npy`, without simulator
state feedback to the policy. The left arm starts timing while the right arm
approaches the first pen. After four right-hand collections, the left arm moves
the far pen into shared reach. The right arm releases the final pen while the
left arm stops timing. An independent replay from the submitted files scored
**1.0/1.0 in 89.4 seconds**, ending with the clock at **01:21:51**.

```bash
python demo.py
python score.py runs/demo-<timestamp>
python -m unittest -v test_task
```

### Recordings

Render saved measured states without rerunning a policy:

```bash
python render.py runs/demo-<timestamp>
python pack_video.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Four synchronized views form
a 1920×1440, 30 FPS video; the fixed 1080p observer is for human review only.
Original and compressed videos are in the run's `videos/`; sampled images are
in `frames/`. Compression preserves all frames and the full duration.
