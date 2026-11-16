# Measure a wooden bar and enter its length

Measure the bar's longest edge using the retractable tape, enter the length as
integer millimetres on the calculator, and press `=`. The calculator starts on
the robot's left, the upright tape on its right, and the bar behind both.
The bar length is sampled uniformly from every integer from **150 to 200 mm**
using `numpy.random.default_rng(episode.seed).integers(150, 201)`. Set the seed
in `dream-defaults.json`; seed 0 selects 193 mm, seed 1 selects 174 mm and seed 6
selects 172 mm. Width and height stay 70 mm. The scene scales the bar geometry
and mass together; no asset rebuild is needed when changing the seed.

## Scoring

The score is the highest stage reached before the first `=` press or the
120-second limit:

| Score | Condition |
| --- | --- |
| 0 | No seated, zero-aligned hook; a correct answer alone earns no credit. |
| 0.25 | The hook contacts the bar and its zero is within 1 mm of either end plane. |
| 0.5 | At the same instant, the hook remains seated, the exposed blade extends beyond the opposite end plane and points along the bar within 10 degrees. |
| 1 | After reaching 0.5, the first `=` submits the correct integer millimetres within 120 seconds. |

The angle is three-dimensional and positions are relative to the moving bar.
Hook contact is a nonzero PhysX hook/bar contact force. There is no minimum
holding time. Completed stages persist, so the tape may be retracted before
typing; an incorrect answer retains the measurement credit. Physical key
travel triggers calculator events. `score.py` recomputes the stages from
recorded tape endpoints, bar poses and contact forces, then checks the submission.

## Setup and reset

The unlabelled bar represents solid pine, modelled as a rigid homogeneous
block with density about 617 kg/m3. The calculator has a plastic case and
physical spring-loaded keys. The tape has a plastic/rubber case, a steel hook
and a graduated steel blade; blade bending and intermediate contact are omitted.

Spawn positions below are in metres in the standard YAM station world frame.
Every prop uses quaternion `(x,y,z,w) = (0,0,-sqrt(0.5),sqrt(0.5))`.

| Object | Position (x, y, z) |
| --- | --- |
| Tape case | (0.235, -0.160, 0.7820) |
| Calculator base | (0.235, 0.180, 0.7480) |
| Bar centre | (0.415, 0, 0.7830) |

The tabletop is at z=0.7475 m. There is no position jitter. Start both arms at
joint angles `(0,1,1,-0.6,0,0)` radians with open grippers, the tape retracted
and calculator cleared. Settle for 2 seconds before starting the clock.

## Environment

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install
[yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)
using its setup instructions, then activate `source <ARENA_DIR>/yam.env` and
install `requirements.txt` inside that environment. Tested with Python 3.12,
Isaac Sim 6.1.0.0, Isaac Lab 3 (package 17.0.2) and Arena 0.3.0 at
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.
At this Arena revision, `ARENA_NUCLEUS_DIR` must point to the production
`ISAACLAB_NUCLEUS_DIR`; `arena-assets.patch` supplies that resource-URL correction
for a fresh installation (`git -C <ARENA_DIR> apply <TASK_DIR>/arena-assets.patch`).
Robot assets and camera calibration come from yam-arena. Task assets are
self-contained. Asset sources and licenses:

| Asset | Source and license |
| --- | --- |
| Task meshes, wood texture, tape scale and key-label images | Original procedural assets in `model.py`; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution and scope in [LICENSE](LICENSE). |
| DejaVu Sans Bold glyphs in the labels | [DejaVu](https://dejavu-fonts.github.io/); Bitstream Vera license, DejaVu changes public domain; [notice](FONT-LICENSE.txt). Font file not bundled. |
| YAM robot URDF, meshes and derived collision/USD assets | I2RT Robotics, [v1.3.6 / 120c3c814001](https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058), supplied through yam-arena; [MIT](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/LICENSE). Not bundled. |
| YAM black/white link meshes | I2RT meshes distributed with dreamlabs 1.0.23; [source and hashes](https://github.com/kairosiann/yam-isaac-lab-arena/blob/bd62771e60320433893bd1d2df0fc79b5ddcd6d1/yam_arena/assets/yam/dream/SOURCE); MIT under the I2RT license above. Not bundled. |
| Arm and camera calibration | ARISE-Initiative/yamlab, commit `ec0455d2b4ce`, via Dream's standard station and yam-arena; [MIT](https://github.com/ARISE-Initiative/yamlab/blob/ec0455d2b4ce35f21fc126418ea5e74ac567133d/LICENSE). Not bundled. |
| Standard YAM workstation | Standard benchmark environment; MIT. Supplied externally, not bundled. |

Task source code retains its [MIT license](LICENSE). Isaac Lab-Arena is
Apache-2.0; Isaac Lab is BSD-3-Clause. Product appearance references and further
notices are in `ASSETS.md`; no product photographs or downloaded product meshes
are distributed.

Run commands from this directory after activating the environment. Use a fresh
output directory for each execution. Both evaluations use `dream-defaults.json`:
fixed poses, seed 0 (193 mm), CPU PhysX TGS at 240 Hz, native YAM actuators and the same
14-dimensional joint-position/gripper interface at 20 Hz.

```bash
python run.py check runs/check
python run.py adapter-check runs/adapter-check
python -m unittest test_scoring.py
```

Both checks run without model calls. The limit is 2,400 control steps, or 120
simulation seconds, including Dream's observation preroll. API waits do not
advance physics.

## Evaluation

The recorded Dream-1 beta trial scored **0.0/1.0 in 120 simulation seconds**
(about 293 seconds wall time on RTX 5090); it did not seat the hook or make a calculator submission.
Dream receives only three calibrated robot camera views and 14 measured
joint/gripper positions. Native 640x360 images become 320x176 JPEGs at the model
interface. The installed client reads `~/.dreamlabs/credentials`.

```bash
python run.py dream runs/dream
python score.py runs/dream
```

The demo scored **1.0/1.0 in 113.5 simulation seconds**, submitting `193` at
109.883 seconds (about 77 seconds wall time on RTX 5090, excluding app startup).
It measures the bar, unhooks and releases the tape for passive
return, places the case on the table, and presses `1`, `9`, `3`, `=`.
Replay sends 2,270 fixed commands from `demo_actions.npz`; policy execution
uses no simulator observations. This action file is for seed 0 only; replay
fails immediately with a different seed. Independent replay reproduced all
27,241 recorded physical states from the development execution. A separate
answer-only run submitted the correct `193` without measuring and scored 0.

```bash
python run.py replay runs/demo
python score.py runs/demo
```

Each execution saves states, actions, calculator events, its result and a
source/configuration/asset snapshot. Dream also saves the exact input JPEGs and
complete returned action chunks. Render saved states without repeating inference:

```bash
python run.py video runs/demo
python pack_video.py runs/demo/videos
python run.py video runs/dream
python pack_video.py runs/dream/videos
```

The fixed observer and three native robot views form a synchronized 1920x1440,
30 FPS video. `videos/original.mp4` preserves the original encoding;
`videos/video.mp4` is the full-length compressed version, limited to 10 MB.
The observer is for human review and is never supplied to Dream.
