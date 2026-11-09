# Flip a steak at the right time

Use a spatula to turn a steak in a heated pan. Keep its entire initial bottom
face against the pan for at least 50 seconds, then completely separate that face
from the pan by 70 seconds. Leave the steak resting in the pan, cooked side up.

The pan starts in the middle of a black tabletop heater, with a count-up timer
at the upper right and a black spatula standing handle-up in a light-gray holder
at the lower right. The task uses dual YAM arms and three calibrated robot cameras.
See `task.json` for object dimensions, materials, the fixed layout and reset details.

## Scoring

Score the final state, up to 1.0:

- **0.5:** The original bottom face completely leaves the pan and the steak ends
  resting on its opposite face in the pan, released from hands and the spatula.
- **0.5:** The first loss of full-face contact occurs at or after 50 seconds,
  and complete separation of the original bottom occurs at or before 70 seconds.
  This credit requires a completed flip.

No complete separation, an airborne or tool-supported steak, or a steak outside
the pan scores zero. An early or late completed flip scores 0.5. The timer starts
at zero after scene settling; the two departure events are measured from that
instant. There are no per-region cumulative cooking timers.

The scorer checks the flat face boundary against the 135 mm-radius pan floor
with a 0.8 mm contact tolerance and verifies clearance from the pan walls before
counting complete departure. A final resting steak must have linear/angular
speed below 0.02 m/s and 0.2 rad/s and no hand/tool contact above 0.05 N. The pan
must face upward (its local up axis has world Z greater than 0.95). These
numerical tolerances implement contact and rest; they do not change the 50/70 s
limits. `score.py` recomputes the events and final score from every recorded
240 Hz physics state and contact force, without calling a model.

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
Task assets are self-contained; robot assets and camera calibration come from yam-arena.

Run all commands below from this directory after activating the environment:

```bash
export OPENBLAS_NUM_THREADS=1
python -m unittest test_scoring
python run.py check
```

## Evaluation

Both evaluations use the fixed initial layout, seed 0, CPU PhysX at 240 Hz and
20 Hz control. Both use native YAM actuator settings: gripper stiffness 5000 N/m,
damping 100 N s/m and effort limit 50 N. Objects move through physical contact
without grasp attachments. Reset settles the scene for two seconds before the
episode clock begins; the 120-second limit includes final task settling.

### Dream-1 model

Dream-1 beta receives three calibrated camera images and 14 measured robot
joint/gripper positions. Native 640x360 images are resized to 320x176 JPEGs at the
model interface. API waits pause physics. The full prompt is `config.INSTRUCTION`,
and the installed client reads `~/.dreamlabs/credentials`.

The recorded trial scored **0.0/1.0 in 120 simulation seconds** (268 seconds wall
time on RTX 5090). The original bottom face never completely left the pan.

```bash
python dream_eval.py --check
python dream_eval.py
python score.py runs/dream-eval-<timestamp>
```

### Demo policy

The demo replays 1,604 fixed joint-position and gripper commands from
`demo_actions.npy` at 20 Hz, lasting 80.2 simulation seconds. Its commands and
timing are fixed. Simulator states are read only for independent scoring and
recording; those measurements do not feed back into the policy.

Physics replay takes about one minute of wall time on RTX 5090, excluding video
rendering. The verified open-loop replay scored **1.0/1.0**: full contact ended at 58.40 s,
complete departure occurred at 60.529 s, and the flipped steak settled at 72.65 s.

```bash
python run.py demo
python score.py runs/demo-<timestamp>
```

### Recordings

Render saved physical states without rerunning a trial:

```bash
python render.py runs/demo-<timestamp>
python encode_video.py runs/demo-<timestamp>
```

The same commands accept a Dream run directory. Videos are saved in each run's
`videos/`, sampled images in `frames/`. The fixed 1080p observer and three native
640x360 robot views form a 1920x1440, 30 FPS video. The observer is for human review
only. `run.py check` saves a native 1080p calibrated Top Camera initial screenshot
in its new run directory. The heater is already on; its knob and the timer's
buttons are not task controls. Cooking appearance is a two-state surface change,
not a thermal or food-deformation simulation; see `ASSETS.md` for licenses.
