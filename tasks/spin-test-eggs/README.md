# Classify an intact egg as raw or hard-boiled and sort it

> Determine whether the egg is raw or hard-boiled by spinning it and stopping it abruptly. Place it in the correctly labeled basket.

## Setup and reset

Use a dry chicken egg at 20–22 °C, about 58 × 42 mm and 60 g. Draw the class randomly from intact raw and hard-boiled specimens from the same carton. Keep the class record private. Lay one egg on its side about 335 mm forward of the YAM station midpoint, with its long axis across the station between the two arms. Vary position by up to 8 mm in each horizontal direction and yaw by up to 7°.

Center a level 280 × 310 × 8 mm polished glass board 365 mm forward of the station midpoint. Use dry glass with finished edges. Place two 246 × 174 mm slotted rigid plastic baskets 360 mm forward and 246 mm to each side. Each has a 6 mm floor and a 30 mm rim above the floor. Mark the left basket Raw and the right basket Hard-boiled. Keep the shell visible in the camera views.

Use washable finger covers when handling food and keep spill containment outside the board. Inspect the shell before and after each trial. Stop a trial if the shell cracks or leaks; that trial scores 0. Remove the specimen and clean and dry the board and covers before resetting.

## Scoring

The three stages award cumulative credit:

- **0.4:** Spin the egg through at least one complete tabletop revolution.
- **+0.3:** Abruptly stop the spinning egg by robot contact.
- **+0.3:** After timely release and a hands-off response interval on the board, finish with the egg fully inside the correctly labeled basket and released.

Later stages require the earlier ones. Without a full revolution, every outcome scores 0. After a full revolution but without a qualifying abrupt stop, every outcome scores 0.4, including correct basket placement. A completed spin and qualifying stop earns 0.7. The final 0.3 also requires complete loss of robot contact within 0.5 seconds after the egg is stopped, followed by at least 0.25 continuous seconds resting on the board without robot contact before regrasping. Gripping past that deadline, immediate carrying, an incomplete response interval, wrong placement or an egg still held in its basket cannot earn that final 0.3. Any shell crack or leakage at any point overrides all partial credit: the entire trial scores 0.

Count a full turn from the shell's continuous orientation on the board; smaller repeated rocking does not accumulate credit. A qualifying abrupt stop starts from at least 2 revolutions per second immediately before the first braking contact. Robot contact must reduce vertical angular speed to at most 0.5 rad/s within 0.30 seconds. A slower turn can earn the initial 0.4 but cannot qualify for either later increment. The simulator uses three consecutive 240 Hz samples (12.5 ms) to filter isolated zero crossings. Brief contact gaps within the braking interval are allowed. Natural slowing does not count as an abrupt stop.

For physical grading, track the same non-class-identifying shell marking across video frames. Two revolutions per second corresponds to one turn per 0.5 seconds; use the reference demo and frame timestamps to check pre-braking speed, the visible pause, complete separation within 0.5 seconds of that pause, and the subsequent 0.25-second hands-off interval. A fixed close view and a high-frame-rate phone recording help resolve marginal cases. The 2 revolutions per second minimum is this task's operating requirement, not a universal calibrated threshold for real eggs.

Evaluation uses recorded physical actions and the final basket checked against the private class record. It does not infer the policy's attention or internal reasoning. The contact-free response interval makes the egg's mechanical response available to observe; no claim is made about which information the policy used. Correct basket choice alone cannot substitute for the required spin, stop and release protocol.

The demonstration opens the fingers after arrest and classifies from the shell's response: a raw egg can resume turning, while a hard-boiled egg remains nearly still. The rubric requires the physical response opportunity, without imposing a particular viewing direction or two-hand contact pattern.

## Why it differs

- Opaque bottles are sorted by the large load difference during an upright lift. The egg test uses matched mass and transfers momentum to a supported shell before observing its hands-off response. [Throw out the empty bottles, leave the full ones: they look the same](https://github.com/robocurve/benchmark-tasks/pull/1938)
- The brake must hold a pre-spinning wheel still while the other hand removes a fastener. The egg must be set spinning, released, and classified from its subsequent motion. [Stop a coasting flywheel with its brake lever, not by hand, and keep it braked to unscrew its clamp nut](https://github.com/robocurve/benchmark-tasks/pull/1674)
- The carton task handles whole eggs and tests pointed-end seating and lid closure. The spin test uses a shell's mechanical response to decide where to place it. [Pack six eggs point-down from an egg flat into a half-dozen carton and close its lid](https://github.com/robocurve/benchmark-tasks/pull/2384)

## Validation

Dream-1 scores 0.0 at 2,400 steps (120 seconds), seed 1178019216: the egg does not complete a full tabletop revolution. The scripted demo scores 1.0 in 377 steps (18.85 seconds) on the same raw-egg seed, and 1.0 in 375 steps on hard-boiled seed 817639092. The demo classifies from measured shell motion; the hidden class is used only for reset and scoring.

The raw and hard-boiled demos turn at 2.23 and 3.51 revolutions per second immediately before braking, respectively. They lose all robot contact 0.033 and 0.046 seconds after arrest. Both complete the 0.25-second contact-free board interval before regrasping.

## Simulation and reproduction

The station uses CPU PhysX at 240 Hz and native YAM position and jaw actions at 20 Hz. Both classes share shell geometry, mass and contact materials. A raw egg initially uses the same full rigid inertia as a boiled egg. At a vertical spin speed of 2 revolutions per second, the raw model transfers existing vertical angular momentum into a passive internal rotor: 20% effective shell inertia and 80% internal inertia. A linear, equal-and-opposite torque relaxes their speed difference over 30 ms. A short shell arrest leaves internal rotation available to restart the shell after release. This threshold rule is a phenomenological approximation, not resolved fluid flow or a fit to measured tabletop egg dynamics. It resets with the episode. Native contact loads and uncalibrated overload markers are diagnostic only. The shared board static and dynamic friction coefficient is 0.0037, an explicit low-friction approximation without physical tabletop spin calibration. [Background on the physical spin test](https://arxiv.org/html/2202.10243#S4)

Shells are rigid and do not fracture or leak in this simulator. The scorer latches a separate `shell_broken` event and makes the entire score 0; the rigid-shell simulation emits false by construction. In physical trials, visible cracking or leakage supplies that event. The `overloaded` telemetry is an uncalibrated load diagnostic, not a fracture detector or evidence of shell integrity.

The scripted policy uses measured shell motion to classify. Dream-1 receives the three calibrated wrist and top camera images and 14-value joint/gripper state. The scene, scorer and model observations share one configuration.

Dream captures the three robot views once per 20 Hz observation. The same images supply the policy encoder and the native camera recording. The journal saves measured poses, physics samples and executed actions in one-second batches. The external observer panel is rendered separately from the recorded robot link and egg poses. The 30 fps presentation holds the latest recorded sample between timestamps.

Task-local board, egg and basket meshes are authored from `assets.py`. For the evaluation environment, use Isaac Lab 17.0.2, Isaac Lab Arena 0.3.0, inspect-robots 0.60.0 and inspect-robots-dream 0.1.0. The shared YAM station and calibration come from the existing YAM Arena checkout at commit `bd62771e60320433893bd1d2df0fc79b5ddcd6d1`. Its included i2rt robot mesh is available under MIT. [YAM Arena](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1)

The runner also uses SciPy, Pillow and imageio-ffmpeg, and video compression requires FFmpeg and FFprobe. Set `PRODUCTION_GPU_RUN_ID` to the runner-provided identifier in production. The episode limit is 2,400 control steps at 20 Hz. Each scored run draws a fresh evaluator seed by default. That seed controls starting position, yaw and the hidden class of one egg. The model receives its normal observations and instruction. The evaluator stores the seed in `trial.json` and `result.json`.

```bash
python run_trial.py demo /path/to/new-demo-run
python run_trial.py dream /path/to/new-dream-run
python render_dream_media.py /path/to/new-dream-run
python pack_video.py /path/to/new-dream-run/videos/original.mp4 /path/to/dream.mp4
python run_trial.py screenshot /path/to/screenshot.png RECORDED_DEMO_SEED
python run_trial.py demo /path/to/demo-replay 1178019216
python pair.py /path/to/matched-pair 0
```

- [Exploratorium, Take an Egg for a Spin](https://annex.exploratorium.edu/cooking/eggs/activity-spin.html)
- [Bertho, Darbois Texier and Pauchard, Egg-speriments: Stretch, crack, and spin, §IV](https://arxiv.org/html/2202.10243#S4)
