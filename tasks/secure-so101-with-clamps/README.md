# Secure an SO-101 robot arm to a small table edge with plastic clamps

Raise an unpowered, folded SO-101 follower from its forward-resting position and secure its native base directly to a small work table. The left YAM gripper raises and steadies the arm while the right gripper opens and fits two spring clamps. Each upper pad bears on a base foot and each lower pad bears under the tabletop. Both hands withdraw before the retention check. Table clamps are included in the [SO-101 bill of materials](https://github.com/TheRobotStudio/SO-ARM100/blob/a758567c3978dfeefe282ede0500085a48fe8f78/README.md).

## Why it differs

- **Fit and proof-lift a glass vacuum filter holder** captures two annular glass flanges and verifies their joint by lifting the assembled flask from the upper funnel. Here the left hand raises and prevents an unfastened folded robot from toppling while the other aims portable jaws below a fixed table edge, transferring support to clamps on two native base feet. [PR #2697](https://github.com/robocurve/benchmark-tasks/pull/2697)

- **Clamp a printed part in a vise at a marked height** and **Hold a block in a vise while the jaw closes** also require simultaneous holding and fastening. Their fixed vise guides the moving jaw toward a workpiece held at a height. Here the support hand resists the cantilever moment while the other hand transports and aims the entire sprung mechanism around the edge. It must establish a load path from the pad on top of the native base to the pad under the table. [PR #282](https://github.com/robocurve/benchmark-tasks/pull/282), [PR #669](https://github.com/robocurve/benchmark-tasks/pull/669)
- **Clamp two boards edge to edge by pumping a one-hand bar clamp's trigger** and **Clamp up a glued panel with the clamps alternating below and above** draw supported boards together in-plane. Here the pads close across a base foot and the table thickness while the other hand keeps an overhanging center of mass from rotating off the edge. [PR #771](https://github.com/robocurve/benchmark-tasks/pull/771), [PR #1450](https://github.com/robocurve/benchmark-tasks/pull/1450)
- **Clamp a part to the mill table with a strap clamp, its heel on the step-block tread that levels it** sets a strap on a stud with a supported heel. Here a free clamp approaches horizontally and finds the underside of the edge while the base is hand-supported. [PR #1313](https://github.com/robocurve/benchmark-tasks/pull/1313)
- **Hold an SO-101 arm's 3D-printed shoulder bracket up with one hand and push a servo motor into its pocket with the other** centers and inserts a servo into a held part. Here the full arm is assembled. The support hand brings its base into position and transfers overturning load to two external clamps. [PR #836](https://github.com/robocurve/benchmark-tasks/pull/836)
- **Mount a phone in a spring-clamp car holder, holding its sprung arms open while the phone goes in** holds open an installed mechanism and inserts the payload. Here the payload must be held at the table edge while the spring tools themselves are carried into place. **Jump-start a flat battery with sprung booster clamps in the safe order** also applies portable spring clamps, to rigid terminals in an electrical sequence. Here each clamp closes across two separate objects to fasten a hand-supported load to the bench. [PR #1815](https://github.com/robocurve/benchmark-tasks/pull/1815), [PR #1879](https://github.com/robocurve/benchmark-tasks/pull/1879)
- **Unload a cantilevered plank without tipping it** controls a shifting center of mass while removing loads. Here the arm's mass distribution stays fixed and the grippers must add a load-bearing connection around the edge before they can let go. [PR #36](https://github.com/robocurve/benchmark-tasks/pull/36)

## Setup and reset

Use the calibrated bimanual YAM station. Dimensions below are in its world frame: +X away from the robot bases, +Y toward the left arm, +Z upward. The standard work surface is at Z = 747.5 mm. Fix the small table and clamp staging rail to it.

| Item | Specification |
| --- | --- |
| Small table | Plywood top 255 × 322 × 16 mm, X = 363–618 mm, Y = −168 to +154 mm, top Z = 929 mm; four 36 mm square timber legs, clear height 165.5 mm. |
| SO-101 follower | Full-size printed links, native base and six STS3215 servos; total simulated mass 0.632 kg. Fix shoulder lift at −1.5525 rad, elbow flex at 1.5 rad, wrist flex at 0.8 rad, gripper at −0.17 rad; remaining joints at zero. Disconnect power and retain this folded pose. |
| Initial arm | Starting root position (387.988, −0.0747, 946.359) mm; quaternion XYZW (−0.0006315, 0.1369928, 0.0011553, 0.9905713). The folded arm rests approximately 15.75° forward, supported by two base contacts and its forward end on the tabletop. |
| Mounting target | Native base datum (383, 0, 929) mm with zero roll, pitch and yaw. Clamp the two side feet directly at approximately Y = ±44 mm. The foot tops are 17.5 mm above the base datum. |
| Two spring clamps | Original nylon-lever geometry with elastomer pads (17 × 22 × 6 mm), approximately 50 mm opening. Mechanism reference: BESSEY Clippix XCL2. Handles face the YAM bases. |
| Staging rail | Plywood 114 × 181 × 25 mm, near edge X = 367 mm, Y = −386 to −205 mm, top Z = 938 mm. Clamp hinge origins start at X = 351 mm, Y = −244 and −341 mm, Z = 925.5 mm; hinge opening 0.5 rad. |

Reset the folded arm to its three-contact forward rest and park both clamps on the rail. Use the screenshot to reproduce the orientation. For physical trials, verify the table and rail are secure, keep the SO-101 unpowered, and use a slack catch tether or catch mat below its sweep. It must not support the arm or interfere with the clamps during scoring. [Clamp mechanism reference](https://bessey.de/de-de/bessey-tool/produkte/spannwerkzeuge/federzwingen-und-ratschenzwingen/federzwinge-clippix-lang-und-spitz-xcl).

## Scoring

Score 0.5 per distinct base foot successfully clamped, subject to the following conditions. Follow the left gripper in side or wrist video: both fingers must remain in contact on opposing sides of the folded arm continuously from each clamp's open straddle until its loaded closure. The base must remain upright and aligned during this interval. This is a visible grasp/contact condition; no calibrated upward-force threshold is required. The right gripper must manipulate that same clamp. A later regrasp cannot repair an unsupported installation.

At closure, the upper working pad must bear on the corresponding native base foot, and the lower working pad must bear under the table. Check contact with side/wrist video and, where obscured, a thin paper strip or feeler gauge. In base coordinates, qualifying upper contacts are X = −22.5 to +65.5 mm, Z = 17.5 ±3 mm, and Y = +20 to +56.5 mm for the left foot or −56.5 to −20 mm for the right foot. Qualifying underside contacts lie 0–65 mm in from the table edge, within 25 mm along Y of the corresponding foot center, at Z = 913 ±3 mm. The simulator checks physical contact on the actual working pad faces, not just a jaw's bounding box. Contact magnitudes are diagnostic; no calibrated minimum force is part of the task.

After both hands have no contact and at least 25 mm clearance, maintain the base for two continuous seconds within 6 mm X, 8 mm Y and 3 mm height of its target, tilt ≤3°, yaw ≤5°, and translation drift ≤3 mm. If a clamp loses its seat or moves over 4 mm relative to either surface, it needs a new supported opening and closure. One qualifying retained foot earns 0.5; two earn 1.0; otherwise score 0.

## Simulation and execution

The SO-101 uses full-size CAD meshes, URDF mass/inertia and compound convex collision geometry. Its folded joints are fixed within one free rigid assembly for this unpowered mounting task. The native base has a 48-piece CAD-derived convex decomposition that retains its concave mounting recess. Clamps have passive preloaded revolute springs and rigid pad contacts. Spring stiffness, friction and rigid pad compliance are uncalibrated approximations. Both demo and Dream use the same assets, physics and native YAM position actions, drive gains and force limits.

Dependencies: Isaac Sim 6.1, Isaac Lab 17.0.2, Isaac Lab-Arena 0.3.0, the calibrated [YAM Arena adapter](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1), Inspect Robots 0.60.0 and its Dream adapter 0.1.0, NumPy, SciPy, Pillow and imageio-ffmpeg. Dream uses its standard credential configuration. Run from this task directory in the configured environment, with DISPLAY and WAYLAND_DISPLAY unset on headless hosts. Each output directory must be new.

```sh
python test_score.py -v
python demo.py demo runs/demo
python dream_eval.py check runs/check
python dream_eval.py dream runs/dream
python render.py runs/demo runs/screenshot.png
python render.py runs/dream
```

Seed 0; fixed layout. Limit 2,400 actions at 20 Hz (120 simulated seconds), including retention. PhysX runs at 240 Hz. The scripted policy reads robot and object poses for IK feedback. Dream-1 receives only the three calibrated robot RGB views (rendered 640 × 360, sent as 320 × 176 JPEGs) and the 14 robot joint/gripper readings. Per-action scores and actual position targets are recorded. Measured 30 Hz trajectories can be replayed to render synchronized videos without another model run.

## Asset licenses

- `assets/so101.usdc`: follower printed parts and servo meshes from [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101), Apache-2.0; posed and collision-decomposed. See [license](assets/LICENSE-SO-ARM100.txt).
- Clamp and table/rail geometry are original procedural assets. BESSEY is the mechanism reference.
- The runtime YAM station supplies I2RT YAM meshes and Dream color variants under MIT, with attribution in `yam_arena/assets/yam/LICENSE.i2rt`.

## Results

The scripted demo scored **1.0** in 944 actions (47.2 s), ending with both native base feet clamped and both hands clear for 3.1 s. Dream-1 scored **0.0** in 2,400 actions (120 s). Execution results are recorded in each run's `result.json` and per-action `scorer.jsonl`.
