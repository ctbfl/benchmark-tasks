# Secure an SO-101 robot arm to a small table edge with plastic clamps

Mount a complete SO-101 follower to a small work table before commissioning it. One YAM gripper supports the arm at the mounting marks while the other carries each spring clamp, opens its jaws, threads them above and below the edge, and lets them close. After withdrawal, the two clamps and the table carry the assembly. The SO-101 bill of materials includes table clamps for workstation assembly. [SO-101 bill of materials](https://github.com/TheRobotStudio/SO-ARM100/blob/a758567c3978dfeefe282ede0500085a48fe8f78/README.md)

## Why it differs

- **Clamp a printed part in a vise at a marked height** and **Hold a block in a vise while the jaw closes** also require simultaneous holding and fastening. Their fixed vise guides the moving jaw toward a workpiece held at a height. Here the support hand resists the cantilever moment while the other hand transports and aims the entire sprung mechanism around the edge. It must establish a load path from the pad on top of the adapter to the pad under the table. [PR #282](https://github.com/robocurve/benchmark-tasks/pull/282), [PR #669](https://github.com/robocurve/benchmark-tasks/pull/669)
- **Clamp two boards edge to edge by pumping a one-hand bar clamp's trigger** and **Clamp up a glued panel with the clamps alternating below and above** draw supported boards together in-plane. Here the pads close across the table thickness while the other hand keeps an overhanging center of mass from rotating off the edge. [PR #771](https://github.com/robocurve/benchmark-tasks/pull/771), [PR #1450](https://github.com/robocurve/benchmark-tasks/pull/1450)
- **Clamp a part to the mill table with a strap clamp, its heel on the step-block tread that levels it** sets a strap on a stud with a supported heel. Here a free clamp approaches horizontally and finds the underside of the edge while the mounting plate is hand-supported. [PR #1313](https://github.com/robocurve/benchmark-tasks/pull/1313)
- **Hold an SO-101 arm's 3D-printed shoulder bracket up with one hand and push a servo motor into its pocket with the other** centers and inserts a servo into a held part. Here the full arm is assembled. The support hand brings its base into position and transfers overturning load to two external clamps. [PR #836](https://github.com/robocurve/benchmark-tasks/pull/836)
- **Mount a phone in a spring-clamp car holder, holding its sprung arms open while the phone goes in** holds open an installed mechanism and inserts the payload. Here the payload must be held at the table edge while the spring tools themselves are carried into place. **Jump-start a flat battery with sprung booster clamps in the safe order** also applies portable spring clamps, to rigid terminals in an electrical sequence. Here each clamp closes across two separate objects to fasten a hand-supported load to the bench. [PR #1815](https://github.com/robocurve/benchmark-tasks/pull/1815), [PR #1879](https://github.com/robocurve/benchmark-tasks/pull/1879)
- **Unload a cantilevered plank without tipping it** controls a shifting center of mass while removing loads. Here the arm's mass distribution stays fixed and the grippers must add a load-bearing connection around the edge before they can let go. [PR #36](https://github.com/robocurve/benchmark-tasks/pull/36)

## Setup and reset

Use the calibrated bimanual YAM station. Measurements below use +X away from the YAM bases, +Y toward the left arm and +Z upward. The origin is midway between the YAM bases in plan view, with the standard YAM work surface at Z = 747.5 mm. The small table and rack are bolted down. Their legs sit clear of the near edges.

| Item | Dimensions and placement |
| --- | --- |
| Small table | 255 × 322 × 16 mm plywood top, X = 363–618 mm, Y = −168 to +154 mm, top Z = 929 mm. Four 36 mm square timber legs with 165.5 mm clear height. |
| SO-101 and adapter | Full-size follower with six STS3215 servos and printed links, bolted to a 172 × 284 × 9 mm birch plywood blank, cut back to X = −6 mm except for the left support tongue (Y = +58 to +129 mm). Relative to the mounting datum, board edges are X = −61 to +111 mm and Y = −155 to +129 mm. The arm's base-link datum is 61 mm inboard and 32 mm left of that datum, yawed 180° so the arm extends toward the edge. Hold its joint angles at −0.76 rad shoulder lift, +1.0 rad elbow flex and zero for the other URDF joints. |
| Tape | Add a start witness line at X = 515 mm, and two mounting witness marks just outside the board sides at X = 474 mm. Mark 26 mm squares on the board at X = 389 mm in the final pose, at Y = −53 and −120 mm. |
| Clamps and staging rail | Two one-hand nylon spring clamps with elastomer pads, able to open 40 mm or more and bite 25 mm inboard. Use a 50 mm clamp such as the BESSEY Clippix XCL2 as the mechanism reference. Park them at Y = −244 and −341 mm on a 114 × 181 × 25 mm plywood rail, near edge X = 367 mm, top Z = 938 mm. The handles face the YAM bases. |

Set the board flat with its far edge against the start line and its two long edges parallel to the table. The adapter plus arm weighs 0.853 kg in simulation. At the final marks the calculated center of mass is 13 mm outboard, with a nominal overturning moment of 0.106 N·m. For physical setup, verify the balance point on a straight timber edge with a hand ready to catch the arm. Use the same pose and start offset and confirm the unloaded table is stable. Secure the SO-101 joints at the listed angles for the mounting trial and keep the power disconnected. Use a slack safety tether or a catch mat below the sweep in every physical trial. Keep it clear of the mount and grippers during scoring. [BESSEY spring-clamp specifications](https://bessey.de/de-de/bessey-tool/produkte/spannwerkzeuge/federzwingen-und-ratschenzwingen/federzwinge-clippix-lang-und-spitz-xcl)

## Scoring

Follow each physical clamp from its open straddle of a marked square to loaded closure. In side video, check that the left gripper continuously grasps the support tongue with both fingers and bears upward on it throughout that interval. If the upward load is ambiguous, use thin calibrated pressure strips on the gripper faces and check for more than 0.4 N of net upward load. Use the right wrist or observer view to follow the other gripper acting on that same clamp. After the jaws close, verify that the upper pad bears within the square on the plate and the lower pad bears under the tabletop, 11–45 mm in from the edge. The overhead view follows the square as the plate moves. Use a thin feeler gauge to check the loaded portion of a pad when its edge is partly hidden. A regrasp after closure cannot earn credit. If a clamp is replaced, loses its seat or moves over 4 mm relative to either surface, observe a new supported opening and closure before counting it.

The simulator checks each 20 Hz sample for upward load and opposing left finger contact from the open straddle through loaded closure. It measures right finger contact separately for each clamp. At closure, the force carried by actual upper-pad contacts inside the 26 mm square is computed in the measured plate frame. Actual lower-pad contacts are checked in the tabletop frame.

Use a square, ruler and side video to verify that for two continuous seconds after both hands withdraw the board remains within 6 mm along X and 8 mm along Y of its marks, within 3 mm of the tabletop, within 3° of level and 5° in yaw, and drifts at most 3 mm. Score 0.5 for one qualifying clamp and 1.0 for two on distinct squares. All other outcomes score 0.

In the scripted run, clamp A straddled the edge at step 521 and seated at 564. Clamp B straddled at 807 and seated at 865. The least upward support in those windows was 5.69 N and 5.64 N. Both clamps retained load on their squares for 3.2 seconds after the hands withdrew. In the original physical release controls, the plate tilted 64.78° with both clamps still parked and 38.52° with a clamp held open across the edge. Both controls scored zero. The scorer's CPU regression checks cover late regrasp after unsupported installation, a loaded upper pad outside its moving square, an interrupted support grasp and loss of credit after relocation or replacement.

## Simulation and execution

The SO-101 uses full-size visual meshes and a compound convex collision shell. The assembled inertia comes from the URDF masses and inertias plus the plywood adapter. The arm pose is rigid for this unpowered mounting task. The clamps use passive, preloaded revolute springs with rigid rubber-pad contact geometry. Clamp stiffness, friction and the fixed joint pose are uncalibrated simulation approximations. The native YAM position action interface and actuator limits are used in both trials.

Dependencies are Isaac Sim 6.1, Isaac Lab 3.0, Isaac Lab-Arena 0.3.0, the calibrated [YAM Arena adapter](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1), Inspect Robots 0.60 and the Inspect Robots Dream adapter. Rendering also uses Pillow, NumPy and imageio-ffmpeg. The Dream client reads its standard credentials. Run the following from this task directory in that environment. On a headless host, run with DISPLAY and WAYLAND_DISPLAY unset. Use a fresh output directory name for each run.

```sh
python test_score.py -v
python demo.py support runs/support
python demo.py unsupported runs/unsupported
python demo.py unclamped runs/unclamped
python demo.py demo runs/demo
python dream_eval.py check runs/check
python dream_eval.py dream runs/dream
python render.py runs/demo runs/screenshot.png
python render.py runs/dream
```

The seed is 0 and the layout is fixed. The shared limit is 2,400 actions at 20 Hz (120 simulated seconds), including the stability interval. The scripted policy reads measured robot and object poses for IK feedback. Dream-1 receives only three calibrated RGB feeds and the 14 joint and gripper readings. PhysX advances at 240 Hz. The scorer writes per-action measurements and the final result. Replay uses measured poses sampled every eight physics ticks to produce synchronized 30 FPS video.

## Asset licenses

- `assets/so101.usdc` derives its follower printed parts and servo meshes from [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101), Apache-2.0. The meshes were posed, simplified and combined with original adapter geometry. See [license](assets/LICENSE-SO-ARM100.txt).
- The clamp, table, adapter and fixture marks are original procedural geometry. BESSEY is a reference for the mechanism.
- The host YAM station uses I2RT YAM meshes and their Dream-format color variants under MIT. The runtime adapter supplies their license in `yam_arena/assets/yam/LICENSE.i2rt`.

## Results

The scripted demo scored **1.0** after 979 actions (48.95 s). Dream-1 scored **0.0** after 2400 actions (120.00 s) in its original execution. An offline check of all archived force samples found zero upper-jaw reaction for both clamps throughout the model run, which also excludes credit under the revised rules.
