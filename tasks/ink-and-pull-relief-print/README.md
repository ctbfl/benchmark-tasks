# Ink a relief block with a roller and pull a clear print without smearing

Make a small lighthouse print as in a school art lesson or card-making project. Roll a thin film of ink onto the raised block, press a clean sheet against it, and lift the sheet from one corner. Load, even film thickness, and motion at separation determine whether the same block gives a usable proof for a card or an edition. [Speedball classroom lesson](https://www.speedballart.com/wp-content/uploads/2019/03/BPLessonPlan_Elem_FINAL.pdf), [Speedball demonstration](https://www.speedballart.com/wp-content/uploads/2019/10/Block-Printing-MT-Lecture-Demo-Manual.pdf)

## Why it differs

- **Finish painting a panel a coworker started, without repainting their work** uses a preloaded roller to fill bare parts of a panel and grades the coat on that panel. This task requires charging a dry brayer on a slab, coating raised details, and releasing a separate sheet with an intact wet print. [PR #1530](https://github.com/robocurve/benchmark-tasks/pull/1530)
- **Apply a water-transfer temporary tattoo** wets a preprinted film and peels off its backing on a curved forearm. The robot here must create the image from wet pigment on a carved block. [PR #2541](https://github.com/robocurve/benchmark-tasks/pull/2541)
- **Set a date stamp to the work order's date and stamp it** inks a preset-size die on felt and makes one impression after setting the wheels. Relief printing charges a free-running brayer, levels ink around its circumference, maps it onto raised ridges across a stationary block, and separates the moving paper from those ridges. [PR #877](https://github.com/robocurve/benchmark-tasks/pull/877)
- **Stamp Seal** places a handheld seal at a selected mat. The upstream success check uses seal position and gripper release. Here the quality target is the actual ink footprint, which can degrade during removal even after an accurate placement. [RoboTwin 2.0 task definition](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/stamp_seal.py)
- **Register a PCB stencil and print solder paste** registers an apertured metal foil and uses a squeegee to fill openings onto PCB pads. Relief printing needs controlled rolling contact that inks protrusions, then transfers that film by direct loaded contact with flexible paper. [PR #2563](https://github.com/robocurve/benchmark-tasks/pull/2563)
- **Press a stencil flat with one hand while the other dabs paint through its three holes** holds a mask against the output and applies pigment through openings with a dauber. Here the block holds the ink, the recesses define white space, and the paper must leave the inked surface without smearing. [PR #758](https://github.com/robocurve/benchmark-tasks/pull/758)
- **Set a word in wood type in a chase so that it prints the right way round** assembles the printing forme from separate mirrored letters. Here the design is already carved. The robot makes, transfers and separates a wet film. [PR #1529](https://github.com/robocurve/benchmark-tasks/pull/1529)

## Physical setup

Use a bench with the standard dual-arm YAM mounts, 610 mm apart, and the three calibrated cameras. The origin is the midpoint between bases at table height. X points forward, Y left and Z up. Coordinates below are in millimetres. Use water-washable dark relief ink.

| Piece | Physical build and place at reset |
| --- | --- |
| Brayer | 3 in / 76 mm soft rubber roller, nominal Ø38 mm, free-running axle and 95 g frame. Use a 47 × 26 × 24 mm handle with accessible parallel faces. At rest the roller axis lies along Y at (227, 211, 26), with the grip centre 71 mm to the rear and 43 mm above the axle. |
| Inking slab | 302 × 102 × 7 mm acrylic plate centred at (300, 211). For physical resets, dose 0.10 mL of washable relief ink with a graduated syringe and spread it across a nominal 24 × 76 mm loading band centred at X = 264 mm. |
| Relief block | 96 × 66 mm plywood-backed linoleum block with image face at Z = 26 mm and cuts 4 mm deep. Centre (318, −35). The union of the ten rectangles in `geometry.py` defines the original lighthouse image. All rectangle dimensions are in millimetres from this centre. Secure the block and slab to the bench with edge tape, as in a printmaking bench setup. |
| Paper | 124 × 102 mm blank uncoated printmaking paper, nominally 120 g/m², with the long edge along X. Add a folded 27 × 12 × 22 mm cardstock pull tab near one corner. Locate its centre 46 mm rear of the sheet centre and 39 mm to the right, keeping the underside clear for printing. |
| Paper rest | 132 × 104 mm flat board with top 39.2 mm above the table. Its centre is at (318, −217). Rest the loose sheet at (318, −227), leaving the tab side slightly overhanging. |

A 3 in brayer is a standard commercially available size. The block dimensions allow a single centred roller stroke to span all protrusions. The bench layout uses a short print sheet that can be lifted by a parallel gripper via its reinforced corner. [Speedball relief supplies catalogue](https://www.speedballart.com/wp-content/uploads/2022/07/Speedball-BlockPrint-2022-Update.pdf)

Reset with a clean brayer, relief, slab and paper rest, a fresh sheet and a fresh measured ink dose. Start both grippers open at the station ready pose and use fixed seed 0. The two ink-contacting tools and paper are physically separate at reset. The paper and brayer are loose and held by ordinary grasp contact.

## Scoring the proof

**Contact eligibility.** The raised paper tab is attached to the back. Show it and the opposite print face at reset, then follow the sheet's orientation through the trial. Film the entire attempt from above and from a side angle that keeps the slab top, roller sleeve and sheet in view together during close approaches. The top of the slab is loaded from dosing. The rubber sleeve is loaded from its first ink pickup. Each stays in scope for the rest of the trial. Use the side view to check whether the print face actually touches either loaded surface, including brief touches during pickup, transport, pressing or withdrawal. Keep the view clear of the wrist and fixture during any approach where contact could occur.

One such contact ends eligibility for all print tiers, even if the final image looks clean. Back-face contact and contact with dry material leave eligibility intact. If an approach where contact could occur is hidden or the face cannot be identified in the recording, withhold print credit. Relief inking can still earn its own credit. In simulation the same rule latches the first measured contact with the loaded surface on the underside of a paper panel.

**Photo scoring.** Take a perpendicular photograph with a millimetre scale or scan the flattened sheet face up. Mark all visible ink on the 2 mm grid at 37% normalized darkening, using matching blank paper and a saturated ink sample as references. Calibrate physical scale from the ruler. Prepare the relief template once in its face-up proof convention, which reverses the relief as seen from above.

Register the whole visible photograph with one in-plane rotation and translation at the calibrated scale. The scorer searches the full circle in 1° steps and translations in 1 mm steps. Select the placement with the highest covered fraction of the template, using the least excess ink to break ties. The template has fixed handedness and dimensions. Use that one placement for both measures.

- **Coverage** is the fraction of registered template cells with visible ink.
- **Excess ink** is the count of all marked cells outside the registered template and its one-cell edge allowance, divided by template area. This counts drag marks, offset impressions and other extra ink. The unrotated template contains 447 cells, or 1,788 mm².
- **Separation** requires the whole sheet to stay at least 8 mm above the relief or outside its footprint with a 2 mm lateral allowance for 0.4 s. Check the last in-trial condition in the video.

For eligible prints, award 1.0 for at least 90% coverage and at most 4% excess with separation, 0.65 for at least 65% coverage and at most 18% excess with separation, or 0.4 for at least 45% coverage and at most 18% excess. If the sheet is still on the block at the time limit, record that state before carefully lifting it for inspection. Award 0.2 if at least 60% of the relief reached the inking threshold and the paper earns less. Use a top photograph with a 55% normalized darkening threshold for this relief tier. Other outcomes score 0.

## Simulation and reproduction

The same assets and scorer serve the demo and Dream-1. CPU PhysX at 240 Hz uses a passive revolute roller. Three low-mass sheet panels with passive bending springs approximate flexible paper. Their effective collision thickness is 1.6 mm. The palette and relief have actual colliders, including the carved depth. Native YAM joint, gripper and effort limits apply. The scorer integrates finite pigment exchanged at loaded contacts. Rotation and translational travel jointly govern pickup. Each printing sample maps loaded relief to the actual sheet coordinates. Continued contact while sliding deposits additional ink at the new coordinates. The renderer reads those changing film arrays. The ink law is qualitative and its coefficients have not been calibrated against measured ink rheology. The physical ink dose and paper weight above are reproducible starting specifications.

Use Isaac Sim 6.1, Isaac Lab 17.0.2, Isaac Lab-Arena 0.3.0 and the calibrated [YAM Arena station](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1). Dream uses Inspect Robots 0.60.0, its Dream adapter 0.1.0 and the Dream SDK. It receives three 640 × 360 camera renders resampled to 320 × 176 by its adapter, plus the 14 measured joint and gripper values. The scripted demo reads poses and roller pressure for IK and load feedback. Physics runs only when actions are applied. Both use the same 14 absolute joint/gripper targets at 20 Hz and a 2,400-step / 120 s limit.

From this task directory in the activated simulation environment with no display server variables set, use a fresh output path for each run. Set a run identifier explicitly when reproducing locally.

```bash
PRODUCTION_GPU_RUN_ID=local-demo python run.py demo ./reproduction-demo
PRODUCTION_GPU_RUN_ID=local-dream python run.py dream ./reproduction-dream
python run.py screenshot ./screenshot.png
python run.py render ./reproduction-demo
python test_scoring.py
PRODUCTION_GPU_RUN_ID=local-rescore python rescore.py ./recorded-run ./rescored-run
```

Each trial saves `result.json`, per-control measurements and a trajectory containing the executed actions, physical contacts and measured states. The scorer can recompute the result from a saved recording. USD sources can be rebuilt with `python assets.py` and OpenUSD.

## Evaluation

The recorded YAM demo scored **1.0** in 2,254 control steps (112.7 s), with 420 of 447 target cells covered (94.0%) and no excess ink. The recorded Dream-1 trial scored **0.0** after 2,400 control steps (120 s), leaving the sheet blank. Both recordings satisfy the print-face contact eligibility rule. The supplied scorer reproduces the results from the recorded poses and contacts.

## Asset sources and licenses

The lighthouse artwork and the task's procedural block, brayer, slab, paper and rest are original. The runtime YAM meshes and black/white covers are by I2RT Robotics under MIT; the adapter supplies the notice in `yam_arena/assets/yam/LICENSE.i2rt`. The calibrated station's table, floor and camera stand are procedural assets supplied by the YAM Arena dependency. No third-party artwork or scanned task assets are bundled.
