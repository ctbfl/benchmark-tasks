# Slice a cinnamon-roll log by drawing floss across it without squashing the spiral

A loose length of unflavored floss starts on the board in front of a small cinnamon dough log. Feed the floss underneath at the mark, cross the ends overhead and pull them apart to cinch the loop. Aim for two round 45 mm portions. The underfeed and opposite-direction pull follow the baking school recipe. [King Arthur Baking, Brunch Bakes workbook, p. 9](https://shop.kingarthurbaking.com/content/class-pdfs/Brunch-Bakes-Workbook-King-Arthur-Baking.pdf)

## Why it differs

- **Split a loaf cake into two layers with a wire** holds a straight line taut and level as it translates lengthwise through a baked loaf. This task needs slack management for the underfeed, a crossover above a rounded raw log, and tension on a contracting circumferential loop. Success measures transverse separation and cross-section shape. [PR #2576](https://github.com/robocurve/benchmark-tasks/pull/2576)
- **Divide a 0.9 kg dough log into four pieces with bench scrapers, chopping at the marks** uses two rigid blades and scores the locations and completeness of four portions. The floss here must stay below the log during insertion and transfer tension around the circumference after the crossing. [PR #2198](https://github.com/robocurve/benchmark-tasks/pull/2198)
- **Floss two model teeth past their tight contact without snapping into the gum, wiping both sides** threads a tight dental gap and wipes the adjacent surfaces. The ends here cross to form a shrinking loop that must sever the enclosed food and preserve the spiral. [PR #1341](https://github.com/robocurve/benchmark-tasks/pull/1341)

## Setup and reset

On the calibrated YAM station, measure forward from the line through the two arm bases and laterally from their midpoint. Center a 300 × 426 × 8 mm silicone-faced board at 293 mm forward. Place a 90 × 62 mm cinnamon dough log lengthwise along the forward axis, centered on the board. Lightly score its midpoint. On each side of the log, draw a center tick and two shorter ticks 10 mm to either side, measured along the log axis. Tie a 256 mm exposed length of unflavored floss to the bottom of two 22 × 26 × 28 mm wood grip blocks. Set the blocks 195 mm forward and 116 mm to either side of center, with the cord slack toward the arms.

For a real trial, use the King Arthur dough and filling from one batch, rolled at room temperature after the first rise. Portion 90 mm test logs from the assembled log and target a diameter of 62 ± 4 mm. Keep unused logs covered. Use a fresh test log each trial and restore the same marks, floss and grip positions. Clean the board between trials and wipe the gripper pads after dough contact. The grip blocks provide parallel-jaw contact surfaces for the fine floss ends.

## Scoring

Measure the height and width of both ends before the run. Hold a ruler beside the dough without touching it, or take a straight-on photo with the ruler in the plane of the end. After recording final placement, measure each cut end the same way with its board-facing side down. If the log is still joined, measure its outer ends in place. Each height and width must be 89–111% of the corresponding starting measurement.

Confirm from video that the floss makes the cut after underfeed and crossover. If the rolls separate, check each roll's shortest and longest thickness along the original log axis with a ruler. Both must be 35–55 mm. Score zero for a failed dimension check, dough left off the board, or a cut beyond the guide band. A visible open slit beyond the band after the floss moves clear also earns zero. A temporary indent that closes after unloading permits continued scoring. Leave any joined dough in place for scoring.

- **0.35** for a loop fed under the mark and crossed above it, visibly taut against both sides inside the guide ticks for at least 0.25 s. Judge this from video with the dough in place.
- **0.75** for a completed in-band cut. Both portions lift independently.
- **1.0** if, in addition, both portions rest within the board, their midpoint is within 27 mm of the start, both grips are on the board and both grippers are open. Each gripper's fingertip center must stay at least 40 mm from the center of each block for half a second.

## Simulation and reproduction

The tension-only cord has solved node and link contacts with reaction forces on native YAM-grasped handles. A small underside clearance in its analytic collider proxies soft dough yielding during underfeed. An enclosed taut loop advances the loaded cap only while all bearing contacts lie within ±10 mm and the cap receives upward load. It parts at a 3 mm ligament. Independently of that gate, loaded contacts outside the band accumulate plastic groove depth in 4 mm axial intervals. A 2 mm groove earns zero even if the loop later completes the marked cut. This score-only proxy uses force above a local yield threshold. The simulator uses rigid bulk portions and approximate visible flattening. The mat adds contact-dependent drag, and the render thickens the cord for visibility.

Both entries use seed 0, 240 Hz PhysX, 20 Hz native YAM joint and grip targets, and the same starting state. The 1,800-action limit is 90 seconds of simulation time. The demo uses Cartesian waypoints solved with YAM IK and reads each portion's position to align the pull and show that the slices move independently after release. Dream-1 receives only calibrated left wrist, top and right wrist camera images at 320 × 176 and the standard joint and gripper state. Each trial writes raw actions, physics, scorer readings and synchronized 30 FPS media into its output directory.

Use the calibrated YAM Arena adapter at commit bd62771e60320433893bd1d2df0fc79b5ddcd6d1 with Isaac Sim 6.1, Isaac Lab 17.0.2, Inspect Robots 0.60.0, Inspect Robots Dream 0.1.0 and dreamlabs-client 1.0.14. Python dependencies also include NumPy, Numba, Pillow and imageio-ffmpeg. Use Coverage 7.13.1 if Coverage is installed alongside the Numba version in this environment. Dream credentials use the SDK default ~/.dreamlabs/credentials.

```bash
env -u DISPLAY -u WAYLAND_DISPLAY PRODUCTION_GPU_RUN_ID=local-demo python run.py demo runs/demo
env -u DISPLAY -u WAYLAND_DISPLAY PRODUCTION_GPU_RUN_ID=local-dream python run.py dream runs/dream
```

The board, marks, grip blocks, dough surfaces and cinnamon spiral are original procedural assets.

- I2RT Robotics' YAM meshes, including the split arm meshes used by the workstation, are licensed under MIT. [Mesh license](https://github.com/kairosiann/yam-isaac-lab-arena/blob/bd62771e60320433893bd1d2df0fc79b5ddcd6d1/yam_arena/assets/yam/LICENSE.i2rt)
- The YAM Arena adapter is an external runtime dependency. The upstream README at the pinned revision declares no adapter license, and this task package contains no adapter source or mesh files. [Adapter license statement](https://github.com/kairosiann/yam-isaac-lab-arena/blob/bd62771e60320433893bd1d2df0fc79b5ddcd6d1/README.md#license)

## Recorded trials

The demo finished at 1.0 after 519 actions (25.95 s), with a maximum advancing cut offset of 9.06 mm. Dream-1 scored 0.0 in 1,800 actions without underfeeding the floss.

The [recorded score verification](recorded-score-verification.json) ties the scorer replay to each delivered video by its SHA-256 hash and frame ticks. The demo's peak off-band groove depth was 0.00 mm against the 2.00 mm limit. `intact` was true for all 11 qualifying control steps, actions 509–519. The tenth qualifying step was action 518 at physics tick 6,216, when the credited score reached 1.0. The replay recomputes the cord and grading from recorded poses and contact forces and uses the recorded placement and handoff checks. It reproduces the score on every action of both trials.
