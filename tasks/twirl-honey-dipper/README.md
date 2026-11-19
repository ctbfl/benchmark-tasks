# Carry honey on a rotating dipper, stop to drizzle onto toast, then return the dipper to its rest

Pick up the prepared dipper by its dry handle. Carry it across the gap while rolling the shaft around its long axis. Over the toast, stop rolling and sweep slowly to spread the drizzle. Return the dipper to its original pot rest, release the handle, and withdraw both grippers. The operation lets someone serve honey at breakfast without dripping it across the table. [Honey supplier's technique guide](https://alsafahoney.net/guides/honey-dipper)

## Why it differs

- The syrup task pours 10 mL by tipping a bottle while using the other hand to level the cup. It scores the amount in the cup and the final placement of both vessels. Here the tilted vessel is replaced by a rolling, exposed liquid film on the dipper. The operator must keep that film on the moving tool, then let it drain by stopping the tool's axial rotation. [Pour a 10 mL dose of syrup into its dosing cup, held level, and put both down](https://github.com/robocurve/benchmark-tasks/pull/2254)
- The toothpaste task meters paste by compressing a tube over a toothbrush. This task starts with honey already on the outside of the dipper. It measures suppression of passive dripping during transport and release over the food by changing wrist roll. [Put a pea-sized dab of toothpaste on a child's toothbrush, squeezing 0.15 to 0.35 g from a kids' tube](https://github.com/robocurve/benchmark-tasks/pull/2381)
- The soup task requires matching the two arms as they lift, tip and lower a heavy pot. Here even a stable horizontal grasp can drip during transport. The challenge is to use axial roll to keep an exposed coating on a light hand tool. [Pour a full 3 kg pot of soup into a tureen without spilling and set the pot back on the hob](https://github.com/robocurve/benchmark-tasks/pull/463)
- The blind task turns a hanging wand through successive pinch-and-regrasp strokes to set slat angle. Here the tool moves between two locations while rolling, and stopping the same motion changes the flow of the payload. [Open a mini blind's slats by twirling its hanging tilt wand](https://github.com/robocurve/benchmark-tasks/pull/2527)
- The BEHAVIOR-1K activity shares the breakfast outcome. Its goal predicates cover cooked toast on a plate with syrup and cinnamon. This task measures the manipulation of a loaded grooved tool during the transfer and drizzle. [make cinnamon toast](https://github.com/StanfordVL/BEHAVIOR-1K/blob/cc8f4ce0c89fa5a2f4be510b187989aa0dc69761/bddl3/bddl/activity_definitions/make_cinnamon_toast/problem0.bddl)

## Setup and reset

Use the dual-arm YAM station with parallel jaws. The work surface is at z = 747.5 mm. In station coordinates, place the 106 mm diameter, 140 mm tall teal ceramic footed honey dish at (490, −265) mm and the 154 mm diameter ceramic plate at (490, −20) mm. The 117 × 94 mm slice sits on the plate, with its rounded crown toward +x and its upper surface 21 mm above the table. The station keeps the pot, plate and slice in place.

The 155 mm maple dipper has a 15 mm dry handle, a 32 mm maximum head diameter, five raised collars and four annular grooves. Lay it across the ceramic rest and shallow well along +x with the butt toward the robot bases. Its root is at (366, −265) mm and the shaft is about 141 mm above the table.

### Wet reset for a physical trial

1. Use **Rowse Runny Honey (Clear Honey, jar)** from one lot for the session. Use clear liquid honey without visible crystals. Equilibrate the closed jar, dipper and dish for at least two hours in a 22 °C room. Verify **22.0 ± 0.5 °C** in a sample from the jar before filling the dish. [Manufacturer's Runny Honey specification](https://rowsehoney.co.uk/products/clear-honey)
2. Clean and dry the dipper, plate and spill catchment. Use a fresh slice of plain white sandwich bread from the same loaf, 15 ± 2 mm thick, toasted to an even light gold. Cool on a rack for at least 5 min and until it is at room temperature (22 ± 1 °C), then tare within 2 min of starting the trial. Fill the dish to its **8.0 ± 0.5 mm** mark above the centre of the well floor, equivalent to 131 mm above the station tabletop. Check the level with a depth rule. Tare the toast, plate and spill catchment separately.
3. Hold the dry dipper horizontal on a rack over the source dish. Preload four capped dosing syringes with **1.00 ± 0.05 g per syringe**, verified by before-and-after weighing during dose preparation. Starting at the groove nearest the handle, empty one syringe into each groove over one revolution of the shaft. Start the four doses at **0, 3, 6 and 9 seconds** and complete each revolution within its 3 s slot. Keep the neck and handle dry.
4. Put the dipper in its indexed horizontal rest at **13.0 ± 0.5 s** from first dose, using the same roll-marker orientation each time. Start the robot and both grading videos **1.0 ± 0.2 s after placement** with a shared sync cue. Repeat setup if the timing or fill level falls outside tolerance.

These are bench setup specifications. The simulator starts with an idealized 4.0 g uniform coating across the grooves, and the model coefficients are estimates. The bench timings and liquid properties require physical validation.

## Scoring

Use a synchronized overhead and oblique-side video pair, a 0.01 g scale and a ruler. Set the oblique camera so the 15 mm shaft spans at least 40 pixels at the carry and toast positions. Mark the dry/wet boundary 97 mm from the butt. Put two narrow, contrasting longitudinal paint stripes a quarter-turn apart on the exposed dry shaft, between the jaw and boundary. Use these stripes as clock-face marks. Place or overlay four lines perpendicular to the pot-to-toast direction at 45, 95, 145 and 195 mm from the centre of the pot. A ruler visible at table height gives the spacing. The marks support video grading. Simulation samples the same checkpoints and pose ranges from the recorded solid motion.

- **Dry handling throughout.** The parallel pads bracket the dry shaft on the butt side of the line. Treat a pad as engaged within 2 mm of the shaft with jaw gap at most 20 mm. Either hand may grip the dry region. Any jaw or palm touching the wet zone makes the whole trial 0. For consistent inspection, use the 32 mm diameter head envelope from the boundary line through the tip, with 2 mm video/geometry tolerance. A wet-end regrasp counts even after a successful drizzle.
- **Clean carry.** Take the dry grip with the head centre over the 43 mm radius of the pot opening and no more than 35 mm above the liquid. Keep the grip and keep the shaft within 20° of horizontal until the first arrival over the toast. Pause the recording at the head centre's first forward crossing of each of the four lines. Compare the clock-face orientation of the shaft at each adjacent pair. Each of the three changes must be at least 30° by the shorter arc, in either direction. Limit final spill to 0.15 g.
- **Stopped, contact-free drizzle.** Find one continuous interval of at least 1 s in which the dry grip is maintained and the shaft stays within 20° of horizontal. The shaft stripes stay within 20° of their roll orientation at the start of that interval. The head centre stays over the toast and more than 25 mm above it, with the solid tool at least 8 mm clear. Compare the toast in the overhead view at three points. It must be bare at the start of the interval, have visible honey at the end, and show the same deposits in the final overhead view. Judge the interval and the toast changes from the paired views. Tool or gripper contact with the toast disqualifies the dose. The operator may sweep sideways within the stopped interval.
- **Return and release.** After drizzling, lay the dry shaft across both notches of the original teal pot rest, with the grooved head inside the bowl and the butt pointing toward the gripper bases. The shaft points within 12° of the original +x direction. At the two supports (x = 411 and 441 mm), its centreline is within 4.5 mm laterally of the centre of the notch and 3.5 mm vertically of the seated centreline (139.9 mm above the table). Use the fixture centre lines and the calibrated ruler in the two views. The butt may shift by up to 20 mm fore or aft, provided both notches carry the dry shaft. The head centre must be within 20 mm horizontally of the bowl centre and 127–148 mm above the table. Open the gripper and move both grippers and palms at least 20 mm off the tool. For the final 1 s, the tool may shift at most 2 mm or rotate at most 3° from its initial released pose.
- **Final measurements.** Weigh the toast after the run. Count its gain as the valid dose if the stopped-drizzle check passes. Use the final overhead image and a ruler for the outermost honey spots along the pot-to-toast axis. Use the gain of the plate, tared liners and removable catchments over the reachable work area, including beneath the front edge and on protected fixtures, for spill mass. Honey recovered in the source dish stays separate from the spill catchment. Use thin removable pad covers if needed to protect the rig.

| Score | Conditions |
| --- | --- |
| 1.00 | Clean carry, valid final dose ≥0.70 g, spread ≥30 mm, and completed return, release and 1 s hold |
| 0.75 | Clean carry, valid final dose ≥0.30 g, and full score not met |
| 0.50 | Clean carry, valid final dose ≥0.20 g and <0.30 g |
| 0.25 | Valid final dose ≥0.20 g without a clean carry |
| 0.00 | Wet-end contact, valid dose below 0.20 g, or any other outcome |

The trial ends within 1,200 actions at 20 Hz (60 s). An adjudicator watches the video and weighs at the end of the trial. The carry checkpoint test uses four frames and the drizzle test uses a bounded motion interval and three toast views. All mass thresholds use final weights.

## Simulation

Isaac Lab PhysX simulates the solid dipper and its five collision collars, the supporting ceramic well and its rest, the station fixtures and native YAM contacts and actuators. The evaluator passes only the three calibrated RGB views and 14 robot joint/gripper values to Dream. The same native joint position actions drive both runs. The scripted demo reads the measured reset pose to plan its grasp and the measured dipper position to align its approach to the rest. On the return leg it rolls the dipper above the table, stops when it reaches the source dish, then aligns with both notches. It drives the placement, release, and withdrawal through ordinary arm and gripper actions.

The four wet grooves use a mass-conserving azimuthal thin-film approximation. A measured axial roll advects the film in the direction of rotation. Viscous gravity flux brings it to the underside, and capillary smoothing redistributes thickness. A pendant volume grows below a sufficiently thick film and detaches once its weight exceeds a capillary neck force. The reduced model also allows roll to take up a pendant. The net solid motion used for the liquid model is averaged with a 60 ms time constant. That makes the calculation insensitive to the dry contact solver's rapid chatter. The solid angles and speeds come from measured poses, and the rates and capillary coefficients are approximations. The model has independent grooves and discrete falling drops. It omits flow between grooves, splashes, absorption and the reaction of the light coating on robot dynamics. Surface geometry classifies actual impacts. [Experiment on retaining viscous fluid on a rotating rod](https://dxwl.bnu.edu.cn/EN/10.16854/j.cnki.1000-0712.230145)

The matched simulator comparison shares 2.846 g on the dipper at the start, and reuses the exact positions, linear velocities and dwell from a measured robot trial. Only the measured roll changes. Rolling through transport produced 0 g off target versus 0.202 g when stopping halfway through the carry. After rotating transport, the stop produced 0.602 g on toast versus 0 g with continued roll. The four native executions and matched material calculation are reproducible with the commands below.

## Measured runs

The scripted demonstration achieves **1.0/1.0** in 450 control steps (22.5 s). The three carry-band turns measure 56°, 103° and 45°. It deposits 0.802 g of honey across 47.0 mm of toast and spills 0 g. On the return leg, the dipper rolls through 5.78 rad, with a mean absolute roll rate of 2.36 rad/s while translating. It is placed in the original rest, released, and remains stable with both grippers clear for 2.0 s.

Dream-1 beta achieves **0.0/1.0** after 1,200 control steps (60 s), with 0 g of honey on the toast and 0.451 g spilled. Both trials begin at the prescribed task reset. The reported scores are computed from their recorded physical trajectories using the supplied scorer, and the videos show those same trajectories.

### Validation controls

A drain-then-carry control holds the lifted dipper motionless over the source dish for 32 s, allowing 1.203 g of honey to drain back. It then passes the rotation checkpoints and returns the dipper to the rest. The toast gains 0 g, yielding a score of 0.0 after 1,090 actions.

A wet-end handling control initializes a separate simulator at the measured end-of-drizzle state. The left gripper closes on the grooved head, the right gripper releases the dry handle, and the left gripper lifts the dipper. The scorer assigns 0 for wet-end contact. This endpoint initialization applies only to the handling control; the demonstration and Dream-1 trial use the prescribed task reset.

## Reproduce

Use Python 3.12, Isaac Sim 6.1.0, Isaac Lab-Arena and the dependencies in `requirements.txt`. Use the calibrated [YAM Isaac Lab Arena station at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) and its standard converted YAM USD. Activate the installed environment and run from this task directory. A Dream Labs credential configured for their normal SDK client is needed for `dream`. The diagnostic, demo, check and Dream use fixed seed 0.

```bash
unset DISPLAY WAYLAND_DISPLAY
python probe.py runs/probe
python probe.py runs/drain-control drain
python compare.py runs/probe
python run.py check runs/check
python run.py demo runs/demo
python wet_regrasp.py runs/demo runs/wet-regrasp
python run.py dream runs/dream
python scoring.py runs/demo
python scoring.py runs/dream
python render.py runs/demo runs/screenshot.png
python render.py runs/dream
```

Each run directory retains measured 240 Hz tool poses and liquid state summaries, the native action sequence and scored landings. The columns of `trajectory.npy` are time in seconds, left and right joint positions (8 each including both fingers), tool position and XYZW quaternion, finite-difference linear and angular velocity, film-effective roll rate in rad/s, and honey retained on the dipper in grams. The matched control trace stores time, effective roll rate, retained grams and tool XYZ. The media renderer replays those poses, checks the computed landings against the recorded ones, and draws the three station cameras and a fixed external view at 30 FPS. The screenshot uses the station's calibrated top camera at 1920 × 1080.

Task-specific assets were generated with `build_assets.py` for this submission. No third-party task meshes, images or textures are included. The station and robot ship with the upstream runtime under their own licenses. See [YAM Arena's asset and license notes](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1#license).
