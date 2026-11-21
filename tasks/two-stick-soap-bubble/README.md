# Make a free-floating soap bubble with a two-stick string wand

Pick up the two connected sticks, dip the cord, and sweep the opened loop against a gentle breeze. Close the tips on the inflated neck and let the bubble drift away. The two hands carry, spread and close the suspended loop together, so the opening and air loading change with their motion. This is the weighted two-stick wand used in outdoor play. [Northumbria University NUSTEM activity](https://nustem.uk/activity/giant-bubbles-adult-led-activity/)

## Why it differs

- **Herd three cans into a bay with a seine rope** uses the rope's bight to push upright rigid cans across a table into a target. Here the lifted cord supports a soap film. Both arms have to keep its opening exposed to airflow, then shrink the opening while the inflated film is still intact so it seals and separates. [PR #578](https://github.com/robocurve/benchmark-tasks/pull/578)
- **Braid a practice mannequin head's long hair: three sections, six crossings, outer over middle each time, that stay put** tests forming and preserving a braid in three sections of a mannequin head's hair. Here the two tips tension opposite ends of one loop and vary its live aperture to shape and release an unsupported film. [PR #2296](https://github.com/robocurve/benchmark-tasks/pull/2296)

## Setup and reset

Use the standard 110 × 130 cm tabletop and two YAMs. The midpoint of the bases is the horizontal origin. Forward is +X and left is +Y. Heights in this table are above the tabletop.

| Part | Build and placement |
| --- | --- |
| Two sticks | Cut two 244 × 12 mm wooden dowels. Give each an 84 mm long, 24 mm OD silicone sleeve with its center 53 mm from the rear. Use green on the left and orange on the right. Each tip tie is 185 mm ahead of the corresponding sleeve center. |
| Weighted loop | Use 4 mm braided cotton and an approximately 3 g smooth washer or nut. With fixed knots at both stick ties, leave 246 mm of exposed cord on top and 396 mm below, measured along the cord between knots. Thread the weight onto the lower strand and tie it at the midpoint. |
| Cradle | Use four anchored wood or printed plastic posts with saddles. Put post centers at X = 150 and 288 mm and Y = ±103 mm. Use 19 × 32 mm flat saddles at 214 mm above the table, with 10 mm high side lips spaced 18 mm apart. Set the wand sleeve centers at X = 205 mm, Y = ±103 mm, Z = 220 mm with both sticks horizontal and the ties pointing forward. |
| Dipping basin | Use a rigid plastic basin with a 244 × 330 mm inner opening and roughly 120 mm inside depth. The inside spans X = 430–674 mm and Y = ±165 mm. Rim height is 129 mm. Fill to 103 mm above the table, about 7.6 L for the flat-sided model. Place a nonslip spill mat under the basin in real trials. |
| Solution and air | Use commercial giant-bubble solution from one lot, or mix a batch using the NUSTEM recipe at the same ratios. Use clean room-temperature solution and remove surface foam between trials. Set a guarded fan behind the station to blow forward at about 1.5 m/s through the working opening. Check it with a handheld anemometer before the trials. |

Reset both arms to the standard raised ready pose with jaws open. Dry the cord and sleeves before seating the sticks for a new scored run. Keep the arms and grip sleeves outside the bath. Allow open space in front of the station for the bubble to drift. A detailed nominal model of the cradle, sticks, knots, and basin is in `assets/wand-kit.usda`.

## Scoring

- **0.5** when a continuous soap film spans the loop after it is dipped and lifted out of the solution.
- **A further 0.5** when the film closes and separates into a bubble that floats independently in the air.
- **0** if neither event occurs. Keep the highest score achieved within 40 s (800 steps at 20 Hz). A later burst does not cancel a completed event.

For real trials, watch the loop and the released bubble on video. Use a side or front view to confirm that the film spans the lifted loop, then confirm a closed bubble separates from the loop and moves independently through the air. A film still joined to the loop earns only the first event.

## Simulation and reproduction

The wand and the cord have colliders. Thirty-six light capsule links with passive spherical joints approximate a limp cotton loop. The wetting model reads immersion of those links and both ties. The pressure model takes relative normal air speed and the moving cord's projected aperture, then integrates air influx against a capillary threshold and leak. Neck closure is based on narrowing relative to the inflated opening and bringing the ties together. A separated bubble moves independently and can burst on contact. The bath height is fixed and the model uses parameterized thin-film surfaces and contact envelopes. Liquid circulation, film thickness, and aerodynamic load on the arms are omitted. The parameters have no physical bench calibration. [Salkin et al., Generating Soap Bubbles by Blowing on Soap Films](https://doi.org/10.1103/PhysRevLett.116.077801)

Use Isaac Sim 6.1.0.0 with Isaac Lab `91a0906f93802eb859192dcb0bca879f891e92d6`, Isaac Lab Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, and YAM Arena `bd62771e60320433893bd1d2df0fc79b5ddcd6d1`. Install the pinned Python clients from `requirements.txt` in that environment. Runtime uses seed 0, CPU PhysX at 240 Hz and an RTX-capable GPU for cameras. Both controllers use the native 14-dimensional position interface with the upstream robot limits. The demo reads actual joint angles to trim tracking error. Dream receives RGB from the calibrated top and wrist cameras and its measured joint/gripper vector.

From this task directory in the configured environment, launch Kit headlessly without a display connection.

```bash
python -B demo.py runs/demo
env -u DISPLAY -u WAYLAND_DISPLAY python -B dream_eval.py --check runs/camera-check
env -u DISPLAY -u WAYLAND_DISPLAY python -B dream_eval.py runs/dream
python -B test_score.py -v
python -B score.py runs/demo
python -B score.py runs/dream
env -u DISPLAY -u WAYLAND_DISPLAY python -B render.py runs/demo
env -u DISPLAY -u WAYLAND_DISPLAY python -B render.py runs/dream
```

The Dream client reads its usual credentials. The demo and Dream use the same native-physics environment; the supplied scorer grades the saved physics measurements. The 30 FPS videos replay measured states, and the screenshot uses the native calibrated Top Camera projection at 1920 × 1080.

## Results

The recorded scripted trial scored **1.0**: a film spans the lifted loop at 13.24 s and a closed bubble separates and floats independently at 18.12 s. The recorded Dream-1 beta trial scored **0.0** over 800 actions and showed no wetting event. The supplied scorer grades the saved physics measurements and checks that film and bubble motion match the recorded video states. See `recorded-scores.json` for the event results.

The demo used a 0.20 normalized opening command to hold the sleeves. In the air-motion phase, the finger-drive force estimates logged at 20 Hz had a 12.79 N median and 14.14 N peak. The same trial also registered a 58.67 N solver-measured normal contact peak at the left wand during inflation at 16.850 s. Replaying its exact actions reproduced every measured body pose. At 240 Hz, that pad's contact vector was (+0.62, −58.66, +0.16) N in station coordinates and the opposite pad registered (+0.50, +58.62, −1.18) N. The longest run above 50 N was one 4.17 ms substep, with a highest 0.25 s mean pad force of 20.96 N. The substep drive estimate peaked at 14.66 N. The native 50 N drive setting is retained. These are separate simulator quantities.

## Asset licenses

The task's dowels, cord, cradle, basin and films are original procedural assets licensed CC BY 4.0. Standard YAM meshes, including the black/white covers, are by I2RT Robotics and distributed under MIT. Standard camera and arm calibration is from ARISE-Initiative/yamlab under MIT. The table, floor and camera stand are procedural geometry from the YAM Arena dependency. [License and upstream references](LICENSE)
