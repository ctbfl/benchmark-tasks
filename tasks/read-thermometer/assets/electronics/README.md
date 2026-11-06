# Tabletop electronic instruments

Dimensions are metres; Z is up. Both assets are free-base articulations with independent spring-return buttons. No robot or global solver settings are overridden.

- `timer.usda`: original 140 mm wide white timer model with a rounded shell; face normal points towards -X and 25 degrees upward. The blue buttons are START/PAUSE, M, S from left to right, with black geometric lettering below them. Initial display is `00:15`, already running. The rear stand is modelled in its open position as rigid support geometry.
- `calculator.usda`: wrapper for `calculator_model.usda`, from the MIT-licensed Tape measure task. Its purple shell measures 101.6 by 139.7 mm and carries 21 keys. Rotate the asset -90 degrees about Z in this scene so the screen top points towards +X. Its rigid root body is `base`.
- `metadata.json`: key-to-joint mapping, neutral key centres, orientation, normals, travel and thresholds.
- `textures/`: the 21 original key-label textures copied with the tape-measure calculator. Timer lettering is opaque font-outline triangle geometry made with the locally installed DejaVu Sans font.

Timer keys have 3 mm travel, a 350 N/m spring at zero, 0.5 N s/m damping and 6 g moving mass. Their thresholds are 1.6 mm press and 0.7 mm release. The calculator retains its original 1.5 mm travel, 600 N/m spring, 0.4 N s/m damping and 2 g moving mass, with 0.8 mm press and 0.25 mm release thresholds. These are engineering approximations, not measured product parameters. Connected body/key collisions are disabled by their joint; other articulation collisions remain enabled.

Integration uses `electronics.pressed_buttons({key: measured_joint_position}, previous_pressed, press_threshold=..., release_threshold=...)` with thresholds from each device's metadata, followed by `state.update_buttons(pressed)`. Feed actual simulator measurements into this function, not desired positions. Call `TimerState.advance(physics_dt)` once per simulated physics step. Display helpers update the seven-segment geometry; no image overlay is used.

`TimerState` implements separate minute/second increments and wraparound, start/pause/resume, overlapping M+S reset and stop at zero. There is no count-up mode. M/S remain usable while running. Held buttons trigger once until released.

`CalculatorState` implements decimal entry, sequential basic arithmetic, repeated equals, sign change, clear, clear entry, percentage, square root and error display. `last_submission` contains the numeric value submitted by `=`, or `None`; `submission_count` counts equal-button evaluations. The task scorer checks the submitted temperature.

Regenerate and check from the task directory after activating the shared `yam.env`:

```bash
python electronics.py
python -m unittest -v test_electronics.py
```

The seven CPU tests cover timer logic, held buttons and chords, physical displacement hysteresis, arithmetic, temperature entry, authored joints and display updates. Simulator contact and settling validation is performed by the integrated task scene.
