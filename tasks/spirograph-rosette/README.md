# Draw a closed Spirograph rosette while keeping the wheel meshed inside its ring

Use a pen to drive the loose wheel inside the drawing ring. The eccentric through-hole transfers motion from the pen to the wheel while the nib draws on paper. Two circuits of the wheel center make one closed three-lobed rosette. The board holds the ring still, as with a drawing toy's fixed or magnetically secured frame. [Hasbro Spirograph instructions](https://instructions.hasbro.com/en-us/instruction/spirograph)

## Why it differs

- **Scribe a full circle with spring dividers, keeping the point seated in the center-punch mark all the way round** uses a stationary center point and a fixed radius to trace a circle on steel. Here the wheel center travels around the ring, the eccentric pen hole counter-rotates with the wheel, and success depends on continuous tooth engagement through both orbits. [PR #1557](https://github.com/robocurve/benchmark-tasks/pull/1557)
- **Mesh a three-gear train, finding each gear's tooth phase by contact** seats separate gears on fixed shafts and checks final engagement. Here the starting wheel is already seated and free to translate. The pen has to maintain rolling tooth registration under load while leaving a closed mark. [PR #893](https://github.com/robocurve/benchmark-tasks/pull/893)
- **Trace an oblique path with two XY plotter knobs** coordinates two axes through fixed controls to trace a marked lane. Here the hand drives the pen directly through a moving hole and maintains simultaneous pen, paper and gear contact to generate the figure. [PR #408](https://github.com/robocurve/benchmark-tasks/pull/408)

- **Draw a right-angled triangle on an Etch A Sketch** coordinates two knobs to move an internal stylus along three straight sides. The rosette uses a hand-held pen that loads the bore of a free rolling gear while its nib leaves a continuous line on paper. [PR #2571](https://github.com/robocurve/benchmark-tasks/pull/2571)

## Setup and reset

Use a calibrated dual-arm YAM worktable. Set a 254 × 230 × 10 mm rigid drawing board with its center 350 mm forward of the YAM base line and 100 mm toward the right arm from the station midline. Fasten a blank 198 × 172 mm sheet and clamp the 142 mm outside-diameter blue ring flat on it. The ring has 54 teeth, 2.2 mm module and 25° pressure angle. Seat the green 36-tooth plastic wheel flat, center 19.8 mm forward of the ring center, with the 6.7 mm bore facing forward. The ring and wheel are 4.8 mm thick. The bore is 27.5 mm off center.

Start the orange 90 mm felt-tip pen upright in its socket, 66 mm toward the arms and 88 mm toward the right arm from board center. Its barrel is 14.2 mm across the grip, and the 4.8 mm ferrule fits through the bore. Reset the arms to the station ready pose and replace the sheet after each trial. The included millimetre-scale meshes and source can be used to produce the two plastic gears, holder and clamps. Paper, a short felt-tip refill and a printed barrel suffice for a physical reproduction.

## Scoring

A full score requires the same uninterrupted ink segment to cover at least 12.466 rad of wheel-center travel with the gear in rolling phase and close within 2 mm of the first mark. The longest valid segment earns its fraction of two circuits, capped at 0.95 before closure. More than 8 mm of extra ink laid down outside valid contact scores zero.

The scorer samples the lowest point of the nib and the normal contact forces at 240 Hz. Ink requires at least 0.030 N of paper force. A detectable lift or an unmarked excursion wider than the 0.8 mm mark splits both the rendered line and the scored arc. A 0.05 mm height tolerance accounts for simulated contact chatter. When the nib stays at the surface and within one mark width, brief pressure dropouts can preserve an overlapping line. A return to the previous mark after a lift or wider excursion starts a new segment.

The wheel center must stay within 19.8 ± 1.05 mm of the ring center, within 1 mm of the sheet height, and within 3.6° of flat. The shaft must be within 1.45 mm of the bore axis at wheel mid-height and within 13.1° of upright. Ring and bore force each need a sample of at least 0.012 N within 0.12 s. Geometric disengagement, a gap of more than 0.12 s between qualifying samples or a rolling phase error of 4.8° also ends the qualified arc.

For a physical trial, use a felt tip with a 0.8 mm mark and record an overhead or oblique view that shows the pen and wheel. Use the bore and index mark to check wheel travel and rolling phase, and treat a visible pen lift as a new segment. After the trial, the evaluator can lift the loose wheel and photograph the sheet against a millimetre ruler to check for uncovered gaps and measure closure. Count only the longest qualifying segment on the video and sheet.

## Reproduce

Use Linux, Python 3.12, RTX and FFmpeg. Set up [yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) with its `setup.sh`. Tested with Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Arena 0.3.0 at `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. Use the Arena production asset server for the standard workstation. In the activated environment, run these from this task directory, choosing new output directories for each trial.

```bash
unset DISPLAY WAYLAND_DISPLAY
python -m unittest test_scoring
python trial.py check runs/check
python trial.py demo runs/demo
python trial.py dream runs/dream
python scoring.py runs/demo/physics.f32
python scoring.py runs/dream/physics.f32
python media.py runs/demo/result.json demo.mp4
python media.py runs/dream/result.json dream.mp4
python media.py runs/demo/result.json screenshot.png
```

The production runner supplies `PRODUCTION_GPU_RUN_ID` for provenance. For a standalone reproduction, export a local run identifier before launching the trial. For Dream-1, configure the client's standard `~/.dreamlabs/credentials`. It receives three calibrated robot camera frames at 640 × 360 resized to 320 × 176 JPEGs and measured 14-D robot joint/gripper state. The scripted demo reads wheel, pen and robot poses to command the same native absolute joint position and gripper targets at 20 Hz. Both use seed 0, 240 Hz CPU PhysX, the native YAM gains, and a shared 2,400-action limit (120 seconds). Evaluations write scores, raw contact measurements and executed actions. The media command replays measured positions and contact marks at 30 fps.

The gear, board, holder, paper and pen USDs are original and self-contained. The simulated ink is a visible trail of measured nib contact. The rigid gears use convex pieces along their tooth flanks and around the eccentric bore. The standard workstation and camera calibration come from the Arena/YAM environment. I2RT robot meshes are MIT-licensed by I2RT Robotics, as documented in the YAM adapter's `yam_arena/assets/yam/LICENSE.i2rt`. No external task-object meshes or textures are bundled.

## Recorded results

- **Dream-1:** 0.0/1.0 in 2,400 actions (120 s), rescored from the recorded rollout under the revised ink rule. No pen-to-bore contact was measured. The video replays those same recorded states.
- **Scripted demo:** 1.0/1.0 in 1,197 actions (59.85 s). A brief mark during seating is a separate segment. The qualified rosette spans 12.499 rad and ends 0.212 mm from its start. Full credit first occurs at 55.525 s. The arm then lifts the pen and moves aside. The seated wheel covers part of the drawing in the final frame. The simulated nib force has a 2.84 N median and a 6.58 N P95 over qualifying samples. A single 4.17 ms physics step at paper recontact reached 12.19 N.
