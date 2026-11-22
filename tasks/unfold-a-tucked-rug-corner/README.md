# Unfold a tucked rug corner inside a taped boundary

Retrieve the corner folded underneath a small entry mat and lay the entire mat flat inside its red tape outline. This is a mat-handling task: expose material hidden beneath the supported body while preserving its placement footprint.

## Setup

Use the native bimanual YAM station and its calibrated top and wrist cameras. The task object is a 600 × 400 × 5 mm soft, backed entry mat (nominal mass 264 g), with a teal woven face and dark bound edges. It rests on the level station tabletop at z = 0.7475 m. The mat centre is (0.45, 0) m; its long side follows world x, pointing away from the robot.

Fold the near-right corner down underneath along a diagonal joining points 120 mm from that corner on both adjacent edges. The rest of the mat remains on the support; the corner is concealed below it. Red paper tape, 18 mm wide and 0.2 mm thick, marks a 640 × 440 mm rectangle. Its **inner** edges are x = 0.13–0.77 m and y = −0.22–0.22 m. The tape is a placement marker, not a retaining fixture. Reset restores this layout and allows five seconds of passive settling before the episode clock starts. No tools, weights or attachments are supplied.

## Scoring

Evaluate the current final state:

| Score | Condition |
| --- | --- |
| **1.0** | The whole mat is unfolded and flat, and every part lies inside or touches the tape's inner edge. |
| **0.5** | The whole mat is unfolded and flat, but any part crosses the inner edge. |
| **0.0** | The mat is not unfolded and flat, including a still-tucked corner, regardless of containment. |

For the 5 mm mat, flatness allows at most 2 mm of extra surface height: all pile and backing points must lie between −2 mm and +7 mm relative to the support. The projected material area must be at least 99% of 0.24 m² (0.2376 m²). The scorer uses signed, oriented triangle area so material folded underneath cancels instead of being counted twice. Containment checks the entire pile and backing surfaces, not just the centre or corners. Credit does not latch an earlier state; temporary boundary crossings do not independently disqualify the final result.

For physical trials, use a calibrated overhead photo, with the tape as the scale, to inspect the exposed footprint, corner and boundary. Low views and a ruler or 2 mm clearance gauge check raised folds relative to an adjacent supported area of the same 5 mm mat. Inspect all four edges. Merely moving the folded mat into the frame earns no credit.

## Why it differs

- [**Blanket Flatten 01** (RoboARC / Embodied First Exam)](https://github.com/mll-lab-nu/embodied-first-exam/blob/e6db5737f72627597713a96087f27b1b17b06063/tasks/roboarc/roboarc-blanket-flatten-01-i01-standard/instruction.md) asks the robot to spread a blanket without overlaps and release it. Both tasks manipulate deformable sheets. Here the initial obstruction is one corner trapped underneath a backed mat; the robot must gain underside access and recover that corner while keeping the whole mat within its placement outline. The physical manipulation and final containment are both exercised by the demo and scorer.
- [**Strip and replace the fitted sheet on a bassinet mattress**](https://github.com/robocurve/benchmark-tasks/pull/2232) removes a used sheet and stretches a clean elastic hem around four mattress corners. This task recovers an underfold in a free mat on a level surface; it has no mattress, elastic pocket fitting or laundry transfer.

## Simulation fidelity

The mat is a passive native PhysX surface FEM mesh: 10,121 nodes and 19,824 triangles, a flat unstressed rest shape and a folded initial shape. Its collision and displayed thickness are 5 mm. Density is 220 kg/m³, in-plane Young's modulus 2 MPa, Poisson ratio 0.3 and dynamic friction 0.6. Bending is represented by an effective 1 mm backing within the single 5 mm shell, giving a surface bend parameter of 1465.20 Pa. Pile and backing are not separately resolved materials. This is an explicit soft-mat approximation, not a measured material calibration.

Self-contact, gravity and robot contact evolve in the native solver. The task preserves native YAM drives and force limits. Display surfaces follow measured mesh nodes; model images show those same nodes and measured robot joints. The expert plans pad poses with inverse kinematics, reads measured robot joints for position-error compensation, and commands only the shared 14-dimensional joint-position/gripper interface. The recorded demo uses the right hand to insert, grasp, lift, lower, release and retract; the corner unfolds through the resulting passive contact motion.

## Run and reproduce

Use Python 3.12, CUDA-capable Isaac Sim / Isaac Lab with native surface-deformable support, Isaac Lab-Arena and [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena). Install the compatible Inspect Robots, inspect-robots-dream and dreamlabs-client packages for Dream evaluation. The scripts also use NumPy, SciPy, Pillow and imageio-ffmpeg. See `runtime-versions.json` for the tested revisions and package versions. Robot/station assets are supplied by the adapter; task-owned assets are included locally.

Run from this directory in that environment:

```bash
python -B expert.py runs/demo
python -B dream_eval.py runs/dream
python -B score_run.py runs/demo
python -B render.py runs/demo
python -m unittest test_scoring
```

Each run directory must be new. Both policies share `dream-defaults.json`: seed 0, fixed layout without jitter, 480 Hz physics, 20 Hz control and a 2,400-control-step / 120-second limit. API waiting does not advance physics. Allow the simulator's initial five-second settling period in addition to the episode time; wall time also depends on GPU rendering and API latency. Dream uses its SDK's normal local credential configuration; credentials are not part of this task. The recorded 120-second model episode took about 14 minutes of wall time on an RTX 5090, including simulator startup and live observations; video export is separate.

`trajectory.npz` records measured cloth nodes and robot joints at 30 Hz, plus requested and applied actions at 20 Hz. `result.json` contains the final live score. `score_run.py` regrades the final measured state. `render.py` draws the recorded states at normal speed without policy calls or extra physics; the output is `videos/demo-original.mp4` for either run type. Videos combine a fixed external observer with the calibrated left wrist, top and right wrist cameras. Only the latter three RGB views and 14 robot joint/gripper values are available to Dream; the SDK sends 320 × 176 JPEGs from 640 × 360 renders.

## Recorded results

The scripted demo reaches **1.0 in 500 control steps (25.0 seconds)**, including five seconds of passive settling after withdrawal. Its final projected-area ratio is 0.999985, maximum surface height is 5.000 mm and boundary overrun is 0 mm.

Dream-1 scores **0.0 at the 2,400-control-step limit (120.0 seconds)**. The mat remains inside the frame with its corner still tucked underneath. Both runs use the same scene, material, native robot drives, action interface and scorer.

## Assets and licenses

`assets/rug.usda` and `assets/weave.png` are original task assets released under CC0 1.0. The runtime robot and station dependency notices and source links are listed in [ASSET-LICENSE.md](ASSET-LICENSE.md). No third-party meshes are copied into this task directory.
