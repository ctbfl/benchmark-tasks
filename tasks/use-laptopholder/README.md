# Unfold a laptop stand and set its highest position

Fully spread the laptop stand, seat both support rods in the highest-setting slots, and leave the stand released on the tabletop inside the red rectangle. Dual YAM arms operate the passive linkage using the calibrated three-camera workstation.

The physical reference is the [Tonmom adjustable folding laptop stand, ASIN B08G84P8ZX](https://www.amazon.com/Tonmom-Laptop-Stand-Adjustable-Riser/dp/B08G84P8ZX?th=1), with its stock hinges and seven height settings. Start folded and centered at tabletop (0.46, 0) m, long axis along X and hooks toward negative X. The 12 mm tape encloses a 340 x 240 mm rectangle. See `task.json` for the full setup and reset procedure.

## Scoring

Score only the final state, up to 1.0. Lifting, tilting or shaking the stand to unfold it is allowed. The stand need not remain in contact with the tabletop during manipulation; full credit requires both base rails resting on the tabletop and both grippers released and clear at the end:

- **0.25:** Fully spread the stand: use a ruler perpendicular to the two parallel support rails to measure the transverse distance between their outer side edges along the straight sections. At least **18 cm** earns 0.25; below 18 cm earns 0 for this item. Each rail is 2.5 cm wide, so this is equivalent to an inner-edge gap of at least **13 cm**. Nominal fully open outer width is 18.5 cm.
- **0.25:** Seat the left support rod in its highest-setting slot.
- **0.25:** Seat the right support rod in its highest-setting slot.
- **0.25:** With all three setup conditions satisfied, leave the entire top-down outline inside the tape's inner edges, both base rails supported by the tabletop, and both grippers released and clear. Touching the tape's inner edge is allowed.

The first three items earn credit independently; a stand below the spread threshold can still earn either rod-seating item. For physical evaluation, measure spread with a ruler and inspect the frame boundary from above. From the side, check each rod rests inside slot 1 (nearest the front hinge), rather than on its rim. After the grippers release and move clear, inspect the final overhead and side views to confirm the rods remain seated, both base rails rest on the table, and the stand supports itself without robot contact. The final quarter represents the stand placed and ready for use.

In simulation, the same outer-width threshold is 180 mm (185 mm nominal). Slot 1, nearest the front hinge, is highest; each pin center lies within its 10 x 8 mm opening, between the 4 mm floor and 8 mm rim. `task.Score` checks the complete collision-geometry footprint and uses PhysX contact reports to confirm robot release and tabletop support at both bases. Contact detection uses a 0.01 N numerical threshold. These checks run at every control step.

## Environment and execution

Use Linux, an NVIDIA RTX GPU and FFmpeg. Install [yam-arena at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1) using its `setup.sh`, activate `source <ARENA_DIR>/yam.env`, and install `requirements.txt`. Tested with Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab package 17.0.2 and Arena 0.3.0 at `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`.

Run from this directory:

```bash
python dream_eval.py --check       # Scene, cameras, state and action interface
python run.py expert              # Scripted robot demonstration
python dream_eval.py              # Dream-1 evaluation
```

Trials use the fixed initial layout, seed 0, CPU PhysX TGS at 960 Hz and a 2,400-step limit at 20 Hz (120 simulated seconds). Dream-1 receives the top and wrist camera images plus 14 measured joint/gripper positions. Native 640x360 images become 320x176 JPEGs at the model interface. Both policies execute 14-value robot joint/gripper targets; the scripted expert plans from measured mechanism state. The model uses rigid-body geometry with estimated material and joint parameters.

Dream uses the client's `~/.dreamlabs/credentials`. Each evaluation saves its instruction, source snapshot, actions, measurements and scores under `outputs/`. Inspect the results with:

```bash
python -c 'import json; print(json.load(open("outputs/expert/headless/result.json"))["final"]["score"])'
python -m json.tool outputs/dream-highest-<timestamp>/metrics.json
python verify_dream.py outputs/dream-highest-<timestamp>
```

A headless demonstration takes about three minutes on an RTX 5090; Dream evaluation wall time also includes model responses. `metrics.json` records each trial's actual runtime and score.

Dream-1 evaluated with the current instruction and rubric on 2026-11-16 UTC scores **0.0/1.0** after 2,400 steps and 120 simulated seconds (438.74 seconds wall time on an RTX 5090). All 2,400 recorded scores were independently recomputed. The robot demonstration scores **1.0/1.0** at 105.05 seconds. A complete 2,101-step replay of its recorded actions with the release-and-support scorer also finishes at **1.0/1.0**, with both bases supported and the robot clear.

Validation and video commands:

```bash
python run.py validate            # Mechanism checks across all seven settings
python run.py exercise            # Generate the mechanism scoring fixture
python check_scoring.py           # Partial credit, release and support cases
python run.py benchmark           # Scene and physical contact-sensor checks
python run.py camera_ready        # Four-camera robot demonstration
python verify_camera_ready.py     # Video, timing and calibration checks
python replay_dream.py outputs/dream-highest-<timestamp>
```

Videos use a 1080p observer view and three 640x360 robot views at 30 FPS. Generated data and media are stored in `outputs/`.

## Assets and licenses

| Asset | Source | License |
| --- | --- | --- |
| Laptop stand geometry and materials, `assets/laptop_holder.usda` | Original asset authored for this task and generated by `model.py`; the Tonmom product is a physical reference | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| YAM robot URDF, meshes and derived collision/USD assets | I2RT Robotics, [v1.3.6 / 120c3c814001](https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058), supplied through yam-arena | [MIT](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/LICENSE) |
| YAM black/white link meshes | I2RT meshes distributed with dreamlabs 1.0.23; [source and hashes](https://github.com/kairosiann/yam-isaac-lab-arena/blob/bd62771e60320433893bd1d2df0fc79b5ddcd6d1/yam_arena/assets/yam/dream/SOURCE) | MIT, under the I2RT license above |
| Arm and camera calibration | ARISE-Initiative/yamlab, commit `ec0455d2b4ce`, via Dream's standard station and yam-arena | [MIT](https://github.com/ARISE-Initiative/yamlab/blob/ec0455d2b4ce35f21fc126418ea5e74ac567133d/LICENSE) |


The table, tape and camera-stand geometry are generated by `benchmark.py` using the standard workstation dimensions. Attribute the original stand asset to this task, include its CC BY 4.0 license link, and identify modifications. Framework dependencies retain their upstream licenses.
