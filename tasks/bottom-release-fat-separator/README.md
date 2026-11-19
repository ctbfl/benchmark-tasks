# Drain gravy through a bottom-release fat separator and stop before the fat reaches the outlet

Recover the lower layer of pan drippings for gravy while leaving the floating fat in the separator. Hold down the squeeze paddle to drain into the bowl. Let go before the descending fat reaches the outlet. The paddle springs back and the last of the gravy already below the valve continues into the bowl. [OXO bottom-release separator instructions](https://www.oxo.com/blog/cooking-and-baking/oxo-gravy-thanksgiving-fat-separator)

## Why it differs

- **Skim the fat off stock without wasting the broth** removes the upper layer with a ladle, so the operator controls scoop depth and repeated transfer. Here the lower layer is drawn from a fixed bottom outlet. The robot has to judge when it can release the spring-loaded valve and still land nearly all the gravy without fat breakthrough. [PR #2437](https://github.com/robocurve/benchmark-tasks/pull/2437)
- **Decant an aged wine off its sediment: pour slowly and stop when the sediment reaches the bottle's shoulder** uses bottle tilt to maintain a gentle pour and watches sediment advance toward the shoulder. Here the vessel stays upright. Valve travel controls the gravity-fed flow, and the interface descends vertically as the lower layer drains. [PR #1475](https://github.com/robocurve/benchmark-tasks/pull/1475)
- **Dispense 150 g of tea using a digital scale** uses a tap to deliver a target mass of one liquid. Here the amount to recover depends on the starting state of two phases. A fixed cup mass can award the wrong amount or include fat. [PR #461](https://github.com/robocurve/benchmark-tasks/pull/461)
- **Nursing preparation: purge air while preserving the drawn-up liquid** uses syringe reorientation and coordinated plunger motion to purge a gas pocket while keeping the sample in the barrel. The gravy task recovers a liquid from an upright vessel through a gravity-fed aperture and anticipates passive spring closure. [PR #1151](https://github.com/robocurve/benchmark-tasks/pull/1151)

## Simulation setup and reset

Use the calibrated dual-YAM station. The axes are x forward from the arm bases, y toward the left arm and z up from the tabletop. The separator and bowl share a vertical axis at (411, -85) mm. The clear polymer reservoir has a 98 mm internal diameter above a 30 mm tapered sump with a 40 mm floor diameter. Its floor and rim are 164 and 274 mm above the table. The bottom outlet is 7.6 mm in diameter and ends at z=135 mm. The dark polymer cradle holds the separator upright over a fixed ceramic bowl with 142 mm internal diameter and an 81 mm rim height. The 46 × 38 mm green paddle faces the arms. Its initial top is at (275, -85, 226) mm. The paddle moves 16 mm vertically against a 330 N/m return spring. The paddle and its external linkage move above the fixed nozzle. Geometry is defined in `geometry.py`.

Reset with both arms at READY and open grippers, the bowl empty, and the paddle closed. The geometry stays fixed and the charge varies by seed.

| Seed | Starting water (g) | Starting vegetable oil (g) | Minimum water in bowl (g) |
| --- | ---: | ---: | ---: |
| 0 | 290.00 | 80.99 | 261.00 |
| 1 | 185.00 | 111.93 | 166.50 |
| 2 | 395.00 | 50.05 | 355.50 |

The simulated drippings are modelled as water at 1.00 g/mL and vegetable oil at 0.91 g/mL. The water target is 90% of the starting water mass for the selected seed.

## Physical setup

Use the clear 1 L Cuisipro 747301 fat separator with its factory handle, linkage and bottom plug. Downward pressure on the red stock lever lifts the plug. Releasing it lets the plug reseat. Its published outer dimensions are 229 × 152 × 146 mm. Support the factory foot on a clear 220 × 210 mm deck with a centred outlet opening and rear hold-downs. The fixture leaves the low sidewall exposed for wrist viewing and the bowl sits below the outlet. Weigh the listed starting masses of room-temperature dyed water and canola oil separately. The [physical setup and scoring protocol](BENCH_PROTOCOL.md) gives the construction dimensions, pose and visibility checks, and how the real lever maps to the simulated paddle. [Cuisipro product specifications](https://www.cuisiprousa.com/collections/roasting-tools/products/cuisipro-black-tritan-material-fat-separator-7-25-747301) · [Cuisipro operating instructions](https://www.williams-sonoma.com/netstorage/pdf/Cuisipro-4-Cup-Fat-Seperator-79461-User-Manual.pdf)

## Scoring

A full score requires at least the table's water mass and no visible pooled oil on the bowl surface. A total projected oil area of **25 mm² or greater** counts as visible. This is the area of a 5 × 5 mm square. Oil at or above this area or failure to finish with the valve closed and no flow or drops for 0.35 s scores zero. The closed paddle is within 1.5 mm of rest. An underfill with a clean surface and a settled valve scores `0.95 × bowl water mass / minimum water mass`.

The simulated score uses the same planar oil patch area drawn in the bowl. On the bench, inspect or photograph the settled surface against a millimetre scale under diffuse light, then weigh the separated water. The [bench protocol](BENCH_PROTOCOL.md#inspect-the-surface-and-weigh-the-water) describes visual inspection and optional colour segmentation. The physical mount and surface measurement have not been bench-tested.

## Evaluation

Use Linux with an NVIDIA RTX GPU and the [YAM Arena setup at bd62771](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1). The tested stack was Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 17.0.2 at `91a0906`, Isaac Lab-Arena 0.3.0 at `c7edf3a`, Inspect Robots 0.60.0, Inspect Robots Dream 0.1.0, Inspect Robots YAM 0.37.0, Dream Labs client 1.0.14, and imageio-ffmpeg 0.6.0. The Dream client reads its standard `~/.dreamlabs/credentials` file.

Set `ARENA_DIR` to the Isaac Lab-Arena installation directory and run from this task directory. Each evaluation needs a fresh output directory. The optional final argument selects the fill seed.

```bash
source "$ARENA_DIR/yam.env"
unset DISPLAY WAYLAND_DISPLAY
PRODUCTION_GPU_RUN_ID=local-demo python evaluate.py demo runs/demo 0
PRODUCTION_GPU_RUN_ID=local-dream python evaluate.py dream runs/dream 0
python score_run.py runs/demo
python score_run.py runs/dream
python render.py runs/demo runs/demo-media
python render.py runs/dream runs/dream-media
```

Both runs use 240 Hz CPU PhysX, the same scene and native YAM position actuators. The demo reads the measured closure and remaining gravy to adjust its stopping time. Dream-1 beta receives only left wrist, top and right wrist RGB plus 14 measured robot joint and gripper values. The adapter resizes 640 × 360 camera images to 320 × 176 JPEG and holds during network waits. The demo scored **1.0** in 15.8 simulated seconds after recovering 283.802 g of water with 0 mm² of oil. Dream-1 completed 30.0 simulated seconds and scored **0.00899** after recovering 2.470 g of water with 0 mm² of oil. Both ran on seed 0 with the 90% instruction and the visible oil criterion.

## Fidelity and assets

Finger contact, paddle travel and spring return run in PhysX. The vessel and bowl stay fixed. The flow model uses measured paddle travel, hydrostatic head, an approximate inlet capture region and ballistic travel between outlet and bowl. Reservoir geometry maps both conserved phase volumes to visible heights. There is no CFD, emulsification, wetting, splashing or interception of the modelled stream. Spring and liquid parameters are design estimates. The simulated paddle is physical and the wetted plug and passage are represented by the outlet model. The stock bench linkage has its own lever travel, opening force and drainage rate. The renderer shows oil as a compact floating patch with a minimum thickness of 0.7 mm. Scoring uses that patch's horizontal area. The model omits real spreading, merging and breakup.

The separator, cradle and bowl are original procedural meshes with no external textures. The standard workstation geometry and camera calibration are supplied by the YAM Arena dependency. The I2RT robot meshes, including its covers, are provided by that dependency under the [I2RT MIT license](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/LICENSE). The task ships no copies of those assets.
