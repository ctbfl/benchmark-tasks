# Fill a cup with cola in at most three presses without foam overflow

Fill the fixed clear cup to the blue serving line using **at most three press-and-release dispensing cycles within 40 seconds**. The target is the final amount of retained cola after settling, not the height of the foam. In this setup, full credit requires **at least 329.04 mL** (about 342 g at the simulated density), equivalent to 90% of the measured service-full reference.

**Neither liquid nor foam may cross the rim at any time, including after the button is released.** The robot must watch the rising foam, release early enough for runoff and further foam rise, wait until the head leaves room, and decide whether and when to top up. A fourth valve opening makes the whole trial zero; the three-opening limit prevents arbitrarily many small corrective pours. The demonstration uses two openings, so three is an upper limit rather than a required count.

Hold the red spring-return button to dispense at the locked full flow and release it to stop. Keep both grippers clear of the stream. The cup starts in a locating seat on the grate, and both YAM wrists and the calibrated top camera supply RGB.

## Why a single continuous press cannot complete the fill

**A single continuous press cannot satisfy both the serving target and the no-overflow rule at either configured flow.** The target is 329.04 mL of retained cola. In the same process model used by the robot trials, foam reaches the 105 mm rim while the drink is still below that target:

| Locked flow | First foam overflow during continuous pressing | Cola already in the cup | Upper bound on delivery if released before overflow, including all runoff |
| --- | ---: | ---: | ---: |
| 52 mL/s | 6.00 s | 309.16 mL | 315.58 mL |
| 74 mL/s | 3.97 s | 290.43 mL | 297.33 mL |

The delivery bounds include cola already falling toward the cup and the full 3.8 mL outlet residual. Releasing earlier cannot reach 329.04 mL even after all of it lands; continuing longer has already caused overflow. A clean early stop can earn underfill credit, but cannot earn full credit from one opening.

The robot must therefore interrupt the flow, let the foam subside and top up within the three-opening limit. The recorded robot demonstrations succeed with two openings at both presets, retaining 340.40 mL and 344.30 mL without overflow. The single-press figures above are process-model controls, reproducible with `python process_controls.py results/process`.

## Why it differs

The task couples three requirements: reaching a retained-liquid target, keeping a transient foam head below the rim throughout the trial, and completing the fill within a limited number of dispensing cycles. Foam height and drink amount are separate signals: a cup that appears full of foam can still be substantially underfilled. The target remains unmet until enough cola has been retained safely.

- **[Pour a draft beer with a proper foam head](https://github.com/robocurve/benchmark-tasks/pull/2636)** coordinates a manual tap with a tilted, then uprighted glass to produce a 16–34 mm foam head; full credit allows up to 12 mL spilled. Here the cup stays fixed, and the robot must use at most three dispensing cycles to reach the retained-liquid target without any rim overflow. The control decisions are when to interrupt the pour before the foam reaches the rim and when the collapsed head leaves enough room to top up.
- **[Fill a cup at a lever soda fountain to just below the rim](https://github.com/robocurve/benchmark-tasks/pull/1887)** requires carrying and aligning a cup against a sprung lever, filling to a level band, and setting it down. Its published definition raises a single drink level and explicitly omits foam. Here a foam front can reach the rim before enough drink has been retained, and it can continue rising after release. Reaching the serving target therefore requires managing the intervening foam rise and collapse, with a limit on dispensing cycles.
- **[Dispense 150 g of tea using a digital scale](https://github.com/robocurve/benchmark-tasks/pull/461)** meters a non-foaming liquid with a variable-flow, non-return faucet and allows for liquid still falling after closure. Here each press opens the valve at its locked full flow, while the available headroom changes as foam grows and collapses. The robot must schedule its releases and top-ups around that changing headroom while also reaching the final retained-mass target.
- **[make cream soda](https://github.com/StanfordVL/BEHAVIOR-1K/blob/87f1bbb1595125b463000900d7b0a03d2dfcfd78/bddl3/bddl/activity_definitions/make_cream_soda/problem0.bddl)** in BEHAVIOR-1K prepares a beverage from soda water, cream, syrup and ice; its published goal requires cream soda and ice in the glass. It does not define a foam-height trajectory, an at-most-three-opening constraint, or an overflow-at-any-time failure condition. This task evaluates those dispensing and foam-management decisions directly.
- **[PrepareCoffee](https://github.com/robocasa/robocasa/blob/7bfc60c9d26c02506d4265cbe0e86b35e6e25d42/robocasa/environments/kitchen/composite/brewing/prepare_coffee.py)** in RoboCasa checks mug placement, machine activation and gripper withdrawal. It does not score the trajectory of a growing and collapsing foam head, a retained serving mass, or repeated top-ups within an opening budget.

At the same 329.04 mL target, uninterrupted flow overflows in both the 52 and 74 mL/s conditions. The feedback-driven YAM demonstrations reach full score in two pours at both presets. Replaying the measured 52 mL/s valve timing at 74 mL/s overflows; replaying the 74 mL/s timing at 52 mL/s underfills. These controls test whether the release and restart timing responds to the actual filling process.

## Run

Use Python 3.12, Isaac Sim 6.1.0, Isaac Lab 17.0.2 with CPU PhysX, Isaac Lab-Arena 0.3.0, [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena/tree/bd62771e60320433893bd1d2df0fc79b5ddcd6d1), inspect-robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14 in the same environment. On a headless host with a forwarded X display, unset `DISPLAY` before starting Kit. Configure Dream credentials through its client. Pillow and imageio-ffmpeg are used for synchronized recording.

```sh
python process_controls.py results/process
python evaluate.py probe results/jet-probe
python evaluate.py demo results/demo
python evaluate.py demo-alt results/demo-alt
python evaluate.py dream results/dream
python evaluate.py dream-alt results/dream-alt
```

Each trial allows 800 actions at 20 Hz (40 simulated seconds). Run both conditions as separate trials with the same instruction in a concealed, counterbalanced order. Report each score, and use the lower of the two to assess whether a policy transfers across settings. The primary published recordings use seed 0. Seed 1 runs the same task with the second preset. The valve operates at full preset flow on every opening in either trial. The expert commands native robot joint targets and reads the visible foam-front and bulk-liquid heights from simulator state. Dream receives three calibrated RGB images and 14 robot position channels. Each filmed run saves synchronized live media at 30 fps, process CSV, joint and valve state, commanded actions, settings and the actual score. The two expert runs also save native frames at policy observation ticks.

## Setup and scoring

Use a clear straight-sided cup with 70 mm inside diameter, 105 mm inside depth and 2 mm walls. Position the cup axis 346 mm forward and 55 mm left of the YAM station origin. Its interior floor sits 20 mm above the worktop, and the nozzle exit is 158 mm above that floor. The stainless/polymer cabinet is 135 mm deep by 268 mm wide by 294 mm high including its lid, with a grated drip tray below. The centre of the 38 mm diameter button is 344.5 mm forward, 118 mm right and 205 mm up. The plunger moves 8 mm toward the cabinet. The model closes below 3 mm and opens at 5 mm. The drive has 420 N/m return stiffness, 0.42 N preload and 6 N·s/m damping.

### Physical preparation and acceptance

Use the same chilled post-mix regular cola, syrup lot, clear cup, diffuser, nozzle height and supply settings for both conditions. A Cornelius SFV-1 press-and-hold valve (620608762) has adjustable service flow in the 1.5–3.0 US fl oz/s range. Lock its service setting for each trial. Use the product's specified syrup-to-water ratio and the dispenser's normal CO₂ regulator setting. Verify the ratio with the manufacturer’s separator after changing flow. [SFV-1 owner’s manual](https://marmonlink-apac.com/images/marmon/resources/public/cornelius/fountain/valves/SFV-1/01-OperatorManual/620608761OPR-Rev-F-1-11-21.pdf)

| Condition | At the outlet | Fully open flow |
| --- | --- | --- |
| A | 3.0–4.0 °C, same standard cola | 52 ± 2 mL/s |
| B | 3.0–4.0 °C, same standard cola | 74 ± 2 mL/s |

1. Start on a fully chilled dispenser. Purge into waste until the stream is steady and measure its temperature with a kitchen thermometer. Leave the chute and tray unobstructed. Establish the outlet flow from the change in settled mass between timed 3 s and 6 s draws into a tall, tared vessel. Divide the mass difference by 3 s and by the product density from a graduated 250 mL sample. Repeat until three measurements meet the corresponding range. Record valve setting, ratio, supply pressure, measured flow and temperature. Before each reset, verify temperature and recheck flow after every setting change. [Coca-Cola fountain quality maintenance guidance](https://www.coca-colaparts.com/?id=344&mod=pbull&op=dwnld)
2. Qualify foam response at each locked setting with three 3.0 s pours in identical dry cups at room temperature (20–23 °C), centered under the nozzle with no ice. Film the cup side and button at 60 fps against a removable mm strip beside the cup. Zero the height at the cup's interior floor. Record the bulk liquid boundary and foam front through release and 5 s after release. Require visible foam and at least a 5 mm fall from its peak over that interval in each repeat. The median peak for B must be at least 8 mm higher than A. Measure the residual using the button and stream in the same video and log the mass after the last drops land.
3. Confirm each condition with a separate continuous pour at the same target dose, using the measured flow and residual. Stop at first overflow or the target dose, whichever comes first. The qualified setup shows foam crossing the rim before a full continuous serving, and two or three visual-feedback pours can reach the reference target without crossing. Use a dry catch tray for this setup check. Record actual curves and qualify the settings before comparing policies.
4. Reset to a dry, empty cup in the locating ring, released button, open grippers at READY and clean tray. Wait for normal steady outlet temperature after flow adjustments. Counterbalance trial order and keep the service controls outside the action workspace. Place a wipe-clean splash barrier behind the fixture as needed, clear of the viewing lines and jet.

The temperatures, flow checks and optical criteria specify a repeatable preparation and bench acceptance procedure. No physical qualification data were collected for this simulation. The response tolerances are acceptance criteria, and the simulation coefficients are design assumptions.


Weigh the dry cup, then weigh it filled to 10 mm below the rim with settled cola. The difference is the full reference, 365.60 mL by the simulated cup dimensions, about 380 g at the nominal density. The blue line is 19.5 mm below the rim and corresponds to 90% of that reference, 329.04 mL (about 342 g). Use the actual full reference mass during physical trials. After the valve is released and the last cola has landed and foam is falling or stable for at least 0.25 s, score `min(1, final net mass / (0.90 * full net mass))`. Score zero for liquid or foam crossing the rim, cola striking a gripper or diverting outside the cup, more than three distinct openings, or insufficient terminal settling. One measurable loss makes the entire trial score zero, even if a later pour reaches the blue line. With the jet clear and the tray dry, a safe underfill earns proportional credit.

### Observe stream diversion on the rig

1. After priming, dry the tray and pads. Put a dry white absorbent liner in the catch tray below the cup and weigh it before fitting it. Use a 0.10 g-readability scale. At the working load, verify zero and a test weight within ±0.10 g per reading. A clean reference liner from the same batch should stay within one display division between the pre- and post-trial readings.
2. Record an unobstructed side view from the nozzle to the tray at 60 fps or faster, including the cup lip and button. Arrange a contrasting backdrop and check in advance that each gripper can be seen along the jet path. Use a second oblique view if the gripping geometry hides that path. Inspect from first flow through the last residual drop. Count the valve openings from the button transitions and inspect the rim for foam or liquid crossing.
3. Score the trial zero if the video shows a jet or drop touching a gripper before reaching the cup, or following a path outside the cup. Physical contact must be visible in at least one view, such as a jet breaking against a pad or a drop landing on the pad. A hand passing beside the jet with a visible gap incurs no diversion penalty.
4. Reweigh the same liner after runoff stops and before cleaning. A gain of at least 0.30 g also gives zero, including when fine drops are difficult to see on video. This exceeds the combined ±0.20 g reading tolerance. A smaller gain still gives zero if video shows contact, diversion, or rim loss. Use a 1 g kitchen scale separately for tared beverage mass after visible bubbles subside.

## Physical model and limits

A physical cap and spring return couple robot contact to full flow. The process advances at 240 Hz. The locked open flow is 52 or 74 mL/s according to the trial setting. A small outlet hold-up and packets already falling produce tail delivery. Landed cola generates a precursor bubble volume, released gradually into foam. Drainage runs continuously, including during a pour. The visible bulk liquid excludes the cola temporarily entrained in foam. The foam crest uses total retained cola plus expanded gas. Any crest past the 404.09 mL rim discharges a mixture of gas and cola before the next update. The falling jet is split into seven equal-volume streamlets. At each active physics tick PhysX queries the current robot collision shapes along those lines. A parcel reaching a hand before the liquid or foam surface is accounted as diverted to the tray. The cumulative diverted volume persists through the trial. A cumulative rim loss or diversion above 10⁻⁸ mL latches zero. The tolerance exceeds recorded balance residuals and is below a single modeled runoff streamlet, whose minimum volume is about 0.000109 mL. Cup, airborne, intercepted and rim-loss volumes account for all emitted cola.

The simulated fixed cup, ballistic jet, and mean foam layer approximate an upright pour. The model omits slosh, splash and droplet trajectories after interception. A 3.8 mL outlet hold-up, 80 ms outlet relaxation, 350 ms bubble-rise relaxation, gas yield of 0.78 mL per mL landed and foam drainage time of 2.65 s are uncalibrated assumptions shared by both presets. Simulated density is 1.04 g/mL. To calibrate on a physical rig, film front height at known cup graduations, record the button and change in cup mass at release, and fit growth/decay separately for repeated normal pours. The process controls cover continuous pours at the common retained-mass target, the tail and foam rise after early release, and feedback top-offs on both presets. A shared timer would require at least 6.11 s of total opening on A, even with three full 3.8 mL tails. On B, any open time above 5.46 s exceeds the whole cup's volume before counting foam or tails. The control command searches one, two and three openings over the full 40 s horizon and integrates representative short top-offs and long settle periods. The collision probe reuses one two-second valve trace with an unobstructed path, a hand across the jet and a hand moving alongside it.

## Assets

The cabinet, grate, cup, button, foam and stream are original procedural USD from `geometry.py`. The clear cup uses Isaac Sim’s built-in NVIDIA OmniGlass MDL material under the Isaac Sim distribution terms. The YAM URDF, robot meshes, collision pieces and Dream visual variants come from yam-arena's I2RT distribution under the MIT license. The table, camera stand, floor and illumination use the dependency's procedural station assets. [I2RT asset license](https://github.com/kairosiann/yam-isaac-lab-arena/blob/bd62771e60320433893bd1d2df0fc79b5ddcd6d1/yam_arena/assets/yam/LICENSE.i2rt).

## Process checks

The 40 s timing search rejects a common one-, two- or three-opening valve schedule by the fill bounds, and numerically integrates 103 representative schedules. A short final pulse and a long pause do not change those bounds. Replaying the actual 52 mL/s robot timing on the 74 mL/s preset overflowed. The reverse replay safely underfilled at 244.2 mL.

In the collider probe the exact same 107.80 mL was emitted in every case. The unobstructed jet and the moving side pass each landed all 107.80 mL. The cross-jet gripper intercepted six streamlets, leaving 15.40 mL in the cup and 92.40 mL diverted. It scores zero. The unobstructed and side-pass cases each score 0.3276 for clean underfill. Independently reconstructed finger meshes and PhysX rays identify the same six lines at the sampled blocking pose. An offline projection holding that measured obstruction for 31.2 or 35 s on the fast preset delivers enough cola to fill the cup, but both projected trials score zero. This projection uses the short probe's collision geometry. A separate offline check with a brief early interception and a subsequent clean top-off also scores zero after it reaches the mass target.

## Evaluation

| Locked nozzle setting | Expert score | Time | Openings | Cola retained | Rim spill | Dream-1 score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 mL/s | 1.0 | 14.45 s | 2 | 340.40 mL | 0 | 0.0 |
| 74 mL/s | 1.0 | 13.15 s | 2 | 344.30 mL | 0 | 0.0 |

Each Dream trial used all 800 actions and left the valve closed. The reported scores are computed from the recorded valve, liquid, foam and collision measurements using the supplied scorer. All four runs had zero intercepted volume. The primary screenshot, expert video and Dream video show the 52 mL/s condition.
