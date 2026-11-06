# Iron stand and thermometer clamp

Procedural models reused from the Read thermometer task. All dimensions are metres. Rebuild from the task directory with `python stand_clamp.py` inside the shared YAM environment.

`stand.usda` is a free 2 kg rigid body, with a 250 × 150 mm rectangular base and a 12 mm diameter upright. Its origin is the base's bottom centre. The long base edge follows Y, parallel to the table's long edge; the upright is at `(0, 0.090)`, near the left short edge in the calibrated view, and reaches `z=0.440`. The clamp extends rightward along -Y, over the beaker on the right of the upright.

`clamp.usda` is a free articulation rooted at `/clamp/boss`. Load with no external actuators. Its origin lies on the stand upright. The thermometer axis is at `(0, -0.120)` in this frame. The initial 9 mm jaw gap matches the thermometer stem. The clamp does not create any joint to the stand or thermometer.

The boss has a left-opening (+Y) U-shaped rod socket with the original wall thickness and corners retained. Once the screw is loosened, the clamp can slide onto the upright from its right side, without passing over the rod top. The knob faces forward (-X), toward the arms. A rotary knob and a translating pressure pad are coupled by a native articulation `PhysxMimicJointAPI`, with a 2 mm per revolution lead. The screw's initial opening is 11.8 mm around the 12 mm stand rod, providing assembly preload. Angular joint friction approximates thread self-locking. Rotating the knob withdraws or advances the pad; the slider has a 9.2 mm total travel. The shaft's individual thread ridges are visual geometry; collision is carried by the pressure pad. This models the screw's kinematic ratio rather than resolving contact between helical threads.

The thermometer jaw rotates around its vertical pivot. Negative rotation opens it, to a -35° limit. Squeezing its rear handle toward the fixed handle opens the front jaw. A passive torsion spring has a fixed +4° rest target, 0.8 N·m/rad stiffness, 0.012 N·m·s/rad damping and a 0.12 N·m force limit. At the 9 mm initial opening the nominal moving-jaw contact force is approximately 2.2 N. The spring and damping are uncalibrated engineering parameters. The fixed jaw has a longitudinal 90° V groove with two rubber contact faces, while the moving jaw has a flat rubber liner. These three faces centre the round thermometer stem. Rubber liners provide contact friction; there is no hidden attachment to the thermometer. The visible torsion coil is decorative, while the joint drive provides its physical spring force.

Joint-connected metal parts have pairwise collision disabled where they share a pin or bearing. Articulation self-collision and contacts with the stand and thermometer remain enabled. Material and spring values are prototype approximations, not measured product data. Global simulation and robot settings are not changed.

`metadata.json` lists all body and joint paths. The task resets the screw to one open turn (2 mm pad retraction); it must be tightened by physical knob rotation.

Native cross-type articulation coupling follows the `MimicJointDemo.py` supplied with Isaac Sim 6.1. The relation is `travel_m = angle_deg * 0.002 / 360`. The legacy PhysX mimic API supports this mixed rotary/linear pair; the Newton mimic schema is documented for equal joint types only.
