# Assets and scale

The cone and road USD assets are original procedural meshes, distributed under
this task's MIT license. No downloaded models, textures or branding are included.
The workstation and robot are loaded from the separately installed yam-arena package.

Cone height and square base width use a 1:16 scale of the dimensions in the
[JBC 28-inch cone drawing](https://www.jbcsafetyplastic.com/assets/img/product/1/files/RS70025C_with_handle.pdf):
711.2 / 16 = 44.45 mm high; 358.775 / 16 = 22.4234 mm wide. This drawing is a
dimensional reference only; its artwork and product geometry are not redistributed.

The white board is 640 x 700 x 2 mm. The two-lane road is simplified to 240 mm wide;
a nominal 6 m road at the cone scale would be 375 mm wide. The two tangent arcs
have radius 250 mm and turn 60 degrees each, giving a 523.599 mm centreline and
104.720 mm ideal cone spacing.

The tabletop cone is a 12 g prototype, not a mass-scaled road cone. The 0.8 mm
hollow shell, 2 mm base and internal horizontal nesting shoulder provide 16 mm
stack spacing. Four passive radial spring petals approximate compression of the
top 5 mm: 400 N/m stiffness, 0.2 N s/m damping, 0.8 mm travel, and 5 N spring force
limit. Friction coefficients are 0.8 on the shell, 1.0 on the base and 1.2 on the
soft tip. This is an uncalibrated compliance approximation, not a finite-element
model of PVC. Mutual collision between the four quadrants of this single
compressible cap is filtered to avoid jamming the radial approximation; external
contact with fingers, the board and other cones remains enabled. Grasping uses
actual finger contact, without attachments.

Run `python build_assets.py` in the documented Isaac Lab environment to regenerate
both self-contained USD files. Measurements are in metres.
