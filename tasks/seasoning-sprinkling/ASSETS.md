# Assets and licenses

The bottle, independent flip covers, sealing plugs, perforations, plate, steak,
seasoning grains and label/steak textures are original procedural assets generated
by `assets.py`. They are licensed under the included MIT license. No photographs,
commercial product logos or downloaded third-party object models are included.

USD construction and workstation/recording utilities adapt MIT-licensed code from
the Ring Binder Insertion task; its copyright notice is retained in `LICENSE`.
The workstation, YAM robot and calibration remain external dependencies provided
by yam-arena. Fonts used to rasterize labels and video captions are Pillow's
bundled default font; no font files are redistributed.

Seasoning is a coarse rigid-grain approximation: 800 convex, faceted grains of circumscribed radius
1.25 mm and mass 0.0125 g each represent 10 g of pepper salt. Granular contacts,
gravity and the actual opening geometry determine the flow. Fine powder cohesion,
aerodynamic drag, grain fracture and food absorption are not modeled. The steak
is a rigid elliptical slab with a procedural cooked texture. Hinge friction and
material properties are prototype parameters, not measurements of a retail bottle.
