# Asset and dependency provenance

- The circuit and rheostat USD geometry, seven-segment digits and label textures
  are original procedural assets distributed under the accompanying MIT license.
- Labels are rendered with DejaVu Sans. No font file is redistributed. DejaVu's
  Bitstream Vera license permits commercial use and rendering; its changes are
  public domain. `assets.py` regenerates labels using the system DejaVu font.
- The YAM robot, workstation and calibrated camera rig are loaded from the
  separately installed `yam-arena` dependency, not copied into this task. The
  robot meshes originate from I2RT under MIT; the dependency includes
  `assets/yam/LICENSE.i2rt` and source attribution for its Dream appearance meshes.
- Isaac Sim, Isaac Lab, Arena, Inspect Robots and the Dream SDK remain external
  dependencies under their own licenses. This package redistributes none of their
  code, models or credentials.
