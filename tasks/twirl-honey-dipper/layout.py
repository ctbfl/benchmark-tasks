"""Dimensions of the fixed tabletop layout, in metres."""
from yam_arena.station import TABLE_TOP

JAR = (.490, -.265)
JAR_INNER = .043
JAR_RISER = .062
JAR_SURFACE = TABLE_TOP + JAR_RISER + .069
JAR_RIM = TABLE_TOP + JAR_RISER + .078
TOAST = (.490, -.020)
TOAST_TOP = TABLE_TOP + .021
TOAST_OUTLINE = (
    (-.050, -.043), (.027, -.043), (.046, -.047), (.057, -.041),
    (.064, -.026), (.067, -.010), (.067, .010), (.064, .026),
    (.057, .041), (.046, .047), (.027, .043), (-.050, .043),
    (-.056, .037), (-.056, -.037),
)
PLATE_R = .077
PLATE_TOP = TABLE_TOP + .006
DIPPER = (JAR[0] - .124, JAR[1], TABLE_TOP + JAR_RISER + .079)
GROOVES = (.102, .111, .120, .129)
STEM_RADIUS = .0075
GROOVE_RADIUS = .0093
GROOVE_WIDTH = .0058
GRIP_X = .002
