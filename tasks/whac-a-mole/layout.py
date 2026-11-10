"""Fixed device dimensions and layout in metres (no simulator imports)."""

BASE_SIZE = (.220, .260, .040)          # x (depth, away from robot), y (width), height
BUTTON_RADIUS = .024
BUTTON_HEIGHT = .012
BUTTON_GAP = .010                         # cap bottom above the base top at rest
BUTTON_TRAVEL = .008
BUTTON_STIFFNESS = 400.0                  # N/m
BUTTON_DAMPING = 2.0                      # N s/m
# Buttons in device coordinates (x away from the robot, y to the robot's left).
BUTTONS = {
    "near_left": (-.055, .070),
    "near_right": (-.055, -.070),
    "far_left": (.035, .070),
    "far_right": (.035, -.070),
}
DISPLAY_CENTER = (.085, 0.)
SEGMENTS = {"0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
            "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd", " ": ""}
