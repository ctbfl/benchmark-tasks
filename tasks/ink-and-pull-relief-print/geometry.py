"""Nominal dimensions in metres, relative to the tabletop."""
import numpy as np

PITCH = .002
AXLE_R = .019
AXLE_W = .076
AXLE_HOME = (.227, .211, .026)
HANDLE = (-.071, 0, .043)
PALETTE = (.300, .211, .007)
PALETTE_SIZE = (.302, .102)
BLOCK = (.318, -.035, .026)
BLOCK_SIZE = (.096, .066)
RELIEF = .004
# Rectangle union in the X-Y plane, millimetres from the centre. The gaps are carved.
# An asymmetric shoreline, lighthouse and small boat compose the printable image.
CUT = (
    (-43, 40, -29, -23), (-43, -13, -23, -18),
    (-31, -19, -18, 12), (-35, -15, 12, 19), (-31, -19, 19, 24),
    (-10, 28, 13, 17), (-6, 18, 5, 9),
    (2, 34, -12, -7), (6, 30, -17, -12), (15, 19, -7, 4),
)
SHEET_HOME = (.318, -.227, .040)
SHEET_SIZE = (.124, .102)
THICKNESS = .0016
PANELS = 3
TAB = (-.046, -.005, .012)
PAD_TOP = SHEET_HOME[2] - THICKNESS / 2


def grid(size):
    axes = [np.arange(round(s / PITCH)) * PITCH - s / 2 + PITCH / 2 for s in size]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


PALETTE_GRID = grid(PALETTE_SIZE)
BLOCK_GRID = grid(BLOCK_SIZE)
PANEL_SIZE = (SHEET_SIZE[0], SHEET_SIZE[1] / PANELS)
PANEL_GRID = grid(PANEL_SIZE)
BLOCK_MASK = np.zeros(BLOCK_GRID.shape[:2], dtype=bool)
for x0, x1, y0, y1 in CUT:
    x, y = np.moveaxis(BLOCK_GRID * 1000, -1, 0)
    BLOCK_MASK |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
