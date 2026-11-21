"""Register and score a flattened sheet using one rigid in-plane transform."""
from functools import cache, lru_cache

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from scipy.spatial.transform import Rotation

from geometry import BLOCK, BLOCK_MASK, BLOCK_SIZE, PANEL_GRID, PANELS, PITCH, THICKNESS

PRINT_MARK = .14
INKED_BLOCK = .30
SHEET_SHAPE = (PANEL_GRID.shape[0], PANEL_GRID.shape[1]*PANELS)
CELL_MM = round(PITCH * 1000)
PADDING = max(BLOCK_MASK.shape)
EDGE_ALLOWANCE = np.ones((3, 3), dtype=bool)
TARGET_PIXELS = int(BLOCK_MASK.sum())


@cache
def templates():
    """Sample the rotated carve at one-degree and one-millimetre registration steps."""
    coords = np.arange(-PADDING, PADDING + 1) * CELL_MM
    x, y = np.meshgrid(coords, coords, indexing="ij")
    rasters = []
    centre = (np.asarray(BLOCK_MASK.shape)-1)/2
    for angle in range(360):
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            u = np.round(c * (x + dx) + s * (y + dy), 8)
            v = np.round(-s * (x + dx) + c * (y + dy), 8)
            mask = map_coordinates(BLOCK_MASK,
                                   [u/CELL_MM+centre[0], v/CELL_MM+centre[1]],
                                   order=0, mode="grid-constant")
            i, j = np.where(mask)
            lo = (i.min() - 1, j.min() - 1)
            roi = mask[lo[0]:i.max()+2, lo[1]:j.max()+2]
            origin = coords[list(lo)] + [dx, dy]
            rasters.append((angle, origin, int(roi.sum()), roi.astype(np.uint8),
                            binary_dilation(roi, structure=EDGE_ALLOWANCE)))
    return rasters


@lru_cache(maxsize=64)
def register(marked_bytes):
    marked = np.frombuffer(marked_bytes, dtype=np.uint8).reshape(SHEET_SHAPE)
    marked_count = int(marked.sum())
    if not marked_count:
        return 0., 0., 0, None
    marked = np.pad(marked, PADDING)
    best = (-1., 0., 0)
    registration = None
    for angle, origin, area, target, allowance in templates():
        overlap = cv2.matchTemplate(marked, target, cv2.TM_CCORR)
        hits = round(cv2.minMaxLoc(overlap)[1])
        coverage = hits / area
        if coverage < best[0]:
            continue
        # For equal coverage, choose the single placement that leaves the least total ink outside.
        for i, j in np.argwhere(overlap > hits - .5):
            region = marked[i:i+target.shape[0], j:j+target.shape[1]]
            outside = marked_count - int(np.count_nonzero(region & allowance))
            key = (coverage, -outside/area, hits)
            if key > best:
                best = key
                offset = (np.array([i, j])-PADDING)*CELL_MM - origin
                registration = (angle, int(offset[0]), int(offset[1]), outside)

    angle, x, y, outside = registration
    return best[0], abs(best[1]), outside, (angle, x, y)


def print_metrics(paper):
    marked = np.concatenate(paper, axis=1) >= PRINT_MARK
    coverage, ghost, outside, registration = register(marked.tobytes())
    return {
        "coverage": coverage,
        "ghost_ratio": ghost,
        "marked_area_mm2": int(marked.sum()) * CELL_MM**2,
        "ghost_area_mm2": outside * CELL_MM**2,
        "registration": registration,
    }


def grade(block, paper, clear, ever_inked=False, *, print_eligible):
    metrics = print_metrics(paper)
    inked = float((block[BLOCK_MASK] >= INKED_BLOCK).mean())
    coverage, ghost = metrics["coverage"], metrics["ghost_ratio"]
    if print_eligible and clear and coverage >= .90 and ghost <= .04:
        score = 1.
    elif print_eligible and clear and coverage >= .65 and ghost <= .18:
        score = .65
    elif print_eligible and coverage >= .45 and ghost <= .18:
        score = .4
    elif ever_inked or inked >= .60:
        score = .2
    else:
        score = 0.
    return {"score": score, "block_coverage": inked, "sheet_clear": bool(clear),
            "print_eligible": bool(print_eligible), **metrics}


def paper_over_relief(panels, table_z):
    """Check whether any part of the sheet still needs to clear the relief."""
    local = np.c_[PANEL_GRID.reshape(-1, 2), np.full(PANEL_GRID.size//2, -THICKNESS/2)]
    points = np.stack([Rotation.from_quat(p[3:]).apply(local) + p[:3] for p in panels])
    over = np.all(abs(points[..., :2] - BLOCK[:2]) < np.asarray(BLOCK_SIZE)/2 + .002, axis=2)
    low = points[..., 2] < table_z + BLOCK[2] + .008
    return bool((over & low).any())
