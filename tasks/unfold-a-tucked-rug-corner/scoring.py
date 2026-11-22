"""Grade a fully unfolded rug, with full credit only inside the tape's inner edge."""
import numpy as np

LENGTH, WIDTH, THICKNESS = .60, .40, .005
CENTER_X = .45
FRAME_MARGIN = .02
FRAME_MIN = np.array([CENTER_X - LENGTH / 2 - FRAME_MARGIN, -WIDTH / 2 - FRAME_MARGIN])
FRAME_MAX = np.array([CENTER_X + LENGTH / 2 + FRAME_MARGIN, WIDTH / 2 + FRAME_MARGIN])
TAPE_WIDTH, TAPE_THICKNESS = .018, .0002
HEIGHT_TOLERANCE = .002
MIN_PROJECTED_AREA_RATIO = .99


def grade(pile, backing, quads, support_z):
    """Read current world-space surface vertices; do not retain an earlier best score."""
    pile, backing = np.asarray(pile, dtype=float), np.asarray(backing, dtype=float)
    quads = np.asarray(quads, dtype=int)
    if pile.ndim != 2 or pile.shape[1] != 3 or pile.shape != backing.shape or not len(pile):
        raise ValueError("Expected matching nonempty N x 3 pile and backing vertices")
    if quads.ndim != 2 or quads.shape[1] not in (3, 4) or not len(quads):
        raise ValueError("Expected nonempty material triangles or quads with original pile-side winding")
    if quads.min() < 0 or quads.max() >= len(pile):
        raise ValueError("Material face index is outside the vertex array")
    points = np.concatenate((pile, backing))
    if not np.isfinite(points).all() or not np.isfinite(support_z):
        raise ValueError("Geometry and support height must be finite")
    faces = np.concatenate((quads[:, :3], quads[:, [0, 2, 3]])) if quads.shape[1] == 4 else quads
    a, b, c = np.moveaxis(pile[faces], 1, 0)
    # Signed area cancels folded-under material instead of counting overlapping layers twice.
    area = .5 * np.cross(b - a, c - a)[:, 2].sum()
    area_ratio = float(abs(area) / (LENGTH * WIDTH))
    heights = points[:, 2] - support_z
    flat = bool(heights.min() >= -HEIGHT_TOLERANCE
                and heights.max() <= THICKNESS + HEIGHT_TOLERANCE
                and area_ratio >= MIN_PROJECTED_AREA_RATIO)
    overflow = float(np.maximum(np.maximum(FRAME_MIN - points[:, :2],
                                          points[:, :2] - FRAME_MAX), 0.).max())
    inside = overflow <= 1e-7
    return {"score": (1. if inside else .5) if flat else 0.,
            "flat": flat, "inside_frame": bool(inside),
            "projected_area_ratio": area_ratio,
            "maximum_surface_height_mm": float(heights.max() * 1000),
            "outside_inner_edge_mm": overflow * 1000}
