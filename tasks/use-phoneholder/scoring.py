"""Shared final-state scoring for the model, demonstration and saved poses."""

import numpy as np
from scipy.spatial.transform import Rotation

MAX_SECONDS = 180


def conditions(m):
    return {
        "phone_placed_on_holder": bool(m["tray_contact"] and m["phone_released"]),
        "phone_angle_at_least_45": bool(m["phone_angle_deg"] >= 45),
        "phone_bottom_above_3cm": bool(m["phone_bottom_height_m"] > .03),
        "holder_yaw_within_30": bool(abs(m["holder_heading_error_deg"]) <= 30),
        "phone_yaw_within_30": bool(abs(m["phone_heading_error_deg"]) <= 30),
        "within_deadline": bool(0 <= m["time_s"] <= MAX_SECONDS),
    }


def score_placement(row, surface, table_height, time_s):
    pp, cp = np.asarray(row["phone_pos"]), np.asarray(row["cradle_pos"])
    pr = Rotation.from_quat(row["phone_quat"]).as_matrix()
    cr = Rotation.from_quat(row["cradle_quat"]).as_matrix()
    world = surface @ pr.T + pp
    local = (world - cp) @ cr
    tray_distance = np.inf
    # Actual tray and pad collision boxes from phoneholder.usda. Geometric
    # contact remains observable when native sleep makes force reports zero.
    for x in (-.020, .020):
        for center, half_size in (
            ([x, -.019, .010125], [.010, .001, .006125]),
            ([x, -.0178, .00975], [.009, .0002, .00525]),
        ):
            distance = np.linalg.norm(np.maximum(np.abs(local - center) - half_size, 0), axis=1).min()
            tray_distance = min(tray_distance, float(distance))
    measurements = {
        "time_s": time_s,
        "phone_angle_deg": float(np.degrees(np.arccos(np.clip(abs(pr[2, 2]), 0, 1)))),
        "phone_bottom_height_m": float(world[:, 2].min() - table_height),
        "holder_heading_error_deg": row["holder_heading_error_deg"],
        "phone_heading_error_deg": row["phone_heading_error_deg"],
        "tray_distance_m": tray_distance,
        "tray_contact": tray_distance <= .00030,  # Sum of both assets' 0.15 mm contact skins.
        "phone_released": not np.any(np.asarray(row["phone_contacts_N"])[2:]),
    }
    checks = conditions(measurements)
    return {**measurements, "checks": checks, "success": all(checks.values()),
            "score": int(all(checks.values()))}
