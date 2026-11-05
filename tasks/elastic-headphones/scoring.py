"""Final-state score: unfolding .4, each earcup .2, independent hanging .2."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

TABLE_TOP = .7475
JOINTS = ("left_fold", "right_fold", "left_swivel", "right_swivel")


class HeadphoneScore:
    def __init__(self, control_hz):
        self.window = deque(maxlen=5 * control_hz)

    def update(self, sample):
        root, stand = np.asarray(sample["root"]), np.asarray(sample["stand"])
        angles = sample["angles_deg"]
        links = np.asarray(sample["links"])
        assert root.shape == stand.shape == (7,) and links.shape == (15, 7)
        assert set(angles) == set(JOINTS)
        assert np.isfinite(np.r_[root, stand, links.ravel(), list(angles.values())]).all()
        rotation = Rotation.from_quat(stand[3:])
        local = rotation.inv().apply(root[:3] - stand[:3])
        tilt = float(np.degrees(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1))))
        # Dropping the headphones off the work surface invalidates all partial credit.
        on_workspace = bool(links[:, 2].min() >= TABLE_TOP - .03)
        checks = dict(
            unfolded=on_workspace and all(abs(angles[n]) <= 15 for n in JOINTS[:2]),
            left_cup_aligned=on_workspace and abs(angles["left_swivel"]) <= 15,
            right_cup_aligned=on_workspace and abs(angles["right_swivel"]) <= 15,
            crown_inside_cradle=bool(abs(local[0]) <= .024 and abs(local[1]) <= .017
                                     and .240 <= local[2] <= .265),
            stand_upright=bool(tilt <= 5 and abs(stand[2] - TABLE_TOP) <= .005),
            headphones_above_table=bool(links[:, 2].min() >= TABLE_TOP + .02),
            robots_released=sample["robot_contact_N"] < .05,
        )
        candidate = all(checks[name] for name in (
            "crown_inside_cradle", "stand_upright", "headphones_above_table", "robots_released"))
        self.window.append((root[:3].copy(), stand[:3].copy(), candidate, sample["support_force_N"]))
        root_drift = float(np.linalg.norm(np.ptp([x[0] for x in self.window], axis=0)))
        stand_drift = float(np.linalg.norm(np.ptp([x[1] for x in self.window], axis=0)))
        # Sleeping PhysX bodies can stop reporting force. Require observed support
        # within the same five-second window plus continuous geometry and release.
        hanging = (len(self.window) == self.window.maxlen and all(x[2] for x in self.window)
                   and max(x[3] for x in self.window) > .5 and root_drift <= .003 and stand_drift <= .002)
        checks["hanging_stably_5s"] = hanging
        components = dict(unfold=.4 * checks["unfolded"],
            left_earcup=.2 * checks["left_cup_aligned"],
            right_earcup=.2 * checks["right_cup_aligned"], hang=.2 * hanging)
        score = round(sum(components.values()), 1)
        return dict(time_s=sample["time_s"], score=score, success=score == 1., components=components,
            checks=checks, angles_deg=angles, crown_in_stand_m=local.tolist(), stand_tilt_deg=tilt,
            crown_drift_5s_m=root_drift, stand_drift_5s_m=stand_drift,
            support_force_N=sample["support_force_N"], robot_contact_N=sample["robot_contact_N"])


class FinalScore:
    name = "headphones_final_score"

    def __call__(self, record, target):
        from inspect_robots.scorer import Score
        info = record.steps[-1].result.info
        return Score(value=info["score"], explanation="Final state: unfold .4, each earcup .2, hang .2",
                     metadata={"components": info["components"], "checks": info["checks"]})
