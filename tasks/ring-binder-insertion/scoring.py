"""Geometric threading and the three user-requested scoring milestones."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import HOLES_X, HOLE_Y, HOLE_R, WIRE_R, CARD_T, PIVOT_Y, PIVOT_Z, ring_points, MAX_STEPS, HZ


def threading(rail_pose, card_pose):
    """Intersect each physical right half-ring with the plane of its matching hole."""
    rr = Rotation.from_quat(rail_pose[3:]).as_matrix()
    cr = Rotation.from_quat(card_pose[3:]).as_matrix()
    distances=[]
    for x in HOLES_X:
        world=(ring_points(-1)+[x,0,0])@rr.T+rail_pose[:3]
        local=(world-card_pose[:3])@cr
        found=[]
        for a,b in zip(local[:-1],local[1:]):
            if a[2]*b[2]<=0 and abs(b[2]-a[2])>1e-10:
                p=a+(b-a)*(-a[2]/(b[2]-a[2]))
                found.append(np.linalg.norm(p[:2]-[x,HOLE_Y]))
        distances.append(float(min(found)) if found else 1.)
    return np.asarray(distances)


class Score:
    def __init__(self):
        self.opened=False
        self.last=dict(score=0.,opened=False,inserted=False,closed=False,success=False,time_s=0.)

    def update(self, time_s, angles, rail_pose, card_pose, base_pose, card_contact):
        self.opened |= bool(angles[0]<-.45 and angles[1]>.45)
        distances=threading(rail_pose,card_pose)
        local=(card_pose[:3]-base_pose[:3])@Rotation.from_quat(base_pose[3:]).as_matrix()
        # The page must be released on the right, with all six holes actually threaded.
        inserted=bool(self.opened and np.all(distances < HOLE_R-WIRE_R)
                      and local[1]<0 and local[2]<.055 and card_contact<.1)
        tips=[Rotation.from_euler('x',angle).apply(ring_points(side)[-1])+[0,side*PIVOT_Y,PIVOT_Z]
              for side,angle in zip((1,-1),angles[:2])]
        tip_gap=float(np.linalg.norm(tips[0]-tips[1])-2*WIRE_R)
        closed=bool(inserted and tip_gap<CARD_T)
        score=.2*self.opened+.6*inserted+.2*closed
        if time_s > MAX_STEPS/HZ+1e-9:
            raise ValueError('Episode exceeds the shared execution limit')
        self.last=dict(time_s=time_s,score=round(score,2),opened=self.opened,
            inserted=inserted,closed=closed,success=closed,hole_errors_m=distances.tolist(),
            card_local_m=local.tolist(),card_robot_force_N=float(card_contact),tip_clearance_m=tip_gap)
        return self.last

    def result(self):
        return self.last.copy()
