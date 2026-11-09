"""Final resting seasoning mass on the steak, measured in grams."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import GRAIN_RADIUS, GRAIN_MASS_G, GRAIN_COUNT, STEAK_AXES, STEAK_HEIGHT


def score(grains, steak, bottle):
    assert grains.shape == (GRAIN_COUNT, 13) and np.isfinite(grains).all()
    rotation = Rotation.from_quat(steak[3:7])
    local = rotation.inv().apply(grains[:,:3]-steak[:3])
    relative_speed = np.linalg.norm(grains[:,7:10]-steak[7:10],axis=1)
    footprint = np.square(local[:,:2]/STEAK_AXES).sum(axis=1) <= 1
    # Geometric contact tolerance; airborne grains do not count as deposited mass.
    supported = footprint & (abs(local[:,2]-STEAK_HEIGHT-GRAIN_RADIUS)<.0006)
    supported &= relative_speed < .04
    # Include resting grains supported by another grain, rather than the meat directly.
    candidates = footprint & (local[:,2]>STEAK_HEIGHT) & (relative_speed<.04)
    indices = np.flatnonzero(candidates)
    points = local[indices]
    distance = np.linalg.norm(points[:,None]-points[None,:],axis=2)
    below = points[None,:,2] < points[:,None,2]-.0001
    touching = (distance<2*GRAIN_RADIUS+.0004) & below
    resting = supported[indices]
    for _ in indices:
        grown = resting | np.any(touching & resting[None,:],axis=1)
        if np.array_equal(grown,resting): break
        resting = grown
    supported[indices] = resting
    mass = round(int(supported.sum())*GRAIN_MASS_G,6)
    local_bottle = Rotation.from_quat(bottle[3:7]).inv().apply(grains[:,:3]-bottle[:3])
    in_bottle = ((abs(local_bottle[:,0])<.022) & (abs(local_bottle[:,1])<.030)
                 & (local_bottle[:,2]>0) & (local_bottle[:,2]<.112))
    seasoned, safe = mass>0, 0<mass<=2.
    return dict(score=.5*seasoned+.5*safe, success=bool(seasoned and safe),
        seasoned=bool(seasoned),dose_safe=bool(safe),steak_mass_g=mass,
        steak_grains=int(supported.sum()),bottle_mass_g=float(in_bottle.sum()*GRAIN_MASS_G),
        total_mass_g=GRAIN_COUNT*GRAIN_MASS_G)
