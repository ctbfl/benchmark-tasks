"""YAM arm kinematics for the scripted demo: IK, Cartesian paths and arm-to-rod clearance (no simulator)."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp
from yam_arena import ik, station
from yam_arena.constants import READY

LO, UP = ik.LOWER, ik.UPPER
BASES = station.BASES
TABLE = station.TABLE_TOP
READY = np.array(READY, float)
SEEDS = [READY, [0, 1.6, 1.4, -.2, 0, 0], [.5, 1.2, 1.2, -.3, 0, 0], [-.5, 1.2, 1.2, -.3, 0, 0],
         [0, .8, 1.2, .4, 0, 0], [1., 1.5, 1.5, 0, .8, 0], [-1., 1.5, 1.5, 0, -.8, 0],
         [0, 2., 2., -.5, 0, 1.5], [0, 2., 2., -.5, 0, -1.5], [.8, 1., 1.5, .5, 1., 1.], [-.8, 1., 1.5, .5, -1., -1.]]


def unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v)


def frame(approach, closing):
    """Gripper rotation: X closing axis, Z approach axis."""
    a = unit(approach)
    c = np.asarray(closing, float)
    c = unit(c-a*(c@a))
    return np.column_stack((c, np.cross(a, c), a))


def fk(side, q):
    return ik.tcp_pose(np.asarray(q, float), BASES[side])


def solve(side, p, r, seed=None, all_seeds=False):
    """IK. Returns q or None. Prefers the solution closest to seed."""
    p, r = np.asarray(p, float), np.asarray(r, float)

    def res(q):
        pp, rr = fk(side, q)
        return np.r_[10*(pp-p), R.from_matrix(r@rr.T).as_rotvec()]
    best = None
    seeds = ([seed] if seed is not None else [])+([] if (seed is not None and not all_seeds) else SEEDS)
    for s in seeds:
        f = least_squares(res, np.clip(s, LO+1e-6, UP-1e-6), bounds=(LO, UP), max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        e = res(f.x)
        if np.linalg.norm(e[:3])/10 < 3e-4 and np.linalg.norm(e[3:]) < 3e-3:
            d = 0 if seed is None else np.abs(f.x-seed).max()
            if best is None or d < best[1]:
                best = (f.x, d)
            if seed is not None and s is seeds[0] and d < .5:
                break
    return None if best is None else best[0]


def interp(p0, r0, p1, r1, n):
    sl = Slerp([0, 1], R.from_matrix(np.array([r0, r1])))
    return [(p0+(p1-p0)*t, sl([t])[0].as_matrix()) for t in np.linspace(0, 1, n)]


def chain(side, poses, seed, jump=.5):
    """Continuous IK along poses; None if any step fails or jumps."""
    qs, q = [], np.asarray(seed, float)
    for p, r in poses:
        qn = solve(side, p, r, q)
        if qn is None or np.abs(qn-q).max() > jump:
            return None
        qs.append(qn)
        q = qn
    return np.array(qs)


def margin(q):
    return np.degrees(np.minimum(q-LO, UP-q))


# ---------------------------------------------------------------- arm collision points vs. the rod
_CLOUD = None


def _clouds():
    """Collision-mesh vertices of the distal links in their own frames (subsampled)."""
    global _CLOUD
    if _CLOUD is None:
        import glob, re, struct
        from pathlib import Path
        from yam_arena import YAM_URDF
        txt = open(YAM_URDF).read()
        root = Path(YAM_URDF).parent
        rng = np.random.default_rng(0)
        _CLOUD = {}
        for link in ('link3', 'link4', 'link5', 'gripper', 'tip_left', 'tip_right'):
            block = re.search(r'<link name="%s".*?</link>' % link, txt, re.S).group(0)
            pts = []
            for col in re.findall(r'<collision>.*?</collision>', block, re.S):
                xyz = np.array(re.search(r'origin xyz="([^"]+)"', col).group(1).split(), float)
                f = root/re.search(r'mesh filename="([^"]+)"', col).group(1)
                b = open(f, 'rb').read()
                n = struct.unpack('<I', b[80:84])[0]
                v = np.frombuffer(b[84:84+50*n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))['v'].reshape(-1, 3)
                pts.append(np.unique(np.round(v, 3), axis=0)+xyz)
            p = np.vstack(pts)
            _CLOUD[link] = p[rng.choice(len(p), min(400, len(p)), replace=False)]
    return _CLOUD


def arm_points(side, q, grip=.35):
    from yam_arena import YAM_URDF
    from yam_arena.urdf import forward_kinematics
    from yam_arena.constants import ARM_JOINTS
    js = dict(zip(ARM_JOINTS, np.asarray(q, float)))
    js['joint7'] = js['joint8'] = -.04695*grip
    T = forward_kinematics(YAM_URDF, js)
    base = np.asarray(BASES[side])
    return np.vstack([c@T[k][:3, :3].T+T[k][:3, 3]+base for k, c in _clouds().items()])


def rod_clearance(side, q, tip, u, length=.781, grip=.35):
    p = arm_points(side, q, grip)-tip
    s = np.clip(p@u, 0, length)
    return float(np.linalg.norm(p-s[:, None]*u, axis=1).min())
