"""Builds the scripted demo's 14-D action sequence (joint targets and grippers at 20 Hz)."""
import numpy as np
from demo_kinematics import *

HZ = 20


def smooth(t):
    return t*t*(3-2*t)


class Seq:
    def __init__(self):
        self.q = {s: READY.copy() for s in ('left', 'right')}          # ideal (uncompensated) joints
        self.off = {s: np.zeros(6) for s in ('left', 'right')}         # current command offset
        self.g = dict(left=1., right=1.)
        self.pose = {s: fk(s, READY) for s in ('left', 'right')}
        self.rows, self.marks = [], []

    @property
    def t(self):
        return len(self.rows)/HZ

    def emit(self):
        self.rows.append(np.r_[self.q['left']+self.off['left'], self.g['left'],
                               self.q['right']+self.off['right'], self.g['right']])

    def mark(self, label, side=None):
        """Record the end of a named segment (for logs and timing)."""
        self.marks.append(dict(label=label, step=len(self.rows), t=self.t, side=side,
                               q=None if side is None else self.q[side].tolist(),
                               off=None if side is None else self.off[side].tolist()))

    def hold(self, seconds, label=None):
        for _ in range(round(seconds*HZ)):
            self.emit()
        if label:
            self.mark(label)

    def grip(self, side, value, seconds, label=None):
        g0, n = self.g[side], max(1, round(seconds*HZ))
        for i in range(n):
            self.g[side] = g0+(value-g0)*(i+1)/n
            self.emit()
        if label:
            self.mark(label)

    def _run(self, plans, seconds, grips=None):
        """plans: {side: joint array (n, 6) incl. offsets handled by caller}."""
        n = max(2, round(seconds*HZ))
        g0 = dict(self.g)
        for i in range(n):
            s = smooth((i+1)/n)
            for side, (qs, off0, off1) in plans.items():
                k = s*(len(qs)-1)
                a = min(int(k), len(qs)-2)
                self.q[side] = qs[a]+(qs[a+1]-qs[a])*(k-a)
                self.off[side] = off0+(off1-off0)*s
            for side, v in (grips or {}).items():
                self.g[side] = g0[side]+(v-g0[side])*s
            self.emit()

    def _plan(self, side, waypoints, label, joint=False):
        """Cartesian path through (p, R) waypoints -> dense joint path."""
        p0, r0 = self.pose[side]
        qs = [self.q[side]]
        for p1, r1 in waypoints:
            p1 = np.asarray(p1, float)
            r1 = r0 if r1 is None else np.asarray(r1, float)
            if joint:
                q1 = solve(side, p1, r1, qs[-1], all_seeds=True)
                assert q1 is not None, (label, side, 'unreachable', p1)
                qs += list(qs[-1]+(q1-qs[-1])*np.linspace(0, 1, 13)[1:, None])
            else:
                ang = np.linalg.norm(R.from_matrix(r1@r0.T).as_rotvec())
                n = int(max(4, np.linalg.norm(p1-p0)/.008, ang/np.radians(3)))
                c = chain(side, interp(p0, r0, p1, r1, n+1)[1:], qs[-1])
                assert c is not None, (label, side, 'no continuous path to', p1.round(4).tolist())
                qs += list(c)
            p0, r0 = p1, r1
        self.pose[side] = (p0, r0)
        off1 = self.off[side].copy()
        return np.array(qs), self.off[side].copy(), off1

    def move(self, side, p, r=None, seconds=2., label=None, joint=False, grip=None, via=()):
        label = label or f'm{len(self.marks)}'
        self._run({side: self._plan(side, [*via, (p, r)], label, joint)}, seconds,
                  None if grip is None else {side: grip})
        self.mark(label, side)

    def rel(self, side, d, seconds=2., label=None, r=None):
        self.move(side, self.pose[side][0]+np.asarray(d, float), r, seconds, label)

    def both(self, left, right, seconds, label):
        """left/right: (p, R) or list of waypoints."""
        plans = {}
        for side, w in (('left', left), ('right', right)):
            if w is not None:
                w = w if isinstance(w, list) else [w]
                plans[side] = self._plan(side, w, label+'/'+side)
        self._run(plans, seconds)
        for side in plans:
            self.mark(label+'/'+side, side)

    def joints(self, side, q1, seconds, label=None):
        q1 = np.asarray(q1, float)
        self._run({side: (np.array([self.q[side], q1]), self.off[side].copy(),
                          np.zeros(6))}, seconds)
        self.pose[side] = fk(side, q1)
        self.mark(label or f'j{len(self.marks)}', side)

    def transit(self, side, p, r, seconds, label, rod, clearance=.02, grip=None, q1=None, away=None):
        """Joint-space move that keeps the arm ``clearance`` away from the rod (tip, u); adds a retreat
        waypoint when the direct move would pass too close."""
        import demo_kinematics as kin
        tip, u = rod
        if q1 is None:
            cands = [solve(side, p, r, self.q[side], all_seeds=True)]+[solve(side, p, r, np.asarray(sd, float)) for sd in SEEDS]
            cands = [c for c in cands if c is not None]
        else:
            cands = [np.asarray(q1, float)]
        assert cands, (label, 'target unreachable')
        g = self.g[side] if grip is None else max(grip, self.g[side])
        def clear(qa, qb, n=16):
            return min(kin.rod_clearance(side, qa+(qb-qa)*k/n, tip, u, grip=g) for k in range(n+1))
        q0 = self.q[side]
        p0, r0 = self.pose[side]
        extra = [] if away is None else [p0+away*k+np.array(v) for k in (.04, .07) for v in ((0, 0, 0), (-.05, 0, -.05), (0, 0, -.06))]
        vias = [None]+extra+[p0+np.array(v) for v in ((-.06, 0, -.06), (-.08, 0, 0), (0, 0, -.08), (-.06, -.06, -.04),
                                               (-.10, 0, -.10), (0, -.08, -.04), (-.06, 0, .06))]
        best = None
        jvias = [q0+np.eye(6)[j]*dv for j, dv in ((1, -.4), (2, .4), (1, -.6), (2, .6), (1, -.25), (2, .25))]
        c0_ = kin.rod_clearance(side, q0, tip, u, grip=g)
        for via in vias+jvias:
            for qe in cands:
                if via is None:
                    c = clear(q0, qe)
                    if c > clearance and (best is None or c > best[0]):
                        best = (c, None, qe)
                    continue
                if len(via) == 6:
                    qv = np.clip(via, LO, UP)
                else:
                    qv = solve(side, via, r0, q0)
                    if qv is None or np.abs(qv-q0).max() > .6:
                        continue
                if clear(q0, qv, 6) < min(c0_, clearance)-.002:   # the retreat leg must not get closer than the start
                    continue
                c = clear(qv, qe)
                if c > clearance and (best is None or c > best[0]):
                    best = (c, qv, qe)
            if best is not None:
                break
        assert best is not None, (label, 'no rod-clear transit')
        _, qv, qe = best
        if qv is not None:
            self.follow(side, np.array([qv]), seconds*.4, label+' (retreat)')
            seconds *= .7
        n = max(2, round(seconds*HZ))
        self._run({side: (np.array([self.q[side], qe]), self.off[side].copy(), self.off[side].copy())}, seconds,
                  None if grip is None else {side: grip})
        self.pose[side] = fk(side, qe)
        self.mark(label, side)

    def follow(self, side, qs, seconds, label):
        """Execute a prepared joint path (n, 6)."""
        qs = np.vstack([self.q[side], qs])
        self._run({side: (qs, self.off[side].copy(), self.off[side].copy())}, seconds)
        self.pose[side] = fk(side, qs[-1])
        self.mark(label, side)
