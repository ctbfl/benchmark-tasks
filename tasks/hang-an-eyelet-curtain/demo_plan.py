"""Scripted demo plan. Nominal poses below were measured from the fixed initial layout; when a simulator
`sense` callback is given, the handoff and the second ring use measured ring and rod poses."""
import sys
import numpy as np
from demo_kinematics import *
from demo_sequence import Seq

Z = np.array([0, 0, 1.])
Y = np.array([0, 1., 0])
RI, RO = .015, .018
RIM = (RI+RO)/2
R1 = np.array([.25, .051, TABLE+.0020])       # ring 1 (left) centre on the table
DOWN_Y = frame(-Z, [0, -1, 0])                # top-down, closing along y
# Presentation of ring 1 by the left hand. The ring leans back (front normal tilted up by LEAN) so the
# unsupported fabric corner with ring 0 droops to one definite side; the top edge is nearly level.
LEAN, EPS = np.radians(20), np.radians(5)
_lean = R.from_euler('x', LEAN).as_matrix()
E1 = _lean@[np.cos(EPS), 0, np.sin(EPS)]      # ring-1 centre -> left pinch point (far side)
XC = _lean@[-np.sin(EPS), 0, np.cos(EPS)]     # fabric top-edge normal
N1 = _lean@[0, 1., 0]                         # ring front normal
PRESENT = np.array([.24, -.01, .975])         # left TCP while presenting
PLANE = .012                                  # ring plane lies this far beyond the left TCP
BETA = np.radians(-25)                       # right pinch point: 25 deg from the top-edge normal towards ring 0
PIVOT = np.array([.3615, .599, .8525])         # rod axis inside the left cradle
ROD_TIP0 = np.array([.3615, -.131, .8525])
GRASP_FROM_TIP = .38
THETA, PSI = 11.6, 10.                        # raised rod: pitch and yaw about the left cradle
TIP = np.array([.2364, -.1047, .9880])         # measured raised rod tip and inward axis
U = unit([.1694, .9681, -.1849])
HELD1 = np.array([-.0012, .0001, .0245])      # measured ring-1 centre in the right gripper frame
S1 = .052                                     # ring-1 centre this far inside the tip when released
C0 = np.array([.184, -.058, .913])            # measured hanging ring 0: centre, normal, fabric top-edge normal
N0 = unit([-.80, .55, .23])
T0 = unit([-.38, -.75, .54])
PHI0 = np.radians(32)                         # pinch point, measured round the rim from the top edge towards ring 1
TAU0 = np.radians(75)                         # where that pinch point ends up on the rod, from up towards the robot
TR0 = .024
S0 = .13                                      # ring 0 is pushed this far inside the tip
PRES_C = np.array([.2249, -.0229, .9694])     # measured presented ring-1 centre and normal
PRES_N = unit([-.047, .968, .246])
TCP_R = .024                                  # right TCP distance from the ring centre


def build(stage=99, sense=None):
    s = Seq()
    s.sense = sense
    s.hold(.5)
    # ---- 1. left hand pinches the left rim of ring 1 from above and lifts it
    rim = R1+[0, RIM, 0]
    s.grip('left', .20, .5)
    s.move('left', rim+[0, 0, .08], DOWN_Y, 2.5, 'L above ring1')
    s.move('left', rim+[0, 0, .012], None, 2., 'L at ring1 rim')
    s.hold(.3)
    s.grip('left', 0., 1.2)
    s.hold(.4, 'L pinched')
    s.rel('left', [0, 0, .10], 2.5, 'L lifted')
    if stage <= 1:
        s.hold(1.)
        return s
    # ---- 2. left presents the ring: gripper points right, ring vertical, fabric hangs freely
    lift = s.pose['left'][0]
    s.move('left', lift, frame(-Z, [-1, 0, 0]), 2.5, 'L wrist turned')
    s.move('left', PRESENT, frame(-N1, -E1), 4., 'L presents')
    s.hold(1.5, 'L presents settle')
    if stage <= 2:
        return s
    # ---- 3. right hand pinches both faces of the ring near the fabric top edge; left lets go
    centre, normal = (PRES_C, PRES_N) if PRES_C is not None else (PRESENT-E1*RIM-N1*PLANE, N1)
    xc = unit(XC-normal*(XC@normal)); e1 = np.cross(xc, normal)
    d = np.cos(BETA)*xc-np.sin(BETA)*e1              # ring centre -> right pinch point
    rr = frame(-d, -normal)
    grasp = centre+d*TCP_R
    s.grip('right', .22, .5)
    s.move('right', grasp+d*.05, rr, 3.5, 'R before ring1')
    if s.sense is not None:                          # aim at the measured ring
        m = s.sense()
        c, n = np.asarray(m['c1']), np.asarray(m['n1'])
        if n@normal < 0:
            n = -n
        xc = unit(XC-n*(XC@n)); e1 = np.cross(xc, n)
        d = np.cos(BETA)*xc-np.sin(BETA)*e1
        rr = frame(-d, -n)
        grasp = c+d*TCP_R
        s.move('right', grasp+d*.02, rr, 1.5, 'R re-aimed at ring1')
    s.move('right', grasp, rr, 2., 'R at ring1')
    s.hold(.3)
    if s.sense is not None:                          # cancel the measured tracking error before closing
        for _ in range(2):
            m = s.sense()
            p, r = fk('right', m['qr'])
            c, n = np.asarray(m['c1']), np.asarray(m['n1'])
            if n@normal < 0:
                n = -n
            xc = unit(XC-n*(XC@n)); e1 = np.cross(xc, n)
            dd = np.cos(BETA)*xc-np.sin(BETA)*e1
            want = c+dd*TCP_R
            cp, cr = s.pose['right']
            s.move('right', cp+want-p, frame(-dd, -n)@r.T@cr, .6, 'R corrected at ring1')
            s.hold(.3)
    s.grip('right', 0., 1.2)
    s.hold(.5, 'R pinched ring1')
    s.grip('left', .30, .8)
    s.hold(.5, 'L released')
    s.rel('left', N1*.07, 2., 'L withdrawn')
    s.hold(1.0, 'R holds ring1')
    if stage <= 3:
        return s
    # ---- 4. left hand grips the rod and raises its right end out of the cradle, pivoting on the left cradle;
    #         meanwhile the right hand carries the ring to the right of the rod tip
    grip_at = PIVOT+(ROD_TIP0-PIVOT)*(1-GRASP_FROM_TIP/np.linalg.norm(ROD_TIP0-PIVOT))
    a = unit([np.sin(np.radians(20)), 0, -np.cos(np.radians(20))])
    rg = frame(a, np.cross([0, 1, 0], a))          # closing axis crosses the rod
    uh = unit(U*[1, 1, 0])
    ri = R.from_rotvec(np.cross(normal, uh)/np.linalg.norm(np.cross(normal, uh))
                       * np.arccos(normal@uh)).as_matrix()@rr     # ring upright, facing along the raised rod
    held = lambda c, r=ri: c-r@HELD1               # right TCP for a ring-1 centre
    lift_off = -ri[:, 2]*.003                      # hole centre rides 3 mm towards the fingers
    s.both((grip_at-a*.07, rg), (held(TIP-U*.05+lift_off), ri), 4., 'to rod / ring1 beside tip')
    s.move('left', grip_at, None, 2., 'L at rod')
    s.grip('left', 0., 1.2)
    s.hold(.4, 'L grips rod')
    def rod_pose(theta, psi):
        rot = (R.from_euler('z', -psi, degrees=True)*R.from_euler('x', -theta, degrees=True)).as_matrix()
        return PIVOT+rot@(grip_at-PIVOT), rot@rg
    s.move('left', *rod_pose(6, 0), 2.5, 'L rod out of cradle')
    s.move('left', *rod_pose(THETA, PSI), 4., 'L rod raised', via=[rod_pose(.5*(6+THETA), .5*PSI)])
    s.hold(1., 'L rod raised settle')
    if stage <= 4:
        return s
    # ---- 5. right hand threads ring 1 along the rod, past the knob, and lets go
    s.move('right', held(TIP-U*.03+lift_off), None, 1.5, 'R ring1 at tip')
    s.hold(.5, 'R ring1 at tip settle')
    s.move('right', held(TIP+U*S1+lift_off), None, 4., 'R ring1 past knob')
    s.hold(.5, 'R ring1 past knob settle')
    s.grip('right', .25, .8)
    s.hold(.5, 'R released ring1')
    s.rel('right', -ri[:, 2]*.018, 1., 'R withdrawn')
    s.hold(3., 'ring1 hangs')
    if stage <= 5:
        return s
    # ---- 6-7. right hand pinches the hanging ring 0, turns it to face opposite to ring 1 while carrying
    #           it round the rod tip, and threads it. Ring and rod poses come from the simulator when live.
    m = s.sense() if s.sense is not None else dict(c0=C0, n0=N0, t0=T0, tip=TIP, u=U, c1=None)
    import copy
    chosen = None
    for sigma, phi, tau, bk in CANDIDATES:
        trial = copy.deepcopy(s)
        trial.sense = None
        try:
            second_ring(trial, m, sigma, phi, tau, stage, bk)
        except AssertionError:
            continue
        chosen = (sigma, phi, tau, bk)
        break
    assert chosen is not None, 'no reachable second-ring plan'
    s.marks.append(dict(label='second ring plan %s' % (chosen,), step=len(s.rows), t=s.t, side=None, q=None, off=None))
    second_ring(s, m, *chosen[:3], stage, chosen[3])
    if stage <= 7:
        return s
    # ---- 8. left hand returns the rod to both cradles and lets go
    s.transit('right', None, None, 3.5, 'R home', getattr(s, 'rod', (TIP, U)), q1=READY, away=getattr(s, 'away', None))
    s.move('left', *rod_pose(6, 0), 5., 'L rod above cradle', via=[rod_pose(.5*(6+THETA), .5*PSI)])
    s.move('left', *rod_pose(0, 0), 3., 'L rod seated')
    s.hold(.5)
    s.grip('left', .35, 1.)
    s.rel('left', -rg[:, 2]*.08, 2., 'L clear of rod')
    s.joints('left', READY, 3., 'L home')
    s.hold(3., 'final')
    return s


CANDIDATES = [(sg, ph, ta, bk) for bk in (.06, .03) for ph in (0, 30, -30, 60, 90, 120, -60, 150, -90, 180) for ta in (75, 65, 85, 55) for sg in (1, -1)]


def second_ring(s, m, sigma, phi, tau, stage, back=.06):
    c0, n0, t0, tip, u = (np.asarray(m[k], float) for k in ('c0', 'n0', 't0', 'tip', 'u'))
    live = s.sense is not None

    def pinch(c, n, t, sg):
        t = unit(t-n*(t@n))
        w = np.cross(n, t)
        if m.get('c1') is not None and w@(np.asarray(m['c1'])-c) < 0:
            w = -w                                     # rotate the pinch point towards ring 1
        d = np.cos(np.radians(phi))*t+np.sin(np.radians(phi))*w
        return d, frame(-d, -sg*n)

    def update(sg):
        mm = s.sense()
        c, n, t = (np.asarray(mm[k], float) for k in ('c0', 'n0', 't0'))
        if (-sg*n)@s.pose['right'][1][:, 0] < 0:      # ring turned past 90 deg: pair the other face
            sg = -sg
        return c, n, t, sg, mm

    d0, rs = pinch(c0, n0, t0, sigma)
    s.rod = (tip, u)
    s.transit('right', c0+d0*(TR0+back), rs, 4.5, 'R before ring0', (tip, u), grip=.35)

    def settle(sg):
        """Wait until ring 0 hangs still (at most 4 s), then return its pose."""
        prev = None
        for _ in range(8):
            c, n, t, sg, mm = update(sg)
            if prev is not None and np.linalg.norm(c-prev) < .002:
                break
            prev = c
            s.hold(.5)
        return c, n, t, sg, mm

    for attempt in range(4 if live else 1):
        try:
            if live:
                for dist, sec in ((.03, 1.2), (.012, 1.2)):
                    c0, n0, t0, sigma, _ = settle(sigma)
                    d0, rs = pinch(c0, n0, t0, sigma)
                    s.move('right', c0+d0*(TR0+dist), rs, sec, 'R approach ring0')
                c0, n0, t0, sigma, _ = settle(sigma)
                d0, rs = pinch(c0, n0, t0, sigma)
            s.move('right', c0+d0*TR0, rs, 1.2 if live else 2., 'R at ring0')
            s.hold(.2)
            if live:
                c0, n0, t0, sigma, mm = update(sigma)
                d0, rs = pinch(c0, n0, t0, sigma)
                p, r = fk('right', mm['qr'])
                cp, cr = s.pose['right']
                try:
                    s.move('right', cp+c0+d0*TR0-p, rs@r.T@cr, .5, 'R corrected at ring0')
                except AssertionError:
                    pass
            s.grip('right', 0., 1.2)
            s.hold(.5, 'R pinched ring0')
            if not live:
                break
            s.rel('right', [0, 0, .015], .6, 'R test lift ring0')
            s.hold(.3)
            mm = s.sense()
            p, r = fk('right', mm['qr'])
            miss = np.linalg.norm(np.asarray(mm['c0'])-(p+r[:, 2]*TR0))
            s.marks.append(dict(label=f'ring0 grasp check {miss*1000:.1f} mm', step=len(s.rows), t=s.t, side=None, q=None, off=None))
            if miss < .008:
                break
        except AssertionError:
            if not live:
                raise
        s.grip('right', .35, .5)
        try:
            s.rel('right', -s.pose['right'][1][:, 2]*.04, 1., 'R back off ring0')
        except AssertionError:
            pass
        s.hold(1.)
    else:
        raise AssertionError('ring 0 grasp failed')
    if live:
        c0, n0, t0, sigma, _ = update(sigma)
    if stage <= 6:
        return
    Zv = np.array([0, 0, 1.])
    up_p = unit(Zv-u*(Zv@u)); near_p = np.cross(up_p, u)
    df = np.sin(np.radians(tau))*near_p+np.cos(np.radians(tau))*up_p   # final pinch direction, across the rod
    # gripper closing axis is -sigma*n; ring 0 must end facing -u (ring 1 faces +u)
    rf = frame(-df, sigma*u)
    on_rod = lambda sx, off: tip+u*sx+off+df*TR0
    ride, hang = df*.003, -up_p*.009
    g0 = c0-tip-u*((c0-tip)@u)
    cm = tip-u*.015+unit(g0)*.055
    ce = tip-u*.025+ride
    qs_ = s.q['right']
    qe = solve('right', ce+df*TR0, rf, qs_, all_seeds=True)
    assert qe is not None, 'fold end unreachable'
    path, q, n = [], qs_, 40
    try:
        for k in range(1, n+1):
            f = k/n
            _, r = fk('right', qs_+(qe-qs_)*f)
            c = c0+(cm-c0)*(2*f) if f < .5 else cm+(ce-cm)*(2*f-1)
            qn = solve('right', c-r[:, 2]*TR0, r, q)
            assert qn is not None and np.abs(qn-q).max() < .35, ('fold path', k)
            path.append(qn); q = qn
    except AssertionError:
        # fallback: straight joint-space turn, kept clear of the rod until the ring is outside the tip
        best = None
        for qe2 in [qe]+[solve('right', ce+df*TR0, rf, np.asarray(sd, float)) for sd in SEEDS]:
            if qe2 is None:
                continue
            path = [qs_+(qe2-qs_)*k/n for k in range(1, n+1)]
            clear = []
            for qq in path:
                p, r = fk('right', qq)
                pts = [p+r[:, 2]*TR0, p+r[:, 2]*.013+r[:, 0]*.012, p+r[:, 2]*.013-r[:, 0]*.012, p-r[:, 2]*.05]
                worst = 1.
                for x in pts:
                    sx = (x-tip)@u
                    if sx < -.008:
                        continue                       # beyond the free tip
                    worst = min(worst, np.linalg.norm((x-tip)-u*np.clip(sx, 0, .78)))
                clear.append(worst)
            if min(clear) > .025 and (best is None or min(clear) > best[0]):
                best = (min(clear), path)
        assert best is not None, 'no clear fold'
        path = best[1]
    s.follow('right', np.array(path), 6., 'R ring0 turned at tip')
    s.pose['right'] = (ce+df*TR0, rf)
    s.hold(.5, 'R ring0 at tip settle')
    s.move('right', on_rod(.015, hang), None, 2.5, 'R ring0 on stub')
    s.grip('right', .07, .5)
    s.hold(1., 'R ring0 loose on stub')
    s.grip('right', 0., .8)
    s.hold(.3, 'R ring0 repinched')
    s.move('right', on_rod(.015, ride), None, 1., 'R ring0 centred')
    s.move('right', on_rod(S0, ride), None, 6., 'R ring0 deep')
    s.move('right', on_rod(S0, hang), None, 1., 'R ring0 lowered')
    s.grip('right', .3, .8)
    s.hold(.4, 'R released ring0')
    s.rel('right', df*.025, 1.2, 'R clear of ring0')
    s.away = df
    s.hold(1., 'both rings hang')
