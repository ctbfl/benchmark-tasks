"""Boundary and failure cases for the user-defined event sequence."""
import unittest
import numpy as np
from config import DT, PAN_FLOOR, STEAK_HEIGHT
from scoring import Scorer

PAN=np.r_[0.,0.,0.,0.,0.,0.,1.,np.zeros(6)]
RAW=PAN.copy(); RAW[2]=PAN_FLOOR
FLIPPED=RAW.copy(); FLIPPED[2]+=STEAK_HEIGHT; FLIPPED[3:7]=[1.,0.,0.,0.]


class ScoringTests(unittest.TestCase):
    def scenario(self,insertion,departure):
        scorer=Scorer(RAW,PAN)
        from scipy.spatial.transform import Rotation
        from config import outline
        tilted=RAW.copy()
        rot=Rotation.from_euler('y',15,degrees=True)
        tilted[3:7]=rot.as_quat()
        tilted[2]=PAN_FLOOR-rot.apply(np.c_[outline(),np.zeros(len(outline()))])[:,2].min()
        scorer.ticks=round(insertion/DT)-1
        scorer.update(tilted,PAN)
        raised=RAW.copy(); raised[2]+=.05
        scorer.ticks=round(departure/DT)-1
        scorer.update(raised,PAN)
        scorer.update(FLIPPED,PAN)
        return scorer

    def test_exact_time_boundaries(self):
        self.assertEqual(self.scenario(50,70).result()['score'],1.)
        self.assertEqual(self.scenario(50-DT,70).result()['score'],.5)
        self.assertEqual(self.scenario(50,70+DT).result()['score'],.5)

    def test_never_departed_is_failure(self):
        scorer=Scorer(RAW,PAN)
        scorer.ticks=round(65/DT)
        scorer.update(RAW,PAN)
        self.assertIsNone(scorer.departure_s)
        self.assertEqual(scorer.result()['score'],0.)

    def test_tilt_with_one_edge_still_touching(self):
        from scipy.spatial.transform import Rotation
        from config import outline
        tilted=RAW.copy()
        rot=Rotation.from_euler('y',15,degrees=True)
        tilted[3:7]=rot.as_quat()
        tilted[2]=PAN_FLOOR-rot.apply(np.c_[outline(),np.zeros(len(outline()))])[:,2].min()
        scorer=Scorer(RAW,PAN)
        scorer.ticks=round(55/DT)-1
        scorer.update(tilted,PAN)
        self.assertEqual(scorer.insertion_s,55)
        self.assertIsNone(scorer.departure_s)
        self.assertEqual(scorer.result()['score'],0.)

    def test_bottom_still_touching_pan_wall_is_not_departed(self):
        from config import outline
        touching=RAW.copy()
        touching[2]=.035
        touching[0]=(.130+.018*(.035-.003)/.037)*np.cos(np.pi/64)-outline()[:,0].max()
        scorer=Scorer(RAW,PAN)
        scorer.ticks=round(60/DT)-1
        scorer.update(touching,PAN)
        self.assertIsNone(scorer.departure_s)
        self.assertEqual(scorer.result()['score'],0.)

    def test_airborne_or_supported_by_tool_fails(self):
        scorer=self.scenario(55,60)
        air=FLIPPED.copy(); air[2]+=.02
        scorer.update(air,PAN)
        self.assertEqual(scorer.result()['score'],0.)
        scorer.update(FLIPPED,PAN,unsupported=True)
        self.assertEqual(scorer.result()['score'],0.)

    def test_original_side_down_fails(self):
        scorer=self.scenario(55,60)
        scorer.update(RAW,PAN)
        self.assertEqual(scorer.result()['score'],0.)

    def test_off_pan_fails(self):
        scorer=self.scenario(55,60)
        off=FLIPPED.copy(); off[0]=.4
        scorer.update(off,PAN)
        self.assertEqual(scorer.result()['score'],0.)


if __name__=='__main__': unittest.main()
