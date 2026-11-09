"""Dose-boundary, support and spill checks independent of the simulator."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from config import GRAIN_COUNT,GRAIN_RADIUS,STEAK_HEIGHT
from scoring import score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.grains=np.zeros((GRAIN_COUNT,13))
        self.grains[:,0]=1.
        self.grains[:,6]=1.
        self.steak=np.zeros(13);self.steak[6]=1.
        self.bottle=np.array([1.,0.,0.,0.,0.,0.,1.])

    def deposit(self,n):
        points=np.array([(x,y,STEAK_HEIGHT+GRAIN_RADIUS)
                         for x in np.arange(-.04,.041,.004)
                         for y in np.arange(-.03,.031,.004)])
        self.grains[:n,:3]=points[:n]

    def result(self):
        return score(self.grains,self.steak,self.bottle)

    def test_zero_earns_no_credit(self):
        self.assertEqual(self.result()['score'],0.)
        self.assertFalse(self.result()['dose_safe'])
        self.assertFalse(self.result()['success'])

    def test_positive_dose_below_limit_earns_full_credit(self):
        self.deposit(1)
        self.assertEqual(self.result()['score'],1.)
        self.assertTrue(self.result()['dose_safe'])

    def test_inclusive_two_gram_boundary(self):
        self.deposit(160)
        self.assertEqual(self.result()['steak_mass_g'],2.)
        self.assertEqual(self.result()['score'],1.)
        self.deposit(161)
        self.assertEqual(self.result()['score'],.5)
        self.assertFalse(self.result()['dose_safe'])

    def test_airborne_and_table_spill_do_not_count(self):
        self.deposit(12)
        self.grains[:6,2]+=.01
        self.grains[6:12,2]=GRAIN_RADIUS
        self.assertEqual(self.result()['steak_mass_g'],0.)
        self.assertEqual(self.result()['score'],0.)

    def test_stack_is_included(self):
        self.deposit(1)
        self.grains[1,:3]=self.grains[0,:3]+[0,0,2*GRAIN_RADIUS]
        self.assertEqual(self.result()['steak_grains'],2)

    def test_score_follows_the_actual_steak_pose(self):
        self.deposit(40)
        rotation=Rotation.from_euler('z',.7)
        shift=np.array([.1,.2,.5])
        self.grains[:,:3]=rotation.apply(self.grains[:,:3])+shift
        self.steak[:3]=shift;self.steak[3:7]=rotation.as_quat()
        self.assertEqual(self.result()['steak_grains'],40)


if __name__=='__main__':unittest.main()
