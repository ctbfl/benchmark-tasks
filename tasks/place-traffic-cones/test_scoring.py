"""Adversarial geometric scoring cases and independent nearest-point checks."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from config import LENGTH, curve, nearest, BOARD_SIZE
from scoring import measure


class ScoringTests(unittest.TestCase):
    def roots(self):
        return np.column_stack((curve(np.linspace(0,LENGTH,6)),
            np.full(6,.75+BOARD_SIZE[2]),np.tile([0.,0.,0.,1.],(6,1))))

    def score(self, roots, contact=None, time=100):
        return measure(roots,np.zeros(6) if contact is None else contact,.75,time)['score']

    def test_success_and_permutation(self):
        self.assertEqual(self.score(self.roots()[[3,0,5,1,4,2]]),1.)

    def test_lifted_tipped_and_held_fail(self):
        roots=self.roots(); roots[1,2]+=.02
        self.assertEqual(self.score(roots),0.)
        roots=self.roots(); roots[1,3:]=Rotation.from_euler('x',90,degrees=True).as_quat()
        self.assertEqual(self.score(roots),0.)
        self.assertEqual(self.score(self.roots(),np.ones(6)),0.)

    def test_lateral_error_and_missing_endpoint(self):
        roots=self.roots(); roots[3,0]+=.04
        self.assertEqual(self.score(roots),0.)
        roots=self.roots(); roots[0,:2]=curve(.021)
        self.assertEqual(self.score(roots),0.)

    def test_maximum_gap_only(self):
        roots=self.roots()
        # One small gap is permitted: only the maximum gap is compared to L/5.
        roots[:,:2]=curve(np.array([0.,.01,.14,.27,.40,LENGTH]))
        self.assertEqual(self.score(roots),1.)
        roots[:,:2]=curve(np.array([0.,.01,.02,.27,.40,LENGTH]))
        self.assertEqual(self.score(roots),0.)

    def test_time_limit(self):
        self.assertEqual(self.score(self.roots(),time=120),1.)
        self.assertEqual(self.score(self.roots(),time=120.05),0.)

    def test_exact_projection_against_dense_independent_search(self):
        rng=np.random.default_rng(0)
        points=rng.uniform([-.5,-.6],[.9,.7],(300,2))
        dense=curve(np.linspace(0,LENGTH,30001))
        actual,s=nearest(points)
        brute=np.min(np.linalg.norm(dense[None,:,:]-points[:,None,:],axis=-1),axis=1)
        np.testing.assert_allclose(actual,brute,atol=2e-5)
        np.testing.assert_allclose(np.linalg.norm(curve(s)-points,axis=-1),actual,atol=1e-12)


if __name__=='__main__':
    unittest.main()
