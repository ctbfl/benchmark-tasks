"""Counterexamples for the physically meaningful insertion score."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import OPEN_ANGLE, HOLE_Y, ring_points, PIVOT_Y, PIVOT_Z, HOLE_R, WIRE_R
from scoring import Score, threading


def test_threading_and_unreleased_card():
    rail=np.r_[[0,-PIVOT_Y,PIVOT_Z],Rotation.from_euler('x',OPEN_ANGLE).as_quat()]
    curve=Rotation.from_quat(rail[3:]).apply(ring_points(-1))+rail[:3]
    point=curve[-4]
    card=np.r_[[0,point[1]-HOLE_Y,point[2]],[0,0,0,1]]
    assert np.max(threading(rail,card))<1e-8
    score=Score()
    result=score.update(1.,[-OPEN_ANGLE,OPEN_ANGLE],rail,card,np.array([0,0,0,0,0,0,1.]),1.)
    assert result['score']==.2 and not result['inserted']
    result=score.update(2.,[-OPEN_ANGLE,OPEN_ANGLE],rail,card,np.array([0,0,0,0,0,0,1.]),0.)
    assert result['score']==.8
    turned=card.copy()
    yaw=Rotation.from_euler('z',.08)
    hole_center=card[:3]+[0,HOLE_Y,0]
    turned[:3]=hole_center-yaw.apply([0,HOLE_Y,0])
    turned[3:]=yaw.as_quat()
    errors=threading(rail,turned)
    assert 0<np.count_nonzero(errors<HOLE_R-WIRE_R)<6
    result=score.update(2.5,[-OPEN_ANGLE,OPEN_ANGLE],rail,turned,np.array([0,0,0,0,0,0,1.]),0.)
    assert result['score']==.2
    card[0]+=.01
    result=score.update(3.,[-OPEN_ANGLE,OPEN_ANGLE],rail,card,np.array([0,0,0,0,0,0,1.]),0.)
    assert result['score']==.2


def test_closed_initial_binder_does_not_score():
    rail=np.array([0,-PIVOT_Y,PIVOT_Z,0,0,0,1.])
    card=np.array([0,-.08,.01,0,0,0,1.])
    assert Score().update(0.,[0,0],rail,card,np.array([0,0,0,0,0,0,1.]),0.)['score']==0.


if __name__=='__main__':
    test_threading_and_unreleased_card()
    test_closed_initial_binder_does_not_score()
    print('Scoring counterexamples passed')
