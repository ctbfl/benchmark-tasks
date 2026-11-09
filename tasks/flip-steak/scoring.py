"""Score the initial full contact, insertion, and complete departure events."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import PAN_FLOOR, PAN_RADIUS, STEAK_HEIGHT, DT, outline

CONTACT_TOLERANCE = .0008


def face_contact(steak, pan, height):
    """Flat convex face against the pan floor; boundary extrema define separation."""
    sr=Rotation.from_quat(steak[3:7]).as_matrix()
    pr=Rotation.from_quat(pan[3:7]).as_matrix()
    points=np.c_[outline(),np.full(len(outline()),height)]
    local=(points@sr.T+steak[:3]-pan[:3])@pr
    gap=local[:,2]-PAN_FLOOR
    inside=np.linalg.norm(local[:,:2],axis=1)<=PAN_RADIUS
    full=bool(inside.all() and np.max(np.abs(gap))<=CONTACT_TOLERANCE)
    # Check the portion below the rim against the pan's flared inner wall.
    # This prevents a face still touching the wall from counting as departed.
    lip=.041+CONTACT_TOLERANCE
    nxt=np.roll(local,-1,axis=0)
    crossing=(local[:,2]<=lip)!=(nxt[:,2]<=lip)
    edge=local[crossing]; end=nxt[crossing]
    intersections=edge+(end-edge)*((lip-edge[:,2])/(end[:,2]-edge[:,2]))[:,None]
    low=np.vstack([local[local[:,2]<=lip],intersections])
    inner=np.minimum(.1475,.130+.018*np.clip((low[:,2]-.003)/.037,0,1))
    # Inner surfaces use 64 planar facets, so use their actual apothems.
    normals=np.c_[np.cos((np.arange(64)+.5)*2*np.pi/64),np.sin((np.arange(64)+.5)*2*np.pi/64)]
    wall_clear=bool(np.all(low[:,:2]@normals.T < inner[:,None]*np.cos(np.pi/64)-CONTACT_TOLERANCE))
    outside=np.linalg.norm(local[:,:2].mean(axis=0))>.1545+np.max(np.linalg.norm(outline(),axis=1))
    separated=bool((gap.min()>CONTACT_TOLERANCE and wall_clear)
                   or local[:,2].max()<-CONTACT_TOLERANCE or outside)
    return full,separated


class Scorer:
    def __init__(self, steak, pan):
        assert face_contact(steak,pan,0)[0], 'The standard initial steak must fully contact the pan'
        self.ticks=0
        self.insertion_s=None
        self.departure_s=None
        self.first_flip_s=None
        self.flipped=False
        self.cooked=False

    def update(self, steak, pan, unsupported=False):
        self.ticks+=1
        time=self.ticks*DT
        full,separated=face_contact(steak,pan,0)
        if self.insertion_s is None and not full: self.insertion_s=time
        if self.departure_s is None and separated: self.departure_s=time
        full_duration=time if self.insertion_s is None else self.insertion_s
        self.cooked=self.cooked or full_duration+1e-9>=50
        sr=Rotation.from_quat(steak[3:7]).as_matrix()
        pr=Rotation.from_quat(pan[3:7]).as_matrix()
        self.flipped=bool(self.departure_s is not None and sr[2,2]<0 and pr[2,2]>.95
            and face_contact(steak,pan,STEAK_HEIGHT)[0]
            and np.linalg.norm(steak[7:10])<.02 and np.linalg.norm(steak[10:13])<.2 and not unsupported)
        if self.flipped and self.first_flip_s is None: self.first_flip_s=time

    def result(self):
        full=self.ticks*DT if self.insertion_s is None else self.insertion_s
        timing=full+1e-9>=50 and self.departure_s is not None and self.departure_s<=70+1e-9
        return dict(score=.5*int(self.flipped)+.5*int(self.flipped and timing),flipped=self.flipped,
            timing_valid=bool(timing),full_contact_s=full,insertion_s=self.insertion_s,
            departure_s=self.departure_s,first_flip_s=self.first_flip_s,
            cooked=self.cooked,time_s=self.ticks*DT)
