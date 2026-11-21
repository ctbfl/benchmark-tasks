"""Scorer fixtures using known geometry and ink provenance."""
import numpy as np
from scipy.ndimage import affine_transform

from contact_rule import PrintContact
from geometry import BLOCK_MASK, THICKNESS
from ink import Film
from scoring import SHEET_SHAPE, grade


def panels(image):
    return np.stack(np.split(image, 3, axis=1)) * .7


def proof(angle=0, shift=(0, 0), mirror=False):
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    inverse = np.array([[c, s], [-s, c]])
    source = BLOCK_MASK[:, ::-1] if mirror else BLOCK_MASK
    centre = (np.asarray(source.shape)-1)/2
    destination = (np.asarray(SHEET_SHAPE)-1)/2 + shift
    return affine_transform(source, inverse, offset=centre-inverse@destination,
                            output_shape=SHEET_SHAPE, order=0, mode="grid-constant")


class ContactSample:
    def __init__(self, surface, *, back=False):
        self.surface = surface
        self.side = 1 if back else -1

    def get_contact_data(self, _dt):
        counts = np.zeros((1,2), dtype=int)
        counts[0,self.surface] = 1
        return (np.array([.6]), np.array([[0.,0.,self.side*THICKNESS/2]]),
                np.array([[0.,0.,-self.side]]), np.array([0.]), counts, np.zeros_like(counts))


def observe(rule, sample, time):
    rule.observe(sample, np.array([0.,0.,0.,0.,0.,0.,1.]), 0, 1/240, time)


def main():
    dry = np.zeros_like(BLOCK_MASK)
    full = proof()
    ideal = panels(full)
    inked = .4*BLOCK_MASK
    wet = Film(0., 1/240)
    clean = PrintContact()
    clean.update_loaded(wet)
    print_result = grade(dry, ideal, True, print_eligible=True)
    assert print_result["score"] == 1.

    slab = np.zeros(SHEET_SHAPE, dtype=bool)
    slab[12:24,1:39] = True
    slab_rule = PrintContact()
    slab_rule.update_loaded(wet)
    observe(slab_rule, ContactSample(0), .1)
    assert slab_rule.first["surface"] == "slab"
    marked = grade(dry, panels(slab), True, print_eligible=slab_rule.first is None)
    assert marked["score"] == 0. and marked["marked_area_mm2"] == 1824
    print("slab-only", marked)

    # The same final visible image after a slab contact and a relief pull remains ineligible.
    observe(slab_rule, ContactSample(0), 4.)
    assert slab_rule.first["time_s"] == .1
    mixed = grade(inked, ideal, True, print_eligible=slab_rule.first is None)
    assert mixed["score"] == .2 and mixed["coverage"] == print_result["coverage"] == 1.
    assert mixed["ghost_ratio"] == print_result["ghost_ratio"] == 0.
    assert grade(dry, ideal, True, print_eligible=False)["score"] == 0.
    print("identical final pixels", print_result, mixed)

    # A dry sleeve and a clean back contact remain eligible.
    observe(clean, ContactSample(1), .2)
    observe(clean, ContactSample(0, back=True), .3)
    assert clean.first is None
    blank = Film(0., 1/240)
    blank.palette.fill(0.)
    no_ink = PrintContact()
    no_ink.update_loaded(blank)
    observe(no_ink, ContactSample(0), .5)
    assert no_ink.first is None

    wet.roller.fill(.2)
    clean.update_loaded(wet)
    observe(clean, ContactSample(1), .6)
    assert clean.first["surface"] == "roller"
    assert grade(inked, ideal, True, print_eligible=False)["score"] == .2
    print("inked roller face latched", clean.first)

    for angle, shift in ((0,(0,0)),(17,(1.25,-1.9)),(37,(-2,2)),
                         (89,(2.5,0)),(123,(-1.1,1.8)),(180,(0,0)),(271,(-2.3,1.2))):
        raster = panels(proof(angle, shift))
        result = grade(dry, raster, True, print_eligible=True)
        assert result["score"] == 1., (angle, result)
        assert result["ghost_ratio"] == 0.
        if angle in (0, 180):
            assert result["coverage"] == 1.
        print(f"clean {angle} degrees", result)
    assert grade(dry, panels(proof(mirror=True)), True, print_eligible=True)["score"] < 1.

    partial = full.copy()
    partial[28:] = False
    assert grade(dry, panels(partial), False, print_eligible=True)["score"] == .4
    dirty = grade(dry, panels(full | slab), True, print_eligible=True)
    assert dirty["ghost_ratio"] > .18 and dirty["score"] == 0.
    assert grade(dry, panels(np.zeros(SHEET_SHAPE)), True,
                 ever_inked=True, print_eligible=False)["score"] == .2
    print("back and dry contacts stayed eligible, excess cap applied", dirty)


if __name__ == "__main__":
    main()
