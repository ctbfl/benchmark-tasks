"""Coin detection from the top camera's RGB image, used by expert.py to plan the demo.

Pixels are classified by colour (copper pennies, near-neutral silver quarters, the tan table and
red tape), resampled onto a top-down grid of the table plane with the camera's calibrated
intrinsics and pose, and individual fully visible coins are found as peaks of the distance
transform of the coin mask. No simulator object state is used.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from build_assets import COINS

ROI = (.18, .50, -.175, .175)       # stays clear of the tape frames' inner edge at |y| = 0.19          # table x range, y range searched for coins
COIN_TOP = .0016                     # coin top surface height above the table


class Camera:
    def __init__(self, K, position, quat_ros_xyzw, table_z, shift=(0., 0.)):
        self.K = np.asarray(K, float)
        self.shift = np.asarray(shift, float)   # rendered minus modelled pixel position
        self.position = np.asarray(position, float)
        self.rotation = Rotation.from_quat(quat_ros_xyzw)   # camera (ROS optical) -> world
        self.table_z = table_z

    @classmethod
    def from_sensor(cls, sensor, table_z):
        return cls(sensor.data.intrinsic_matrices.torch[0].cpu().numpy(),
                   sensor.data.pos_w.torch[0].cpu().numpy(),
                   sensor.data.quat_w_ros.torch[0].cpu().numpy(), table_z)

    def to_dict(self):
        return {"K": self.K.tolist(), "position": self.position.tolist(),
                "quat_ros_xyzw": self.rotation.as_quat().tolist(), "table_z": self.table_z,
                "shift": self.shift.tolist()}

    def table_points(self, u, v, height=COIN_TOP):
        """Pixel centres (u = column, v = row) -> world XY on the plane table_z + height."""
        u, v = np.asarray(u, float) - self.shift[0], np.asarray(v, float) - self.shift[1]
        rays = np.linalg.solve(self.K, np.vstack([u, v, np.ones_like(u)]))
        rays = self.rotation.apply(rays.T)
        scale = (self.table_z + height - self.position[2]) / rays[:, 2]
        return self.position[:2] + rays[:, :2] * scale[:, None]

    def pixel(self, xy, height=COIN_TOP):
        world = np.array([xy[0], xy[1], self.table_z + height])
        camera = self.rotation.inv().apply(world - self.position)
        p = self.K @ camera
        return p[:2] / p[2] + self.shift

    def calibrate(self, image, search=10.):
        """Fit the pixel shift that best overlays the known tape frames on the red tape pixels.

        Samples inside each tape strip should be red and samples just outside should not; the
        score difference peaks sharply at the true shift even though the tape is many pixels wide.
        """
        from build_assets import FRAMES, TAPE_WIDTH
        rgb = image.astype(float)
        red = (rgb[..., 0] > 120) & (rgb[..., 1] < .35*rgb[..., 0]) & (rgb[..., 2] < .35*rgb[..., 0])
        inside, outside = [], []
        w = TAPE_WIDTH
        for (cx, cy), (sx, sy) in FRAMES.values():
            for along in np.linspace(-.4, .4, 40):
                for sign in (-1, 1):
                    for across, target in ((0., inside), (-.3*w, inside), (.3*w, inside),
                                           (-.5*w - .004, outside), (.5*w + .004, outside)):
                        # Strips parallel to Y at x = cx +- (sx + w)/2, and parallel to X.
                        target.append((cx + sign*(sx + w)/2 + across, cy + along*sy))
                        target.append((cx + along*sx, cy + sign*(sy + w)/2 + across))
        self.shift = np.zeros(2)
        base_in = np.array([self.pixel(p, .0003) for p in inside])
        base_out = np.array([self.pixel(p, .0003) for p in outside])

        def fraction(base, du, dv):
            u = np.round(base[:, 0] + du).astype(int)
            v = np.round(base[:, 1] + dv).astype(int)
            return red[np.clip(v, 0, image.shape[0]-1), np.clip(u, 0, image.shape[1]-1)].mean()

        steps = np.arange(-search, search + .25, .5)
        scores = [(fraction(base_in, du, dv) - fraction(base_out, du, dv), du, dv)
                  for du in steps for dv in steps]
        best = max(scores)
        self.shift = np.array(best[1:], float)
        return best[0]


def classify_pixels(image):
    """Return (coin mask, copper mask) from an RGB uint8 image."""
    rgb = image.astype(float) / 255.
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high, low = rgb.max(-1), rgb.min(-1)
    chroma = high - low
    saturation = chroma / np.maximum(high, 1e-6)
    hue = np.where(high == r, 60*((g - b) / np.maximum(chroma, 1e-6)) % 360, 0.)
    # Rendered table: hue 42 deg, saturation 0.19, value 0.9. Quarters are nearly neutral; pennies
    # are a darker, more saturated copper (hue about 30 deg, saturation about 0.47, value about 0.6).
    neutral = (saturation < .12) & (high > .22)
    copper = (hue > 15) & (hue < 36) & (saturation > .30) & (high > .18) & (high < .82)
    red_tape = (r > .4) & (g < .25*r)
    return (neutral | copper) & ~red_tape, copper & ~red_tape


GRID = .00075                        # table-plane resampling step in metres


def rectify(image, camera, roi=ROI):
    """Sample the coin and copper masks on a regular table-plane grid over the ROI."""
    xs = np.arange(roi[0], roi[1], GRID)
    ys = np.arange(roi[2], roi[3], GRID)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    world = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, camera.table_z + COIN_TOP)])
    cam = camera.rotation.inv().apply(world - camera.position)
    pix = (camera.K @ cam.T).T
    u = np.round(pix[:, 0] / pix[:, 2] + camera.shift[0]).astype(int)
    v = np.round(pix[:, 1] / pix[:, 2] + camera.shift[1]).astype(int)
    u = np.clip(u, 0, image.shape[1] - 1)
    v = np.clip(v, 0, image.shape[0] - 1)
    coin, copper = classify_pixels(image)
    return xs, ys, coin[v, u].reshape(gx.shape), copper[v, u].reshape(gx.shape)


def find_coins(image, camera, roi=ROI):
    """Individual coins, including ones that touch others, plus all coin points.

    In the top-down grid every fully visible coin is a disc; the distance transform of the coin
    mask peaks near the disc centre with a value close to the coin radius. Peaks are accepted
    greedily from the largest; the coin type comes from the colour inside the disc. A disc that
    is partly covered by another coin has a clearly smaller peak and is not reported.
    """
    xs, ys, coin, copper = rectify(image, camera, roi)
    coin = ndimage.binary_opening(coin, iterations=1)
    distance = ndimage.distance_transform_edt(coin) * GRID
    peaks = (distance == ndimage.maximum_filter(distance, size=7)) & (distance > .8*COINS["penny"]["radius"])
    order = np.argsort(-distance[peaks])
    candidates = np.column_stack(np.nonzero(peaks))[order]
    found = []
    ii, jj = np.mgrid[0:coin.shape[0], 0:coin.shape[1]]
    for i, j in candidates:
        centre = np.array([xs[i], ys[j]])
        if any(np.linalg.norm(centre - f["xy"]) < .015 for f in found):
            continue
        disc = (ii - i)**2 + (jj - j)**2 <= (COINS["penny"]["radius"]*.8 / GRID)**2
        copper_fraction = float(copper[disc].mean())
        kind = "penny" if copper_fraction > .5 else "quarter"
        radius = COINS[kind]["radius"]
        disc = (ii - i)**2 + (jj - j)**2 <= (radius*.9 / GRID)**2
        covered = float(coin[disc].mean())
        clean = distance[i, j] > .85*radius and covered > .95 and (copper_fraction > .85 or copper_fraction < .15)
        found.append({"kind": kind, "xy": centre, "peak": float(distance[i, j]), "clean": clean})
    points = np.column_stack([np.repeat(xs, len(ys)).reshape(coin.shape)[coin],
                              np.tile(ys, len(xs)).reshape(coin.shape)[coin]])
    return [f for f in found if f["clean"]], points, found
