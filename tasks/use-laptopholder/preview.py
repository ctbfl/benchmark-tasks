"""Render the actual USD/PhysX stand, including a close view of the curved nose."""

import cv2
import omni.replicator.core as rep
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

import model
from runtime import simulation, step


def run():
    sim = simulation()
    model.build(sim.stage, "/World/Holder", model.OPEN_WIDTH, 4)
    camera = rep.create.camera(position=(.45,-.5,.35),look_at=(.08,0,.07),
                               focal_length=35,clipping_range=(.005,10))
    product = rep.create.render_product(camera,(1200,900))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    sim.reset()
    step(sim,1)
    folder = model.ROOT/"outputs/preview"
    folder.mkdir(exist_ok=True)
    for name,pos,target in (("overview",(.45,-.5,.35),(.08,0,.07)),
                            ("front_detail",(-.13,-.20,.10),(-.014,-.08,.016))):
        with camera:
            rep.modify.pose(position=pos,look_at=target)
        for _ in range(12):
            sim.render()
            ensure_isaac_rtx_render_update(force=True)
        pixels = rgb.get_data()
        assert pixels.size, "Camera did not render"
        cv2.imwrite(str(folder/(name+".png")),cv2.cvtColor(pixels[...,:3],cv2.COLOR_RGB2BGR))
    sim.stop()
