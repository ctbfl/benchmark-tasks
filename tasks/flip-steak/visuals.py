"""Clock and texture changes affect appearance only."""
from pxr import UsdGeom, UsdShade
from config import ROOT

SEGMENTS={'0':'abcdef','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg',
          '5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg'}


def update(stage, seconds, cooked):
    value=int(seconds+1e-7)
    display=f'{value//60:02d}{value%60:02d}'
    for i,c in enumerate(display):
        path=ROOT+f'/timer/body/display/digit_{i}'
        for seg in 'abcdefg':
            UsdGeom.Imageable(stage.GetPrimAtPath(path+'/'+seg)).GetVisibilityAttr().Set(
                'inherited' if seg in SEGMENTS[c] else 'invisible')
        UsdGeom.Imageable(stage.GetPrimAtPath(path+'/dot')).GetVisibilityAttr().Set('invisible')
    material=UsdShade.Material(stage.GetPrimAtPath(ROOT+'/steak/Looks/'+('Seared' if cooked else 'Raw')))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ROOT+'/steak/bottom')).Bind(material)


def prepare_render():
    """Discard temporal reconstruction history after a clock or pose change."""
    import carb.eventdispatcher
    import carb.settings
    carb.settings.get_settings().set_bool('/rtx-transient/post/dlss/forceParamReset',True)
    carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name='omni.rtx.ForceResetAccumulation')
