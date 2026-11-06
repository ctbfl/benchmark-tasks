"""Archive the exact source and environment information used by an execution."""
import hashlib
import importlib.metadata
import json
import shutil
from config import HERE, CONFIG


def snapshot(out):
    source=out/'source'
    source.mkdir()
    paths=[*HERE.glob('*.py'),HERE/'dream-defaults.json',*sorted((HERE/'assets').glob('*.usda'))]
    if (HERE/'demo_actions.npy').is_file():
        paths.append(HERE/'demo_actions.npy')
    hashes={}
    for path in paths:
        relative=path.relative_to(HERE)
        target=source/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(config=CONFIG,source_sha256=hashes,
        versions={name:importlib.metadata.version(name) for name in
                  ('numpy','scipy','Pillow','isaaclab','isaaclab_arena','inspect-robots','inspect-robots-dream','dreamlabs-client')}),indent=2)+'\n')
