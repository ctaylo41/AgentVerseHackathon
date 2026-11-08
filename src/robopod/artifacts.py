"""Versioned robot/scene identity, recorded independently of observed geometry."""
import hashlib
from pathlib import Path

ASSET = Path(__file__).parent / 'assets/room.xml'
ARTIFACT_VERSION = 2


def identity():
    return dict(artifact_version=ARTIFACT_VERSION, scene_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                rig='cartesian-gantry-6dof-v1', perception='semantic-parts-learned-association-v2')


def require_compatible(value, *, expected=None):
    expected = identity() if expected is None else expected
    if value != expected:
        raise ValueError('Survey/scene version is missing or incompatible with this robot. Run Explore scene again; old artifacts are preserved.')
