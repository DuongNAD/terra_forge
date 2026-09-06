"""
Inspector subpackage for TerraForge.
Unconditionally exports pure-Python QualityValidator and inspection camera utilities.
"""

from terra_forge.inspector.validator import QualityValidator
from terra_forge.inspector.cameras import setup_inspection_cameras, INSPECTION_PRESETS
from terra_forge.inspector.vision_rig import render_inspection_snapshots, build_2x2_vision_grid

__all__ = [
    "QualityValidator",
    "setup_inspection_cameras",
    "INSPECTION_PRESETS",
    "render_inspection_snapshots",
    "build_2x2_vision_grid",
]
