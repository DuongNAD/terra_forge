"""
Export Pipeline for TerraForge:
1. glTF/GLB production asset (Three.js r128 compatible).
2. Production .blend master file.
3. Simulation Elevation Matrix & Navmesh JSON (for game engines like Genesis Zero).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
try:
    import bpy
except ImportError:
    bpy = None


from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import MapConfig


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _resolve_path(output_path: str) -> Path:
    p = Path(output_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def export_production_glb(output_path: str) -> str:
    """
    Exports clean, optimized GLB file compatible with Three.js r128:
    - Draco disabled (zero CPU decompression overhead on client)
    - Vertex colors COLOR_0 preserved
    - PBR materials linked
    """
    out_file = _resolve_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Exclude prototype collection before GLB export
    proto_col = bpy.data.collections.get("TerraForge_Prototypes")
    if proto_col:
        proto_col.hide_render = True

    # Use standard glTF exporter
    bpy.ops.export_scene.gltf(
        filepath=str(out_file),
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_draco_mesh_compression_enable=False,
        export_image_format="AUTO",
        export_lights=True,
        export_cameras=True,
    )
    return str(out_file)


def export_blend_file(output_path: str) -> str:
    """Saves the master production .blend file."""
    out_file = _resolve_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_file))
    return str(out_file)


def export_simulation_heightmap_json(output_path: str, hf: Heightfield2D, config: MapConfig) -> str:
    """
    Exports 2D elevation grid and walkability/passability matrix for simulation engines (e.g. genesis.world).
    Passability classification:
    - 0: Deep Water / Unpassable
    - 1: Walkable Soil / Grass / Basin
    - 2: Steep Cliff (Slope > 40 deg)
    - 3: High Alpine Mountain Peak
    """
    out_file = _resolve_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    slopes = hf.compute_slope_degrees()
    lake_lvl = config.hydrology.lake_level
    sea_lvl = config.hydrology.sea_level

    # Passability matrix
    passability = np.ones_like(hf.Z, dtype=int)

    # Accurate spatial water mask:
    # 1. Central lake basin: points within lake perimeter and below/at water level
    lx, ly = config.hydrology.lake_center
    lr = config.hydrology.lake_radius
    dist_sq_lake = (hf.X - lx)**2 + (hf.Y - ly)**2
    in_lake = (dist_sq_lake <= (lr * 1.05)**2) & (hf.Z <= lake_lvl + 0.1)

    # 2. Coastal marine bay & submerged depressions
    dist_sq_bay = (hf.X - 38.0)**2 + (hf.Y - (-42.0))**2
    in_bay = (dist_sq_bay <= 25.0**2) & (hf.Z <= sea_lvl + 0.1)
    submerged_sea = (hf.Z < sea_lvl - 0.5)
    in_sea = in_bay | submerged_sea

    # 3. Active river channel
    in_river = np.zeros_like(in_sea, dtype=bool)
    if config.hydrology.river_enabled:
        from terra_forge.core.hydrology import HydrologyEngine
        he = HydrologyEngine(hf, config.hydrology)
        t_pts = np.linspace(0.0, 1.0, 80)
        rx, ry, rz, rw = he.evaluate_river_spline(t_pts)
        for i in range(len(rx)):
            dist_sq = (hf.X - rx[i])**2 + (hf.Y - ry[i])**2
            in_river |= (dist_sq <= (rw[i] * 0.75)**2) & (hf.Z <= rz[i] + 0.3)

    water_mask = in_lake | in_sea | in_river
    passability[water_mask] = 0

    # Steep cliffs
    cliff_mask = (slopes > 40.0) & (~water_mask)
    passability[cliff_mask] = 2

    # High Alpine
    alpine_mask = (hf.Z > 22.0) & (~cliff_mask) & (~water_mask)
    passability[alpine_mask] = 3

    # Downsample matrix slightly for compact JSON (e.g. 64x64 or 128x128)
    export_res = min(128, hf.res_x)
    step = hf.res_x // export_res

    z_sub = np.round(hf.Z[::step, ::step], 2).tolist()
    pass_sub = passability[::step, ::step].tolist()

    data: Dict[str, Any] = {
        "generator": "TerraForge",
        "name": config.name,
        "theme": config.theme,
        "bounds": {
            "width": config.diorama.width,
            "depth": config.diorama.depth,
            "z_base": config.diorama.z_base,
            "min_x": -config.diorama.width / 2.0,
            "max_x": config.diorama.width / 2.0,
            "min_y": -config.diorama.depth / 2.0,
            "max_y": config.diorama.depth / 2.0,
        },
        "grid_size": [export_res, export_res],
        "water_levels": {
            "lake": lake_lvl,
            "sea": sea_lvl,
            "cave_pool": config.cavern.pool_level,
        },
        "elevation_matrix": z_sub,
        "passability_matrix": pass_sub,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(out_file)
