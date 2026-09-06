"""
Unified Blender Runner for TerraForge.
This script is executed inside Blender (either via -b -P in headless mode
or dynamically sent to Blender MCP in live mode).
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Add terra_forge to sys.path if not present
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from terra_forge.schema.map_config import MapConfig
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.core.biomes import BiomeEngine
from terra_forge.core.diorama_slab import DioramaSlabBuilder

from terra_forge.blender.mesh_builder import build_diorama_terrain_object
from terra_forge.blender.water_builder import build_water_bodies
from terra_forge.blender.cave_builder import build_karst_cavern
from terra_forge.blender.asset_fetcher import resolve_prototype_object
from terra_forge.blender.geonodes import apply_biome_scatter_modifier
from terra_forge.blender.exporter import (
    export_production_glb,
    export_blend_file,
    export_simulation_heightmap_json,
)
from terra_forge.inspector.cameras import setup_inspection_cameras
from terra_forge.inspector.validator import QualityValidator
from terra_forge.inspector.vision_rig import render_inspection_snapshots, build_2x2_vision_grid


def run_pipeline(config_dict_or_path: dict | str, render_preview: bool = False, preview_dir: str = "previews") -> dict:
    """Executes the complete end-to-end TerraForge diorama generation pipeline."""
    t0 = time.perf_counter()

    if isinstance(config_dict_or_path, (str, Path)):
        with open(config_dict_or_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    else:
        cfg_data = config_dict_or_path

    config = MapConfig.model_validate(cfg_data)
    print(f"[TerraForge] Starting generation: {config.name} (theme: {config.theme}, seed: {config.seed})...")

    import bpy
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    # Step 1: Synthesize Base Heightfield Topography
    t_topo_0 = time.perf_counter()
    hf = Heightfield2D(
        width=config.diorama.width,
        depth=config.diorama.depth,
        resolution=config.diorama.resolution,
    )

    # Add subtle micro-roughness bedrock foundation
    hf.add_perlin_octaves(seed=config.seed, octaves=3, amplitude=2.5)

    for peak in config.peaks:
        hf.add_peak(
            peak.position[0],
            peak.position[1],
            peak.height,
            peak.radius,
            peak.shape,
            peak.ridge_angle_deg,
        )

    # Carve Lake Basin & Form Natural Retaining Berm
    hf.carve_lake_basin(
        config.hydrology.lake_center[0],
        config.hydrology.lake_center[1],
        config.hydrology.lake_radius,
        config.hydrology.lake_level,
        config.hydrology.lake_depth,
    )
    t_topo_1 = time.perf_counter()
    print(f"[TerraForge] Base topography synthesized in {t_topo_1 - t_topo_0:.3f}s")

    # Step 2: Geological Erosion (Hydraulic Droplets & Thermal Talus)
    t_ero_0 = time.perf_counter()
    if config.erosion.enabled:
        ero = ErosionSimulator(hf, config.erosion)
        ero.run_hydraulic_erosion(config.seed)
        ero.run_thermal_erosion()
    t_ero_1 = time.perf_counter()
    print(f"[TerraForge] Geological erosion completed in {t_ero_1 - t_ero_0:.3f}s")

    # Step 3: Hydrology Channel Carving & Water Distance Field
    t_hyd_0 = time.perf_counter()
    hydro = HydrologyEngine(hf, config.hydrology)
    hydro.carve_river_channel()
    dist_water = hydro.compute_water_distance_field()
    t_hyd_1 = time.perf_counter()
    print(f"[TerraForge] Hydrology & river carved in {t_hyd_1 - t_hyd_0:.3f}s")

    # Step 4: Build Watertight Diorama Slab Mesh Data
    t_mesh_0 = time.perf_counter()
    slab_builder = DioramaSlabBuilder(hf, config.diorama)
    mesh_data = slab_builder.build_mesh_data()
    terrain_obj = build_diorama_terrain_object(mesh_data, config)
    t_mesh_1 = time.perf_counter()
    print(f"[TerraForge] Diorama terrain mesh constructed in {t_mesh_1 - t_mesh_0:.3f}s")

    # Step 5: Build Multi-Tier Water Bodies
    water_objs = build_water_bodies(config.hydrology, config.shaders, diorama_bounds=config.diorama, hf=hf)
    print(f"[TerraForge] {len(water_objs)} water bodies built.")

    # Step 6: Build Karst Cavern & Speleothems
    cave_objs = build_karst_cavern(config.cavern, config.shaders)
    print(f"[TerraForge] Karst cavern system built ({len(cave_objs)} elements).")

    # Step 7: Parametric Geometry Nodes Scattering
    t_bio_0 = time.perf_counter()
    water_cutoff = max(float(config.hydrology.lake_level), float(config.hydrology.sea_level))
    for biome_cfg in config.biomes:
        proto_obj = resolve_prototype_object(biome_cfg.prototype)
        apply_biome_scatter_modifier(terrain_obj, proto_obj, biome_cfg, water_level=water_cutoff)
    t_bio_1 = time.perf_counter()
    print(f"[TerraForge] {len(config.biomes)} parametric scatter biomes applied in {t_bio_1 - t_bio_0:.3f}s")

    # Step 8: Setup Inspection Cameras & Validate Invariants
    setup_inspection_cameras(config=config, hf=hf)
    validator = QualityValidator(config, hf)
    validation_results = validator.validate_all()
    print(f"[TerraForge] Validation: Passed={validation_results['passed']}, Errors={len(validation_results['errors'])}, Warnings={len(validation_results['warnings'])}")

    # Step 9: Production Exports
    export_outputs = {}
    if config.export.export_blend:
        p = export_blend_file(config.export.blend_path)
        export_outputs["blend"] = p
        print(f"[TerraForge] Saved master .blend: {p}")

    if config.export.export_glb:
        p = export_production_glb(config.export.glb_path)
        export_outputs["glb"] = p
        print(f"[TerraForge] Exported Three.js GLB: {p}")

    if config.export.export_heightmap_json:
        p = export_simulation_heightmap_json(config.export.heightmap_path, hf, config)
        export_outputs["heightmap_json"] = p
        print(f"[TerraForge] Exported simulation heightmap JSON: {p}")

    # Step 10: Render Previews if requested
    preview_grid_path = None
    if render_preview:
        print("[TerraForge] Rendering multi-angle inspection snapshots...")
        snap_paths = render_inspection_snapshots(preview_dir, config=config, hf=hf)
        grid_file = os.path.join(preview_dir, f"{config.name}_preview_2x2.png")
        preview_grid_path = build_2x2_vision_grid(snap_paths, grid_file)
        print(f"[TerraForge] Built 2x2 vision grid preview: {preview_grid_path}")

    total_time = time.perf_counter() - t0
    print(f"[TerraForge] Pipeline successfully completed in {total_time:.2f}s!")

    return {
        "status": "success",
        "name": config.name,
        "theme": config.theme,
        "execution_time_sec": round(total_time, 2),
        "validation": validation_results,
        "exports": export_outputs,
        "preview_grid": preview_grid_path,
    }


if __name__ == "__main__":
    # If run directly via Blender: blender -b -P runner.py -- <config.json>
    cfg_file = None
    render_flag = False

    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        custom_args = args[idx + 1:]
        if len(custom_args) > 0:
            cfg_file = custom_args[0]
        if "--render" in custom_args:
            render_flag = True

    if not cfg_file:
        # Default fallback to alpine_fjord preset
        cfg_file = str(script_dir / "terra_forge" / "schema" / "presets" / "alpine_fjord.json")

    try:
        results = run_pipeline(cfg_file, render_preview=render_flag)
        print("RESULT_JSON:" + json.dumps(results))
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
