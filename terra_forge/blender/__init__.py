"""
Blender subpackage for TerraForge.
BlenderBridge is accessible outside Blender.
Builders, shaders, and exporters are loaded when running inside Blender (with bpy/bmesh).
"""

from terra_forge.blender.bridge import BlenderBridge

__all__ = ["BlenderBridge"]

try:
    import bpy
    import bmesh
    from terra_forge.blender.mesh_builder import build_diorama_terrain_object
    from terra_forge.blender.water_builder import build_water_bodies
    from terra_forge.blender.cave_builder import build_karst_cavern
    from terra_forge.blender.shaders import (
        create_triplanar_terrain_shader,
        create_water_shader,
        create_bioluminescent_fungus_shader,
    )
    from terra_forge.blender.geonodes import build_biome_geonodes_tree, apply_biome_scatter_modifier
    from terra_forge.blender.asset_fetcher import resolve_prototype_object
    from terra_forge.blender.exporter import (
        export_production_glb,
        export_blend_file,
        export_simulation_heightmap_json,
    )

    __all__ += [
        "build_diorama_terrain_object",
        "build_water_bodies",
        "build_karst_cavern",
        "create_triplanar_terrain_shader",
        "create_water_shader",
        "create_bioluminescent_fungus_shader",
        "build_biome_geonodes_tree",
        "apply_biome_scatter_modifier",
        "resolve_prototype_object",
        "export_production_glb",
        "export_blend_file",
        "export_simulation_heightmap_json",
    ]
except ImportError:
    pass
