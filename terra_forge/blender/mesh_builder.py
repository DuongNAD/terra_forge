"""
Blender Diorama Mesh Construction.
Converts Heightfield & Diorama Slab data into high-performance BMesh geometry,
applies geological vertex color stratification and PBR materials.
"""

from __future__ import annotations
from typing import Dict, Any
import bmesh
import bpy
from terra_forge.schema.map_config import MapConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader


def build_diorama_terrain_object(mesh_data: Dict[str, Any], config: MapConfig) -> bpy.types.Object:
    """
    Constructs or updates the 'Diorama_Island_Block' mesh object in Blender.
    Applies vertex colors (COLOR_0), smooth face shading, and triplanar material.
    """
    obj_name = "Diorama_Island_Block"
    existing_obj = bpy.data.objects.get(obj_name)

    mesh_name = "Diorama_Island_Mesh"
    mesh = bpy.data.meshes.get(mesh_name)
    if mesh is None:
        mesh = bpy.data.meshes.new(mesh_name)
    else:
        mesh.clear_geometry()

    # Load vertices and faces
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    colors = mesh_data.get("vertex_colors", [])

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Apply Smooth Shading
    for poly in mesh.polygons:
        poly.use_smooth = True

    # Apply Strata Vertex Color Layer (COLOR_0)
    if colors and len(colors) == len(vertices):
        if not mesh.color_attributes.get("COLOR_0"):
            color_attr = mesh.color_attributes.new(name="COLOR_0", type="FLOAT_COLOR", domain="POINT")
        else:
            color_attr = mesh.color_attributes.get("COLOR_0")

        # Assign vertex colors
        for i, col in enumerate(colors):
            color_attr.data[i].color = col

    # Create or update Object
    if existing_obj is None:
        obj = bpy.data.objects.new(obj_name, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        obj = existing_obj
        obj.data = mesh

    # Assign Triplanar PBR Shader
    terrain_mat = create_triplanar_terrain_shader(config.shaders)
    if not obj.data.materials:
        obj.data.materials.append(terrain_mat)
    else:
        obj.data.materials[0] = terrain_mat

    return obj
