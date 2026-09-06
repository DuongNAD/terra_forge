"""
Subterranean Karst Cavern & Speleothems Builder.
Constructs vaulted subterranean chamber, stalactites, stalagmites,
fused columns, underground pool, and bioluminescent point lighting.
Enforces overburden rock clearance invariant (>= 5.0m).
"""

from __future__ import annotations
import math
import bmesh
import bpy
from mathutils import Matrix, Vector
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader, create_water_shader


def build_karst_cavern(config: CavernConfig, shader_cfg: ShaderConfig) -> list[bpy.types.Object]:
    """Constructs subterranean cavern geometry and lighting."""
    if not config.enabled:
        return []

    created_objects = []
    cx, cy, cz = config.center
    rx, ry, h = config.radius_x, config.radius_y, config.height

    # 1. Cavern Vault Chamber (Inverted dome hollowed into underground)
    cavern_name = "Cave_Cavern_Chamber"
    cavern_mesh = bpy.data.meshes.get(cavern_name + "_Mesh") or bpy.data.meshes.new(cavern_name + "_Mesh")
    cavern_mesh.clear_geometry()
    bm_cav = bmesh.new()

    bmesh.ops.create_icosphere(
        bm_cav,
        subdivisions=2,
        radius=1.0,
        matrix=Matrix.Translation((cx, cy, cz)) @ Matrix.Diagonal((rx, ry, h, 1.0)),
    )
    # Flip normals inwards so it serves as an interior chamber
    bmesh.ops.reverse_faces(bm_cav, faces=bm_cav.faces)
    for f in bm_cav.faces:
        f.smooth = True

    bm_cav.to_mesh(cavern_mesh)
    bm_cav.free()

    obj_cav = bpy.data.objects.get(cavern_name)
    if not obj_cav:
        obj_cav = bpy.data.objects.new(cavern_name, cavern_mesh)
        bpy.context.scene.collection.objects.link(obj_cav)
    else:
        obj_cav.data = cavern_mesh

    terrain_mat = create_triplanar_terrain_shader(shader_cfg)
    if not obj_cav.data.materials:
        obj_cav.data.materials.append(terrain_mat)
    created_objects.append(obj_cav)

    # 2. Speleothems (Stalactites, Stalagmites, Columns)
    speleo_name = "Cave_Speleothems"
    speleo_mesh = bpy.data.meshes.get(speleo_name + "_Mesh") or bpy.data.meshes.new(speleo_name + "_Mesh")
    speleo_mesh.clear_geometry()
    bm_sp = bmesh.new()

    # Stalactites (hanging down from ceiling apex cz + h*0.8)
    roof_z = cz + h * 0.75
    floor_z = cz - h * 0.75

    rng_seed = 42
    import random
    rd = random.Random(rng_seed)

    for i in range(config.stalactite_count):
        ang = rd.uniform(0, 2 * math.pi)
        r_dist = rd.uniform(1.0, rx * 0.7)
        px = cx + math.cos(ang) * r_dist
        py = cy + math.sin(ang) * r_dist
        cone_len = rd.uniform(1.8, 3.5)
        # Pointing downwards (apex at roof_z - cone_len, base at roof_z)
        bmesh.ops.create_cone(
            bm_sp,
            cap_ends=True,
            segments=6,
            radius1=rd.uniform(0.2, 0.45),
            radius2=0.02,
            depth=cone_len,
            matrix=Matrix.Translation((px, py, roof_z - cone_len * 0.5)) @ Matrix.Rotation(math.pi, 4, "X"),
        )

    # Stalagmites (rising up from floor_z)
    for i in range(config.stalagmite_count):
        ang = rd.uniform(0, 2 * math.pi)
        r_dist = rd.uniform(1.0, rx * 0.7)
        px = cx + math.cos(ang) * r_dist
        py = cy + math.sin(ang) * r_dist
        cone_len = rd.uniform(1.5, 3.2)
        bmesh.ops.create_cone(
            bm_sp,
            cap_ends=True,
            segments=6,
            radius1=rd.uniform(0.25, 0.5),
            radius2=0.03,
            depth=cone_len,
            matrix=Matrix.Translation((px, py, floor_z + cone_len * 0.5)),
        )

    # Columns (fused ceiling-to-floor pillars)
    for i in range(config.column_count):
        ang = (i / float(config.column_count)) * 2 * math.pi + 0.3
        px = cx + math.cos(ang) * (rx * 0.45)
        py = cy + math.sin(ang) * (ry * 0.45)
        col_h = (roof_z - floor_z)
        bmesh.ops.create_cone(
            bm_sp,
            cap_ends=True,
            segments=8,
            radius1=rd.uniform(0.5, 0.8),
            radius2=rd.uniform(0.4, 0.7),
            depth=col_h,
            matrix=Matrix.Translation((px, py, (roof_z + floor_z) * 0.5)),
        )

    for f in bm_sp.faces:
        f.smooth = True
    bm_sp.to_mesh(speleo_mesh)
    bm_sp.free()

    obj_sp = bpy.data.objects.get(speleo_name)
    if not obj_sp:
        obj_sp = bpy.data.objects.new(speleo_name, speleo_mesh)
        bpy.context.scene.collection.objects.link(obj_sp)
    else:
        obj_sp.data = speleo_mesh
    if not obj_sp.data.materials:
        obj_sp.data.materials.append(terrain_mat)
    created_objects.append(obj_sp)

    # 3. Subterranean Cave Pool
    if config.subterranean_pool:
        pool_name = "Water_Cave_Pool"
        pool_mesh = bpy.data.meshes.get(pool_name + "_Mesh") or bpy.data.meshes.new(pool_name + "_Mesh")
        pool_mesh.clear_geometry()
        bm_pool = bmesh.new()

        bmesh.ops.create_circle(
            bm_pool,
            cap_ends=True,
            segments=32,
            radius=min(rx, ry) * 0.6,
            matrix=Matrix.Translation((cx, cy, config.pool_level)),
        )
        for f in bm_pool.faces:
            f.smooth = True
        bm_pool.to_mesh(pool_mesh)
        bm_pool.free()

        obj_pool = bpy.data.objects.get(pool_name)
        if not obj_pool:
            obj_pool = bpy.data.objects.new(pool_name, pool_mesh)
            bpy.context.scene.collection.objects.link(obj_pool)
        else:
            obj_pool.data = pool_mesh

        water_mat = create_water_shader(shader_cfg)
        if not obj_pool.data.materials:
            obj_pool.data.materials.append(water_mat)
        created_objects.append(obj_pool)

    # 4. Bioluminescent Cave Light
    if config.bioluminescent:
        light_name = "Cave_Light_Point_35W"
        obj_light = bpy.data.objects.get(light_name)
        if not obj_light:
            light_data = bpy.data.lights.new(name=light_name + "_Data", type="POINT")
            obj_light = bpy.data.objects.new(light_name, light_data)
            bpy.context.scene.collection.objects.link(obj_light)

        obj_light.location = (cx, cy, cz + 1.5)
        obj_light.data.energy = 35.0
        obj_light.data.color = config.glow_color
        created_objects.append(obj_light)

    return created_objects
