"""
Subterranean Karst Cavern & Speleothems Builder.
Constructs vaulted subterranean chamber, rocky arch entrance, fluted speleothems,
fused columns, underground pool, and bioluminescent point lighting.
Enforces overburden rock clearance invariant (>= 5.0m).
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

try:
    import bmesh
    import bpy
    from mathutils import Matrix, Vector
    HAS_BPY = True
except ImportError:
    bmesh = None
    bpy = None
    Matrix = None
    Vector = None
    HAS_BPY = False

from terra_forge.ai.vault import AbioticType
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.shaders import (
    create_cavern_mineral_shader,
    create_triplanar_terrain_shader,
    create_water_shader,
)
from terra_forge.blender.asset_fetcher import (
    _build_cavern_column_mesh,
    _build_cavern_mineral_mesh,
    _build_stalactite_mesh,
    _build_stalagmite_mesh,
)


def build_karst_cavern(
    config: CavernConfig,
    shader_cfg: ShaderConfig,
    hf: Optional[Any] = None,
    terrain_obj: Optional[bpy.types.Object] = None,
    vault: Optional[Any] = None,
) -> list[bpy.types.Object]:
    """
    Constructs subterranean cavern geometry, rocky arch entrance, fluted speleothems,
    mineral crystal clusters, and 35W bioluminescent point lights.
    Enforces rock overburden thickness invariant: z_terrain(cx, cy) - (cz + h) >= 5.0m.
    Seamlessly carves passage into terrain_obj via Boolean and adds DATA_TRANSFER normal blending.
    """
    if not config.enabled:
        return []

    cx, cy, cz = config.center
    rx, ry, h = config.radius_x, config.radius_y, config.height

    # Invariant assertion: Rock overburden thickness >= 5.0m
    if hf is not None:
        surface_z = hf.sample_bilinear(cx, cy)
        roof_z_apex = cz + h
        overburden = surface_z - roof_z_apex
        min_clearance = getattr(config, "min_roof_clearance", 5.0)
        assert overburden >= min_clearance, (
            f"Cavern overburden breach at ({cx:.1f}, {cy:.1f}): "
            f"surface {surface_z:.2f}m - roof {roof_z_apex:.2f}m = {overburden:.2f}m "
            f"< required {min_clearance:.2f}m"
        )

    if not HAS_BPY or bpy is None:
        return []

    created_objects = []
    terrain_mat = create_triplanar_terrain_shader(shader_cfg)

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

    if not obj_cav.data.materials:
        obj_cav.data.materials.append(terrain_mat)
    else:
        obj_cav.data.materials[0] = terrain_mat
    created_objects.append(obj_cav)

    # 2. Feature 10 & 16: Seamless Karst Arch Tunnel Entrance & Cliff Carving
    arch_name = "Cave_Arch_Entrance"
    arch_mesh = bpy.data.meshes.get(arch_name + "_Mesh") or bpy.data.meshes.new(arch_name + "_Mesh")
    arch_mesh.clear_geometry()
    bm_arch = bmesh.new()

    ex, ey, ez = config.entrance_pos
    start_pt = Vector((ex, ey, ez))
    end_pt = Vector((cx, cy, cz))
    dir_vec = end_pt - start_pt
    dist = dir_vec.length
    dir_norm = dir_vec.normalized() if dist > 1e-4 else Vector((1, 0, 0))

    # Internal clearance constraint: radius >= 2.5m everywhere along centerline
    arch_r = max(2.5, config.entrance_radius)

    n_rings = 20
    n_segs = 20
    up_ref = Vector((0, 0, 1)) if abs(dir_norm.z) < 0.9 else Vector((0, 1, 0))
    right_v = dir_norm.cross(up_ref).normalized()
    up_v = right_v.cross(dir_norm).normalized()

    # Build natural karst arch portal with flaring outer flanges and porous karst geometry
    for s in range(n_rings):
        t = s / float(n_rings - 1)
        center_s = start_pt + dir_vec * t
        # Front portal flange expansion at entrance to overlap and blend with cliff cuts
        flange = 1.35 if s == 0 else (1.20 if s == 1 else (1.10 if s == 2 else 1.0))
        for i in range(n_segs):
            theta = 2.0 * math.pi * i / n_segs
            # Horseshoe / natural karst arch modulation with guaranteed clearance >= 2.5m
            r_mod = max(2.5, arch_r * (1.0 + 0.12 * math.cos(3.0 * theta))) * flange
            pt = center_s + right_v * (r_mod * math.cos(theta)) + up_v * (r_mod * math.sin(theta))
            bm_arch.verts.new(pt)

    bm_arch.verts.ensure_lookup_table()

    # Create inverted (interior-facing) quad walls
    for s in range(n_rings - 1):
        r0 = s * n_segs
        r1 = (s + 1) * n_segs
        for i in range(n_segs):
            i_next = (i + 1) % n_segs
            bm_arch.faces.new([
                bm_arch.verts[r0 + i],
                bm_arch.verts[r1 + i],
                bm_arch.verts[r1 + i_next],
                bm_arch.verts[r0 + i_next],
            ])

    for f in bm_arch.faces:
        f.smooth = True

    bm_arch.to_mesh(arch_mesh)
    bm_arch.free()

    obj_arch = bpy.data.objects.get(arch_name)
    if not obj_arch:
        obj_arch = bpy.data.objects.new(arch_name, arch_mesh)
        bpy.context.scene.collection.objects.link(obj_arch)
    else:
        obj_arch.data = arch_mesh

    # Apply Boolean carve on terrain_obj if present
    if terrain_obj is not None and getattr(terrain_obj, "data", None):
        try:
            cutter_mesh = bpy.data.meshes.new("Cave_Cutter_Temp_Mesh")
            bm_cut = bmesh.new()
            cut_len = min(15.0, max(5.0, dist * 0.5))
            rot_mat = up_ref.rotation_difference(dir_norm).to_matrix().to_4x4()
            bmesh.ops.create_cone(
                bm_cut,
                cap_ends=True,
                cap_tris=False,
                segments=16,
                radius1=arch_r * 0.95,
                radius2=arch_r * 0.95,
                depth=cut_len,
                matrix=Matrix.Translation(start_pt + dir_norm * (cut_len * 0.5)) @ rot_mat,
            )
            bm_cut.to_mesh(cutter_mesh)
            bm_cut.free()

            cutter_obj = bpy.data.objects.new("Cave_Cutter_Temp", cutter_mesh)
            bpy.context.scene.collection.objects.link(cutter_obj)

            bool_mod = terrain_obj.modifiers.new(name="Boolean_Karst_Entrance", type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = cutter_obj
            bool_mod.solver = 'EXACT'

            ctx = bpy.context
            ctx.view_layer.objects.active = terrain_obj
            bpy.ops.object.modifier_apply(modifier="Boolean_Karst_Entrance")

            bpy.data.objects.remove(cutter_obj, do_unlink=True)
            bpy.data.meshes.remove(cutter_mesh)
        except Exception:
            pass

    # Assign triplanar terrain shader across arch
    if not obj_arch.data.materials:
        obj_arch.data.materials.append(terrain_mat)
    else:
        obj_arch.data.materials[0] = terrain_mat

    # Add DATA_TRANSFER normal blending modifier from terrain
    if terrain_obj is not None:
        try:
            dt_mod = obj_arch.modifiers.new(name="NormalTransfer_Terrain", type='DATA_TRANSFER')
            dt_mod.object = terrain_obj
            dt_mod.use_loop_data = True
            dt_mod.data_types_loops = {'CUSTOM_NORMAL'}
            dt_mod.loop_mapping = 'NEAREST_POLYNOR'
        except Exception:
            pass

    obj_arch["abiotic_type"] = int(AbioticType.KarstArch)
    obj_arch["entrance_radius"] = float(arch_r)
    created_objects.append(obj_arch)

    # 3. Speleothems (Fluted Stalactites, Layered Stalagmites, Flowstone Columns)
    speleo_name = "Cave_Speleothems"
    speleo_mesh = bpy.data.meshes.get(speleo_name + "_Mesh") or bpy.data.meshes.new(speleo_name + "_Mesh")
    speleo_mesh.clear_geometry()
    bm_sp = bmesh.new()

    roof_z = cz + h * 0.75
    floor_z = cz - h * 0.75

    rng_seed = 42
    rd = random.Random(rng_seed)

    # Stalactites (hanging down from ceiling apex cz + h*0.75)
    for i in range(config.stalactite_count):
        ang = rd.uniform(0, 2 * math.pi)
        r_dist = rd.uniform(1.0, rx * 0.7)
        px = cx + math.cos(ang) * r_dist
        py = cy + math.sin(ang) * r_dist
        cone_len = rd.uniform(1.8, 3.5)

        s_verts, s_faces = _build_stalactite_mesh(target_height=cone_len)
        base_v_idx = len(bm_sp.verts)
        for v in s_verts:
            bx = px + v[0]
            by = py + v[2]
            bz = roof_z - (cone_len - v[1])
            bm_sp.verts.new((bx, by, bz))
        bm_sp.verts.ensure_lookup_table()
        for f in s_faces:
            bm_sp.faces.new([bm_sp.verts[base_v_idx + idx] for idx in reversed(f)])

    # Stalagmites (rising up from floor_z)
    for i in range(config.stalagmite_count):
        ang = rd.uniform(0, 2 * math.pi)
        r_dist = rd.uniform(1.0, rx * 0.7)
        px = cx + math.cos(ang) * r_dist
        py = cy + math.sin(ang) * r_dist
        cone_len = rd.uniform(1.5, 3.2)

        sm_verts, sm_faces = _build_stalagmite_mesh(target_height=cone_len)
        base_v_idx = len(bm_sp.verts)
        for v in sm_verts:
            bx = px + v[0]
            by = py + v[2]
            bz = floor_z + v[1]
            bm_sp.verts.new((bx, by, bz))
        bm_sp.verts.ensure_lookup_table()
        for f in sm_faces:
            bm_sp.faces.new([bm_sp.verts[base_v_idx + idx] for idx in f])

    # Columns (fused ceiling-to-floor flowstone pillars)
    col_h = roof_z - floor_z
    for i in range(config.column_count):
        ang = (i / float(max(1, config.column_count))) * 2 * math.pi + 0.3
        px = cx + math.cos(ang) * (rx * 0.45)
        py = cy + math.sin(ang) * (ry * 0.45)

        c_verts, c_faces = _build_cavern_column_mesh(target_height=col_h)
        base_v_idx = len(bm_sp.verts)
        for v in c_verts:
            bx = px + v[0]
            by = py + v[2]
            bz = floor_z + v[1]
            bm_sp.verts.new((bx, by, bz))
        bm_sp.verts.ensure_lookup_table()
        for f in c_faces:
            bm_sp.faces.new([bm_sp.verts[base_v_idx + idx] for idx in f])

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
    else:
        obj_sp.data.materials[0] = terrain_mat
    created_objects.append(obj_sp)

    # 4. Subterranean Cave Pool
    if config.subterranean_pool:
        pool_name = "Water_Cave_Pool"
        pool_mesh = bpy.data.meshes.get(pool_name + "_Mesh") or bpy.data.meshes.new(pool_name + "_Mesh")
        pool_mesh.clear_geometry()
        bm_pool = bmesh.new()

        pool_z = float(getattr(config, "pool_level", cz - h * 0.5))
        pool_r = min(rx, ry) * 0.6

        bmesh.ops.create_circle(
            bm_pool,
            cap_ends=True,
            segments=32,
            radius=pool_r,
            matrix=Matrix.Translation((cx, cy, pool_z)),
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
        else:
            obj_pool.data.materials[0] = water_mat
        created_objects.append(obj_pool)

    # 5. Mineral Crystal Clusters (12 faceted druse clusters along pool edge and walls)
    min_name = "Cave_Mineral_Clusters"
    min_mesh = bpy.data.meshes.get(min_name + "_Mesh") or bpy.data.meshes.new(min_name + "_Mesh")
    min_mesh.clear_geometry()
    bm_min = bmesh.new()

    mineral_positions = []
    pool_r = min(rx, ry) * 0.6 if config.subterranean_pool else rx * 0.4
    cluster_count = 12

    for i in range(cluster_count):
        theta = (i / float(cluster_count)) * 2.0 * math.pi + 0.15
        if i % 2 == 0:
            dist_c = pool_r * 0.95
        else:
            dist_c = min(rx, ry) * 0.78
        px = cx + math.cos(theta) * dist_c
        py = cy + math.sin(theta) * dist_c
        pz = floor_z
        mineral_positions.append((px, py, pz))

        m_height = rd.uniform(1.2, 1.8)
        m_verts, m_faces = _build_cavern_mineral_mesh(target_height=m_height)
        base_v_idx = len(bm_min.verts)
        for v in m_verts:
            bx = px + v[0]
            by = py + v[2]
            bz = pz + v[1]
            bm_min.verts.new((bx, by, bz))
        bm_min.verts.ensure_lookup_table()
        for f in m_faces:
            bm_min.faces.new([bm_min.verts[base_v_idx + idx] for idx in f])

    for f in bm_min.faces:
        f.smooth = True
    bm_min.to_mesh(min_mesh)
    bm_min.free()

    obj_min = bpy.data.objects.get(min_name)
    if not obj_min:
        obj_min = bpy.data.objects.new(min_name, min_mesh)
        bpy.context.scene.collection.objects.link(obj_min)
    else:
        obj_min.data = min_mesh

    mineral_mat = create_cavern_mineral_shader(config.glow_color, emission_strength=4.5)
    if not obj_min.data.materials:
        obj_min.data.materials.append(mineral_mat)
    else:
        obj_min.data.materials[0] = mineral_mat
    created_objects.append(obj_min)

    # 6. Point Lights (35W point lights illuminating the crystal clusters)
    if config.bioluminescent:
        for i, (px, py, pz) in enumerate(mineral_positions):
            if i == 0:
                light_name = "Cave_Light_Point_35W"
            else:
                light_name = f"Cave_Light_Point_35W_{i+1:02d}"

            obj_light = bpy.data.objects.get(light_name)
            if not obj_light:
                light_data = bpy.data.lights.new(name=light_name + "_Data", type="POINT")
                obj_light = bpy.data.objects.new(light_name, light_data)
                bpy.context.scene.collection.objects.link(obj_light)

            obj_light.location = (px, py, pz + 1.2)
            obj_light.data.energy = 35.0
            obj_light.data.color = config.glow_color
            created_objects.append(obj_light)

    return created_objects

