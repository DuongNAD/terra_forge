"""
Multi-Tier Hydrology & Water Mesh Construction.
Generates 3D water geometry for lakes, river ribbons, cascades, marine bays, and cave pools.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import bmesh
    import bpy
    HAS_BPY = True
except ImportError:
    bmesh = None
    bpy = None
    HAS_BPY = False

import numpy as np

from terra_forge.blender.shaders import create_water_shader
from terra_forge.schema.map_config import HydrologyConfig, ShaderConfig


def build_water_bodies(
    config: HydrologyConfig,
    shader_cfg: ShaderConfig,
    diorama_bounds: Any = None,
    hf: Any = None,
) -> list[Any]:
    """
    Constructs all continuous hydrological water meshes:
    1. Watertight Central Lake body with retaining freeboard perimeter collar.
    2. Dynamic Marine Bay / Coastal plane with downward diorama skirt walls.
    3. Dynamic River Meander Ribbon respecting arbitrary river_stages per preset.
    """
    if not HAS_BPY or bpy is None or bmesh is None:
        return []
    water_mat = create_water_shader(shader_cfg)
    created_objects = []

    # Diorama boundary extents
    d_width = getattr(diorama_bounds, "width", 160.0) if diorama_bounds else 160.0
    d_depth = getattr(diorama_bounds, "depth", 160.0) if diorama_bounds else 160.0
    z_floor = getattr(diorama_bounds, "z_base", -16.0) if diorama_bounds else -16.0

    # 1. Central Lake Mesh (Watertight Manifold Cylinder)
    if config.lake_radius > 0.0:
        lake_name = "Water_Lake_Central"
        lake_mesh = bpy.data.meshes.get(lake_name + "_Mesh") or bpy.data.meshes.new(lake_name + "_Mesh")
        lake_mesh.clear_geometry()
        bm_lake = bmesh.new()

        cx, cy = float(config.lake_center[0]), float(config.lake_center[1])
        z_top = float(config.lake_level)
        z_bot = z_top - max(1.5, float(config.lake_depth))
        segments = 48
        radius = float(config.lake_radius)

        # Generate top and bottom rim vertices
        top_verts = []
        bot_verts = []
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            top_verts.append(bm_lake.verts.new((x, y, z_top)))
            bot_verts.append(bm_lake.verts.new((x, y, z_bot)))

        bm_lake.verts.ensure_lookup_table()

        # Top surface cap face
        bm_lake.faces.new(top_verts)
        # Bottom basin floor face (reversed winding for correct outward normals)
        bm_lake.faces.new(list(reversed(bot_verts)))

        # Watertight perimeter skirt retaining collar
        for i in range(segments):
            i_next = (i + 1) % segments
            v_tl = top_verts[i]
            v_tr = top_verts[i_next]
            v_br = bot_verts[i_next]
            v_bl = bot_verts[i]
            bm_lake.faces.new([v_tl, v_bl, v_br, v_tr])

        for f in bm_lake.faces:
            f.smooth = True

        bmesh.ops.recalc_face_normals(bm_lake, faces=bm_lake.faces)
        bm_lake.to_mesh(lake_mesh)
        bm_lake.free()

        obj_lake = bpy.data.objects.get(lake_name)
        if not obj_lake:
            obj_lake = bpy.data.objects.new(lake_name, lake_mesh)
            bpy.context.scene.collection.objects.link(obj_lake)
        else:
            obj_lake.data = lake_mesh

        if not obj_lake.data.materials:
            obj_lake.data.materials.append(water_mat)
        else:
            obj_lake.data.materials[0] = water_mat
        created_objects.append(obj_lake)

    # 2. Marine Bay Coastal Plane with Watertight Freeboard Skirts
    if getattr(config, "bay_enabled", True) and config.sea_level > -50.0:
        bay_name = "Water_Bay_Marine"
        bay_mesh = bpy.data.meshes.get(bay_name + "_Mesh") or bpy.data.meshes.new(bay_name + "_Mesh")
        bay_mesh.clear_geometry()
        bm_bay = bmesh.new()

        z_sea = float(config.sea_level)
        x_max = d_width * 0.5
        y_min = -d_depth * 0.5
        x_min = x_max - 50.0
        y_max = y_min + 50.0

        # Subdivided surface grid for wave deformation
        res_grid = 8
        grid_verts = []
        for iy in range(res_grid + 1):
            row = []
            v_frac = iy / res_grid
            vy = y_min + (y_max - y_min) * v_frac
            for ix in range(res_grid + 1):
                u_frac = ix / res_grid
                vx = x_min + (x_max - x_min) * u_frac
                row.append(bm_bay.verts.new((vx, vy, z_sea)))
            grid_verts.append(row)

        bm_bay.verts.ensure_lookup_table()

        # Grid surface quads
        for iy in range(res_grid):
            for ix in range(res_grid):
                v00 = grid_verts[iy][ix]
                v10 = grid_verts[iy][ix + 1]
                v11 = grid_verts[iy + 1][ix + 1]
                v01 = grid_verts[iy + 1][ix]
                bm_bay.faces.new([v00, v10, v11, v01])

        # Watertight downward skirt on outer diorama boundaries (South edge y=y_min, East edge x=x_max)
        z_skirt_bot = max(z_floor, z_sea - 14.0)

        # South edge skirt
        for ix in range(res_grid):
            vt0 = grid_verts[0][ix]
            vt1 = grid_verts[0][ix + 1]
            vb0 = bm_bay.verts.new((vt0.co.x, vt0.co.y, z_skirt_bot))
            vb1 = bm_bay.verts.new((vt1.co.x, vt1.co.y, z_skirt_bot))
            bm_bay.faces.new([vt0, vb0, vb1, vt1])

        # East edge skirt
        for iy in range(res_grid):
            vt0 = grid_verts[iy][res_grid]
            vt1 = grid_verts[iy + 1][res_grid]
            vb0 = bm_bay.verts.new((vt0.co.x, vt0.co.y, z_skirt_bot))
            vb1 = bm_bay.verts.new((vt1.co.x, vt1.co.y, z_skirt_bot))
            bm_bay.faces.new([vt0, vt1, vb1, vb0])

        for f in bm_bay.faces:
            f.smooth = True

        bmesh.ops.recalc_face_normals(bm_bay, faces=bm_bay.faces)
        bm_bay.to_mesh(bay_mesh)
        bm_bay.free()

        obj_bay = bpy.data.objects.get(bay_name)
        if not obj_bay:
            obj_bay = bpy.data.objects.new(bay_name, bay_mesh)
            bpy.context.scene.collection.objects.link(obj_bay)
        else:
            obj_bay.data = bay_mesh

        if not obj_bay.data.materials:
            obj_bay.data.materials.append(water_mat)
        else:
            obj_bay.data.materials[0] = water_mat
        created_objects.append(obj_bay)

    # 3. Dynamic River Ribbon Geometry (Respecting arbitrary river_stages)
    if config.river_enabled and config.river_stages:
        river_name = "Water_River_Meander"
        river_mesh = bpy.data.meshes.get(river_name + "_Mesh") or bpy.data.meshes.new(river_name + "_Mesh")
        river_mesh.clear_geometry()
        bm_riv = bmesh.new()

        from terra_forge.core.heightfield import Heightfield2D
        from terra_forge.core.hydrology import HydrologyEngine

        active_hf = hf if hf is not None else Heightfield2D(resolution=16)
        he = HydrologyEngine(active_hf, config)

        stages = config.river_stages
        num_stages = len(stages)
        t_prev = 0.0

        def add_ribbon_strip(t_vals: np.ndarray, is_outlet: bool = False):
            rx, ry, rz, rw = he.evaluate_river_spline(t_vals)
            steps = len(t_vals)
            if steps < 2:
                return

            left_verts = []
            right_verts = []
            pz_prev = None

            for i in range(steps):
                if i < steps - 1:
                    tx = float(rx[i + 1] - rx[i])
                    ty = float(ry[i + 1] - ry[i])
                else:
                    tx = float(rx[i] - rx[i - 1])
                    ty = float(ry[i] - ry[i - 1])

                t_len = math.sqrt(tx**2 + ty**2) + 1e-6
                nx = -ty / t_len
                ny = tx / t_len
                half_w = float(rw[i]) * 0.5

                px = float(rx[i])
                py = float(ry[i])

                if hf is not None:
                    z_bed = float(hf.sample_bilinear(px, py))
                    # Water rests snugly in the carved channel
                    pz_candidate = z_bed + min(0.20, max(0.08, float(config.river_carve_depth) * 0.15))

                    # Near central lake, blend seamlessly to lake_level
                    if config.lake_radius > 0.0:
                        lx, ly = float(config.lake_center[0]), float(config.lake_center[1])
                        d_lake = math.hypot(px - lx, py - ly)
                        if d_lake <= config.lake_radius + 4.0:
                            blend = max(0.0, min(1.0, (d_lake - config.lake_radius) / 4.0))
                            pz_candidate = float(config.lake_level) * (1.0 - blend) + pz_candidate * blend

                    # Monotonic downhill flow (water doesn't flow uphill)
                    if pz_prev is not None and not (is_outlet and i == 0):
                        pz = min(pz_candidate, pz_prev)
                    else:
                        pz = pz_candidate
                    pz_prev = pz
                else:
                    pz = float(rz[i])

                vl = bm_riv.verts.new((px - nx * half_w, py - ny * half_w, pz))
                vr = bm_riv.verts.new((px + nx * half_w, py + ny * half_w, pz))
                left_verts.append(vl)
                right_verts.append(vr)

            bm_riv.verts.ensure_lookup_table()
            uv_lay = bm_riv.loops.layers.uv.verify()
            for i in range(steps - 1):
                v0 = left_verts[i]
                v1 = right_verts[i]
                v2 = right_verts[i + 1]
                v3 = left_verts[i + 1]
                face = bm_riv.faces.new([v0, v1, v2, v3])
                u0 = i / float(max(1, steps - 1))
                u1 = (i + 1) / float(max(1, steps - 1))
                for loop in face.loops:
                    if loop.vert == v0:
                        loop[uv_lay].uv = (u0, 0.0)
                    elif loop.vert == v1:
                        loop[uv_lay].uv = (u0, 1.0)
                    elif loop.vert == v2:
                        loop[uv_lay].uv = (u1, 1.0)
                    elif loop.vert == v3:
                        loop[uv_lay].uv = (u1, 0.0)

        # Dynamically build ribbons for all non-submerged river stages
        for i, stage in enumerate(stages):
            t_start = float(stage.get("t_start", t_prev))
            t_end = float(stage.get("t_end", (i + 1.0) / num_stages if i < num_stages - 1 else 1.00))
            t_prev = t_end

            stage_name = stage.get("name", "")
            is_lake_transit = stage.get("is_lake_transit", False) or ("Transit" in stage_name) or (stage_name == "Central Lake Transit") or ("Lake" in stage_name and "Inflow" not in stage_name and "Cascades" not in stage_name and "Gorge" not in stage_name)
            if is_lake_transit and config.lake_radius > 0.0:
                continue

            is_outlet = stage.get("is_outlet", False) or ("Outlet" in stage_name) or ("Waterfall" in stage_name)
            stage_span = max(0.05, t_end - t_start)
            num_steps = max(16, int(64 * (stage_span / 0.50)))
            t_samples = np.linspace(t_start, t_end, num_steps)
            add_ribbon_strip(t_samples, is_outlet=is_outlet)

        if bm_riv.faces:
            for f in bm_riv.faces:
                f.smooth = True
            bmesh.ops.recalc_face_normals(bm_riv, faces=bm_riv.faces)
            bm_riv.to_mesh(river_mesh)
        bm_riv.free()

        obj_riv = bpy.data.objects.get(river_name)
        if not obj_riv:
            obj_riv = bpy.data.objects.new(river_name, river_mesh)
            bpy.context.scene.collection.objects.link(obj_riv)
        else:
            obj_riv.data = river_mesh

        if not obj_riv.data.materials:
            obj_riv.data.materials.append(water_mat)
        else:
            obj_riv.data.materials[0] = water_mat
        created_objects.append(obj_riv)

    return created_objects
