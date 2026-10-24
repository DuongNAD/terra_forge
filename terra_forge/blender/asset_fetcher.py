"""
Procedural Botanical, Detritus & Aquatic Asset Generator for Blender.
Constructs built-in zero-dependency 3D prototypes:
- Canopy trees: alpine conifers, deciduous oaks, tropical palms
- Understory shrubs and bushes
- Rich ground flora: wildflowers, dense grass clumps, ferns, forest mushrooms, rock-clinging moss
- Tree decay & windthrow: fallen decaying logs, snapped weathered stumps
- Underwater ecosystems: aquatic seaweed/waterweeds, shoreline reeds, rounded riverbed pebbles, coral reefs
- Geological detritus: scree talus boulders

Each prototype has smooth shading, clean topology, and tailored PBR materials.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import bpy
    import bmesh
    from mathutils import Matrix, Vector
    HAS_BPY = True
except ImportError:
    bpy = None
    bmesh = None
    Matrix = None
    Vector = None
    HAS_BPY = False

import random

if HAS_BPY:
    try:
        from terra_forge.blender.shaders import (
            create_bioluminescent_fungus_shader,
            create_triplanar_terrain_shader,
        )
    except ImportError:
        create_bioluminescent_fungus_shader = None
        create_triplanar_terrain_shader = None
else:
    create_bioluminescent_fungus_shader = None
    create_triplanar_terrain_shader = None

from terra_forge.ai.vault import (
    DEFAULT_ABIOTIC_PROFILES,
    DEFAULT_FLORA_PROFILES,
    AbioticMetadata,
    AbioticType,
    AssetEntry,
    FloraType,
    LocalAssetVault,
)
from terra_forge.ecology.primordial import (
    AbioticPromptEngine,
    validate_abiotic_prompt,
)
from terra_forge.ai.normalizer import (
    CollisionPrimitives,
    GeometryNormalizer,
    NormalizedMesh,
    build_glb_from_mesh,
    create_procedural_primitive,
    extract_mesh_from_glb,
)

logger = logging.getLogger("terra_forge.blender.asset_fetcher")


def get_or_create_prototype_collection() -> Optional[Any]:
    """Gets or creates the 'TerraForge_Prototypes' collection and excludes it from diorama view layer."""
    if not HAS_BPY or bpy is None:
        return None
    col_name = "TerraForge_Prototypes"
    col = bpy.data.collections.get(col_name)
    if col is None:
        col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(col)
        layer_col = bpy.context.view_layer.layer_collection.children.get(col_name)
        if layer_col:
            layer_col.exclude = True
    return col


def _get_or_create_material(
    mat_name: str,
    base_color: tuple,
    roughness: float = 0.6,
    specular: float = 0.5,
    emission_color: Optional[tuple] = None,
    emission_strength: float = 0.0,
) -> Optional[Any]:
    """Helper to create or fetch a standard PBR material with Principled BSDF."""
    if not HAS_BPY or bpy is None:
        return None
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (base_color[0], base_color[1], base_color[2], 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = specular
            elif "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = specular
            if emission_color is not None and emission_strength > 0.0:
                if "Emission Color" in bsdf.inputs:
                    bsdf.inputs["Emission Color"].default_value = (emission_color[0], emission_color[1], emission_color[2], 1.0)
                elif "Emission" in bsdf.inputs:
                    bsdf.inputs["Emission"].default_value = (emission_color[0], emission_color[1], emission_color[2], 1.0)
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


# =============================================================================
# 1. Canopy Trees
# =============================================================================

def create_pine_tree(name: str = "Proto_Pine_Tall", height: float = 5.0) -> Any:
    """Builds an authentic stylized alpine conifer tree with flared bark trunk and tiered conifer branches."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Trunk (flared base root cylinder to upper canopy)
    f_start = set(bm.faces)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=8,
        radius1=0.28,
        radius2=0.08,
        depth=height * 0.70,
        matrix=Matrix.Translation((0, 0, height * 0.35)),
    )
    for f in bm.faces:
        f.material_index = 0

    f_trunk = set(bm.faces)

    # 2. Foliage branch tiers (5 tiered whorls of downward-angled conifer branches)
    tiers = [
        (height * 0.28, height * 0.24, 1.65),
        (height * 0.42, height * 0.22, 1.40),
        (height * 0.56, height * 0.20, 1.15),
        (height * 0.70, height * 0.18, 0.85),
        (height * 0.82, height * 0.22, 0.45),
    ]
    for i, (z_base, tier_h, r_base) in enumerate(tiers):
        rot_z = Matrix.Rotation(math.radians(24 * (i % 2)), 4, "Z")
        trans = Matrix.Translation((0, 0, z_base + tier_h * 0.45))
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=8,
            radius1=r_base,
            radius2=0.06 if i < 4 else 0.0,
            depth=tier_h,
            matrix=trans @ rot_z,
        )

    for f in bm.faces:
        if f not in f_trunk:
            f.material_index = 1
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_bark = _get_or_create_material("MAT_Proto_Wood_Bark", (0.16, 0.09, 0.05), roughness=0.85)
    mat_needles = _get_or_create_material("MAT_Proto_Pine_Needles", (0.04, 0.20, 0.08), roughness=0.55)
    if mat_bark:
        obj.data.materials.append(mat_bark)
    if mat_needles:
        obj.data.materials.append(mat_needles)
    return obj


def create_broadleaf_tree(name: str = "Proto_Broadleaf", height: float = 4.2) -> Any:
    """Builds an organic deciduous broadleaf canopy tree with spreading branches and clustered foliage."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Main trunk
    f_start = set(bm.faces)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=8,
        radius1=0.32,
        radius2=0.18,
        depth=height * 0.45,
        matrix=Matrix.Translation((0, 0, height * 0.225)),
    )

    # 3 Primary scaffold branches
    branches = [
        (Vector((0, 0, height * 0.38)), Vector((-0.8, -0.5, height * 0.68)), 0.14, 0.07),
        (Vector((0, 0, height * 0.40)), Vector((0.9, -0.4, height * 0.70)), 0.13, 0.07),
        (Vector((0, 0, height * 0.42)), Vector((0.1, 0.9, height * 0.74)), 0.14, 0.07),
    ]
    for p0, p1, r1, r2 in branches:
        v = p1 - p0
        mid = (p0 + p1) * 0.5
        rot = Vector((0, 0, 1)).rotation_difference(v.normalized()).to_matrix().to_4x4()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=6,
            radius1=r1,
            radius2=r2,
            depth=v.length,
            matrix=Matrix.Translation(mid) @ rot,
        )

    for f in bm.faces:
        f.material_index = 0

    f_wood = set(bm.faces)

    # 2. Multi-lobe canopy clusters nestled at branch ends
    clusters = [
        ((-0.9, -0.6, height * 0.70), 0.95),
        ((1.0, -0.4, height * 0.72), 0.90),
        ((0.1, 1.0, height * 0.76), 1.00),
        ((0.0, 0.0, height * 0.88), 1.20),
        ((-0.5, 0.4, height * 0.80), 0.85),
        ((0.6, 0.3, height * 0.78), 0.85),
    ]
    for pos, rad in clusters:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=2,
            radius=rad,
            matrix=Matrix.Translation(pos),
        )

    for f in bm.faces:
        if f not in f_wood:
            f.material_index = 1
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_bark = _get_or_create_material("MAT_Proto_Wood_Bark", (0.18, 0.12, 0.07), roughness=0.82)
    mat_leaves = _get_or_create_material("MAT_Proto_Broadleaf_Crown", (0.13, 0.36, 0.09), roughness=0.52)
    if mat_bark:
        obj.data.materials.append(mat_bark)
    if mat_leaves:
        obj.data.materials.append(mat_leaves)
    return obj


def create_palm_tree(name: str = "Proto_Palm_Tropical", height: float = 4.8) -> Any:
    """Builds a tropical palm tree with curved ringed trunk, radiating fronds, and coconuts."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Curved segmented trunk (5 segments)
    f_start = set(bm.faces)
    seg_count = 5
    dz = (height * 0.8) / seg_count
    curr_pos = Vector((0.0, 0.0, 0.0))
    for i in range(seg_count):
        r1 = 0.22 * (1.0 - 0.07 * i)
        r2 = 0.22 * (1.0 - 0.07 * (i + 1))
        lean_x = 0.08 * (i + 1)
        next_pos = Vector((curr_pos.x + lean_x, curr_pos.y, curr_pos.z + dz))
        mid_pos = (curr_pos + next_pos) * 0.5

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=7,
            radius1=r1,
            radius2=r2,
            depth=dz,
            matrix=Matrix.Translation(mid_pos),
        )
        curr_pos = next_pos

    for f in bm.faces:
        f.material_index = 0

    f_trunk = set(bm.faces)

    # 2. Palm fronds (8 radiating arched leaves)
    frond_count = 8
    crown_top = curr_pos
    for i in range(frond_count):
        angle = (i / frond_count) * 2.0 * math.pi
        rad_mat = Matrix.Rotation(angle, 4, "Z")
        tilt_mat = Matrix.Rotation(math.radians(30 + 10 * (i % 2)), 4, "Y")
        trans = Matrix.Translation((crown_top.x + math.cos(angle) * 0.8, crown_top.y + math.sin(angle) * 0.8, crown_top.z - 0.1))

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.22,
            radius2=0.02,
            depth=1.6,
            matrix=trans @ rad_mat @ tilt_mat,
        )

    for f in bm.faces:
        if f not in f_trunk:
            f.material_index = 1

    f_fronds = set(bm.faces)

    # 3. Coconut cluster at crown apex
    for cx, cy in [(0.12, 0.1), (-0.1, 0.14), (0.05, -0.15)]:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=1,
            radius=0.12,
            matrix=Matrix.Translation((crown_top.x + cx, crown_top.y + cy, crown_top.z - 0.05)),
        )

    for f in bm.faces:
        if f not in f_fronds:
            f.material_index = 2
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_trunk = _get_or_create_material("MAT_Proto_Palm_Trunk", (0.24, 0.18, 0.13), roughness=0.80)
    mat_fronds = _get_or_create_material("MAT_Proto_Palm_Tropical", (0.12, 0.44, 0.14), roughness=0.45)
    mat_coconuts = _get_or_create_material("MAT_Proto_Coconut", (0.18, 0.10, 0.04), roughness=0.70)
    if mat_trunk:
        obj.data.materials.append(mat_trunk)
    if mat_fronds:
        obj.data.materials.append(mat_fronds)
    if mat_coconuts:
        obj.data.materials.append(mat_coconuts)
    return obj


# =============================================================================
# 2. Understory Shrubs & Bushes
# =============================================================================

def create_shrub(name: str = "Proto_Understory_Shrub", height: float = 1.6) -> Any:
    """Builds a dense multi-stem understory shrub with woody branches, lush foliage clusters, and accent blossoms."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Diverging woody stems
    f_start = set(bm.faces)
    stems = [
        (Vector((0, 0, 0)), Vector((-0.25, -0.15, height * 0.42)), 0.07, 0.035),
        (Vector((0, 0, 0)), Vector((0.28, -0.12, height * 0.45)), 0.065, 0.03),
        (Vector((0, 0, 0)), Vector((0.04, 0.30, height * 0.44)), 0.065, 0.03),
    ]
    for p0, p1, r1, r2 in stems:
        v = p1 - p0
        mid = (p0 + p1) * 0.5
        rot = Vector((0, 0, 1)).rotation_difference(v.normalized()).to_matrix().to_4x4()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=5,
            radius1=r1,
            radius2=r2,
            depth=v.length,
            matrix=Matrix.Translation(mid) @ rot,
        )

    for f in bm.faces:
        f.material_index = 0

    f_wood = set(bm.faces)

    # 2. Clustered lush foliage mounds
    clumps = [
        ((-0.28, -0.18, height * 0.55), 0.55),
        ((0.32, -0.14, height * 0.58), 0.58),
        ((0.05, 0.32, height * 0.56), 0.52),
        ((0.0, 0.0, height * 0.72), 0.68),
        ((-0.1, 0.12, height * 0.62), 0.50),
    ]
    for pos, rad in clumps:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=2,
            radius=rad,
            matrix=Matrix.Translation(pos),
        )

    for f in bm.faces:
        if f not in f_wood:
            f.material_index = 1

    f_foliage = set(bm.faces)

    # 3. Accent blossoms / berries
    accents = [
        (-0.35, -0.22, height * 0.68),
        (0.42, -0.18, height * 0.70),
        (0.08, 0.40, height * 0.66),
        (0.0, 0.0, height * 0.92),
    ]
    for pos in accents:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=1,
            radius=0.08,
            matrix=Matrix.Translation(pos),
        )

    for f in bm.faces:
        if f not in f_foliage:
            f.material_index = 2
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_wood = _get_or_create_material("MAT_Proto_Wood_Bark", (0.18, 0.12, 0.07), roughness=0.82)
    mat_foliage = _get_or_create_material("MAT_Proto_Understory_Shrub", (0.16, 0.34, 0.12), roughness=0.60)
    mat_berry = _get_or_create_material("MAT_Proto_Berry_Blossom", (0.85, 0.22, 0.25), roughness=0.35)
    if mat_wood:
        obj.data.materials.append(mat_wood)
    if mat_foliage:
        obj.data.materials.append(mat_foliage)
    if mat_berry:
        obj.data.materials.append(mat_berry)
    return obj


# =============================================================================
# 3. Rich Ground Flora
# =============================================================================

def create_wildflower(name: str = "Proto_Wildflower", height: float = 0.45) -> Any:
    """Builds delicate wildflowers with slender green stem, basal leaves, vibrant petals, and golden center disc."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Slender stem & basal leaves (Material 0)
    f_start = set(bm.faces)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=5,
        radius1=0.025,
        radius2=0.015,
        depth=height,
        matrix=Matrix.Translation((0, 0, height * 0.5)),
    )
    # 2 Basal leaves
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=4,
        radius1=0.04,
        radius2=0.005,
        depth=0.16,
        matrix=Matrix.Translation((0.07, 0, 0.12)) @ Matrix.Rotation(math.radians(50), 4, "Y"),
    )
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=4,
        radius1=0.04,
        radius2=0.005,
        depth=0.16,
        matrix=Matrix.Translation((-0.07, 0, 0.14)) @ Matrix.Rotation(math.radians(-50), 4, "Y"),
    )
    for f in bm.faces:
        f.material_index = 0

    f_stem = set(bm.faces)

    # 2. Radiating Petals (Material 1)
    disc_z = height + 0.02
    petal_count = 6
    for i in range(petal_count):
        angle = (i / petal_count) * 2.0 * math.pi
        px = math.cos(angle) * 0.08
        py = math.sin(angle) * 0.08
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=1,
            radius=0.035,
            matrix=Matrix.Translation((px, py, disc_z)) @ Matrix.Scale(1.5, 4, (math.cos(angle), math.sin(angle), 0.3)),
        )

    for f in bm.faces:
        if f not in f_stem:
            f.material_index = 1

    f_petals = set(bm.faces)

    # 3. Golden center disc (Material 2)
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=1,
        radius=0.04,
        matrix=Matrix.Translation((0, 0, disc_z + 0.01)),
    )
    for f in bm.faces:
        if f not in f_petals:
            f.material_index = 2
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_stem = _get_or_create_material("MAT_Proto_Stem_Green", (0.14, 0.38, 0.10), roughness=0.50)
    mat_petals = _get_or_create_material("MAT_Proto_Wildflower_Petals", (0.82, 0.28, 0.72), roughness=0.40)
    mat_core = _get_or_create_material("MAT_Proto_Flower_Core", (0.95, 0.80, 0.12), roughness=0.40)
    if mat_stem:
        obj.data.materials.append(mat_stem)
    if mat_petals:
        obj.data.materials.append(mat_petals)
    if mat_core:
        obj.data.materials.append(mat_core)
    return obj


def create_grass_clump(name: str = "Proto_Grass_Clump", height: float = 0.5) -> Any:
    """Builds a dense cluster of curved, tapered grass blades."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 7 radiating curved grass blades
    blade_count = 7
    for i in range(blade_count):
        angle = (i / blade_count) * 2.0 * math.pi
        h = height * (0.8 + 0.3 * ((i % 3) / 2.0))
        rad_mat = Matrix.Rotation(angle, 4, "Z")
        tilt_mat = Matrix.Rotation(math.radians(18 + 8 * (i % 3)), 4, "Y")
        trans = Matrix.Translation((math.cos(angle) * 0.08, math.sin(angle) * 0.08, h * 0.45))

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.035,
            radius2=0.005,
            depth=h,
            matrix=trans @ rad_mat @ tilt_mat,
        )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Grass_Blade", (0.22, 0.48, 0.14), roughness=0.55)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_fern(name: str = "Proto_Fern") -> Any:
    """Builds arched understory fern fronds."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    for i in range(6):
        angle = (i / 6.0) * 2.0 * math.pi
        rad_mat = Matrix.Rotation(angle, 4, "Z")
        tilt_mat = Matrix.Rotation(math.radians(35), 4, "Y")
        trans = Matrix.Translation((math.cos(angle) * 0.4, math.sin(angle) * 0.4, 0.2))

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.15,
            radius2=0.01,
            depth=0.7,
            matrix=trans @ rad_mat @ tilt_mat,
        )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Fern", (0.12, 0.45, 0.16), roughness=0.55)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_forest_mushroom(name: str = "Proto_Forest_Mushroom", height: float = 0.35) -> Any:
    """Builds an earthy forest mushroom with curved fleshy stalk and wide umbrella cap."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # Stem
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=6,
        radius1=0.045,
        radius2=0.03,
        depth=height * 0.7,
        matrix=Matrix.Translation((0, 0, height * 0.35)),
    )

    # Wide mushroom cap (dome)
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=0.16,
        matrix=Matrix.Translation((0, 0, height * 0.7)) @ Matrix.Scale(0.65, 4, (0, 0, 1)),
    )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Forest_Mushroom", (0.55, 0.32, 0.22), roughness=0.5)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_glowing_fungus(name: str = "Proto_Fungus_Glow", glow_color=(0.1, 0.9, 0.4)) -> Any:
    """Builds subterranean bioluminescent mushroom with glowing cap."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=6,
        radius1=0.08,
        radius2=0.05,
        depth=0.45,
        matrix=Matrix.Translation((0, 0, 0.225)),
    )

    bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=0.25,
        matrix=Matrix.Translation((0, 0, 0.45)),
    )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = create_bioluminescent_fungus_shader(glow_color)
    obj.data.materials.append(mat)
    return obj


def create_rock_moss(name: str = "Proto_Rock_Moss", radius: float = 0.6) -> Any:
    """Builds a low-profile undulating moss cushion adhering to rock surfaces."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # Flattened ellipsoidal cushion
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=radius,
        matrix=Matrix.Scale(1.3, 4, (1, 0, 0)) @ Matrix.Scale(0.9, 4, (0, 1, 0)) @ Matrix.Scale(0.22, 4, (0, 0, 1)) @ Matrix.Translation((0, 0, radius * 0.15)),
    )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Rock_Moss", (0.24, 0.44, 0.12), roughness=0.85)
    if mat:
        obj.data.materials.append(mat)
    return obj


# =============================================================================
# 4. Tree Decay & Windthrow
# =============================================================================

def create_fallen_log(name: str = "Proto_Fallen_Log", length: float = 3.5, radius: float = 0.28) -> Any:
    """Builds a horizontal decaying fallen log with weathered bark, heartwood end rings, and branch nubs."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Main outer bark cylinder (without cap ends so caps can have heartwood material)
    f_start = set(bm.faces)
    rot_y = Matrix.Rotation(math.radians(90), 4, "Y")
    trans = Matrix.Translation((0, 0, radius * 0.85))

    bmesh.ops.create_cone(
        bm,
        cap_ends=False,
        segments=8,
        radius1=radius,
        radius2=radius * 0.75,
        depth=length,
        matrix=trans @ rot_y,
    )

    # Snapped branch nub
    nub_rot = Matrix.Rotation(math.radians(45), 4, "Z") @ Matrix.Rotation(math.radians(50), 4, "X")
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=5,
        radius1=radius * 0.35,
        radius2=radius * 0.15,
        depth=0.5,
        matrix=Matrix.Translation((0.4, 0.15, radius * 1.4)) @ nub_rot,
    )

    for f in bm.faces:
        f.material_index = 0

    f_bark = set(bm.faces)

    # 2. Exposed cut/broken heartwood ends
    half_l = length * 0.5
    bmesh.ops.create_circle(
        bm,
        cap_ends=True,
        segments=8,
        radius=radius,
        matrix=Matrix.Translation((-half_l, 0, radius * 0.85)) @ Matrix.Rotation(math.radians(-90), 4, "Y"),
    )
    bmesh.ops.create_circle(
        bm,
        cap_ends=True,
        segments=8,
        radius=radius * 0.75,
        matrix=Matrix.Translation((half_l, 0, radius * 0.85)) @ Matrix.Rotation(math.radians(90), 4, "Y"),
    )

    for f in bm.faces:
        if f not in f_bark:
            f.material_index = 1
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_bark = _get_or_create_material("MAT_Proto_Fallen_Log", (0.22, 0.16, 0.11), roughness=0.88)
    mat_heart = _get_or_create_material("MAT_Proto_Heartwood", (0.42, 0.32, 0.20), roughness=0.75)
    if mat_bark:
        obj.data.materials.append(mat_bark)
    if mat_heart:
        obj.data.materials.append(mat_heart)
    return obj


def create_snapped_stump(name: str = "Proto_Snapped_Stump", height: float = 1.2, radius: float = 0.35) -> Any:
    """Builds a weathered tree stump with jagged splintered top and flared buttress roots."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Main stump trunk (outer bark)
    f_start = set(bm.faces)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=8,
        radius1=radius * 1.25,
        radius2=radius * 0.85,
        depth=height,
        matrix=Matrix.Translation((0, 0, height * 0.5)),
    )
    # Flared root buttresses
    for i in range(3):
        angle = (i / 3.0) * 2.0 * math.pi + 0.3
        rx = math.cos(angle) * (radius * 1.1)
        ry = math.sin(angle) * (radius * 1.1)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.12,
            radius2=0.03,
            depth=0.35,
            matrix=Matrix.Translation((rx, ry, 0.15)),
        )

    for f in bm.faces:
        f.material_index = 0

    f_bark = set(bm.faces)

    # 2. Splintered jagged top peaks (exposed lighter wood)
    for i in range(4):
        angle = (i / 4.0) * 2.0 * math.pi
        px = math.cos(angle) * (radius * 0.5)
        py = math.sin(angle) * (radius * 0.5)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.08,
            radius2=0.01,
            depth=0.28,
            matrix=Matrix.Translation((px, py, height + 0.12)),
        )

    for f in bm.faces:
        if f not in f_bark:
            f.material_index = 1
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_bark = _get_or_create_material("MAT_Proto_Snapped_Stump", (0.24, 0.17, 0.12), roughness=0.88)
    mat_heart = _get_or_create_material("MAT_Proto_Heartwood", (0.42, 0.32, 0.20), roughness=0.75)
    if mat_bark:
        obj.data.materials.append(mat_bark)
    if mat_heart:
        obj.data.materials.append(mat_heart)
    return obj


# =============================================================================
# 5. Underwater Ecosystems & Riverbed
# =============================================================================

def create_aquatic_seaweed(name: str = "Proto_Aquatic_Seaweed", height: float = 1.8) -> Any:
    """Builds undulating aquatic kelp/seaweed ribbons anchored to holdfast."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # Holdfast base stone
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=1,
        radius=0.15,
        matrix=Matrix.Translation((0, 0, 0.08)),
    )

    # 4 undulating ribbons
    ribbons = [
        ((0.0, 0.0, 0.0), (0.1, 0.05, height * 0.5), (0.2, -0.05, height)),
        ((0.05, -0.05, 0.0), (-0.08, 0.12, height * 0.45), (-0.15, 0.2, height * 0.9)),
        ((-0.04, 0.06, 0.0), (0.05, -0.1, height * 0.4), (0.1, -0.18, height * 0.85)),
        ((0.02, 0.04, 0.0), (-0.06, -0.08, height * 0.55), (0.0, -0.12, height * 1.05)),
    ]
    for p0, p1, p2 in ribbons:
        # Segment 1
        v1 = Vector(p1) - Vector(p0)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.08,
            radius2=0.06,
            depth=v1.length,
            matrix=Matrix.Translation((Vector(p0) + Vector(p1)) * 0.5),
        )
        # Segment 2
        v2 = Vector(p2) - Vector(p1)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=4,
            radius1=0.06,
            radius2=0.02,
            depth=v2.length,
            matrix=Matrix.Translation((Vector(p1) + Vector(p2)) * 0.5),
        )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Aquatic_Seaweed", (0.08, 0.34, 0.22), roughness=0.35, specular=0.6)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_reeds(name: str = "Proto_Reeds") -> Any:
    """Builds wetland/riparian reeds cluster with slender amber-green stalks and velvet cattail seed heads."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # 1. Stalks (Material 0)
    f_start = set(bm.faces)
    stalk_positions = [(-0.18, -0.12), (0.15, -0.10), (-0.08, 0.18), (0.20, 0.12), (0.0, 0.0), (-0.12, 0.08), (0.08, -0.15)]
    for sx, sy in stalk_positions:
        h = 1.3 + 0.3 * (sx + sy)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=5,
            radius1=0.03,
            radius2=0.008,
            depth=h,
            matrix=Matrix.Translation((sx, sy, h * 0.5)),
        )

    for f in bm.faces:
        f.material_index = 0

    f_stalks = set(bm.faces)

    # 2. Cattail seed heads (Material 1) on top 3 stalks
    for sx, sy in [stalk_positions[0], stalk_positions[3], stalk_positions[4]]:
        h = 1.3 + 0.3 * (sx + sy)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=6,
            radius1=0.045,
            radius2=0.042,
            depth=0.25,
            matrix=Matrix.Translation((sx, sy, h - 0.15)),
        )

    for f in bm.faces:
        if f not in f_stalks:
            f.material_index = 1
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat_stalk = _get_or_create_material("MAT_Proto_Reeds", (0.24, 0.38, 0.14), roughness=0.55)
    mat_cattail = _get_or_create_material("MAT_Proto_Cattail_Head", (0.22, 0.12, 0.06), roughness=0.90)
    if mat_stalk:
        obj.data.materials.append(mat_stalk)
    if mat_cattail:
        obj.data.materials.append(mat_cattail)
    return obj


def create_riverbed_pebble(name: str = "Proto_Riverbed_Pebble", radius: float = 0.35) -> Any:
    """Builds a water-worn, smooth oblate rounded riverbed/lakebed stone."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # Oblate spheroid
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=radius,
        matrix=Matrix.Scale(1.3, 4, (1, 0, 0)) @ Matrix.Scale(0.9, 4, (0, 1, 0)) @ Matrix.Scale(0.55, 4, (0, 0, 1)) @ Matrix.Translation((0, 0, radius * 0.5)),
    )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Riverbed_Pebble", (0.38, 0.39, 0.42), roughness=0.25, specular=0.7)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_coral_reef(name: str = "Proto_Coral_Reef", radius: float = 1.0) -> Any:
    """Builds a submerged coral or pitted marine rock reef formation."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    # Central reef base
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=radius * 0.7,
        matrix=Matrix.Scale(1.2, 4, (1, 0, 0)) @ Matrix.Scale(0.7, 4, (0, 0, 1)) @ Matrix.Translation((0, 0, radius * 0.4)),
    )

    # 3 branching lobes
    lobes = [
        ((0.45, 0.3, radius * 0.6), 0.38),
        ((-0.4, 0.35, radius * 0.55), 0.32),
        ((-0.1, -0.5, radius * 0.5), 0.40),
    ]
    for pos, rad in lobes:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=1,
            radius=rad,
            matrix=Matrix.Translation(pos),
        )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Coral_Reef", (0.82, 0.45, 0.42), roughness=0.7)
    if mat:
        obj.data.materials.append(mat)
    return obj


def create_rock_boulder(name: str = "Proto_Rock_Boulder") -> Any:
    """Builds organic scree/talus rock boulder."""
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to generate 3D models.")
    existing = bpy.data.objects.get(name)
    if existing:
        return existing

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()

    bmesh.ops.create_icosphere(
        bm,
        subdivisions=1,
        radius=0.9,
        matrix=Matrix.Scale(1.4, 4, (1, 0, 0)) @ Matrix.Scale(0.7, 4, (0, 0, 1)),
    )

    for f in bm.faces:
        f.smooth = True

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    col = get_or_create_prototype_collection()
    if col:
        col.objects.link(obj)

    mat = _get_or_create_material("MAT_Proto_Boulder_Rock", (0.25, 0.24, 0.23), roughness=0.85)
    if mat:
        obj.data.materials.append(mat)
    return obj


# =============================================================================
# Prototype Resolution Mapping
# =============================================================================

PROTOTYPE_FACTORIES: Dict[str, Any] = {
    # Canopy
    "pine_tall": create_pine_tree,
    "pine_medium": create_pine_tree,
    "conifer": create_pine_tree,
    "alpine_pine": create_pine_tree,
    "broadleaf": create_broadleaf_tree,
    "broadleaf_mature": create_broadleaf_tree,
    "oak": create_broadleaf_tree,
    "deciduous": create_broadleaf_tree,
    "palm": create_palm_tree,
    "palm_tropical": create_palm_tree,
    "tropical_palm": create_palm_tree,
    # Understory
    "shrub": create_shrub,
    "bush": create_shrub,
    "krummholz": create_shrub,
    "understory": create_shrub,
    # Ground flora
    "wildflower": create_wildflower,
    "flower": create_wildflower,
    "grass": create_grass_clump,
    "dense_grass": create_grass_clump,
    "turf_grass": create_grass_clump,
    "grass_clump": create_grass_clump,
    "fern": create_fern,
    "mushroom": create_forest_mushroom,
    "forest_mushroom": create_forest_mushroom,
    "fungus": create_glowing_fungus,
    "glow_fungus": create_glowing_fungus,
    "rock_moss": create_rock_moss,
    "moss": create_rock_moss,
    # Decay & Windthrow
    "fallen_log": create_fallen_log,
    "log": create_fallen_log,
    "rotting_wood": create_fallen_log,
    "decay": create_fallen_log,
    "windthrow": create_fallen_log,
    "stump": create_snapped_stump,
    "snapped_stump": create_snapped_stump,
    "snapped_trunk": create_snapped_stump,
    "weathered_stump": create_snapped_stump,
    # Underwater & Aquatic
    "seaweed": create_aquatic_seaweed,
    "waterweed": create_aquatic_seaweed,
    "aquatic_seaweed": create_aquatic_seaweed,
    "kelp": create_aquatic_seaweed,
    "reeds": create_reeds,
    "reed": create_reeds,
    "pebble": create_riverbed_pebble,
    "riverbed_pebble": create_riverbed_pebble,
    "lakebed_pebble": create_riverbed_pebble,
    "coral_reef": create_coral_reef,
    "coral": create_coral_reef,
    "submerged_reef": create_coral_reef,
    # Detritus
    "rock_boulder": create_rock_boulder,
    "rock": create_rock_boulder,
    "boulder": create_rock_boulder,
}


# =============================================================================
# Prototype to Canonical FloraType Mapping
# =============================================================================

PROTOTYPE_TO_FLORA_TYPE: Dict[str, FloraType] = {
    # Canopy
    "pine_tall": FloraType.Pine,
    "pine_medium": FloraType.Pine,
    "conifer": FloraType.Pine,
    "alpine_pine": FloraType.Pine,
    "broadleaf": FloraType.Round,
    "broadleaf_mature": FloraType.Round,
    "oak": FloraType.Round,
    "deciduous": FloraType.Round,
    "palm": FloraType.Palm,
    "palm_tropical": FloraType.Palm,
    "tropical_palm": FloraType.Palm,
    # Understory
    "shrub": FloraType.Bush,
    "bush": FloraType.Bush,
    "krummholz": FloraType.Bush,
    "understory": FloraType.Bush,
    # Ground flora
    "wildflower": FloraType.Bush,
    "flower": FloraType.Bush,
    "grass": FloraType.Tuft,
    "dense_grass": FloraType.Tuft,
    "turf_grass": FloraType.Tuft,
    "grass_clump": FloraType.Tuft,
    "fern": FloraType.Bush,
    "mushroom": FloraType.Bush,
    "forest_mushroom": FloraType.Bush,
    "fungus": FloraType.Bush,
    "glow_fungus": FloraType.Bush,
    "rock_moss": FloraType.Bush,
    "moss": FloraType.Bush,
    # Decay & Windthrow
    "fallen_log": FloraType.DeadTree,
    "log": FloraType.DeadTree,
    "rotting_wood": FloraType.DeadTree,
    "decay": FloraType.DeadTree,
    "windthrow": FloraType.DeadTree,
    "stump": FloraType.DeadTree,
    "snapped_stump": FloraType.DeadTree,
    "snapped_trunk": FloraType.DeadTree,
    "weathered_stump": FloraType.DeadTree,
    # Underwater & Aquatic
    "seaweed": FloraType.Kelp,
    "waterweed": FloraType.Kelp,
    "aquatic_seaweed": FloraType.Kelp,
    "kelp": FloraType.Kelp,
    "reeds": FloraType.Reed,
    "reed": FloraType.Reed,
    "pebble": FloraType.Rock,
    "riverbed_pebble": FloraType.Rock,
    "lakebed_pebble": FloraType.Rock,
    "coral_reef": FloraType.Coral,
    "coral": FloraType.Coral,
    "submerged_reef": FloraType.Coral,
    # Detritus
    "rock_boulder": FloraType.Rock,
    "rock": FloraType.Rock,
    "boulder": FloraType.Rock,
}


# =============================================================================
# Pure-Python Standalone Proxy Objects (for Headless / Non-Blender Execution)
# =============================================================================

class PurePythonVertexProxy:
    """Mock vertex object mimicking bpy.types.MeshVertex."""
    def __init__(self, x: float, y: float, z: float):
        self.co = Vector((x, y, z)) if Vector is not None else (x, y, z)
        self.x = x
        self.y = y
        self.z = z


class PurePythonPolygonProxy:
    """Mock polygon object mimicking bpy.types.MeshPolygon."""
    def __init__(self, vertices: Sequence[int]):
        self.vertices = list(vertices)


class PurePythonMaterialProxy:
    """Mock material object mimicking bpy.types.Material."""
    def __init__(self, name: str):
        self.name = name


class PurePythonMeshProxy:
    """Mock mesh object mimicking bpy.types.Mesh."""
    def __init__(self, name: str, vertices: np.ndarray, triangles: np.ndarray, material_names: Optional[List[str]] = None):
        self.name = name
        self.vertices = [PurePythonVertexProxy(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        self.polygons = [PurePythonPolygonProxy(tri) for tri in triangles]
        mat_names = material_names or ["MAT_Default_Organic"]
        self.materials = [PurePythonMaterialProxy(m) for m in mat_names]


class PurePythonObjectProxy:
    """Mock object mimicking bpy.types.Object for pure-Python environments."""
    def __init__(self, name: str, mesh: PurePythonMeshProxy, custom_props: Optional[Dict[str, Any]] = None):
        self.name = name
        self.data = mesh
        self._props = custom_props or {}

    def __getitem__(self, item: str) -> Any:
        return self._props[item]

    def __setitem__(self, key: str, value: Any) -> None:
        self._props[key] = value

    def __contains__(self, item: str) -> bool:
        return item in self._props

    def get(self, key: str, default: Any = None) -> Any:
        return self._props.get(key, default)


# =============================================================================
# Authentic Biological Mesh Generator (Pure-Python / NumPy)
# =============================================================================

def _build_cylinder_mesh(
    radius_bottom: float,
    radius_top: float,
    height: float,
    segments: int = 8,
    z_offset: float = 0.0,
) -> Tuple[List[List[float]], List[List[int]]]:
    """Helper to generate a tapered cylinder mesh along vertical Z axis."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        c, s = math.cos(theta), math.sin(theta)
        verts.append([radius_bottom * c, z_offset, radius_bottom * s])
        verts.append([radius_top * c, z_offset + height, radius_top * s])

    for i in range(segments):
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * ((i + 1) % segments)
        t1 = 2 * ((i + 1) % segments) + 1
        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])

    return verts, faces


def _build_icosphere_mesh(radius: float, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[List[List[float]], List[List[int]]]:
    """Helper to generate an organic icosphere mesh."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    cx, cy, cz = center
    raw_v = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ]
    verts = []
    for x, y, z in raw_v:
        mag = math.sqrt(x * x + y * y + z * z)
        verts.append([cx + (x / mag) * radius, cy + (y / mag) * radius, cz + (z / mag) * radius])

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return verts, faces


def create_biological_mesh(
    prototype_type: str,
    target_height: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an authentic procedural biological 3D mesh (vertices, triangles)
    in pure Python/NumPy without requiring Blender GUI or bpy.
    """
    clean = prototype_type.lower()
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    if "pine" in clean or "conifer" in clean:
        # 1. Flared trunk
        t_verts, t_faces = _build_cylinder_mesh(0.30, 0.08, target_height * 0.75, segments=8, z_offset=0.0)
        verts.extend(t_verts)
        faces.extend(t_faces)

        # 2. 4 tiered conical branch tiers
        tiers = [
            (target_height * 0.28, target_height * 0.26, 1.8),
            (target_height * 0.44, target_height * 0.24, 1.5),
            (target_height * 0.60, target_height * 0.22, 1.1),
            (target_height * 0.76, target_height * 0.24, 0.6),
        ]
        for z_base, tier_h, radius in tiers:
            top_idx = len(verts)
            verts.append([0.0, z_base + tier_h, 0.0])
            base_start = len(verts)
            segs = 8
            for i in range(segs):
                th = 2.0 * math.pi * i / segs
                verts.append([radius * math.cos(th), z_base, radius * math.sin(th)])
            for i in range(segs):
                nxt = (i + 1) % segs
                faces.append([top_idx, base_start + i, base_start + nxt])
                faces.append([top_idx, base_start + nxt, base_start + i])

    elif "broadleaf" in clean or "oak" in clean or "deciduous" in clean:
        # Trunk + 3 scaffold branches + 5 icosphere canopy lobes
        t_verts, t_faces = _build_cylinder_mesh(0.35, 0.20, target_height * 0.50, segments=8, z_offset=0.0)
        verts.extend(t_verts)
        faces.extend(t_faces)

        # Canopy lobes
        lobes = [
            (0.0, target_height * 0.82, 0.0, 1.3),
            (-0.8, target_height * 0.65, -0.6, 0.9),
            (0.9, target_height * 0.68, -0.4, 0.95),
            (0.1, target_height * 0.70, 0.9, 0.9),
            (-0.5, target_height * 0.75, 0.5, 0.8),
        ]
        for lx, ly, lz, lr in lobes:
            l_verts, l_faces = _build_icosphere_mesh(lr, center=(lx, ly, lz))
            base_idx = len(verts)
            verts.extend(l_verts)
            faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in l_faces])

    elif "palm" in clean:
        # Curved segmented trunk + radiating fronds
        curr_y = 0.0
        curr_x = 0.0
        segs = 6
        seg_h = (target_height * 0.8) / segs
        for s in range(segs):
            r1 = 0.24 * (1.0 - 0.06 * s)
            r2 = 0.24 * (1.0 - 0.06 * (s + 1))
            c_verts, c_faces = _build_cylinder_mesh(r1, r2, seg_h, segments=7, z_offset=curr_y)
            base_idx = len(verts)
            for v in c_verts:
                verts.append([v[0] + curr_x, v[1], v[2]])
            faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in c_faces])
            curr_y += seg_h
            curr_x += 0.08

        # Fronds
        frond_count = 8
        crown_y = curr_y
        for i in range(frond_count):
            angle = (i / frond_count) * 2.0 * math.pi
            fx = curr_x + math.cos(angle) * 1.2
            fz = math.sin(angle) * 1.2
            fy = crown_y - 0.2
            top_idx = len(verts)
            verts.append([curr_x, crown_y, 0.0])
            f_v, f_f = _build_cylinder_mesh(0.18, 0.02, 1.4, segments=4, z_offset=fy)
            b_idx = len(verts)
            for v in f_v:
                verts.append([v[0] + fx * 0.5, v[1], v[2] + fz * 0.5])
            faces.extend([[f[0] + b_idx, f[1] + b_idx, f[2] + b_idx] for f in f_f])

    elif "shrub" in clean or "bush" in clean or "krummholz" in clean:
        # Diverging stems + foliage clusters
        clumps = [
            (0.0, target_height * 0.65, 0.0, target_height * 0.45),
            (-0.35, target_height * 0.50, -0.2, target_height * 0.35),
            (0.40, target_height * 0.52, -0.15, target_height * 0.38),
            (0.05, target_height * 0.55, 0.35, target_height * 0.35),
        ]
        for cx, cy, cz, cr in clumps:
            c_verts, c_faces = _build_icosphere_mesh(cr, center=(cx, cy, cz))
            base_idx = len(verts)
            verts.extend(c_verts)
            faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in c_faces])

    elif "log" in clean or "windthrow" in clean:
        # Horizontal log lying along X axis
        log_r = 0.32
        log_len = target_height * 0.9
        segs = 8
        base_v = len(verts)
        for i in range(segs):
            th = 2.0 * math.pi * i / segs
            cy = log_r * math.cos(th)
            cz = log_r * math.sin(th)
            verts.append([-log_len * 0.5, cy, cz])
            verts.append([log_len * 0.5, cy, cz])
        for i in range(segs):
            b0 = base_v + 2 * i
            t0 = base_v + 2 * i + 1
            b1 = base_v + 2 * ((i + 1) % segments if 'segments' in locals() else (i + 1) % segs)
            t1 = base_v + 2 * ((i + 1) % segments if 'segments' in locals() else (i + 1) % segs) + 1
            faces.append([b0, b1, t0])
            faces.append([t0, b1, t1])

    elif "stump" in clean:
        # Flared stump cylinder with jagged top
        s_verts, s_faces = _build_cylinder_mesh(0.45, 0.32, target_height * 0.85, segments=8, z_offset=0.0)
        verts.extend(s_verts)
        faces.extend(s_faces)
        # Jagged peaks
        top_y = target_height * 0.85
        for i in range(4):
            th = (i / 4.0) * 2.0 * math.pi
            px = 0.20 * math.cos(th)
            pz = 0.20 * math.sin(th)
            p_idx = len(verts)
            verts.append([px, top_y + 0.30, pz])
            verts.append([px - 0.08, top_y, pz - 0.08])
            verts.append([px + 0.08, top_y, pz + 0.08])
            faces.append([p_idx, p_idx + 1, p_idx + 2])

    elif "pebble" in clean or "river" in clean:
        # Water-worn rounded oblate spheroid
        r_verts, r_faces = _build_icosphere_mesh(target_height * 0.5)
        for v in r_verts:
            # Flatten Y and elongate X to create authentic river stone
            verts.append([v[0] * 1.35, v[1] * 0.55, v[2] * 0.95])
        faces.extend(r_faces)

    elif "moss" in clean:
        # Flattened undulating cushion
        m_verts, m_faces = _build_icosphere_mesh(target_height * 0.6)
        for v in m_verts:
            verts.append([v[0] * 1.3, v[1] * 0.25, v[2] * 0.9])
        faces.extend(m_faces)

    elif "fern" in clean:
        # Radiating arching fronds
        for i in range(6):
            th = (i / 6.0) * 2.0 * math.pi
            c_verts, c_faces = _build_cylinder_mesh(0.12, 0.02, target_height * 0.8, segments=4, z_offset=0.0)
            base_idx = len(verts)
            for v in c_verts:
                # Tilt outward
                rad_x = v[0] + math.cos(th) * (v[1] * 0.5)
                rad_z = v[2] + math.sin(th) * (v[1] * 0.5)
                verts.append([rad_x, v[1] * 0.8, rad_z])
            faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in c_faces])

    elif "mushroom" in clean or "fungus" in clean:
        # Stalk + cap
        s_verts, s_faces = _build_cylinder_mesh(0.08, 0.06, target_height * 0.65, segments=6, z_offset=0.0)
        verts.extend(s_verts)
        faces.extend(s_faces)
        # Cap
        c_verts, c_faces = _build_icosphere_mesh(target_height * 0.4, center=(0.0, target_height * 0.7, 0.0))
        base_idx = len(verts)
        for v in c_verts:
            verts.append([v[0] * 1.2, v[1] * 0.6, v[2] * 1.2])
        faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in c_faces])

    elif "reed" in clean:
        # Multiple slender stalks with cattails
        for sx, sz in [(-0.15, -0.1), (0.12, -0.08), (-0.05, 0.15), (0.18, 0.1), (0.0, 0.0)]:
            s_verts, s_faces = _build_cylinder_mesh(0.03, 0.01, target_height * 0.9, segments=5, z_offset=0.0)
            base_idx = len(verts)
            for v in s_verts:
                verts.append([v[0] + sx, v[1], v[2] + sz])
            faces.extend([[f[0] + base_idx, f[1] + base_idx, f[2] + base_idx] for f in s_faces])

    elif "coral" in clean:
        # Base + 3 branching lobes
        b_verts, b_faces = _build_icosphere_mesh(target_height * 0.45, center=(0.0, target_height * 0.3, 0.0))
        verts.extend(b_verts)
        faces.extend(b_faces)
        for lx, ly, lz, lr in [
            (0.3, target_height * 0.55, 0.2, target_height * 0.28),
            (-0.3, target_height * 0.50, 0.25, target_height * 0.25),
            (-0.1, target_height * 0.45, -0.35, target_height * 0.30),
        ]:
            l_v, l_f = _build_icosphere_mesh(lr, center=(lx, ly, lz))
            b_idx = len(verts)
            verts.extend(l_v)
            faces.extend([[f[0] + b_idx, f[1] + b_idx, f[2] + b_idx] for f in l_f])

    else:
        # Generic boulder / rock
        r_verts, r_faces = _build_icosphere_mesh(target_height * 0.5)
        for v in r_verts:
            verts.append([v[0] * 1.3, v[1] * 0.8, v[2] * 1.1])
        faces.extend(r_faces)

    v_arr = np.array(verts, dtype=np.float32)
    f_arr = np.array(faces, dtype=np.uint32)
    return v_arr, f_arr


# =============================================================================
# Biological Asset Result & Vault Fetcher
# =============================================================================

@dataclass
class BiologicalAssetResult:
    """Result of loading or generating a normalized biological asset."""
    name: str
    flora_type: Optional[FloraType]
    source: str  # "vault", "procedural_normalized", "blender_bmesh"
    model_path: Optional[Path]
    normalized_mesh: Optional[NormalizedMesh]
    collision_primitives: Dict[str, Any]
    target_height: float
    blender_object: Optional[Any] = None

    @property
    def is_normalized(self) -> bool:
        """Verify min_y == 0.0 and centered."""
        if self.normalized_mesh is not None:
            min_y = float(np.min(self.normalized_mesh.vertices[:, 1]))
            return abs(min_y) < 1e-4
        return True


def _normalize_blender_object(obj: Any, target_height: Optional[float] = None) -> Any:
    """Ensure bottom pivot (min_z == 0), centered (x, y), and extract collision metadata in Blender."""
    if not HAS_BPY or bpy is None or obj is None:
        return obj

    verts = [v.co.copy() for v in obj.data.vertices]
    if not verts:
        return obj

    min_z = min(v.z for v in verts)
    max_z = max(v.z for v in verts)
    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    c_x = (min_x + max_x) / 2.0
    c_y = (min_y + max_y) / 2.0

    raw_h = max_z - min_z
    scale = 1.0
    if target_height and target_height > 0.0 and raw_h > 1e-6:
        scale = target_height / raw_h

    # Shift vertices to bottom pivot (min_z = 0) and center (X, Y)
    for v in obj.data.vertices:
        v.co.x = (v.co.x - c_x) * scale
        v.co.y = (v.co.y - c_y) * scale
        v.co.z = (v.co.z - min_z) * scale
    obj.data.update()

    # Map Blender (X, Y, Z_up) to Normalizer (X, Y_up, Z)
    np_verts = np.array([[v.co.x, v.co.z, v.co.y] for v in obj.data.vertices], dtype=np.float32)
    h_final = target_height if (target_height and target_height > 0) else raw_h
    colliders = GeometryNormalizer.extract_collision_primitives(np_verts, target_height=h_final)

    obj["collision_primitives"] = colliders.to_dict()
    obj["min_z"] = 0.0
    obj["height_meters"] = float(h_final)
    return obj


def fetch_biological_asset(
    prototype_type: Union[FloraType, str],
    vault: Optional[LocalAssetVault] = None,
    target_height: Optional[float] = None,
    save_to_vault: bool = True,
) -> BiologicalAssetResult:
    """
    Fetch an authentic biological asset sourced from LocalAssetVault or generated procedurally,
    guaranteeing min_y = 0 bottom pivot, metric scaling, and collision primitives.

    Works seamlessly in both pure-Python and Blender environments.
    """
    clean_name = str(prototype_type).lower().strip()
    if isinstance(prototype_type, FloraType):
        flora_type = prototype_type
    else:
        flora_type = PROTOTYPE_TO_FLORA_TYPE.get(clean_name) or FloraType.from_any(prototype_type) or FloraType.Pine

    profile = DEFAULT_FLORA_PROFILES.get(flora_type, {})
    target_h = float(target_height) if target_height is not None else float(profile.get("typical_height", 5.0))

    # 1. Attempt lookup in LocalAssetVault
    if vault is not None:
        cached_assets = vault.list_assets(flora_type)
        if cached_assets:
            entry = cached_assets[0]
            try:
                with open(entry.model_path, "rb") as f:
                    glb_bytes = f.read()
                raw_verts, raw_tris = extract_mesh_from_glb(glb_bytes)
                norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
                colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h, flora_type=flora_type)
                norm_mesh = NormalizedMesh(
                    vertices=norm_verts,
                    triangles=raw_tris,
                    original_bounds=stats["original_bounds"],
                    normalized_bounds=stats["normalized_bounds"],
                    scale_factor=stats["scale_factor"],
                    offset=stats["offset"],
                    target_height=target_h,
                    collision_primitives=colliders,
                )

                # In Blender mode, import GLB into scene
                blender_obj = None
                if HAS_BPY and bpy is not None and getattr(bpy, "context", None) is not None:
                    try:
                        col = get_or_create_prototype_collection()
                        bpy.ops.import_scene.gltf(filepath=str(entry.model_path))
                        blender_obj = bpy.context.active_object
                        if blender_obj and col and blender_obj.name not in col.objects:
                            col.objects.link(blender_obj)
                        blender_obj = _normalize_blender_object(blender_obj, target_height=target_h)
                    except Exception as e:
                        logger.warning(f"Could not import GLB into Blender scene: {e}")
                else:
                    mesh_proxy = PurePythonMeshProxy(f"Mesh_{clean_name}", norm_verts, raw_tris, ["MAT_Vault_PBR"])
                    blender_obj = PurePythonObjectProxy(f"Proto_{clean_name}", mesh_proxy, {"collision_primitives": colliders.to_dict()})

                return BiologicalAssetResult(
                    name=clean_name,
                    flora_type=flora_type,
                    source="vault",
                    model_path=entry.model_path,
                    normalized_mesh=norm_mesh,
                    collision_primitives=colliders.to_dict(),
                    target_height=target_h,
                    blender_object=blender_obj,
                )
            except Exception as e:
                logger.warning(f"Failed to load cached asset {entry.asset_id} from vault: {e}")

    # 2. Generate authentic procedural biological mesh
    raw_verts, triangles = create_biological_mesh(clean_name, target_height=target_h)

    # 3. GeometryNormalizer integration (min_y = 0.0 bottom pivot, centered X/Z, metric scaling)
    norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
    colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h, flora_type=flora_type)

    norm_mesh = NormalizedMesh(
        vertices=norm_verts,
        triangles=triangles,
        original_bounds=stats["original_bounds"],
        normalized_bounds=stats["normalized_bounds"],
        scale_factor=stats["scale_factor"],
        offset=stats["offset"],
        target_height=target_h,
        collision_primitives=colliders,
    )

    # 4. Repack into valid glTF 2.0 GLB
    glb_bytes = build_glb_from_mesh(norm_verts, triangles)
    model_path = None

    if vault is not None and save_to_vault:
        prompt = f"primordial biological {clean_name} {flora_type.name}"
        stored_entry = vault.store(
            prompt=prompt,
            seed=42,
            model_glb=glb_bytes,
            flora_type=flora_type,
            dimensions_meters={"height": target_h},
            collision_primitives=colliders.to_dict(),
            source="procedural_normalizer",
        )
        model_path = stored_entry.model_path

    # 5. Build Blender object or PurePythonObjectProxy
    blender_obj = None
    if HAS_BPY and bpy is not None and getattr(bpy, "context", None) is not None:
        # If in Blender, also construct or normalize the Blender BMesh prototype
        factory = PROTOTYPE_FACTORIES.get(clean_name, create_pine_tree)
        try:
            blender_obj = factory()
            blender_obj = _normalize_blender_object(blender_obj, target_height=target_h)
        except Exception as e:
            logger.warning(f"Could not build Blender BMesh: {e}")
    else:
        mesh_proxy = PurePythonMeshProxy(f"Mesh_{clean_name}", norm_verts, triangles, ["MAT_Organic_PBR"])
        blender_obj = PurePythonObjectProxy(f"Proto_{clean_name}", mesh_proxy, {"collision_primitives": colliders.to_dict()})

    return BiologicalAssetResult(
        name=clean_name,
        flora_type=flora_type,
        source="procedural_normalized",
        model_path=model_path,
        normalized_mesh=norm_mesh,
        collision_primitives=colliders.to_dict(),
        target_height=target_h,
        blender_object=blender_obj,
    )


# =============================================================================
# Authentic Abiotic Mesh Generators (Pure-Python / NumPy)
# =============================================================================

def _build_basalt_column_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Cluster of polygonal hexagonal and pentagonal prism columns with stepped heights."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    columns = [
        # (center_x, center_z, radius_ratio, height_ratio, n_sides)
        (0.0, 0.0, 0.40, 1.00, 6),
        (0.65, 0.05, 0.35, 0.88, 6),
        (0.35, 0.60, 0.36, 0.78, 5),
        (-0.35, 0.62, 0.38, 0.92, 6),
        (-0.68, 0.02, 0.34, 0.82, 5),
        (-0.32, -0.60, 0.37, 0.74, 6),
        (0.38, -0.58, 0.35, 0.85, 6),
    ]

    for cx_ratio, cz_ratio, r_ratio, h_ratio, n_sides in columns:
        h = target_height * h_ratio
        r = target_height * r_ratio * 0.35
        cx = target_height * cx_ratio * 0.40
        cz = target_height * cz_ratio * 0.40
        base_idx = len(verts)

        for i in range(n_sides):
            angle = 2.0 * math.pi * i / n_sides
            verts.append([cx + r * math.cos(angle), 0.0, cz + r * math.sin(angle)])
        for i in range(n_sides):
            angle = 2.0 * math.pi * i / n_sides
            verts.append([cx + r * math.cos(angle), h, cz + r * math.sin(angle)])

        c_bottom = len(verts)
        verts.append([cx, 0.0, cz])
        c_top = len(verts)
        verts.append([cx, h, cz])

        for i in range(n_sides):
            b0 = base_idx + i
            b1 = base_idx + (i + 1) % n_sides
            t0 = base_idx + n_sides + i
            t1 = base_idx + n_sides + (i + 1) % n_sides
            faces.append([b0, b1, t0])
            faces.append([t0, b1, t1])
            faces.append([c_bottom, b1, b0])
            faces.append([c_top, t0, t1])

    return verts, faces


def _build_stalactite_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Conical tapering calcite dripstone hanging speleothem."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    n_rings = 8
    n_segs = 12
    max_radius = target_height * 0.22

    for s in range(n_rings):
        t = s / float(n_rings - 1)  # 0 at bottom tip, 1 at top base
        y = target_height * t
        r_base = max_radius * (0.05 + 0.95 * (t ** 1.3))
        for i in range(n_segs):
            theta = 2.0 * math.pi * i / n_segs
            fluting = 1.0 + 0.10 * math.cos(5.0 * theta)
            r = r_base * fluting
            verts.append([r * math.cos(theta), y, r * math.sin(theta)])

    apex_bottom = len(verts)
    verts.append([0.0, 0.0, 0.0])

    center_top = len(verts)
    verts.append([0.0, target_height, 0.0])

    for s in range(n_rings - 1):
        r0 = s * n_segs
        r1 = (s + 1) * n_segs
        for i in range(n_segs):
            i_next = (i + 1) % n_segs
            faces.append([r0 + i, r0 + i_next, r1 + i])
            faces.append([r1 + i, r0 + i_next, r1 + i_next])

    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([apex_bottom, i, i_next])

    top_offset = (n_rings - 1) * n_segs
    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([center_top, top_offset + i_next, top_offset + i])

    return verts, faces


def _build_stalagmite_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Conical layered calcite mound speleothem rising from floor."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    n_rings = 8
    n_segs = 12
    max_radius = target_height * 0.35

    for s in range(n_rings):
        t = s / float(n_rings - 1)  # 0 at bottom, 1 at top
        y = target_height * t * 0.95
        r_base = max_radius * (1.0 - 0.85 * (t ** 0.8))
        for i in range(n_segs):
            theta = 2.0 * math.pi * i / n_segs
            layering = 1.0 + 0.08 * math.cos(4.0 * theta)
            r = r_base * layering
            verts.append([r * math.cos(theta), y, r * math.sin(theta)])

    center_bottom = len(verts)
    verts.append([0.0, 0.0, 0.0])

    apex_top = len(verts)
    verts.append([0.0, target_height, 0.0])

    for s in range(n_rings - 1):
        r0 = s * n_segs
        r1 = (s + 1) * n_segs
        for i in range(n_segs):
            i_next = (i + 1) % n_segs
            faces.append([r0 + i, r0 + i_next, r1 + i])
            faces.append([r1 + i, r0 + i_next, r1 + i_next])

    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([center_bottom, i_next, i])

    top_offset = (n_rings - 1) * n_segs
    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([apex_top, top_offset + i, top_offset + i_next])

    return verts, faces


def _build_cavern_column_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Monumental fused flowstone pillar spanning floor to ceiling."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    n_rings = 10
    n_segs = 16
    base_radius = target_height * 0.18

    for s in range(n_rings):
        t = s / float(n_rings - 1)
        y = target_height * t
        waist_curve = 0.70 + 0.30 * ((2.0 * t - 1.0) ** 2)
        for i in range(n_segs):
            theta = 2.0 * math.pi * i / n_segs
            fluting = 1.0 + 0.15 * math.cos(6.0 * theta)
            r = base_radius * waist_curve * fluting
            verts.append([r * math.cos(theta), y, r * math.sin(theta)])

    center_bottom = len(verts)
    verts.append([0.0, 0.0, 0.0])

    center_top = len(verts)
    verts.append([0.0, target_height, 0.0])

    for s in range(n_rings - 1):
        r0 = s * n_segs
        r1 = (s + 1) * n_segs
        for i in range(n_segs):
            i_next = (i + 1) % n_segs
            faces.append([r0 + i, r0 + i_next, r1 + i])
            faces.append([r1 + i, r0 + i_next, r1 + i_next])

    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([center_bottom, i_next, i])

    top_offset = (n_rings - 1) * n_segs
    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([center_top, top_offset + i, top_offset + i_next])

    return verts, faces


def _build_cavern_mineral_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Faceted fluorite/calcite crystal druse cluster with radiating prismatic crystals."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    slab_radius = target_height * 0.55
    slab_height = target_height * 0.15
    slab_segs = 8
    base_idx = len(verts)
    for i in range(slab_segs):
        theta = 2.0 * math.pi * i / slab_segs
        verts.append([slab_radius * math.cos(theta), 0.0, slab_radius * math.sin(theta)])
    for i in range(slab_segs):
        theta = 2.0 * math.pi * i / slab_segs
        verts.append([slab_radius * 0.9 * math.cos(theta), slab_height, slab_radius * 0.9 * math.sin(theta)])
    c_slab_bot = len(verts)
    verts.append([0.0, 0.0, 0.0])
    c_slab_top = len(verts)
    verts.append([0.0, slab_height, 0.0])

    for i in range(slab_segs):
        i_next = (i + 1) % slab_segs
        b0 = base_idx + i
        b1 = base_idx + i_next
        t0 = base_idx + slab_segs + i
        t1 = base_idx + slab_segs + i_next
        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])
        faces.append([c_slab_bot, b1, b0])
        faces.append([c_slab_top, t0, t1])

    crystal_configs = [
        # (ox, oz, length_ratio, tilt_deg, yaw_deg, radius_ratio)
        (0.0, 0.0, 0.95, 0.0, 0.0, 0.16),
        (0.25, 0.15, 0.80, 22.0, 30.0, 0.13),
        (-0.20, 0.22, 0.72, 28.0, 110.0, 0.12),
        (-0.25, -0.18, 0.85, 20.0, 210.0, 0.14),
        (0.20, -0.22, 0.70, 25.0, 300.0, 0.12),
        (0.42, 0.02, 0.55, 38.0, 15.0, 0.10),
        (-0.02, 0.40, 0.58, 35.0, 95.0, 0.10),
        (-0.38, -0.05, 0.60, 36.0, 195.0, 0.11),
        (0.05, -0.38, 0.52, 40.0, 280.0, 0.09),
    ]

    for ox, oz, l_ratio, tilt_deg, yaw_deg, r_ratio in crystal_configs:
        c_len = target_height * l_ratio
        c_rad = target_height * r_ratio
        c_origin = np.array([ox * target_height * 0.4, slab_height * 0.8, oz * target_height * 0.4])

        tilt_rad = math.radians(tilt_deg)
        yaw_rad = math.radians(yaw_deg)
        dx = math.sin(tilt_rad) * math.cos(yaw_rad)
        dy = math.cos(tilt_rad)
        dz = math.sin(tilt_rad) * math.sin(yaw_rad)
        dir_vec = np.array([dx, dy, dz], dtype=np.float32)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > 1e-6:
            dir_vec /= dir_norm

        up = np.array([0.0, 1.0, 0.0]) if abs(dir_vec[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(dir_vec, up)
        side_norm = np.linalg.norm(side)
        if side_norm > 1e-6:
            side /= side_norm
        ortho = np.cross(dir_vec, side)
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm > 1e-6:
            ortho /= ortho_norm

        c_idx = len(verts)
        k_sides = 6
        for i in range(k_sides):
            angle = 2.0 * math.pi * i / k_sides
            v = c_origin + (side * math.cos(angle) + ortho * math.sin(angle)) * c_rad
            verts.append(v.tolist())
        shoulder_dist = c_len * 0.82
        for i in range(k_sides):
            angle = 2.0 * math.pi * i / k_sides
            v = c_origin + dir_vec * shoulder_dist + (side * math.cos(angle) + ortho * math.sin(angle)) * c_rad
            verts.append(v.tolist())
        apex_idx = len(verts)
        apex_pos = c_origin + dir_vec * c_len
        verts.append(apex_pos.tolist())

        for i in range(k_sides):
            i_next = (i + 1) % k_sides
            b0 = c_idx + i
            b1 = c_idx + i_next
            s0 = c_idx + k_sides + i
            s1 = c_idx + k_sides + i_next
            faces.append([b0, b1, s0])
            faces.append([s0, b1, s1])
            faces.append([apex_idx, s0, s1])

    return verts, faces


def _build_river_pebble_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Cluster of water-worn oblate flattened riverbed pebbles."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    pebble_configs = [
        # (cx, cz, radius_x, radius_y, radius_z, rot_deg)
        (0.0, 0.0, 0.70, 0.28, 0.50, 20.0),
        (0.55, 0.25, 0.50, 0.22, 0.40, -35.0),
        (-0.50, 0.30, 0.58, 0.24, 0.42, 45.0),
        (-0.45, -0.40, 0.48, 0.20, 0.38, -15.0),
        (0.40, -0.45, 0.52, 0.22, 0.35, 60.0),
        (0.85, -0.15, 0.38, 0.16, 0.28, 10.0),
        (-0.80, -0.05, 0.42, 0.18, 0.30, -50.0),
        (0.15, 0.75, 0.35, 0.15, 0.25, 75.0),
        (-0.10, -0.80, 0.40, 0.17, 0.26, -30.0),
    ]

    n_lat = 7
    n_lon = 10

    for cx, cz, rx, ry, rz, rot_deg in pebble_configs:
        scale = target_height / 0.40 if target_height > 0.0 else 1.0
        c_rx = rx * scale * 0.45
        c_ry = ry * scale * 0.90
        c_rz = rz * scale * 0.45
        c_cx = cx * scale * 0.80
        c_cz = cz * scale * 0.80
        rot_rad = math.radians(rot_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        south_pole_idx = len(verts)
        verts.append([c_cx, 0.0, c_cz])

        rings_start_idx = len(verts)
        for lat in range(1, n_lat - 1):
            theta = math.pi * lat / float(n_lat - 1)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for lon in range(n_lon):
                phi = 2.0 * math.pi * lon / float(n_lon)
                lx = c_rx * sin_t * math.cos(phi)
                ly = c_ry * (1.0 - cos_t)
                lz = c_rz * sin_t * math.sin(phi)
                wx = c_cx + (lx * cos_r - lz * sin_r)
                wy = ly
                wz = c_cz + (lx * sin_r + lz * cos_r)
                verts.append([wx, wy, wz])

        north_pole_idx = len(verts)
        verts.append([c_cx, 2.0 * c_ry, c_cz])

        # South pole fan
        for lon in range(n_lon):
            l_next = (lon + 1) % n_lon
            faces.append([south_pole_idx, rings_start_idx + l_next, rings_start_idx + lon])

        # Intermediate cylinder quad strips
        for ring in range(n_lat - 3):
            r0 = rings_start_idx + ring * n_lon
            r1 = rings_start_idx + (ring + 1) * n_lon
            for lon in range(n_lon):
                l_next = (lon + 1) % n_lon
                faces.append([r0 + lon, r0 + l_next, r1 + lon])
                faces.append([r1 + lon, r0 + l_next, r1 + l_next])

        # North pole fan
        last_ring = rings_start_idx + (n_lat - 3) * n_lon
        for lon in range(n_lon):
            l_next = (lon + 1) % n_lon
            faces.append([north_pole_idx, last_ring + lon, last_ring + l_next])

    return verts, faces


def _build_horn_peak_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Sharp pyramidal alpine horn peak with 4 glacial aretes meeting at summit."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    n_tiers = 6
    n_segs = 16
    max_radius = target_height * 0.65

    for tier in range(n_tiers):
        t = (tier / float(n_tiers - 1)) * 0.92
        y = target_height * t
        r_tier = max_radius * ((1.0 - t) ** 1.3)
        for i in range(n_segs):
            angle = 2.0 * math.pi * i / n_segs
            ridge_factor = 1.0 + 0.35 * (math.cos(2.0 * angle) ** 2)
            r = r_tier * ridge_factor
            verts.append([r * math.cos(angle), y, r * math.sin(angle)])

    center_bottom = len(verts)
    verts.append([0.0, 0.0, 0.0])

    summit_apex = len(verts)
    verts.append([0.0, target_height, 0.0])

    for tier in range(n_tiers - 1):
        r0 = tier * n_segs
        r1 = (tier + 1) * n_segs
        for i in range(n_segs):
            i_next = (i + 1) % n_segs
            faces.append([r0 + i, r0 + i_next, r1 + i])
            faces.append([r1 + i, r0 + i_next, r1 + i_next])

    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([center_bottom, i_next, i])

    top_offset = (n_tiers - 1) * n_segs
    for i in range(n_segs):
        i_next = (i + 1) % n_segs
        faces.append([summit_apex, top_offset + i, top_offset + i_next])

    return verts, faces


def _build_rock_crag_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Massive blocky weathered granite crag with planar tectonic jointing facets."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    blocks = [
        # (cx, cz, width, height_ratio, depth)
        (0.0, 0.0, 0.60, 0.95, 0.50),
        (0.20, 0.15, 0.45, 0.75, 0.40),
        (-0.18, -0.12, 0.50, 0.85, 0.45),
    ]

    for cx_ratio, cz_ratio, w_ratio, h_ratio, d_ratio in blocks:
        w = target_height * w_ratio
        h = target_height * h_ratio
        d = target_height * d_ratio
        cx = target_height * cx_ratio
        cz = target_height * cz_ratio
        b_idx = len(verts)
        corners = [
            [-w * 0.5 + cx, 0.0, -d * 0.5 + cz],
            [ w * 0.5 + cx, 0.0, -d * 0.48 + cz],
            [ w * 0.48 + cx, 0.0,  d * 0.5 + cz],
            [-w * 0.52 + cx, 0.0,  d * 0.46 + cz],
            [-w * 0.42 + cx, h, -d * 0.40 + cz],
            [ w * 0.44 + cx, h, -d * 0.42 + cz],
            [ w * 0.38 + cx, h,  d * 0.42 + cz],
            [-w * 0.40 + cx, h,  d * 0.38 + cz],
        ]
        verts.extend(corners)
        box_faces = [
            [b_idx + 0, b_idx + 1, b_idx + 2], [b_idx + 0, b_idx + 2, b_idx + 3],
            [b_idx + 4, b_idx + 6, b_idx + 5], [b_idx + 4, b_idx + 7, b_idx + 6],
            [b_idx + 0, b_idx + 5, b_idx + 1], [b_idx + 0, b_idx + 4, b_idx + 5],
            [b_idx + 1, b_idx + 6, b_idx + 2], [b_idx + 1, b_idx + 5, b_idx + 6],
            [b_idx + 2, b_idx + 7, b_idx + 3], [b_idx + 2, b_idx + 6, b_idx + 7],
            [b_idx + 3, b_idx + 4, b_idx + 0], [b_idx + 3, b_idx + 7, b_idx + 4],
        ]
        faces.extend(box_faces)

    return verts, faces


def _build_riparian_sandbank_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Curved alluvial ripple-marked sediment sand berm."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    nx = 14
    nz = 8
    length_x = target_height * 4.0
    width_z = target_height * 2.2

    for iz in range(nz):
        fz = (iz / float(nz - 1)) * 2.0 - 1.0
        z = fz * (width_z * 0.5)
        cross_profile = max(0.0, 1.0 - (fz ** 2))

        for ix in range(nx):
            fx = (ix / float(nx - 1)) * 2.0 - 1.0
            x = fx * (length_x * 0.5)
            long_profile = max(0.0, 1.0 - (fx ** 2))
            ripple = 0.08 * math.sin(math.pi * 5.0 * fx)
            y = target_height * cross_profile * long_profile * (1.0 + ripple)
            curve_z = 0.25 * width_z * (fx ** 2)
            verts.append([x, max(0.0, y), z + curve_z])

    for iz in range(nz - 1):
        for ix in range(nx - 1):
            v0 = iz * nx + ix
            v1 = iz * nx + (ix + 1)
            v2 = (iz + 1) * nx + ix
            v3 = (iz + 1) * nx + (ix + 1)
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])

    base_center = len(verts)
    verts.append([0.0, 0.0, 0.0])

    for ix in range(nx - 1):
        faces.append([base_center, ix, ix + 1])
        bot_v = (nz - 1) * nx + ix
        faces.append([base_center, bot_v + 1, bot_v])
    for iz in range(nz - 1):
        left_v = iz * nx
        faces.append([base_center, left_v + nx, left_v])
        right_v = iz * nx + (nx - 1)
        faces.append([base_center, right_v, right_v + nx])

    return verts, faces


def _build_karst_arch_mesh(target_height: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Natural rocky karst arch portal with hollow inner passage, flaring front flanges, and porous karst weathering."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    h = max(0.5, float(target_height))
    w_out = h * 0.64
    w_in = h * 0.38
    d = h * 0.65
    y_spring_in = h * 0.30
    r_in_y = h * 0.42
    y_spring_out = h * 0.36
    r_out_y = h * 0.60

    n_z = 16
    layer_in = []
    layer_out = []

    for iz in range(n_z):
        z_norm = iz / float(n_z - 1)
        z = (z_norm - 0.5) * d
        flange = 1.25 if iz == 0 else (1.12 if iz == 1 else 1.0)

        pts_in = []
        pts_out = []

        # Inner wall: right ground to right spring (4 pts)
        for s in range(4):
            t = s / 3.0
            pts_in.append([w_in, t * y_spring_in, z])
        # Inner arch curve (10 pts)
        for s in range(1, 11):
            th = math.pi * (1.0 - s / 11.0)
            x = w_in * math.cos(th)
            y = y_spring_in + r_in_y * math.sin(th)
            pts_in.append([x, y, z])
        # Inner wall: left spring to left ground (4 pts)
        for s in range(4):
            t = 1.0 - s / 3.0
            pts_in.append([-w_in, t * y_spring_in, z])

        # Outer wall: right ground to right spring (4 pts)
        for s in range(4):
            t = s / 3.0
            x = w_out * flange + 0.03 * h * math.sin(z * 3.0 + t * 4.0)
            y = t * (y_spring_out * flange)
            pts_out.append([x, max(0.0, y), z])
        # Outer arch curve (10 pts) with karst erosion
        for s in range(1, 11):
            th = math.pi * (1.0 - s / 11.0)
            karst = 0.04 * h * (math.sin(4.0 * th + z * 2.0) + math.cos(8.0 * th) * 0.5)
            x = (w_out * flange + karst) * math.cos(th)
            y = (y_spring_out + r_out_y + karst) * math.sin(th) * flange
            pts_out.append([x, max(0.0, y), z])
        # Outer wall: left spring to left ground (4 pts)
        for s in range(4):
            t = 1.0 - s / 3.0
            x = -w_out * flange - 0.03 * h * math.sin(z * 3.0 + t * 4.0)
            y = t * (y_spring_out * flange)
            pts_out.append([x, max(0.0, y), z])

        layer_in.append(pts_in)
        layer_out.append(pts_out)

    n_ring = len(layer_in[0])
    ring_out_indices = []
    ring_in_indices = []

    for iz in range(n_z):
        out_start = len(verts)
        verts.extend(layer_out[iz])
        ring_out_indices.append(list(range(out_start, out_start + n_ring)))

        in_start = len(verts)
        verts.extend(layer_in[iz])
        ring_in_indices.append(list(range(in_start, in_start + n_ring)))

    # Connect outer shell along Z
    for iz in range(n_z - 1):
        r0 = ring_out_indices[iz]
        r1 = ring_out_indices[iz + 1]
        for i in range(n_ring - 1):
            faces.append([r0[i], r0[i + 1], r1[i + 1]])
            faces.append([r0[i], r1[i + 1], r1[i]])

    # Connect inner passage along Z
    for iz in range(n_z - 1):
        r0 = ring_in_indices[iz]
        r1 = ring_in_indices[iz + 1]
        for i in range(n_ring - 1):
            faces.append([r0[i], r1[i + 1], r0[i + 1]])
            faces.append([r0[i], r1[i], r1[i + 1]])

    # Front portal cap (iz = 0)
    r_out_0 = ring_out_indices[0]
    r_in_0 = ring_in_indices[0]
    for i in range(n_ring - 1):
        faces.append([r_out_0[i], r_in_0[i], r_in_0[i + 1]])
        faces.append([r_out_0[i], r_in_0[i + 1], r_out_0[i + 1]])

    # Back portal cap (iz = n_z - 1)
    r_out_1 = ring_out_indices[-1]
    r_in_1 = ring_in_indices[-1]
    for i in range(n_ring - 1):
        faces.append([r_in_1[i], r_out_1[i], r_out_1[i + 1]])
        faces.append([r_in_1[i], r_out_1[i + 1], r_in_1[i + 1]])

    # Bottom ground pads
    for iz in range(n_z - 1):
        o0 = ring_out_indices[iz][0]
        o1 = ring_out_indices[iz + 1][0]
        i0 = ring_in_indices[iz][0]
        i1 = ring_in_indices[iz + 1][0]
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])

        o0_l = ring_out_indices[iz][-1]
        o1_l = ring_out_indices[iz + 1][-1]
        i0_l = ring_in_indices[iz][-1]
        i1_l = ring_in_indices[iz + 1][-1]
        faces.append([i0_l, i1_l, o1_l])
        faces.append([i0_l, o1_l, o0_l])

    # Normalize vertical bounds so min_y == 0.0 and max_y == target_height
    y_vals = [v[1] for v in verts]
    min_y = min(y_vals)
    max_y = max(y_vals)
    curr_h = max_y - min_y
    if curr_h > 1e-4:
        scale = target_height / curr_h
        for v in verts:
            v[1] = (v[1] - min_y) * scale
            v[0] *= scale
            v[2] *= scale

    return verts, faces


def create_abiotic_mesh(
    abiotic_type: Union[AbioticType, int, str],
    target_height: Optional[float] = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an authentic procedural abiotic 3D mesh (vertices, triangles)
    in pure Python/NumPy without requiring Blender GUI or bpy.

    Supports all 10 AbioticType categories:
    - HornPeak, RockCrag, BasaltColumn, RiverPebble, RiparianSandbank,
      Stalactite, Stalagmite, CavernColumn, CavernMineral, KarstArch.
    """
    at = AbioticType.from_any(abiotic_type)
    if at is None:
        if isinstance(abiotic_type, str):
            clean = abiotic_type.lower().strip()
            for member in AbioticType:
                if member.name.lower() in clean or clean in member.name.lower():
                    at = member
                    break
        if at is None:
            at = AbioticType.RockCrag

    # Robust target_height parsing and safe bounds clamping
    if target_height is None:
        h = 5.0
    else:
        try:
            h_val = float(target_height)
            if math.isnan(h_val) or math.isinf(h_val) or h_val <= 0.0:
                h = 5.0
            else:
                h = min(h_val, 100000.0)
        except (TypeError, ValueError):
            h = 5.0

    if at == AbioticType.HornPeak:
        raw_v, raw_f = _build_horn_peak_mesh(h)
    elif at == AbioticType.RockCrag:
        raw_v, raw_f = _build_rock_crag_mesh(h)
    elif at == AbioticType.BasaltColumn:
        raw_v, raw_f = _build_basalt_column_mesh(h)
    elif at == AbioticType.RiverPebble:
        raw_v, raw_f = _build_river_pebble_mesh(h)
    elif at == AbioticType.RiparianSandbank:
        raw_v, raw_f = _build_riparian_sandbank_mesh(h)
    elif at == AbioticType.Stalactite:
        raw_v, raw_f = _build_stalactite_mesh(h)
    elif at == AbioticType.Stalagmite:
        raw_v, raw_f = _build_stalagmite_mesh(h)
    elif at == AbioticType.CavernColumn:
        raw_v, raw_f = _build_cavern_column_mesh(h)
    elif at == AbioticType.CavernMineral:
        raw_v, raw_f = _build_cavern_mineral_mesh(h)
    elif at == AbioticType.KarstArch:
        raw_v, raw_f = _build_karst_arch_mesh(h)
    else:
        raw_v, raw_f = _build_rock_crag_mesh(h)

    v_arr = np.array(raw_v, dtype=np.float32)
    f_arr = np.array(raw_f, dtype=np.uint32)

    # Seed-deterministic variation: rotate around vertical Y axis
    if seed is not None:
        rng = np.random.RandomState(int(seed) % (2**32))
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_rot = v_arr[:, 0] * cos_y - v_arr[:, 2] * sin_y
        z_rot = v_arr[:, 0] * sin_y + v_arr[:, 2] * cos_y
        v_arr[:, 0] = x_rot
        v_arr[:, 2] = z_rot

    return v_arr, f_arr


# =============================================================================
# Abiotic Asset Result & Vault Fetcher
# =============================================================================

@dataclass
class AbioticAssetResult:
    """Result of loading or generating a normalized abiotic asset."""
    name: str
    abiotic_type: AbioticType
    source: str  # "vault", "procedural_normalized", "blender_bmesh"
    model_path: Optional[Path]
    normalized_mesh: Optional[NormalizedMesh]
    collision_primitives: Dict[str, Any]
    target_height: float
    blender_object: Optional[Any] = None

    @property
    def is_normalized(self) -> bool:
        """Verify min_y == 0.0 and centered."""
        if self.normalized_mesh is not None:
            min_y = float(np.min(self.normalized_mesh.vertices[:, 1]))
            return abs(min_y) < 1e-4
        return True

    def __getitem__(self, item: str) -> Any:
        if item in self.collision_primitives:
            return self.collision_primitives[item]
        if self.blender_object is not None and hasattr(self.blender_object, "__getitem__"):
            return self.blender_object[item]
        raise KeyError(item)

    @property
    def data(self) -> Any:
        if self.blender_object is not None and hasattr(self.blender_object, "data"):
            return self.blender_object.data
        return None


def fetch_abiotic_asset(
    abiotic_type: Union[AbioticType, int, str],
    target_height: Optional[float] = None,
    vault: Optional[LocalAssetVault] = None,
    save_to_vault: bool = True,
    seed: Optional[int] = None,
) -> AbioticAssetResult:
    """
    Fetch an authentic abiotic asset sourced from LocalAssetVault or generated procedurally,
    guaranteeing min_y = 0 bottom pivot, metric scaling, and collision primitives.

    Supports all 9 AbioticType categories:
    - HornPeak, RockCrag, BasaltColumn, RiverPebble, RiparianSandbank,
      Stalactite, Stalagmite, CavernColumn, CavernMineral.
    """
    at = AbioticType.from_any(abiotic_type)
    if at is None:
        if isinstance(abiotic_type, str):
            clean = abiotic_type.lower().strip()
            for member in AbioticType:
                if member.name.lower() in clean or clean in member.name.lower():
                    at = member
                    break
        if at is None:
            at = AbioticType.RockCrag

    profile = DEFAULT_ABIOTIC_PROFILES.get(at, {})
    target_h = float(target_height) if target_height is not None else float(profile.get("typical_height", 5.0))

    # 1. Attempt lookup in LocalAssetVault
    if vault is not None:
        cached_assets = vault.list_assets(abiotic_type=at)
        if cached_assets:
            entry = cached_assets[0]
            try:
                with open(entry.model_path, "rb") as f:
                    glb_bytes = f.read()
                raw_verts, raw_tris = extract_mesh_from_glb(glb_bytes)
                norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
                colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h)
                norm_mesh = NormalizedMesh(
                    vertices=norm_verts,
                    triangles=raw_tris,
                    original_bounds=stats["original_bounds"],
                    normalized_bounds=stats["normalized_bounds"],
                    scale_factor=stats["scale_factor"],
                    offset=stats["offset"],
                    target_height=target_h,
                    collision_primitives=colliders,
                )

                blender_obj = None
                if HAS_BPY and bpy is not None and getattr(bpy, "context", None) is not None:
                    try:
                        bpy.ops.import_scene.gltf(filepath=str(entry.model_path))
                        blender_obj = bpy.context.active_object
                        prefix = "Cave_" if at in (AbioticType.KarstArch, AbioticType.Stalactite, AbioticType.Stalagmite, AbioticType.CavernColumn, AbioticType.CavernMineral) else "Diorama_"
                        if blender_obj:
                            blender_obj.name = f"{prefix}Abiotic_{at.name}"
                            if blender_obj.name not in bpy.context.scene.collection.objects:
                                bpy.context.scene.collection.objects.link(blender_obj)
                            blender_obj = _normalize_blender_object(blender_obj, target_height=target_h)
                    except Exception as e:
                        logger.warning(f"Could not import GLB into Blender scene: {e}")
                else:
                    mat_name = "MAT_Cavern_Mineral_Emissive" if at == AbioticType.CavernMineral else f"MAT_Abiotic_{at.name}"
                    mesh_proxy = PurePythonMeshProxy(f"Mesh_{at.name}", norm_verts, raw_tris, [mat_name])
                    blender_obj = PurePythonObjectProxy(f"Proto_{at.name}", mesh_proxy, {"collision_primitives": colliders.to_dict(), "abiotic_type": int(at)})

                return AbioticAssetResult(
                    name=at.name,
                    abiotic_type=at,
                    source="vault",
                    model_path=entry.model_path,
                    normalized_mesh=norm_mesh,
                    collision_primitives=colliders.to_dict(),
                    target_height=target_h,
                    blender_object=blender_obj,
                )
            except Exception as e:
                logger.warning(f"Failed to load cached abiotic asset {entry.asset_id} from vault: {e}")

    # 2. Generate authentic procedural abiotic mesh
    raw_verts, triangles = create_abiotic_mesh(at, target_height=target_h, seed=seed)

    # 3. GeometryNormalizer integration (min_y = 0.0 bottom pivot, centered X/Z, metric scaling)
    norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
    colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h)

    norm_mesh = NormalizedMesh(
        vertices=norm_verts,
        triangles=triangles,
        original_bounds=stats["original_bounds"],
        normalized_bounds=stats["normalized_bounds"],
        scale_factor=stats["scale_factor"],
        offset=stats["offset"],
        target_height=target_h,
        collision_primitives=colliders,
    )

    # 4. Repack into valid glTF 2.0 GLB
    glb_bytes = build_glb_from_mesh(norm_verts, triangles)
    model_path = None

    if vault is not None and save_to_vault:
        prompt_engine = AbioticPromptEngine()
        asset_seed = seed if seed is not None else 42
        prompt = prompt_engine.generate_prompt(at, seed=asset_seed)
        stored_entry = vault.store(
            prompt=prompt,
            seed=asset_seed,
            model_glb=glb_bytes,
            abiotic_type=at,
            dimensions_meters={"height": target_h},
            collision_primitives=colliders.to_dict(),
            source="procedural_normalizer",
        )
        model_path = stored_entry.model_path

    # 5. Build Blender object or PurePythonObjectProxy
    blender_obj = None
    if HAS_BPY and bpy is not None and getattr(bpy, "context", None) is not None:
        try:
            mesh_data = bpy.data.meshes.new(f"Mesh_Abiotic_{at.name}")
            blender_coords = [(v[0], v[2], v[1]) for v in norm_verts]
            tri_list = [tuple(int(idx) for idx in tri) for tri in triangles]
            mesh_data.from_pydata(blender_coords, [], tri_list)
            mesh_data.update()

            prefix = "Cave_" if at in (AbioticType.KarstArch, AbioticType.Stalactite, AbioticType.Stalagmite, AbioticType.CavernColumn, AbioticType.CavernMineral) else "Diorama_"
            blender_obj = bpy.data.objects.new(f"{prefix}Abiotic_{at.name}", mesh_data)
            if blender_obj.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(blender_obj)

            if at == AbioticType.CavernMineral:
                mat = _get_or_create_material("MAT_Cavern_Mineral_Emissive", base_color=(0.1, 0.8, 0.7), roughness=0.2, specular=0.8, emission_color=(0.1, 0.9, 0.8), emission_strength=5.0)
            else:
                mat = _get_or_create_material(f"MAT_Abiotic_{at.name}", base_color=(0.35, 0.35, 0.38), roughness=0.85, specular=0.3)
            if mat and blender_obj.data.materials:
                blender_obj.data.materials[0] = mat
            elif mat:
                blender_obj.data.materials.append(mat)

            blender_obj["collision_primitives"] = colliders.to_dict()
            blender_obj["min_z"] = 0.0
            blender_obj["height_meters"] = float(target_h)
            blender_obj["abiotic_type"] = int(at)
        except Exception as e:
            logger.warning(f"Could not build Blender abiotic object: {e}")

    if blender_obj is None:
        mat_name = "MAT_Cavern_Mineral_Emissive" if at == AbioticType.CavernMineral else f"MAT_Abiotic_{at.name}"
        mesh_proxy = PurePythonMeshProxy(f"Mesh_{at.name}", norm_verts, triangles, [mat_name])
        blender_obj = PurePythonObjectProxy(f"Proto_{at.name}", mesh_proxy, {"collision_primitives": colliders.to_dict(), "abiotic_type": int(at)})

    return AbioticAssetResult(
        name=at.name,
        abiotic_type=at,
        source="procedural_normalized",
        model_path=model_path,
        normalized_mesh=norm_mesh,
        collision_primitives=colliders.to_dict(),
        target_height=target_h,
        blender_object=blender_obj,
    )


def resolve_prototype_object(
    prototype_type: str,
    vault: Optional[LocalAssetVault] = None,
) -> Any:
    """
    Maps prototype name string to prototype 3D object.
    Integrates with LocalAssetVault and GeometryNormalizer.
    In Blender: returns a normalized bpy.data.objects instance.
    In Pure Python: returns a PurePythonObjectProxy with vertices, polygons, and materials.
    """
    pt = prototype_type.lower().strip()

    # Check abiotic types first
    at = AbioticType.from_any(pt)
    if at is None:
        for member in AbioticType:
            if member.name.lower() in pt:
                at = member
                break
    if at is not None:
        res = fetch_abiotic_asset(at, vault=vault)
        return res.blender_object

    if HAS_BPY and bpy is not None and getattr(bpy, "context", None) is not None:
        # If vault is provided, check if an asset exists in the vault
        if vault is not None:
            res = fetch_biological_asset(pt, vault=vault)
            if res.blender_object is not None:
                return res.blender_object

        # Otherwise invoke the BMesh factory and normalize bottom pivot
        for key, factory in PROTOTYPE_FACTORIES.items():
            if key in pt:
                obj = factory()
                return _normalize_blender_object(obj)

        obj = create_pine_tree()
        return _normalize_blender_object(obj)

    # Pure-Python headless fallback
    res = fetch_biological_asset(pt, vault=vault)
    return res.blender_object


# =============================================================================
# Abiotic Placement Passes: Deep Crag Embedding & Riverbed Boulder Scattering
# =============================================================================

def instantiate_embedded_crags(
    terrain_obj: Any,
    hf: Any,
    config: Any,
    vault: Optional[LocalAssetVault] = None,
    count: int = 8,
    embed_depth: float = -0.8,
) -> List[Any]:
    """
    Instantiates weathered granite crags deeply embedded into mountain ridges/peaks.
    
    Enforces:
    - Minimum negative vertical offset (default -0.8m) to sink irregular base geometry.
    - Placed on steep rock terrain / mountain summits and arete ridges.
    - Whitelisted object name starting with 'Diorama_RockCrag_'.
    - Assigns master triplanar rock terrain shader.
    - Linked directly to scene collection (never TerraForge_Prototypes).
    """
    if not HAS_BPY or bpy is None or getattr(bpy, "context", None) is None:
        return []

    created_crags = []
    shader_cfg = getattr(config, "shaders", None)
    terrain_mat = None
    if shader_cfg and create_triplanar_terrain_shader:
        terrain_mat = create_triplanar_terrain_shader(shader_cfg)

    rd = random.Random(getattr(config, "seed", 42) + 101)

    # Find candidate coordinates on mountain summits and ridges
    peaks = getattr(config, "peaks", [])
    candidates = []
    for p in peaks:
        px, py = p.position
        pr = p.radius
        for _ in range(4):
            ang = rd.uniform(0, 2 * math.pi)
            dist = rd.uniform(0.15 * pr, 0.65 * pr)
            candidates.append((px + dist * math.cos(ang), py + dist * math.sin(ang)))

    # Also sample high slopes (>25 deg)
    if hf is not None:
        slopes = hf.compute_slope_degrees()
        high_slope_idx = np.argwhere((slopes > 25.0) & (hf.Z > float(np.percentile(hf.Z, 60))))
        if len(high_slope_idx) > 0:
            rd.shuffle(high_slope_idx)
            for iy, ix in high_slope_idx[:count]:
                candidates.append((float(hf.X[iy, ix]), float(hf.Y[iy, ix])))

    if not candidates:
        candidates = [(0.0, 0.0), (10.0, -10.0), (-12.0, 8.0)]

    rd.shuffle(candidates)
    selected_positions = candidates[:count]

    for i, (cx, cy) in enumerate(selected_positions):
        crag_name = f"Diorama_RockCrag_{i+1:02d}"
        target_h = rd.uniform(5.5, 8.0)
        yaw_deg = rd.uniform(0.0, 360.0)

        # Base surface elevation
        surf_z = hf.sample_bilinear(cx, cy) if hf is not None else 0.0
        embed_z = surf_z + embed_depth

        # Fetch from vault or procedural
        crag_res = fetch_abiotic_asset(
            AbioticType.RockCrag,
            target_height=target_h,
            vault=vault,
            seed=getattr(config, "seed", 42) + i * 17,
        )

        mesh_source = crag_res.data or (crag_res.blender_object.data if crag_res.blender_object else None)
        if mesh_source is None:
            continue

        # Instantiate mesh into scene
        obj = bpy.data.objects.get(crag_name)
        if not obj:
            obj = bpy.data.objects.new(crag_name, mesh_source)
            bpy.context.scene.collection.objects.link(obj)
        else:
            obj.data = mesh_source

        obj.location = (cx, cy, embed_z)
        obj.rotation_euler = (
            rd.uniform(-0.08, 0.08),
            rd.uniform(-0.08, 0.08),
            math.radians(yaw_deg),
        )
        obj["embed_depth"] = float(embed_depth)
        obj["target_height"] = float(target_h)
        obj["abiotic_type"] = int(AbioticType.RockCrag)

        if terrain_mat:
            if not obj.data.materials:
                obj.data.materials.append(terrain_mat)
            else:
                obj.data.materials[0] = terrain_mat

        created_crags.append(obj)

    return created_crags


def scatter_riverbed_boulders(
    terrain_obj: Any,
    hf: Any,
    hydrology_config: Any,
    config: Any,
    vault: Optional[LocalAssetVault] = None,
    count: int = 16,
    embed_ratio: float = 0.25,
) -> List[Any]:
    """
    Scatters water-worn fluvial riverbed boulders along river channels.
    
    Enforces:
    - 25% embed depth into alluvial riverbed sediment (height * 0.25).
    - Streamwise alignment along flow velocity tangent vector.
    - Whitelisted object name starting with 'Diorama_RiverPebble_'.
    - Linked directly to scene collection (never TerraForge_Prototypes).
    """
    if not HAS_BPY or bpy is None or getattr(bpy, "context", None) is None:
        return []

    created_boulders = []
    rd = random.Random(getattr(config, "seed", 42) + 202)

    # Collect points along river stages
    river_stages = getattr(hydrology_config, "river_stages", [])
    river_points = []
    for stage in river_stages:
        p_start = stage.get("start")
        p_end = stage.get("end")
        if p_start and p_end:
            p0 = Vector((p_start[0], p_start[1], p_start[2]))
            p1 = Vector((p_end[0], p_end[1], p_end[2]))
            v_dir = p1 - p0
            l = v_dir.length
            n_pts = max(3, int(l / 4.0))
            for step in range(n_pts):
                t = (step + 0.5) / float(n_pts)
                pos = p0 + v_dir * t
                perp = Vector((-v_dir.y, v_dir.x, 0.0)).normalized()
                offset = perp * rd.uniform(-1.5, 1.5)
                pt = pos + offset
                river_points.append((pt.x, pt.y, v_dir.normalized()))

    if not river_points:
        lake_cx, lake_cy = getattr(hydrology_config, "lake_center", (0.0, 0.0))
        lake_r = getattr(hydrology_config, "lake_radius", 20.0)
        for i in range(count):
            ang = (i / float(count)) * 2.0 * math.pi
            river_points.append((
                lake_cx + lake_r * math.cos(ang),
                lake_cy + lake_r * math.sin(ang),
                Vector((-math.sin(ang), math.cos(ang), 0.0)),
            ))

    rd.shuffle(river_points)
    selected = river_points[:count]

    for i, (bx, by, flow_dir) in enumerate(selected):
        boulder_name = f"Diorama_RiverPebble_{i+1:02d}"
        b_h = rd.uniform(0.6, 1.4)
        embed_d = b_h * embed_ratio

        surf_z = hf.sample_bilinear(bx, by) if hf is not None else 0.0
        embed_z = surf_z - embed_d

        pebble_res = fetch_abiotic_asset(
            AbioticType.RiverPebble,
            target_height=b_h,
            vault=vault,
            seed=getattr(config, "seed", 42) + i * 31,
        )

        mesh_source = pebble_res.data or (pebble_res.blender_object.data if pebble_res.blender_object else None)
        if mesh_source is None:
            continue

        obj = bpy.data.objects.get(boulder_name)
        if not obj:
            obj = bpy.data.objects.new(boulder_name, mesh_source)
            bpy.context.scene.collection.objects.link(obj)
        else:
            obj.data = mesh_source

        flow_yaw = math.atan2(flow_dir.y, flow_dir.x)
        jitter_yaw = math.radians(rd.uniform(-25.0, 25.0))

        obj.location = (bx, by, embed_z)
        obj.rotation_euler = (
            rd.uniform(-0.06, 0.06),
            rd.uniform(-0.06, 0.06),
            flow_yaw + jitter_yaw,
        )
        obj["embed_depth"] = float(embed_d)
        obj["embed_ratio"] = float(embed_ratio)
        obj["target_height"] = float(b_h)
        obj["abiotic_type"] = int(AbioticType.RiverPebble)

        mat = _get_or_create_material(
            "MAT_Abiotic_RiverPebble",
            base_color=(0.34, 0.32, 0.30),
            roughness=0.35,
            specular=0.55,
        )
        if mat:
            if not obj.data.materials:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat

        created_boulders.append(obj)

    return created_boulders

