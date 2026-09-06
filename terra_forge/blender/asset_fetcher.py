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
import math
from typing import Optional, Dict, Any

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

if HAS_BPY:
    try:
        from terra_forge.blender.shaders import create_bioluminescent_fungus_shader
    except ImportError:
        create_bioluminescent_fungus_shader = None
else:
    create_bioluminescent_fungus_shader = None


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


def _get_or_create_material(mat_name: str, base_color: tuple, roughness: float = 0.6, specular: float = 0.5) -> Optional[Any]:
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


def resolve_prototype_object(prototype_type: str) -> Any:
    """Maps prototype name string to prototype 3D object."""
    pt = prototype_type.lower()
    for key, factory in PROTOTYPE_FACTORIES.items():
        if key in pt:
            return factory()

    # Fallback to pine tree
    return create_pine_tree()
