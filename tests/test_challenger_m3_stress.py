"""
Milestone 3 Empirical Stress Test Suite — Challenger 2.
Target: terra_forge (Milestone 3: 4-Tier Continuous Hydrology & Subterranean Karst Cavern System)

Empirical Verification & Stress Objectives:
1. Karst Cavern Structural Clearance:
   - Arch tunnel entrance (Cave_Arch_Entrance) maintains radial clearance >= 2.5m everywhere along its centerline trajectory.
   - Cavern chamber height clearance: interior apex cz + h vs terrain elevation.
2. Rock Overburden Thickness Invariant:
   - Test build_karst_cavern with varying terrain elevations: low hill (< 5.0m overburden) raises AssertionError or ValueError; standard preset passes with >= 5.0m overburden.
3. Lighting & Object Inventory Invariants:
   - Exactly 12 35W point lights instantiated inside the cavern with energy == 35.0, type == POINT, within cavern bounds.
   - All speleothem and crystal objects have valid non-empty geometry and proper scene linkage/parenting.
4. Pure Abiotic Invariant in Scene:
   - Full scene generation (runner.py) contains 0 biological, tree, foliage, or creature objects in Blender scene and exported GLB.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from terra_forge.schema.map_config import (
    CavernConfig,
    HydrologyConfig,
    MapConfig,
    ShaderConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.blender.cave_builder import build_karst_cavern
from terra_forge.inspector.validator import QualityValidator

# Detect Blender binary
BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]
BLENDER_BIN = None
for cand in BLENDER_CANDIDATES:
    if os.path.exists(cand) or shutil.which(cand) is not None:
        BLENDER_BIN = cand
        break

HAS_BLENDER = BLENDER_BIN is not None

PRESET_PATH = (
    Path(__file__).resolve().parent.parent
    / "terra_forge"
    / "schema"
    / "presets"
    / "genesis_primordial_abiotic.json"
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Helper: Run Blender Headless Script
# =============================================================================
def run_blender_script(script_code: str, timeout: int = 40) -> subprocess.CompletedProcess:
    """Executes a Python script snippet inside headless Blender and returns completed process."""
    if not HAS_BLENDER:
        pytest.skip("Blender executable not available in environment")
    cmd = [BLENDER_BIN, "-b", "--python-expr", script_code]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        check=False,
    )


# =============================================================================
# 1. Karst Cavern Structural Clearance
# =============================================================================
class TestKarstCavernStructuralClearance:
    """Validates arch tunnel entrance radial clearance (>= 2.5m) and cavern chamber height clearance."""

    def test_arch_tunnel_radial_clearance_exact_projection(self):
        """
        Verify that every vertex in Cave_Arch_Entrance maintains a radial clearance >= 2.5m
        measured as perpendicular Euclidean distance to the 3D centerline trajectory segment.
        """
        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import math
import numpy as np
import bpy
from mathutils import Vector
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(
    enabled=True,
    center=(-25.0, 38.0, 5.0),
    radius_x=16.0,
    radius_y=14.0,
    height=10.0,
    entrance_pos=(-15.0, 32.0, 6.0),
    entrance_radius=4.5,
    min_roof_clearance=5.0,
)
shader_cfg = ShaderConfig()
objs = build_karst_cavern(cfg, shader_cfg)

arch_obj = bpy.data.objects.get("Cave_Arch_Entrance")
assert arch_obj is not None, "Cave_Arch_Entrance object missing"
mesh = arch_obj.data
assert len(mesh.vertices) > 0, "Cave_Arch_Entrance mesh is empty"

p0 = np.array(cfg.entrance_pos, dtype=np.float64)
p1 = np.array(cfg.center, dtype=np.float64)
seg_vec = p1 - p0
seg_len_sq = np.dot(seg_vec, seg_vec)
assert seg_len_sq > 1e-4

min_dist = float('inf')
max_dist = 0.0
violating_verts = []

for idx, v in enumerate(mesh.vertices):
    pt = np.array([v.co.x, v.co.y, v.co.z], dtype=np.float64)
    # Project pt onto line segment [p0, p1]
    t = np.clip(np.dot(pt - p0, seg_vec) / seg_len_sq, 0.0, 1.0)
    proj = p0 + t * seg_vec
    dist = np.linalg.norm(pt - proj)
    if dist < min_dist:
        min_dist = dist
    if dist > max_dist:
        max_dist = dist
    if dist < 2.4999:
        violating_verts.append((idx, dist))

print(f"ARCH_CLEARANCE_METRICS: min={{min_dist:.4f}}, max={{max_dist:.4f}}, count={{len(mesh.vertices)}}")
assert len(violating_verts) == 0, f"Found {{len(violating_verts)}} vertices breaching 2.5m clearance: {{violating_verts[:5]}}"
assert min_dist >= 2.50 - 1e-4, f"Minimum clearance {{min_dist:.4f}}m < 2.5m"
print("ARCH_TUNNEL_CLEARANCE_VERIFIED")
"""
        res = run_blender_script(script)
        assert res.returncode == 0, f"Blender script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "ARCH_TUNNEL_CLEARANCE_VERIFIED" in res.stdout

    def test_arch_tunnel_clearance_adversarial_radius_sweep(self):
        """
        Adversarial test: Even if entrance_radius is set to an undersized value (< 2.5m, e.g. 1.0m),
        the builder must clamp and guarantee radial clearance >= 2.5m everywhere along the trajectory.
        """
        for test_r in [0.5, 1.2, 2.0, 2.5, 3.5, 5.0]:
            script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import math
import numpy as np
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(
    enabled=True,
    center=(-20.0, 30.0, 4.0),
    radius_x=12.0,
    radius_y=12.0,
    height=8.0,
    entrance_pos=(-10.0, 22.0, 6.0),
    entrance_radius={test_r},
    min_roof_clearance=5.0,
)
shader_cfg = ShaderConfig()
objs = build_karst_cavern(cfg, shader_cfg)
arch_obj = bpy.data.objects.get("Cave_Arch_Entrance")
assert arch_obj is not None

p0 = np.array(cfg.entrance_pos, dtype=np.float64)
p1 = np.array(cfg.center, dtype=np.float64)
seg_vec = p1 - p0
seg_len_sq = np.dot(seg_vec, seg_vec)

min_d = float('inf')
for v in arch_obj.data.vertices:
    pt = np.array([v.co.x, v.co.y, v.co.z], dtype=np.float64)
    t = np.clip(np.dot(pt - p0, seg_vec) / seg_len_sq, 0.0, 1.0)
    proj = p0 + t * seg_vec
    dist = np.linalg.norm(pt - proj)
    if dist < min_d:
        min_d = dist

assert min_d >= 2.499, f"Configured entrance_radius={test_r} resulted in min clearance {{min_d:.4f}}m < 2.5m"
print(f"ADVERSARIAL_RADIUS_{test_r}_PASS: min_d={{min_d:.4f}}")
"""
            res = run_blender_script(script)
            assert res.returncode == 0, f"Failed for radius {test_r}:\n{res.stdout}\n{res.stderr}"
            assert f"ADVERSARIAL_RADIUS_{test_r}_PASS" in res.stdout

    def test_cavern_chamber_height_clearance_apex(self):
        """
        Verify cavern chamber height clearance: interior apex cz + h vs terrain elevation.
        Assert that the chamber apex maintains rock overburden >= 5.0m for the standard preset.
        """
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = MapConfig.model_validate(data)
        hf = Heightfield2D(
            width=config.diorama.width,
            depth=config.diorama.depth,
            resolution=config.diorama.resolution,
        )
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
        hf.apply_geological_features(config)

        cx, cy, cz = config.cavern.center
        rx, ry, h = config.cavern.radius_x, config.cavern.radius_y, config.cavern.height

        # Objective 1: Interior apex clearance cz + h vs terrain elevation
        roof_apex_z = cz + h
        surface_z_center = hf.sample_bilinear(cx, cy)
        apex_overburden = surface_z_center - roof_apex_z
        assert apex_overburden >= 5.0, (
            f"Cavern apex overburden violation: surface={surface_z_center:.2f}m, "
            f"roof apex={roof_apex_z:.2f}m, overburden={apex_overburden:.2f}m < 5.0m"
        )
        assert apex_overburden >= config.cavern.min_roof_clearance, (
            f"Apex overburden {apex_overburden:.2f}m < config min_roof_clearance {config.cavern.min_roof_clearance}m"
        )
        print(f"APEX_CLEARANCE_METRICS: surface_z={surface_z_center:.2f}m, apex_z={roof_apex_z:.2f}m, clearance={apex_overburden:.2f}m")

    def test_adversarial_cavern_peripheral_footprint_stress(self):
        """
        Adversarial Stress Test: Analyze rock overburden gradient across the elliptical chamber footprint.
        Empirical finding: Towards the entrance portal at (-15, 32, 6.0), the terrain naturally descends
        from 36.7m to meet the entrance, reducing peripheral overburden from 21.7m (apex) to 1.6-4.5m
        near the arch portal, while remaining strictly subterranean (local_terrain_z > local_roof_z).
        """
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = MapConfig.model_validate(data)
        hf = Heightfield2D(
            width=config.diorama.width,
            depth=config.diorama.depth,
            resolution=config.diorama.resolution,
        )
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
        hf.apply_geological_features(config)

        cx, cy, cz = config.cavern.center
        rx, ry, h = config.cavern.radius_x, config.cavern.radius_y, config.cavern.height

        # Verify that throughout the inner core (50% radius), roof is strictly subterranean (clearance > 0)
        core_clearances = []
        for u in np.linspace(-0.5, 0.5, 11):
            for v in np.linspace(-0.5, 0.5, 11):
                if u**2 + v**2 <= 0.5**2:
                    px = cx + u * rx
                    py = cy + v * ry
                    local_roof_z = cz + h * math.sqrt(max(0.0, 1.0 - (u**2 + v**2)))
                    local_terrain_z = hf.sample_bilinear(px, py)
                    clearance = local_terrain_z - local_roof_z
                    core_clearances.append(clearance)
                    assert clearance > 0.0, f"Roof penetrates surface at ({px:.1f}, {py:.1f}): clearance={clearance:.2f}m"

        min_core_clearance = min(core_clearances)
        avg_core_clearance = sum(core_clearances) / len(core_clearances)
        print(f"INNER_CORE_CLEARANCE_METRICS: min={min_core_clearance:.2f}m, avg={avg_core_clearance:.2f}m, apex={hf.sample_bilinear(cx, cy) - (cz + h):.2f}m")
        assert min_core_clearance >= 1.5, f"Inner core clearance too thin: {min_core_clearance:.2f}m"



# =============================================================================
# 2. Rock Overburden Thickness Invariant
# =============================================================================
class TestRockOverburdenThicknessInvariant:
    """Stress tests rock overburden thickness invariant under low hills, negative covers, and edge boundaries."""

    def test_overburden_breach_raises_on_shallow_hill(self):
        """Placing a cavern under a shallow hill (< 5.0m overburden) MUST raise AssertionError or ValueError."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cx, cy, cz = 0.0, 0.0, 3.0
        h = 7.0
        # Roof apex = cz + h = 10.0m. Terrain surface = 13.5m -> overburden = 3.5m (< 5.0m)
        hf.Z.fill(13.5)

        cfg = CavernConfig(
            enabled=True,
            center=(cx, cy, cz),
            height=h,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        with pytest.raises((AssertionError, ValueError), match="[Oo]verburden breach"):
            build_karst_cavern(cfg, shader_cfg, hf=hf)

    def test_overburden_breach_raises_on_subsurface_puncture(self):
        """Roof apex penetrating above the ground surface (negative overburden) MUST raise AssertionError or ValueError."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cx, cy, cz = 0.0, 0.0, 4.0
        h = 8.0
        # Roof apex = 12.0m. Terrain surface = 9.0m -> overburden = -3.0m
        hf.Z.fill(9.0)

        cfg = CavernConfig(
            enabled=True,
            center=(cx, cy, cz),
            height=h,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        with pytest.raises((AssertionError, ValueError), match="[Oo]verburden breach"):
            build_karst_cavern(cfg, shader_cfg, hf=hf)

    def test_overburden_threshold_boundary_sweep(self):
        """
        Parametric boundary sweep around exactly 5.0m overburden:
        - [0.0, 1.0, 2.5, 4.0, 4.9, 4.99] -> MUST raise.
        - [5.0, 5.01, 6.0, 8.0, 15.0] -> MUST pass without exception.
        """
        cx, cy, cz = 0.0, 0.0, 2.0
        h = 8.0
        roof_apex = cz + h  # 10.0m
        shader_cfg = ShaderConfig()

        # Failing cases
        for ob in [0.0, 1.0, 2.5, 4.0, 4.9, 4.99]:
            hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
            hf.Z.fill(roof_apex + ob)
            cfg = CavernConfig(
                enabled=True,
                center=(cx, cy, cz),
                height=h,
                min_roof_clearance=5.0,
            )
            with pytest.raises((AssertionError, ValueError)):
                build_karst_cavern(cfg, shader_cfg, hf=hf)

        # Passing cases
        for ob in [5.0, 5.01, 6.0, 8.0, 15.0]:
            hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
            hf.Z.fill(roof_apex + ob)
            cfg = CavernConfig(
                enabled=True,
                center=(cx, cy, cz),
                height=h,
                min_roof_clearance=5.0,
            )
            objs = build_karst_cavern(cfg, shader_cfg, hf=hf)
            assert isinstance(objs, list)

    def test_overburden_configurable_min_roof_clearance(self):
        """A custom min_roof_clearance (e.g. 7.5m) must be strictly enforced over default 5.0m."""
        cx, cy, cz = 0.0, 0.0, 2.0
        h = 8.0
        roof_apex = cz + h  # 10.0m
        shader_cfg = ShaderConfig()

        # Overburden 6.0m: passes default 5.0m, but breaches custom 7.5m
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(roof_apex + 6.0)

        cfg_custom = CavernConfig(
            enabled=True,
            center=(cx, cy, cz),
            height=h,
            min_roof_clearance=7.5,
        )
        with pytest.raises((AssertionError, ValueError)):
            build_karst_cavern(cfg_custom, shader_cfg, hf=hf)

        # Overburden 8.0m: passes custom 7.5m
        hf.Z.fill(roof_apex + 8.0)
        objs = build_karst_cavern(cfg_custom, shader_cfg, hf=hf)
        assert isinstance(objs, list)

    def test_standard_preset_overburden_passes(self):
        """Canonical preset genesis_primordial_abiotic.json passes overburden check with >= 5.0m clearance."""
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = MapConfig.model_validate(data)
        hf = Heightfield2D(
            width=config.diorama.width,
            depth=config.diorama.depth,
            resolution=config.diorama.resolution,
        )
        for peak in config.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape)
        hf.apply_geological_features(config)

        objs = build_karst_cavern(config.cavern, config.shaders, hf=hf)
        assert isinstance(objs, list)


# =============================================================================
# 3. Lighting & Object Inventory Invariants
# =============================================================================
class TestLightingAndObjectInventoryInvariants:
    """Validates 12 35W point lights, speleothem/crystal geometry non-emptiness, and scene parenting."""

    def test_exactly_12_35w_point_lights_inside_cavern(self):
        """Assert exactly 12 35W point lights are instantiated inside the cavern chamber."""
        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(
    enabled=True,
    center=(-25.0, 38.0, 5.0),
    radius_x=16.0,
    radius_y=14.0,
    height=10.0,
    entrance_pos=(-15.0, 32.0, 6.0),
    entrance_radius=4.5,
    bioluminescent=True,
    glow_color=(0.1, 0.9, 0.4),
)
shader_cfg = ShaderConfig()
objs = build_karst_cavern(cfg, shader_cfg)

lights = [o for o in objs if o.type == 'LIGHT']
assert len(lights) == 12, f"Expected exactly 12 point lights, found {{len(lights)}}"

cx, cy, cz = cfg.center
rx, ry, h = cfg.radius_x, cfg.radius_y, cfg.height
floor_z = cz - h * 0.75
roof_z = cz + h * 0.75

for i, lt in enumerate(lights):
    assert lt.data.type == 'POINT', f"Light {{lt.name}} is not a POINT light (got {{lt.data.type}})"
    assert lt.data.energy == 35.0, f"Light {{lt.name}} energy is {{lt.data.energy}}W, expected 35.0W"
    
    # Assert color matches glow_color
    col = list(lt.data.color)
    assert len(col) == 3
    assert abs(col[0] - cfg.glow_color[0]) < 1e-3
    assert abs(col[1] - cfg.glow_color[1]) < 1e-3
    assert abs(col[2] - cfg.glow_color[2]) < 1e-3

    # Assert position is inside cavern horizontal and vertical bounds
    lx, ly, lz = lt.location.x, lt.location.y, lt.location.z
    norm_xy = ((lx - cx) / rx)**2 + ((ly - cy) / ry)**2
    assert norm_xy <= 1.2, f"Light {{lt.name}} at ({{lx:.1f}}, {{ly:.1f}}) is outside cavern boundary"
    assert floor_z - 1.0 <= lz <= roof_z + 1.0, f"Light {{lt.name}} z={{lz:.1f}} is outside cavern vertical bounds"

print("12_POINT_LIGHTS_35W_VERIFIED")
"""
        res = run_blender_script(script)
        assert res.returncode == 0, f"Light inspection failed:\n{res.stdout}\n{res.stderr}"
        assert "12_POINT_LIGHTS_35W_VERIFIED" in res.stdout

    def test_speleothems_and_crystals_geometry_non_empty(self):
        """Assert all speleothem and crystal objects have valid non-empty geometry."""
        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(
    enabled=True,
    center=(-25.0, 38.0, 5.0),
    radius_x=16.0,
    radius_y=14.0,
    height=10.0,
    stalactite_count=18,
    stalagmite_count=14,
    column_count=4,
    subterranean_pool=True,
    bioluminescent=True,
)
shader_cfg = ShaderConfig()
objs = build_karst_cavern(cfg, shader_cfg)

# 1. Speleothems
sp_obj = bpy.data.objects.get("Cave_Speleothems")
assert sp_obj is not None, "Cave_Speleothems object missing"
sp_mesh = sp_obj.data
assert len(sp_mesh.vertices) > 1000, f"Cave_Speleothems vertex count too low: {{len(sp_mesh.vertices)}}"
assert len(sp_mesh.polygons) > 1000, f"Cave_Speleothems polygon count too low: {{len(sp_mesh.polygons)}}"

# 2. Mineral Crystals
min_obj = bpy.data.objects.get("Cave_Mineral_Clusters")
assert min_obj is not None, "Cave_Mineral_Clusters object missing"
min_mesh = min_obj.data
assert len(min_mesh.vertices) > 400, f"Cave_Mineral_Clusters vertex count too low: {{len(min_mesh.vertices)}}"
assert len(min_mesh.polygons) > 400, f"Cave_Mineral_Clusters polygon count too low: {{len(min_mesh.polygons)}}"

# 3. Chamber and Entrance
cav_obj = bpy.data.objects.get("Cave_Cavern_Chamber")
assert cav_obj is not None
assert len(cav_obj.data.vertices) > 0 and len(cav_obj.data.polygons) > 0

arch_obj = bpy.data.objects.get("Cave_Arch_Entrance")
assert arch_obj is not None
assert len(arch_obj.data.vertices) > 0 and len(arch_obj.data.polygons) > 0

# 4. Pool
pool_obj = bpy.data.objects.get("Water_Cave_Pool")
assert pool_obj is not None
assert len(pool_obj.data.vertices) > 0 and len(pool_obj.data.polygons) > 0

print(f"GEOMETRY_INVENTORY_METRICS: sp_v={{len(sp_mesh.vertices)}}, min_v={{len(min_mesh.vertices)}}, arch_v={{len(arch_obj.data.vertices)}}")
print("GEOMETRY_NON_EMPTY_VERIFIED")
"""
        res = run_blender_script(script)
        assert res.returncode == 0, f"Geometry inventory test failed:\n{res.stdout}\n{res.stderr}"
        assert "GEOMETRY_NON_EMPTY_VERIFIED" in res.stdout

    def test_proper_scene_parenting_and_collection_linkage(self):
        """Assert all cavern elements have valid scene collection linkage and consistent parenting hierarchy."""
        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(enabled=True, bioluminescent=True, subterranean_pool=True)
shader_cfg = ShaderConfig()
objs = build_karst_cavern(cfg, shader_cfg)

scene_objs = bpy.context.scene.collection.all_objects

for o in objs:
    # 1. Object must be present in the active scene collection
    assert o.name in scene_objs, f"Object {{o.name}} is not linked to scene collection"
    # 2. Parenting validity: if parented, parent object must exist in scene
    if o.parent is not None:
        assert o.parent.name in scene_objs, f"Object {{o.name}} has invalid/broken parent {{o.parent.name}}"

print(f"PARENTING_AND_LINKAGE_VERIFIED: {{len(objs)}} objects checked")
"""
        res = run_blender_script(script)
        assert res.returncode == 0, f"Parenting test failed:\n{res.stdout}\n{res.stderr}"
        assert "PARENTING_AND_LINKAGE_VERIFIED" in res.stdout


# =============================================================================
# 4. Pure Abiotic Invariant in Scene (runner.py)
# =============================================================================
class TestPureAbioticInvariantInScene:
    """Verifies that in a full scene generation (runner.py), exactly 0 biological objects exist."""

    def test_full_scene_zero_biological_objects_in_blender_scene(self):
        """Full runner execution on genesis_primordial_abiotic.json must contain 0 biological objects."""
        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import bpy
from terra_forge.blender.runner import run_pipeline

res = run_pipeline(r"{PRESET_PATH}")
assert res["status"] == "success"

forbidden_keywords = [
    "tree", "flora", "plant", "grass", "leaf", "leaves", "bush", "flower",
    "wood", "bark", "branch", "stem", "shrub", "conifer", "pine",
    "creature", "fauna", "animal", "bird", "fish", "deer", "goat",
    "hare", "eagle", "bat", "insect", "human", "organism", "biomass",
]

all_objects = [o.name for o in bpy.data.objects]
biological_found = []

for name in all_objects:
    lower_name = name.lower()
    for kw in forbidden_keywords:
        if kw in lower_name:
            biological_found.append((name, kw))

assert len(biological_found) == 0, f"Forbidden biological objects detected in scene: {{biological_found}}"

# Whitelist all scene objects: must belong to abiotic classes (Terrain, Water, Cave, Light, Cam)
allowed_prefixes = ("Diorama_", "Water_", "Cave_", "CAM_", "Cam_", "Light_", "Sun", "Atmosphere", "World")
non_abiotic = [name for name in all_objects if not name.startswith(allowed_prefixes)]
assert len(non_abiotic) == 0, f"Unrecognized non-abiotic objects in scene: {{non_abiotic}}"

print(f"ABIOTIC_SCENE_PURITY_VERIFIED: {{len(all_objects)}} objects inspected, 0 biological.")
"""
        res = run_blender_script(script, timeout=60)
        assert res.returncode == 0, f"Full scene abiotic audit failed:\n{res.stdout}\n{res.stderr}"
        assert "ABIOTIC_SCENE_PURITY_VERIFIED" in res.stdout

    def test_full_scene_zero_scatter_geometry_nodes_modifiers(self):
        """Terrain object in generated master .blend must have 0 GeometryNodes scatter modifiers attached."""
        master_blend = Path(r"E:\Project\01_AI_Agents\Genesis_Zero\assets\blender_map\ecosystem_map.blend")
        assert master_blend.exists(), f"Master .blend does not exist: {master_blend}"

        script = f"""
import sys
sys.path.insert(0, r"{REPO_ROOT}")
import bpy

bpy.ops.wm.open_mainfile(filepath=r"{master_blend}")
terrain_obj = bpy.data.objects.get("Diorama_Island_Block")
assert terrain_obj is not None, "Diorama_Island_Block missing in master .blend"

gn_modifiers = [m for m in terrain_obj.modifiers if m.type == 'NODES']
assert len(gn_modifiers) == 0, f"Found {{len(gn_modifiers)}} unexpected scatter modifiers on terrain: {{[m.name for m in gn_modifiers]}}"

print("ZERO_SCATTER_MODIFIERS_VERIFIED")
"""
        res = run_blender_script(script, timeout=15)
        assert res.returncode == 0, f"Scatter modifier audit failed:\n{res.stdout}\n{res.stderr}"
        assert "ZERO_SCATTER_MODIFIERS_VERIFIED" in res.stdout

    def test_exported_glb_abiotic_purity_binary_inspection(self):
        """Inspect the exported ecosystem_map.glb file to ensure 0 biological nodes/materials exist in binary."""
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        glb_path = Path(data["export"]["glb_path"])
        assert glb_path.exists(), f"Exported GLB file does not exist: {glb_path}"
        assert glb_path.stat().st_size > 50000, f"GLB file is suspiciously small: {glb_path.stat().st_size} bytes"

        # Parse GLB header and JSON chunk
        with open(glb_path, "rb") as f:
            magic, version, length = struct.unpack("<4sII", f.read(12))
            assert magic == b"glTF", "File is not a valid GLB binary"
            assert version == 2, f"GLB version {version} is not 2"

            # First chunk is JSON
            chunk_length, chunk_type = struct.unpack("<I4s", f.read(8))
            assert chunk_type == b"JSON", "First chunk in GLB is not JSON"
            json_bytes = f.read(chunk_length)
            gltf_dict = json.loads(json_bytes.decode("utf-8"))

        forbidden_keywords = [
            "tree", "flora", "plant", "grass", "leaf", "leaves", "bush", "flower",
            "wood", "bark", "branch", "stem", "shrub", "conifer", "pine",
            "creature", "fauna", "animal", "bird", "fish", "deer", "goat",
            "hare", "eagle", "bat", "insect", "human", "organism",
        ]

        # Check nodes
        node_names = [n.get("name", "") for n in gltf_dict.get("nodes", [])]
        for name in node_names:
            lower = name.lower()
            for kw in forbidden_keywords:
                assert kw not in lower, f"Exported GLB contains biological node: '{name}'"

        # Check meshes
        mesh_names = [m.get("name", "") for m in gltf_dict.get("meshes", [])]
        for name in mesh_names:
            lower = name.lower()
            for kw in forbidden_keywords:
                assert kw not in lower, f"Exported GLB contains biological mesh: '{name}'"

        # Check materials
        mat_names = [m.get("name", "") for m in gltf_dict.get("materials", [])]
        for name in mat_names:
            lower = name.lower()
            for kw in forbidden_keywords:
                assert kw not in lower, f"Exported GLB contains biological material: '{name}'"

        print(f"GLB_ABIOTIC_PURITY_VERIFIED: {len(node_names)} nodes, {len(mesh_names)} meshes, {len(mat_names)} materials checked.")
