"""
Milestone M4 Deliverables Verification Test Suite.

Verifies:
1. ObstacleGridSynchronizer handles abiotic solid colliders and cavern bounding obstacles.
2. NavMeshReachabilityValidator enforces 35° slope threshold and inland water depth clearance.
3. find_safe_spawn_point guarantees safe spawn (gentle slope < 15°, dry ground, largest component)
   and achieves 4-connected BFS reachability >= 80.0%, preventing mountain cliff traps.
4. GLB production export excludes prototypes, disables animations, and disables Draco.
5. All 4 official visual acceptance renders exist, have valid dimensions (1280x720), and are properly exposed.
6. Offline 3D viewer has 0 remote HTTP/HTTPS dependencies and references local vendor scripts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import numpy as np
from PIL import Image
import pytest

from terra_forge.ai.vault import AbioticType, FloraType
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.navigation.navmesh import (
    CANONICAL_COVERAGE_TARGET,
    DEFAULT_GENTLE_SLOPE_DEG,
    DEFAULT_MAX_WALKABLE_SLOPE_DEG,
    DEFAULT_WALKABLE_SLOPE,
    SOLID_ABIOTIC_TYPES,
    TALL_FLORA_TYPES,
    NavMeshReachabilityValidator,
    ObstacleGridSynchronizer,
    validate_and_prune_navmesh,
)

GENESIS_ZERO_ROOT = Path(r"e:/Project/01_AI_Agents/Genesis_Zero")
BLENDER_MAP_DIR = GENESIS_ZERO_ROOT / "assets" / "blender_map"
GLB_PATH = BLENDER_MAP_DIR / "ecosystem_map.glb"
VIEWER_HTML_PATH = BLENDER_MAP_DIR / "viewer.html"


# =============================================================================
# 1. Abiotic Solid Colliders & Obstacle Synchronization
# =============================================================================

class TestAbioticObstacleGridSynchronization:
    """Verifies that abiotic geological structures are correctly rasterized into obstacle grid."""

    def test_solid_abiotic_types_registered(self):
        """Solid abiotic types (HornPeak, RockCrag, BasaltColumn, Stalagmite, CavernColumn) must register colliders."""
        sync = ObstacleGridSynchronizer(shape=(50, 50), world_bounds=(-50.0, -50.0, 50.0, 50.0))
        
        solid_types = [
            AbioticType.HornPeak,
            AbioticType.RockCrag,
            AbioticType.BasaltColumn,
            AbioticType.Stalagmite,
            AbioticType.CavernColumn,
        ]
        
        for idx, ab_type in enumerate(solid_types):
            obs = sync.register_abiotic_instance(
                instance_index=idx,
                abiotic_type=ab_type,
                x=float(idx * 5 - 10),
                z=float(idx * 5 - 10),
                scale=1.0,
            )
            assert obs is not None, f"Failed to register solid abiotic type: {ab_type.name}"
            assert obs.collider_radius > 0.0
            assert obs.spawn_clearance >= obs.collider_radius

        assert len(sync.solid_obstacles) == len(solid_types)
        assert np.any(sync.obstacle_grid), "Obstacle grid must contain marked cells for abiotic solids"

    def test_pass_through_abiotic_types_not_registered(self):
        """Pass-through types (RiverPebble, RiparianSandbank, TalusScree, AlluvialSand) must return None."""
        sync = ObstacleGridSynchronizer(shape=(50, 50), world_bounds=(-50.0, -50.0, 50.0, 50.0))
        
        pass_through = [
            AbioticType.RiverPebble,
            AbioticType.RiparianSandbank,
            AbioticType.Stalactite,
            AbioticType.CavernMineral,
        ]

        for idx, ab_type in enumerate(pass_through):
            obs = sync.register_abiotic_instance(
                instance_index=idx,
                abiotic_type=ab_type,
                x=float(idx * 4 - 8),
                z=float(idx * 4 - 8),
                scale=1.0,
            )
            assert obs is None, f"Pass-through abiotic type {ab_type.name} should not register solid collider"

        assert len(sync.solid_obstacles) == 0
        assert not np.any(sync.obstacle_grid), "Pass-through types must not mark obstacle grid"

    def test_cavern_obstacle_registration(self):
        """register_cavern_obstacles must register columns and stalagmites."""
        sync = ObstacleGridSynchronizer(shape=(64, 64), world_bounds=(-50.0, -50.0, 50.0, 50.0))
        cols = [(-16.0, 28.0), (-12.0, 32.0)]
        stals = [(-15.0, 27.0, 1.2), (-13.0, 29.0, 0.8)]
        
        sync.register_cavern_obstacles(column_positions=cols, stalagmites=stals)
        assert len(sync.solid_obstacles) == 4
        assert np.any(sync.obstacle_grid)


# =============================================================================
# 2. Slope 35° Threshold & Inland Water Depth Clearance
# =============================================================================

class TestSlopeThresholdAndWaterDepth:
    """Verifies physical slope threshold scaling and inland water depth handling."""

    def test_slope_35_degree_walkability_cutoff(self):
        """Slopes > 35° must be impassable; slopes < 35° must be walkable."""
        dim = 40
        elev = np.zeros((dim, dim), dtype=np.float32)
        
        # Create a linear ramp with known slope tangent:
        # dx = 200 / 39 ≈ 5.128m
        # For slope = 30°: tan(30°) ≈ 0.577 -> dz = 0.577 * 5.128 ≈ 2.96m per cell
        # For slope = 40°: tan(40°) ≈ 0.839 -> dz = 0.839 * 5.128 ≈ 4.30m per cell
        dx_m = 200.0 / (dim - 1)
        
        # Gentle half (slope ~ 25°)
        elev[:20, :] = np.linspace(0.0, 19.0 * dx_m * math.tan(math.radians(25.0)), 20)[:, None]
        # Steep half (slope ~ 45°)
        start_z = float(elev[19, 0])
        elev[20:, :] = start_z + np.linspace(dx_m * math.tan(math.radians(45.0)), 20.0 * dx_m * math.tan(math.radians(45.0)), 20)[:, None]

        val = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=-5.0,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
        )
        sync = ObstacleGridSynchronizer(shape=(dim, dim))
        walkable = val.compute_walkable_mask(sync.obstacle_grid)

        # Gentle rows (0..18) must be walkable
        assert np.all(walkable[1:18, :]), "Slopes < 35° must be walkable"
        # Steep rows (20..38) must be blocked
        assert not np.any(walkable[21:38, :]), "Slopes > 35° must be non-walkable"

    def test_inland_water_depth_blocks_submerged_cells(self):
        """Inland lake/river with water_depth > 0.05 must be marked as water (non-land/non-walkable)."""
        dim = 30
        elev = np.full((dim, dim), 5.0, dtype=np.float32)
        water_depth = np.zeros((dim, dim), dtype=np.float32)
        # Deep lake in the center (10..20, 10..20)
        water_depth[10:20, 10:20] = 2.5

        val = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=0.0,
            water_depth=water_depth,
        )
        sync = ObstacleGridSynchronizer(shape=(dim, dim))
        walkable = val.compute_walkable_mask(sync.obstacle_grid)

        # Lake center must not be land and must not be walkable
        assert not np.any(val.is_land[10:20, 10:20]), "Submerged cells must be classified as water"
        assert not np.any(walkable[10:20, 10:20]), "Submerged lake cells must be impassable"
        # Outer dry land must be walkable
        assert np.all(walkable[:8, :]), "Dry surrounding land must remain walkable"


# =============================================================================
# 3. Safe Spawn Point Guarantee & Mountain Cliff Trap Prevention
# =============================================================================

class TestSafeSpawnPointAndReachability:
    """Verifies that find_safe_spawn_point guarantees >= 80% coverage and escapes cliff traps."""

    def test_safe_spawn_escapes_mountain_summit_cliff_trap(self):
        """
        Adversarial Scenario: A sharp mountain horn peak is centered at (0, 0)
        with impassable cliffs (> 40°).
        The legacy center-closest algorithm gets trapped at the peak (coverage < 5%).
        find_safe_spawn_point must reject the steep peak, find a gentle-slope valley cell,
        and achieve navmeshCoverage >= 80.0%.
        """
        dim = 128
        hf = Heightfield2D(width=160.0, depth=160.0, resolution=dim)
        hf.Z.fill(2.0)  # Gentle flat base
        # Add sharp horn peak at center
        hf.add_peak(0.0, 0.0, height=12.0, radius=25.0, shape="horn")

        val = NavMeshReachabilityValidator(
            elevation=hf.Z,
            sea_level=0.5,
            world_bounds=(-80.0, -80.0, 80.0, 80.0),
            safe_spawn=True,
        )
        sync = ObstacleGridSynchronizer(shape=(dim, dim), world_bounds=(-80.0, -80.0, 80.0, 80.0))
        
        walkable = val.compute_walkable_mask(sync.obstacle_grid)
        spawn_cell = val.find_safe_spawn_point(walkable, sync)
        
        # Assert spawn criteria
        spawn_r, spawn_c = spawn_cell
        assert val.slope_deg[spawn_r, spawn_c] < DEFAULT_GENTLE_SLOPE_DEG, "Spawn slope must be gentle (< 15°)"
        assert val.elevation[spawn_r, spawn_c] > val.water_cutoff + 0.5, "Spawn must have elevation clearance above water"
        
        # Evaluate reachability from safe spawn
        res = val.evaluate(sync, safe_spawn=True)
        assert res.navmesh_coverage >= CANONICAL_COVERAGE_TARGET, (
            f"Expected safe spawn to achieve >= 80% coverage, got {res.navmesh_coverage:.4f}"
        )
        assert res.is_valid is True

    def test_safe_spawn_on_flat_island(self):
        """On a uniform flat island, safe spawn point is placed safely near center."""
        dim = 32
        elev = np.full((dim, dim), 3.0, dtype=np.float32)
        val = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0, safe_spawn=True)
        sync = ObstacleGridSynchronizer(shape=(dim, dim))
        res = val.evaluate(sync, safe_spawn=True)
        assert res.is_valid is True
        assert res.navmesh_coverage >= 0.95
        sr, sc = res.spawn_cell
        assert 8 <= sr <= 24 and 8 <= sc <= 24


# =============================================================================
# 4. GLB Production Deliverable Integrity
# =============================================================================

class TestProductionGLBDeliverable:
    """Verifies ecosystem_map.glb complies with Three.js production constraints."""

    def test_glb_file_exists_and_size(self):
        assert GLB_PATH.is_file(), f"Missing production GLB: {GLB_PATH}"
        size = GLB_PATH.stat().st_size
        assert size > 2_000_000, f"GLB file unexpectedly small: {size} bytes"

    def test_glb_structure_and_abiotic_compliance(self):
        """Asserts glTF 2.0 format, 0 animations, no Draco compression, and >= 8 meshes."""
        with open(GLB_PATH, "rb") as f:
            header = f.read(12)
            magic, version, length = struct.unpack("<4sII", header)
            assert magic == b"glTF", "Must be valid glTF binary"
            assert version == 2, "Must be glTF 2.0"

            chunk_header = f.read(8)
            chunk_len, chunk_type = struct.unpack("<I4s", chunk_header)
            assert chunk_type == b"JSON"
            gltf_json = json.loads(f.read(chunk_len).decode("utf-8"))

        meshes = gltf_json.get("meshes", [])
        assert len(meshes) >= 8, f"Expected at least 8 meshes, found {len(meshes)}"

        # 0 animations: 100% abiotic geosphere
        animations = gltf_json.get("animations", [])
        assert len(animations) == 0, f"Production abiotic GLB must have 0 animations, found {len(animations)}"

        # Draco must not be used (allows instant offline web viewing without WASM decoders)
        extensions = gltf_json.get("extensionsUsed", [])
        assert "KHR_draco_mesh_compression" not in extensions, "Draco compression must be disabled"


# =============================================================================
# 5. Visual Acceptance Renders Verification
# =============================================================================

class TestVisualAcceptanceRenders:
    """Verifies all 4 official visual acceptance renders exist, have correct resolution and illumination."""

    TARGET_RENDERS = [
        "render_iso_overview.png",
        "render_ne_angle.png",
        "render_top_down.png",
        "render_closeup_water_cavern.png",
    ]

    @pytest.mark.parametrize("render_name", TARGET_RENDERS)
    def test_render_image_exists_and_valid(self, render_name: str):
        img_path = BLENDER_MAP_DIR / render_name
        assert img_path.is_file(), f"Missing visual acceptance render: {render_name}"
        
        # Check size > 50 KB
        assert img_path.stat().st_size > 50_000, f"Render file too small: {img_path.stat().st_size} bytes"

        # Check image resolution and non-black pixel values
        with Image.open(img_path) as im:
            assert im.size == (1280, 720), f"Incorrect dimensions: {im.size}, expected (1280, 720)"
            assert im.format == "PNG", f"Expected PNG format, got {im.format}"
            
            extrema = im.getextrema()
            # Check maximum luminance across RGB channels to ensure not pitch-black
            max_lum = max(ch[1] for ch in extrema[:3])
            assert max_lum > 50, f"Render {render_name} is pitch black (max_lum={max_lum})"


# =============================================================================
# 6. 100% Offline 3D Viewer Verification
# =============================================================================

class TestOfflineViewerCompliance:
    """Verifies viewer.html is 100% functional offline with zero remote network calls."""

    def test_vendor_scripts_exist(self):
        three_js = BLENDER_MAP_DIR / "vendor" / "three.min.js"
        gltf_loader = BLENDER_MAP_DIR / "vendor" / "GLTFLoader.js"
        assert three_js.is_file(), f"Missing local vendor script: {three_js}"
        assert gltf_loader.is_file(), f"Missing local vendor script: {gltf_loader}"
        assert three_js.stat().st_size > 100_000
        assert gltf_loader.stat().st_size > 20_000

    def test_viewer_has_zero_remote_urls(self):
        assert VIEWER_HTML_PATH.is_file(), f"Missing viewer.html: {VIEWER_HTML_PATH}"
        content = VIEWER_HTML_PATH.read_text(encoding="utf-8")
        assert "http://" not in content, "viewer.html contains unsecure http:// link"
        assert "https://" not in content, "viewer.html contains remote https:// link"

    def test_viewer_references_local_vendor_scripts(self):
        content = VIEWER_HTML_PATH.read_text(encoding="utf-8")
        assert "vendor/three.min.js" in content
        assert "vendor/GLTFLoader.js" in content


# =============================================================================
# 7. Pure Abiotic Deliverables Directory Verification
# =============================================================================

class TestPureAbioticDeliverablesDirectoryCompliance:
    """Verifies assets/blender_map/ strictly complies with the Pure Abiotic World Foundation (0% flora, 0% fauna)."""

    FORBIDDEN_FILES = [
        "vegetation_instances.json",
        "vegetation_instances.anmi",
    ]

    FORBIDDEN_KEYWORDS = [
        "vegetation",
        "flora",
        "fauna",
        "tree",
        "pine",
        "grass",
        "shrub",
        "bush",
        "reed",
        "flower",
        "creature",
        "animal",
        "deer",
        "village",
        "settlement",
    ]

    CANONICAL_DELIVERABLES = {
        "ecosystem_map.glb",
        "ecosystem_map.blend",
        "viewer.html",
        "simulation_heightmap.json",
        "render_iso_overview.png",
        "render_ne_angle.png",
        "render_top_down.png",
        "render_closeup_water_cavern.png",
    }

    EXPECTED_ABIOTIC_MESHES = {
        "Diorama_Island_Mesh",
        "Water_Lake_Central_Mesh",
        "Water_Bay_Marine_Mesh",
        "Water_River_Meander_Mesh",
        "Cave_Cavern_Chamber_Mesh",
        "Cave_Arch_Entrance_Mesh",
        "Cave_Speleothems_Mesh",
        "Water_Cave_Pool_Mesh",
        "Cave_Mineral_Clusters_Mesh",
    }

    def test_residual_flora_files_completely_absent(self):
        """Asserts vegetation_instances.json and vegetation_instances.anmi are completely deleted/absent."""
        for filename in self.FORBIDDEN_FILES:
            target_path = BLENDER_MAP_DIR / filename
            assert not target_path.exists(), (
                f"Residual flora file '{filename}' exists in deliverables directory {BLENDER_MAP_DIR}. "
                f"Must be purged to satisfy R5 Pure Abiotic requirement."
            )

    def test_no_biological_filenames_in_blender_map(self):
        """Asserts no file in assets/blender_map/ contains flora, fauna, or biological keywords."""
        assert BLENDER_MAP_DIR.is_dir(), f"Missing blender_map directory: {BLENDER_MAP_DIR}"
        
        violating_files = []
        for p in BLENDER_MAP_DIR.iterdir():
            if p.is_dir() and p.name in ("vendor", "renders"):
                continue
            name_lower = p.name.lower()
            for kw in self.FORBIDDEN_KEYWORDS:
                if kw in name_lower:
                    violating_files.append((p.name, kw))
                    break

        assert not violating_files, (
            f"Found {len(violating_files)} files in assets/blender_map/ violating pure abiotic naming: {violating_files}"
        )

    def test_blender_map_contains_only_whitelisted_deliverables(self):
        """Asserts assets/blender_map/ contains only genuine deliverables and vendor directory."""
        actual_files = {p.name for p in BLENDER_MAP_DIR.iterdir() if p.is_file()}
        
        missing = self.CANONICAL_DELIVERABLES - actual_files
        assert not missing, f"Missing expected deliverables in assets/blender_map: {missing}"

        unexpected = actual_files - self.CANONICAL_DELIVERABLES
        assert not unexpected, (
            f"Unexpected/residual files found in assets/blender_map: {unexpected}. "
            f"All obsolete scripts and uncleaned residues must be purged or archived."
        )

        vendor_dir = BLENDER_MAP_DIR / "vendor"
        assert vendor_dir.is_dir(), "Missing vendor directory in assets/blender_map"
        vendor_files = {p.name for p in vendor_dir.iterdir() if p.is_file()}
        assert "three.min.js" in vendor_files
        assert "GLTFLoader.js" in vendor_files

    def test_glb_meshes_strictly_abiotic_names(self):
        """Asserts all meshes in GLB match the 27 abiotic meshes (9 base + 18 vault crags/pebbles) and 0 skeletal skins."""
        with open(GLB_PATH, "rb") as f:
            f.seek(12)
            chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
            gltf_json = json.loads(f.read(chunk_len).decode("utf-8"))

        mesh_names = {m.get("name", "") for m in gltf_json.get("meshes", [])}
        assert len(mesh_names) == 27, f"Expected exactly 27 pure abiotic meshes, found {len(mesh_names)}"
        assert self.EXPECTED_ABIOTIC_MESHES.issubset(mesh_names), (
            f"Missing base abiotic meshes in GLB: {self.EXPECTED_ABIOTIC_MESHES - mesh_names}"
        )

        vault_meshes = mesh_names - self.EXPECTED_ABIOTIC_MESHES
        assert len(vault_meshes) == 18, f"Expected 18 abiotic vault meshes, found {len(vault_meshes)}"
        for name in vault_meshes:
            assert name.startswith("Mesh_Abiotic_"), f"Unexpected non-abiotic vault mesh name: {name}"

        # Strictly enforce pure abiotic invariant: 0% flora, 0% fauna, 0% architecture
        forbidden_tokens = self.FORBIDDEN_KEYWORDS + [
            "flora", "fauna", "shrub", "moss", "leaf", "leaves", "ibex", "goat", "stag",
            "eagle", "fish", "frog", "bat", "bird", "ferret", "skink", "hare", "croc",
            "tarantula", "hunter", "sentinel", "architecture", "building", "house"
        ]
        for name in mesh_names:
            name_lower = name.lower()
            for token in forbidden_tokens:
                assert token not in name_lower, f"Biological/architectural token '{token}' found in mesh name '{name}'"

        assert len(gltf_json.get("skins", [])) == 0, "Abiotic GLB must have 0 skeletal skins"
