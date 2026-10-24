"""
Milestone M5: Anima-Engine Parity & 60 FPS WebGL Deliverables Production Test Suite.

Verifies Features 18 to 23:
1. Feature 18: Binary WorldArtifact v2 (world_256.anmw) with 36-byte header,
   FNV-1a 32-bit checksum (0x861B9B50), 5 contiguous layers, exact 1,114,148 bytes.
2. Feature 19: Map Manifest Schema (assets/map_manifest.json) compliant with
   Anima-Engine Draft-07 schema, with modelAsset, worldArtifact, pure abiotic biomes,
   and safe spawnPoint.
3. Feature 20: Dual Scene Export (ecosystem_map.blend & ecosystem_map.glb),
   save_version = 0 (no .blend1 files), GLB size <= 10 MB, tris <= 300k, 100% abiotic.
4. Feature 21: NavMesh BFS Reachability >= 80.0% (target 100%), safe spawn definition.
5. Feature 22: 60 FPS WebGL Offline Viewer (assets/blender_map/viewer.html) with zero
   external CDN calls, local vendor scripts, OrbitControls, sun lighting, and 60 FPS stats.
6. Feature 23: 4 Visual Acceptance Renders (1280x720) in assets/blender_map/renders/
   (overview.png, northeast_angle.png, topdown.png, closeup_water_cavern.png).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Dict, List

import jsonschema
import numpy as np
from PIL import Image
import pytest

from terra_forge.core.manifest import (
    CANONICAL_VIEW_IDS,
    CANONICAL_VIEW_LABELS,
    generate_map_manifest,
    load_map_manifest,
    save_map_manifest,
    validate_map_manifest,
)
from terra_forge.core.world_artifact import (
    CANONICAL_GRID_DIM,
    CANONICAL_TOTAL_SIZE_256,
    CANONICAL_WORLD_SCALE,
    HEADER_STRUCT,
    WORLD_ARTIFACT_HEADER_LEN,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WorldArtifact,
    fnv1a_32,
)
from terra_forge.navigation.navmesh import (
    NavMeshReachabilityValidator,
    ObstacleGridSynchronizer,
)


GENESIS_ZERO_ROOT = Path(r"e:/Project/01_AI_Agents/Genesis_Zero")
ASSETS_DIR = GENESIS_ZERO_ROOT / "assets"
ANMW_PATH = ASSETS_DIR / "world_256.anmw"
MANIFEST_PATH = ASSETS_DIR / "map_manifest.json"
BLENDER_MAP_DIR = ASSETS_DIR / "blender_map"
GLB_PATH = BLENDER_MAP_DIR / "ecosystem_map.glb"
BLEND_PATH = BLENDER_MAP_DIR / "ecosystem_map.blend"
BLEND1_PATH = BLENDER_MAP_DIR / "ecosystem_map.blend1"
VIEWER_HTML_PATH = BLENDER_MAP_DIR / "viewer.html"
RENDERS_DIR = BLENDER_MAP_DIR / "renders"
VENDOR_DIR = BLENDER_MAP_DIR / "vendor"
SCHEMA_PATH = Path(r"e:/Project/03_Engines_Simulation/Anima-Engine/map_manifest.schema.json")


# =============================================================================
# Feature 18: Binary WorldArtifact v2 (world_256.anmw)
# =============================================================================

class TestFeature18WorldArtifactV2:
    """Verifies WorldArtifact v2 binary layout, FNV-1a checksum, and 5-layer payload."""

    def test_anmw_file_exists_and_exact_size(self):
        assert ANMW_PATH.is_file(), f"Missing production binary: {ANMW_PATH}"
        assert ANMW_PATH.stat().st_size == CANONICAL_TOTAL_SIZE_256, (
            f"Expected {CANONICAL_TOTAL_SIZE_256} bytes, got {ANMW_PATH.stat().st_size}"
        )

    def test_36_byte_header_fields(self):
        data = ANMW_PATH.read_bytes()
        header_bytes = data[:WORLD_ARTIFACT_HEADER_LEN]
        assert len(header_bytes) == 36, f"Expected 36-byte header, got {len(header_bytes)}"

        (
            magic,
            version,
            width,
            height,
            sea_level,
            seed,
            gen_ver,
            world_scale,
            checksum,
        ) = HEADER_STRUCT.unpack(header_bytes)

        assert magic == WORLD_ARTIFACT_MAGIC, f"Expected {WORLD_ARTIFACT_MAGIC}, got {magic}"
        assert version == WORLD_ARTIFACT_VERSION
        assert width == 256
        assert height == 256
        assert pytest.approx(sea_level, abs=1e-5) == 0.0
        assert seed == 1337
        assert gen_ver == 2
        assert pytest.approx(world_scale, abs=1e-3) == 200.0

        # FNV-1a checksum computed over layer payload (bytes 36 to EOF)
        payload = data[WORLD_ARTIFACT_HEADER_LEN:]
        actual_checksum = fnv1a_32(payload)
        assert checksum == actual_checksum, (
            f"FNV-1a checksum mismatch: header={checksum:#010x}, payload={actual_checksum:#010x}"
        )
        assert checksum == 0x861B9B50, f"Expected canonical 0x861B9B50, got {checksum:#010x}"

    def test_5_layers_payload_integrity(self):
        art = WorldArtifact.from_file(ANMW_PATH)
        assert art.width == 256
        assert art.height == 256

        # 5 layers: elevation, moisture, temperature, flow, biome
        assert art.elevation.shape == (256 * 256,)
        assert art.moisture.shape == (256 * 256,)
        assert art.temperature.shape == (256 * 256,)
        assert art.flow.shape == (256 * 256,)
        assert art.biome.shape == (256 * 256,)

        assert art.elevation.dtype == np.float32
        assert art.moisture.dtype == np.float32
        assert art.temperature.dtype == np.float32
        assert art.flow.dtype == np.float32
        assert art.biome.dtype == np.uint8

        # Numerical bounds
        assert float(np.min(art.elevation)) >= 0.0
        assert float(np.max(art.elevation)) <= 10.0
        assert float(np.min(art.moisture)) >= 0.0 and float(np.max(art.moisture)) <= 1.0
        assert float(np.min(art.temperature)) >= 0.0 and float(np.max(art.temperature)) <= 1.0
        assert float(np.min(art.flow)) >= 0.0
        assert np.all(art.biome >= 0) and np.all(art.biome <= 21)

    def test_roundtrip_serialization_deserialization(self):
        art = WorldArtifact.from_file(ANMW_PATH)
        serialized_bytes = art.to_bytes()
        assert len(serialized_bytes) == CANONICAL_TOTAL_SIZE_256

        art2 = WorldArtifact.from_bytes(serialized_bytes)
        assert art2.width == art.width
        assert art2.height == art.height
        assert np.array_equal(art2.elevation, art.elevation)
        assert np.array_equal(art2.biome, art.biome)


# =============================================================================
# Feature 19: Map Manifest Schema (map_manifest.json)
# =============================================================================

class TestFeature19MapManifestSchema:
    """Verifies map manifest Draft-07 schema compliance and required metadata fields."""

    def test_manifest_file_exists_and_loads(self):
        assert MANIFEST_PATH.is_file(), f"Missing map manifest: {MANIFEST_PATH}"
        manifest = load_map_manifest(MANIFEST_PATH)
        assert manifest["schemaVersion"] == 1

    def test_draft07_jsonschema_compliance(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert SCHEMA_PATH.is_file(), f"Missing schema file: {SCHEMA_PATH}"
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(manifest))
        assert len(errors) == 0, f"Draft-07 schema validation violations: {[e.message for e in errors]}"

    def test_internal_validator_passes(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is True, f"Internal validation failed: {errors}"

    def test_model_asset_block(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert "modelAsset" in manifest
        ma = manifest["modelAsset"]
        assert ma["path"] == "assets/blender_map/ecosystem_map.glb"
        assert ma["format"] == "gltf-binary"
        assert ma["bytes"] == GLB_PATH.stat().st_size
        actual_sha = f"sha256:{hashlib.sha256(GLB_PATH.read_bytes()).hexdigest()}"
        assert ma["checksum"] == actual_sha

    def test_world_artifact_block(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert "worldArtifact" in manifest
        wa = manifest["worldArtifact"]
        assert wa["path"] == "assets/world_256.anmw"
        assert wa["magic"] == "ANMW"
        assert wa["version"] == 2
        assert wa["bytes"] == ANMW_PATH.stat().st_size
        actual_sha = f"sha256:{hashlib.sha256(ANMW_PATH.read_bytes()).hexdigest()}"
        assert wa["checksum"] == actual_sha

    def test_abiotic_biomes_and_spawn_point(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert "mapId" in manifest
        assert manifest["mapId"] == "genesis_primordial_abiotic"
        assert "version" in manifest
        assert manifest["version"] == "2.0.0"

        # Pure abiotic profile
        assert "biomes" in manifest
        abiotic_biomes = manifest["biomes"]
        for b in abiotic_biomes:
            assert b.lower() not in ["forest", "jungle", "swamp", "mangrove"], (
                f"Biotic biome found in pure abiotic manifest: {b}"
            )

        # Safe spawn point
        assert "spawnPoint" in manifest
        sp = manifest["spawnPoint"]
        assert len(sp) == 3
        # Ensure spawn coordinates are within world bounds [-100, 100] and elevation [0, 10]
        assert -100 <= sp[0] <= 100
        assert 0 <= sp[1] <= 10
        assert -100 <= sp[2] <= 100


# =============================================================================
# Feature 20: Dual Scene Export (ecosystem_map.blend & ecosystem_map.glb)
# =============================================================================

class TestFeature20DualSceneExport:
    """Verifies master .blend and .glb files, save_version=0, and WebGL budgets."""

    def test_blend_file_exists_and_no_blend1_backups(self):
        assert BLEND_PATH.is_file(), f"Missing master .blend file: {BLEND_PATH}"
        assert BLEND_PATH.stat().st_size > 1_000_000, "Master .blend file is suspiciously small"
        # Verify strict absence of .blend1 auto-backup
        assert not BLEND1_PATH.exists(), (
            f"Violation: Unwanted .blend1 backup file detected at {BLEND1_PATH}. "
            "bpy.context.preferences.filepaths.save_version must be 0."
        )

    def test_glb_file_size_budget(self):
        assert GLB_PATH.is_file(), f"Missing GLB file: {GLB_PATH}"
        glb_size = GLB_PATH.stat().st_size
        max_budget = 10 * 1024 * 1024  # 10 MB
        assert glb_size <= max_budget, f"GLB file size {glb_size} exceeds 10 MB budget"
        # Target is ~8.5-9 MB
        assert glb_size >= 5 * 1024 * 1024, f"GLB file size {glb_size} is suspiciously small"

    def test_glb_triangle_budget_and_abiotic_purity(self):
        with open(GLB_PATH, "rb") as f:
            magic, version, length = struct.unpack("<4sII", f.read(12))
            assert magic == b"glTF", "File is not a valid glTF binary"
            assert version == 2
            chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
            json_bytes = f.read(chunk_len)
            gltf = json.loads(json_bytes)

        # Calculate triangle count
        total_tris = 0
        for m in gltf.get("meshes", []):
            for prim in m.get("primitives", []):
                idx = prim.get("indices")
                if idx is not None:
                    total_tris += gltf["accessors"][idx]["count"] // 3

        assert total_tris <= 300_000, f"Triangle count {total_tris} exceeds 300k budget"
        assert total_tris >= 150_000, f"Triangle count {total_tris} is lower than expected detail"

        # Verify strict 100% abiotic purity across nodes and meshes
        forbidden_biotic_keywords = [
            "tree", "flora", "plant", "pine", "grass", "flower", "bush",
            "animal", "fauna", "creature", "bird", "fish", "deer", "lizard",
            "building", "house", "ruin", "temple", "road", "bridge",
        ]
        for node in gltf.get("nodes", []):
            name = node.get("name", "").lower()
            for kw in forbidden_biotic_keywords:
                assert kw not in name, f"Biotic element '{kw}' found in GLTF node '{name}'"

        for mesh in gltf.get("meshes", []):
            name = mesh.get("name", "").lower()
            for kw in forbidden_biotic_keywords:
                assert kw not in name, f"Biotic element '{kw}' found in GLTF mesh '{name}'"


# =============================================================================
# Feature 21: NavMesh BFS Reachability (>= 80.0%)
# =============================================================================

class TestFeature21NavMeshBFSReachability:
    """Verifies NavMesh BFS reachability over walkable terrain and safe spawn."""

    def test_reachability_coverage_exceeds_80_percent(self):
        art = WorldArtifact.from_file(ANMW_PATH)
        validator = NavMeshReachabilityValidator(
            elevation=art.elevation,
            sea_level=art.sea_level,
            flow=art.flow,
            safe_spawn=True,
        )
        sync = ObstacleGridSynchronizer()
        res = validator.evaluate(sync)

        assert res.is_valid is True, f"NavMesh reachability failed validation: {res}"
        assert res.navmesh_coverage >= 0.80, (
            f"Coverage {res.navmesh_coverage:.3f} is below 80.0% invariant"
        )
        # In fact on the primordial island terrain, reachability is 100%
        assert res.navmesh_coverage >= 0.95
        assert res.total_land > 50_000
        assert res.reach_count >= int(0.80 * res.total_land)

    def test_safe_spawn_point_coordinates(self):
        art = WorldArtifact.from_file(ANMW_PATH)
        validator = NavMeshReachabilityValidator(
            elevation=art.elevation,
            sea_level=art.sea_level,
            flow=art.flow,
            safe_spawn=True,
        )
        sync = ObstacleGridSynchronizer()
        res = validator.evaluate(sync)

        spawn_pos = res.spawn_world_pos
        assert len(spawn_pos) == 3
        # Check spawn position is walkable
        r, c = res.spawn_cell
        assert bool(res.walkable_mask[r, c]) is True
        assert bool(res.reached_mask[r, c]) is True


# =============================================================================
# Feature 22: 60 FPS WebGL Offline Viewer (viewer.html)
# =============================================================================

class TestFeature22WebGLViewer:
    """Verifies offline WebGL viewer integrity, zero CDN calls, and performance stats."""

    def test_viewer_html_exists_and_zero_network_calls(self):
        assert VIEWER_HTML_PATH.is_file(), f"Missing viewer.html: {VIEWER_HTML_PATH}"
        content = VIEWER_HTML_PATH.read_text(encoding="utf-8")

        # Zero external CDN / HTTP calls
        assert "http://" not in content, "External HTTP call detected in viewer.html"
        assert "https://" not in content, "External HTTPS call detected in viewer.html"
        assert "cdn." not in content, "CDN domain detected in viewer.html"
        assert "unpkg.com" not in content, "unpkg.com detected in viewer.html"
        assert "cdnjs.cloudflare.com" not in content, "cdnjs detected in viewer.html"

    def test_vendor_scripts_exist_locally(self):
        three_js = VENDOR_DIR / "three.min.js"
        gltf_loader = VENDOR_DIR / "GLTFLoader.js"
        assert three_js.is_file(), f"Missing local Three.js script: {three_js}"
        assert gltf_loader.is_file(), f"Missing local GLTFLoader script: {gltf_loader}"
        assert three_js.stat().st_size > 100_000
        assert gltf_loader.stat().st_size > 20_000

    def test_viewer_includes_controls_lighting_and_stats(self):
        content = VIEWER_HTML_PATH.read_text(encoding="utf-8")
        # Spherical orbit controls
        assert "updateCameraFromSpherical" in content or "OrbitControls" in content
        # Directional sun lighting
        assert "DirectionalLight" in content
        # Performance stats counter for 60 FPS
        assert "fps-val" in content or "stats-badge" in content
        assert "60" in content
        # Camera presets
        assert "setCamera" in content
        assert "iso" in content
        assert "ne" in content
        assert "topdown" in content
        assert "water_cavern" in content


# =============================================================================
# Feature 23: 4 Visual Acceptance Renders (1280x720)
# =============================================================================

class TestFeature23VisualAcceptanceRenders:
    """Verifies the 4 production acceptance images in assets/blender_map/renders/."""

    TARGET_RENDERS = [
        "overview.png",
        "northeast_angle.png",
        "topdown.png",
        "closeup_water_cavern.png",
    ]

    def test_renders_directory_exists(self):
        assert RENDERS_DIR.is_dir(), f"Missing renders directory: {RENDERS_DIR}"

    @pytest.mark.parametrize("render_name", TARGET_RENDERS)
    def test_render_dimensions_and_file_size(self, render_name: str):
        img_path = RENDERS_DIR / render_name
        assert img_path.is_file(), f"Missing visual acceptance render: {render_name}"

        # File size must be substantial (> 500 KB)
        file_size = img_path.stat().st_size
        assert file_size >= 500_000, (
            f"{render_name} size {file_size} is under 500 KB threshold"
        )

        with Image.open(img_path) as im:
            assert im.size == (1280, 720), (
                f"{render_name} dimensions {im.size} do not match (1280, 720)"
            )
            assert im.mode in ["RGB", "RGBA"]

    @pytest.mark.parametrize("render_name", TARGET_RENDERS)
    def test_render_luminance_not_pitch_black(self, render_name: str):
        img_path = RENDERS_DIR / render_name
        with Image.open(img_path) as im:
            rgb_im = im.convert("RGB")
            arr = np.array(rgb_im)
            mean_lum = float(np.mean(arr))
            max_lum = float(np.max(arr))

            assert max_lum > 150, f"Render {render_name} is too dim (max_lum={max_lum})"
            assert mean_lum > 25.0, f"Render {render_name} has insufficient illumination (mean_lum={mean_lum})"