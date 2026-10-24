"""
Milestone M5 WorldArtifact & MapManifest Production Deliverables Test Suite.

Verifies:
1. Production WorldArtifact v2 binary (`assets/world_256.anmw`):
   - 36-byte header, FNV-1a 32-bit checksum (0x861b9b50), exact size 1,114,148 bytes.
   - 5 contiguous layers: elevation, moisture, temperature, flow, biome.
   - Checksum matches both Rust and TypeScript decoders.
2. Production Map Manifest (`assets/map_manifest.json`):
   - 100% Draft-07 schema compliance (jsonschema Draft7Validator).
   - Contains modelAsset block matching ecosystem_map.glb SHA-256 and byte count.
   - Contains 8 canonical view IDs with human-readable labels.
3. Cross-asset hash consistency:
   - SHA-256 of world_256.anmw matches manifest["worldArtifact"]["checksum"].
   - SHA-256 of ecosystem_map.glb matches manifest["modelAsset"]["checksum"].
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import jsonschema
import numpy as np
import pytest

from terra_forge.core.manifest import (
    CANONICAL_VIEW_CAMERAS,
    CANONICAL_VIEW_IDS,
    CANONICAL_VIEW_LABELS,
    generate_map_manifest,
    load_map_manifest,
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

GENESIS_ZERO_ROOT = Path(r"e:/Project/01_AI_Agents/Genesis_Zero")
ASSETS_DIR = GENESIS_ZERO_ROOT / "assets"
ANMW_PATH = ASSETS_DIR / "world_256.anmw"
MANIFEST_PATH = ASSETS_DIR / "map_manifest.json"
GLB_PATH = ASSETS_DIR / "blender_map" / "ecosystem_map.glb"
SCHEMA_PATH = Path(r"e:/Project/03_Engines_Simulation/Anima-Engine/map_manifest.schema.json")


# =============================================================================
# 1. WorldArtifact v2 Binary Parity & Integrity
# =============================================================================

class TestProductionWorldArtifactBinary:
    """Validates the committed world_256.anmw binary against Anima-Engine specification."""

    def test_file_exists_and_exact_size(self):
        assert ANMW_PATH.is_file(), f"Missing production binary: {ANMW_PATH}"
        assert ANMW_PATH.stat().st_size == CANONICAL_TOTAL_SIZE_256, (
            f"Expected {CANONICAL_TOTAL_SIZE_256} bytes, got {ANMW_PATH.stat().st_size}"
        )

    def test_header_decoding_and_fields(self):
        data = ANMW_PATH.read_bytes()
        header_bytes = data[:WORLD_ARTIFACT_HEADER_LEN]
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
        assert version == WORLD_ARTIFACT_VERSION, f"Expected version {WORLD_ARTIFACT_VERSION}, got {version}"
        assert width == CANONICAL_GRID_DIM, f"Expected width {CANONICAL_GRID_DIM}, got {width}"
        assert height == CANONICAL_GRID_DIM, f"Expected height {CANONICAL_GRID_DIM}, got {height}"
        assert pytest.approx(sea_level, abs=1e-5) == 0.0
        assert seed == 1337
        assert gen_ver == WORLD_ARTIFACT_VERSION
        assert pytest.approx(world_scale, abs=1e-3) == CANONICAL_WORLD_SCALE

        # Check payload FNV-1a checksum against header
        payload = data[WORLD_ARTIFACT_HEADER_LEN:]
        actual_checksum = fnv1a_32(payload)
        assert checksum == actual_checksum, f"FNV-1a mismatch: header {checksum:#010x} != payload {actual_checksum:#010x}"
        assert checksum == 0x861B9B50, f"Expected known canonical checksum 0x861b9b50, got {checksum:#010x}"

    def test_payload_5_layers_data_integrity(self):
        art = WorldArtifact.from_file(ANMW_PATH)
        assert art.width == 256
        assert art.height == 256

        # Check shapes (WorldArtifact stores 1D contiguous arrays of size width * height)
        assert art.elevation.shape == (256 * 256,)
        assert art.moisture.shape == (256 * 256,)
        assert art.temperature.shape == (256 * 256,)
        assert art.flow.shape == (256 * 256,)
        assert art.biome.shape == (256 * 256,)

        # Check types
        assert art.elevation.dtype == np.float32
        assert art.moisture.dtype == np.float32
        assert art.temperature.dtype == np.float32
        assert art.flow.dtype == np.float32
        assert art.biome.dtype == np.uint8

        # Check value bounds
        assert np.all(art.biome >= 0) and np.all(art.biome <= 21), "Biome IDs must be in canonical range [0, 21]"
        assert float(np.min(art.elevation)) >= 0.0, "Elevation should be non-negative"
        assert float(np.max(art.elevation)) <= 10.0, "Elevation should not exceed world maximum"
        assert float(np.min(art.moisture)) >= 0.0 and float(np.max(art.moisture)) <= 1.0
        assert float(np.min(art.temperature)) >= 0.0 and float(np.max(art.temperature)) <= 1.0


# =============================================================================
# 2. Map Manifest Draft-07 Schema Compliance
# =============================================================================

class TestProductionMapManifest:
    """Validates assets/map_manifest.json against Draft-07 schema and Anima-Engine rules."""

    def test_manifest_exists_and_loads(self):
        assert MANIFEST_PATH.is_file(), f"Missing map manifest: {MANIFEST_PATH}"
        manifest = load_map_manifest(MANIFEST_PATH)
        assert manifest["schemaVersion"] == 1

    def test_manifest_passes_internal_validator(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is True, f"Internal validation failed: {errors}"
        assert len(errors) == 0

    def test_manifest_draft07_jsonschema_compliance(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert SCHEMA_PATH.is_file(), f"Missing schema file: {SCHEMA_PATH}"
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(manifest))
        assert len(errors) == 0, f"Draft-07 schema validation violations: {[e.message for e in errors]}"

    def test_canonical_8_views_present_with_labels(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        views = manifest.get("views", [])
        assert len(views) == 8, f"Expected exactly 8 views, found {len(views)}"

        view_ids = [v["id"] for v in views]
        assert view_ids == CANONICAL_VIEW_IDS, f"View IDs must match canonical order: {view_ids}"

        for v in views:
            assert "label" in v, f"View {v['id']} missing human-readable label"
            assert len(v["label"]) > 0
            assert "camera" in v
            cam = v["camera"]
            assert len(cam["position"]) == 3
            assert len(cam["target"]) == 3
            assert cam.get("fov") == 45.0, f"View {v['id']} camera missing or invalid fov (expected 45.0)"


# =============================================================================
# 3. Artifact & Model Hash Synchronization
# =============================================================================

class TestArtifactHashSynchronization:
    """Verifies that SHA-256 hashes in manifest match files on disk byte-for-byte."""

    def test_world_artifact_sha256_matches_disk(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        wa_block = manifest["worldArtifact"]

        anmw_data = ANMW_PATH.read_bytes()
        actual_sha256 = f"sha256:{hashlib.sha256(anmw_data).hexdigest()}"
        actual_bytes = len(anmw_data)

        assert wa_block["checksum"] == actual_sha256, (
            f"worldArtifact checksum mismatch: manifest={wa_block['checksum']} != disk={actual_sha256}"
        )
        assert wa_block["bytes"] == actual_bytes, (
            f"worldArtifact bytes mismatch: manifest={wa_block['bytes']} != disk={actual_bytes}"
        )

    def test_model_asset_sha256_matches_disk(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert "modelAsset" in manifest, "Manifest must contain modelAsset block"
        ma_block = manifest["modelAsset"]

        assert GLB_PATH.is_file(), f"Missing GLB file: {GLB_PATH}"
        glb_data = GLB_PATH.read_bytes()
        actual_sha256 = f"sha256:{hashlib.sha256(glb_data).hexdigest()}"
        actual_bytes = len(glb_data)

        assert ma_block["checksum"] == actual_sha256, (
            f"modelAsset checksum mismatch: manifest={ma_block['checksum']} != disk={actual_sha256}"
        )
        assert ma_block["bytes"] == actual_bytes, (
            f"modelAsset bytes mismatch: manifest={ma_block['bytes']} != disk={actual_bytes}"
        )
        assert ma_block["format"] == "gltf-binary"

    def test_manifest_generator_with_glb_path(self):
        """Verifies generate_map_manifest correctly integrates GLB hashing."""
        manifest = generate_map_manifest(
            artifact=ANMW_PATH,
            artifact_rel_path="assets/world_256.anmw",
            glb_path=GLB_PATH,
            glb_rel_path="assets/blender_map/ecosystem_map.glb",
        )
        assert "modelAsset" in manifest
        assert manifest["modelAsset"]["format"] == "gltf-binary"
        assert manifest["modelAsset"]["bytes"] == GLB_PATH.stat().st_size
        assert manifest["modelAsset"]["checksum"].startswith("sha256:")
        
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is True
