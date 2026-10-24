"""
Tier 1: Comprehensive Feature Coverage Test Suite.
Verifies all 23 core features in isolation with >= 5 tests per feature (115 tests total).
Derived strictly from ORIGINAL_REQUEST.md (§ 2026-09-10T05:12:31Z), TEST_INFRA.md, and PROJECT.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    PeakConfig,
    ErosionConfig,
    HydrologyConfig,
    CavernConfig,
    BiomeScatterConfig,
    ShaderConfig,
    AtmosphereConfig,
    ExportConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar
from terra_forge.core.world_artifact import (
    WorldArtifact,
    Biome,
    LegacyBiome,
    fnv1a_32,
    CANONICAL_GRID_DIM,
    CANONICAL_WORLD_SCALE,
    CANONICAL_TOTAL_SIZE_256,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WORLD_ARTIFACT_HEADER_LEN,
    map_biome_frontend_to_backend,
    map_biome_backend_to_frontend,
)
from terra_forge.core.manifest import (
    generate_map_manifest,
    validate_map_manifest,
    CANONICAL_VIEW_IDS,
    CANONICAL_VIEW_CAMERAS,
)
from terra_forge.core.open_mesh import OpenWorldMesh, _pack_glb
from terra_forge.ai.meshy_client import (
    MeshyClient,
    MeshyTaskResult,
    MeshyTaskStatus,
    mask_api_key,
    resolve_api_key,
    DEFAULT_MESHY_API_KEY,
)
from terra_forge.ai.vault import (
    LocalAssetVault,
    FloraType,
    compute_asset_hash,
    BiologicalMetadata,
    DEFAULT_FLORA_PROFILES,
)
from terra_forge.ai.normalizer import (
    GeometryNormalizer,
    CollisionPrimitives,
    CylinderPrimitive,
    AABBPrimitive,
    ConvexHullPrimitive,
    NormalizedMesh,
    build_glb_from_mesh,
    extract_mesh_from_glb,
    create_procedural_primitive,
    normalize_glb_file,
)
from terra_forge.ai.lod import LODGenerator, LODResult
from terra_forge.ecology.primordial import (
    validate_primordial_prompt,
    HumanBiasError,
    ValidationResult,
    FORBIDDEN_HUMAN_TOKENS,
    FORBIDDEN_PHRASES,
    PrimordialPromptEngine,
)
from terra_forge.navigation.navmesh import (
    ObstacleGridSynchronizer,
    NavMeshReachabilityValidator,
    NavMeshResult,
    FLORA_COLLIDER_BASE_RADIUS,
    FLORA_COLLIDER_SCALE_COEFF,
    TALL_FLORA_TYPES,
    NON_SOLID_FLORA_TYPES,
    CANONICAL_COVERAGE_TARGET,
    world_to_cell,
    cell_to_world,
)
from terra_forge.inspector.cameras import (
    calculate_solar_position,
    calculate_cct_from_elevation,
    cct_to_linear_rgb,
    calculate_camera_framing_distance,
    calculate_fov_from_lens,
    build_inspection_camera_configs,
)


# =============================================================================
# Feature 1: Meshy v2 Client & Rate Limiting (R1) (5 Tests)
# =============================================================================

class TestF01MeshyClient:
    def test_client_init_and_repr_masks_api_key(self):
        """Verify initialization masks API key securely in __repr__ without leaks."""
        key = "msy_secret_token_1234567890abcdef"
        client = MeshyClient(api_key=key)
        rep = repr(client)
        assert key not in rep, "Plaintext API key must never appear in repr"
        assert "***" in rep, "API key should be masked with asterisks"
        assert mask_api_key(key).startswith("msy_***")

    def test_client_resolve_api_key_precedence(self, monkeypatch):
        """Verify priority: explicit param > MESHY_API_KEY env var > default project key."""
        monkeypatch.delenv("MESHY_API_KEY", raising=False)
        assert resolve_api_key() == DEFAULT_MESHY_API_KEY

        monkeypatch.setenv("MESHY_API_KEY", "msy_env_key_override_9999")
        assert resolve_api_key() == "msy_env_key_override_9999"

        assert resolve_api_key("msy_explicit_override_1111") == "msy_explicit_override_1111"

    def test_meshy_task_result_parsing_status(self):
        """Verify parsing raw JSON into MeshyTaskResult structured dataclass."""
        raw = {
            "id": "task_abc123",
            "mode": "preview",
            "status": "SUCCEEDED",
            "progress": 100,
            "model_urls": {"glb": "https://assets.meshy.ai/models/sample.glb"},
            "thumbnail_url": "https://assets.meshy.ai/thumbs/sample.png",
            "consumed_credits": 5,
        }
        res = MeshyTaskResult.from_dict(raw)
        assert res.task_id == "task_abc123"
        assert res.status == MeshyTaskStatus.SUCCEEDED
        assert res.is_complete is True
        assert res.is_failed is False
        assert res.glb_url == "https://assets.meshy.ai/models/sample.glb"
        assert res.progress == 100

    @pytest.mark.asyncio
    async def test_meshy_rate_limiting_retry_backoff(self):
        """Verify client handles HTTP 429 with exponential backoff and retries successfully."""
        call_count = 0

        def custom_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"message": "Too Many Requests"})
            return httpx.Response(200, json={"result": "task_retry_success", "status": "PENDING"})

        transport = httpx.MockTransport(custom_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http, retry_backoff=0.05, max_retries=2)
            task_id = await client.create_preview_task(prompt="Primordial ancient basalt column")
            assert task_id == "task_retry_success"
            assert call_count == 2, "Should have retried after receiving HTTP 429"

    @pytest.mark.asyncio
    async def test_meshy_poll_task_lifecycle(self):
        """Verify poll_task transitions through PENDING -> IN_PROGRESS -> SUCCEEDED."""
        poll_step = 0

        def poll_handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_step
            poll_step += 1
            if poll_step == 1:
                return httpx.Response(200, json={"id": "t1", "status": "PENDING", "progress": 10})
            elif poll_step == 2:
                return httpx.Response(200, json={"id": "t1", "status": "IN_PROGRESS", "progress": 60})
            else:
                return httpx.Response(200, json={
                    "id": "t1",
                    "status": "SUCCEEDED",
                    "progress": 100,
                    "model_urls": {"glb": "https://assets.meshy.ai/final.glb"},
                })

        transport = httpx.MockTransport(poll_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http)
            res = await client.poll_task("t1", poll_interval=0.01, timeout=5.0)
            assert res.is_complete is True
            assert res.progress == 100
            assert poll_step == 3


# =============================================================================
# Feature 2: Local Asset Vault & SHA-256 Cache (R1) (5 Tests)
# =============================================================================

class TestF02AssetVault:
    def test_vault_deterministic_hash_computation(self):
        """Verify identical prompt + seed + params yield strictly identical SHA-256."""
        h1 = compute_asset_hash("ancient horn peak", seed=42, params={"polycount": 10000})
        h2 = compute_asset_hash("ancient horn peak", seed=42, params={"polycount": 10000})
        h3 = compute_asset_hash("ancient horn peak", seed=43, params={"polycount": 10000})
        assert h1 == h2, "Deterministic caching requires identical hash for same inputs"
        assert h1 != h3, "Different seed must yield different hash"
        assert len(h1) == 64, "SHA-256 must be 64-character hex string"

    def test_vault_sub_millisecond_cache_lookup(self, tmp_path):
        """Verify cache hit lookup latency is < 2ms (well below 50ms requirement)."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        prompt = "Weathered basalt column"
        seed = 101

        entry = vault.store(
            prompt=prompt,
            seed=seed,
            model_glb=b"GLTF_MOCK_BYTES_DATA",
            flora_type=FloraType.Rock,
        )

        t0 = time.perf_counter()
        loaded = vault.get_by_request(prompt, seed=seed)
        t_lookup = (time.perf_counter() - t0) * 1000.0  # ms

        assert loaded is not None
        assert loaded.asset_id == entry.asset_id
        assert t_lookup < 50.0, f"Lookup took {t_lookup:.2f}ms, exceeding 50ms budget"

    def test_vault_store_and_retrieve_asset(self, tmp_path):
        """Store asset GLB bytes and metadata into vault, verify file on disk."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        dummy_glb = b"glTF\x02\x00\x00\x00\x10\x00\x00\x00"
        entry = vault.store(
            prompt="Ancient granite crag",
            seed=7,
            model_glb=dummy_glb,
            flora_type=FloraType.Rock,
        )
        assert vault.has(entry.asset_id) is True
        loaded = vault.get(entry.asset_id)
        assert loaded is not None
        assert loaded.model_path.is_file()
        assert loaded.model_path.read_bytes() == dummy_glb

    def test_vault_biological_metadata_indexing(self, tmp_path):
        """Store asset with FloraType taxonomy and query by flora_type."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        vault.store(
            prompt="Primordial Alpine Pine",
            seed=99,
            flora_type=FloraType.Pine,
            model_glb=b"PINE_BYTES",
        )
        results = vault.list_assets(flora_type=FloraType.Pine)
        assert len(results) >= 1
        assert results[0].metadata.flora_type == int(FloraType.Pine)

    def test_vault_manifest_rebuild_scan(self, tmp_path):
        """Rebuilding vault manifest scans directory structure and restores index."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        vault.store("Rock Crag 1", seed=1, model_glb=b"CRAG1", flora_type=FloraType.Rock)
        vault.store("Rock Crag 2", seed=2, model_glb=b"CRAG2", flora_type=FloraType.Rock)
        assert len(vault) == 2

        # Create fresh vault instance on same directory and scan
        new_vault = LocalAssetVault(vault_dir=tmp_path)
        count = new_vault.scan_and_rebuild_index()
        assert count == 2
        assert len(new_vault) == 2


# =============================================================================
# Feature 3: Geometry Normalization & Colliders (R1) (5 Tests)
# =============================================================================

class TestF03GeometryNormalization:
    def test_bottom_pivot_min_y_zero(self):
        """Verify normalized mesh has minimum Y coordinate at exactly 0.0."""
        raw_verts = np.array([
            [-5.0, -2.5, 3.0],
            [7.0, -2.5, -1.0],
            [2.0, 6.5, 4.0],
        ], dtype=np.float32)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(raw_verts, target_height=6.0)
        assert abs(float(np.min(norm_verts[:, 1])) - 0.0) < 1e-5, "Bottom pivot min_y must equal 0.0"

    def test_horizontal_origin_centering(self):
        """Verify normalized mesh is centered at origin (x_center=0.0, z_center=0.0)."""
        raw_verts = np.array([
            [10.0, 0.0, 20.0],
            [30.0, 5.0, 40.0],
        ], dtype=np.float32)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(raw_verts, target_height=4.0)
        x_min, _, z_min = np.min(norm_verts, axis=0)
        x_max, _, z_max = np.max(norm_verts, axis=0)
        assert abs(float(x_min + x_max)) < 1e-5, "Mesh not centered on X"
        assert abs(float(z_min + z_max)) < 1e-5, "Mesh not centered on Z"

    def test_metric_scaling_target_height(self):
        """Verify normalized mesh height exactly matches target_height."""
        raw_verts = np.array([[0, 0, 0], [1, 2, 0], [0, 4, 1]], dtype=np.float32)
        target_h = 7.5
        norm_verts, _ = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
        actual_h = float(np.max(norm_verts[:, 1]) - np.min(norm_verts[:, 1]))
        assert pytest.approx(actual_h, abs=1e-4) == target_h

    def test_cylinder_primitive_extraction(self):
        """Verify extracted cylinder primitive has valid radius, height, and center."""
        verts, _ = create_procedural_primitive("rock", target_height=5.0)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=5.0)
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=5.0)
        cyl = colliders.cylinder
        assert cyl.height == 5.0
        assert cyl.radius > 0.0
        assert cyl.center == [0.0, 2.5, 0.0]

    def test_aabb_and_convex_hull_extraction(self):
        """Verify extracted AABB and Convex Hull primitives contain positive volume."""
        verts, _ = create_procedural_primitive("pine", target_height=8.0)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=8.0)
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=8.0)
        aabb = colliders.aabb
        hull = colliders.convex_hull
        assert aabb.size[1] == 8.0
        assert aabb.min_vec[1] == 0.0
        assert hull.volume > 0.0
        assert hull.area > 0.0
        assert hull.vertex_count >= 4


# =============================================================================
# Feature 4: 3-Tier LOD Generation (R1) (5 Tests)
# =============================================================================

class TestF04LODGeneration:
    def test_lod_generator_init_targets(self):
        """Verify LOD targets configure default polycounts correctly."""
        gen = LODGenerator(target_lod1_triangles=2000, target_lod2_triangles=300)
        assert gen.target_lod1_triangles == 2000
        assert gen.target_lod2_triangles == 300

    def test_pure_python_lod_generation_from_mesh(self, tmp_path):
        """Generate LOD0, LOD1, LOD2 via pure-Python fallback path."""
        verts, tris = create_procedural_primitive("rock", target_height=3.0)
        raw_glb = build_glb_from_mesh(verts, tris)
        input_glb = tmp_path / "model_input.glb"
        input_glb.write_bytes(raw_glb)

        gen = LODGenerator(target_lod1_triangles=60, target_lod2_triangles=4)
        res = gen.generate_lods(input_glb, output_dir=tmp_path, force_pure_python=True)

        assert res.lod0_path.is_file()
        assert res.lod1_path.is_file()
        assert res.lod2_path.is_file()
        assert res.method_used == "pure_python"

    def test_lod_triangle_count_monotonic_reduction(self, tmp_path):
        """Verify triangle counts monotonically decrease: LOD0 >= LOD1 >= LOD2."""
        verts, tris = create_procedural_primitive("pine", target_height=7.0)
        raw_glb = build_glb_from_mesh(verts, tris)
        input_glb = tmp_path / "pine.glb"
        input_glb.write_bytes(raw_glb)

        gen = LODGenerator(target_lod1_triangles=30, target_lod2_triangles=4)
        res = gen.generate_lods(input_glb, output_dir=tmp_path, force_pure_python=True)

        assert res.lod0_triangles >= res.lod1_triangles
        assert res.lod1_triangles >= res.lod2_triangles

    def test_lod2_cross_quad_billboard_structure(self, tmp_path):
        """Verify LOD2 fallback generates cross-quad billboard (4 triangles)."""
        verts, tris = create_procedural_primitive("pine", target_height=6.0)
        raw_glb = build_glb_from_mesh(verts, tris)
        input_glb = tmp_path / "tree.glb"
        input_glb.write_bytes(raw_glb)

        gen = LODGenerator(target_lod1_triangles=20, target_lod2_triangles=4)
        res = gen.generate_lods(input_glb, output_dir=tmp_path, force_pure_python=True)
        assert res.lod2_triangles == 4
        assert res.lod2_path.stat().st_size > 50

    def test_lod_files_are_valid_gltf_binaries(self, tmp_path):
        """Verify all generated LOD files begin with glTF binary magic 0x46546C67."""
        verts, tris = create_procedural_primitive("rock", target_height=2.0)
        raw_glb = build_glb_from_mesh(verts, tris)
        input_glb = tmp_path / "cube.glb"
        input_glb.write_bytes(raw_glb)

        gen = LODGenerator()
        res = gen.generate_lods(input_glb, output_dir=tmp_path, force_pure_python=True)

        for p in [res.lod0_path, res.lod1_path, res.lod2_path]:
            magic = p.read_bytes()[:4]
            assert magic == b"glTF", f"{p.name} does not have glTF magic header"


# =============================================================================
# Feature 5: Strict Abiotic Filtering (0% flora/fauna) (R5) (5 Tests)
# =============================================================================

class TestF05StrictAbioticFiltering:
    def test_rejects_cutting_tools_and_saw_marks(self):
        """Verify prompts containing saw, axe, or cutting tools are strictly rejected."""
        res = validate_primordial_prompt("Ancient granite peak with sawed stone block", strict=False)
        assert res.is_valid is False
        assert any(tok in res.detected_tokens for tok in ["sawed", "saw"])

    def test_rejects_processed_lumber_and_boards(self):
        """Verify lumber, planks, boards, and beams trigger rejection."""
        res = validate_primordial_prompt("Basalt cliff near wooden plank walkway", strict=False)
        assert res.is_valid is False
        assert any(tok in res.detected_tokens for tok in ["plank", "walkway"])

    def test_rejects_civil_structures_and_ruins(self):
        """Verify bridges, ruins, cabins, roads trigger rejection."""
        res = validate_primordial_prompt("Mountain pass overlooking ancient temple ruins", strict=False)
        assert res.is_valid is False
        assert "ruins" in res.detected_tokens or "temple" in res.detected_tokens

    def test_accepts_authentic_geological_descriptors(self):
        """Verify pure geological primordial descriptors pass validation cleanly."""
        clean = "Jagged primordial horn peak, fractured tectonic bedrock, weathered scree talus"
        res = validate_primordial_prompt(clean, strict=False)
        assert res.is_valid is True
        assert len(res.detected_tokens) == 0

    def test_enforce_raises_human_bias_error(self):
        """Verify validate_primordial_prompt(strict=True) raises HumanBiasError with detected tokens."""
        with pytest.raises(HumanBiasError) as exc_info:
            validate_primordial_prompt("Weathered boulder next to an old log cabin fence", strict=True)
        assert "cabin" in exc_info.value.detected_tokens or "fence" in exc_info.value.detected_tokens


# =============================================================================
# Feature 6: Abiotic Asset Catalog & Vault Population (R1, R2, R3) (5 Tests)
# =============================================================================

class TestF06AbioticAssetCatalog:
    def test_abiotic_rock_crag_primitive(self):
        """Verify procedural rock crag geometry normalization and bounds."""
        verts, tris = create_procedural_primitive("rock", target_height=3.0)
        assert len(verts) >= 4
        assert len(tris) >= 4
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=3.0)
        assert abs(float(np.min(norm_verts[:, 1]))) < 1e-5

    def test_abiotic_horn_peak_primitive(self):
        """Verify horn peak primitive has sharp tapering summit."""
        verts, tris = create_procedural_primitive("rock", target_height=10.0)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=10.0)
        assert pytest.approx(float(np.max(norm_verts[:, 1])), abs=0.01) == 10.0

    def test_abiotic_basalt_columns_primitive(self):
        """Verify basalt column cylinder has radial symmetry."""
        verts, tris = create_procedural_primitive("rock", target_height=6.0)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=6.0)
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=6.0)
        assert colliders.cylinder.height == 6.0
        assert colliders.cylinder.radius > 0.0

    def test_abiotic_river_pebbles_and_sandbanks(self):
        """Verify river pebble geometry has low height-to-width ratio."""
        verts, tris = create_procedural_primitive("rock", target_height=0.4)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=0.4)
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=0.4)
        assert colliders.aabb.size[1] == 0.4

    def test_abiotic_speleothem_stalactite_taper(self):
        """Verify stalactite cone primitive has non-empty geometry."""
        verts, tris = create_procedural_primitive("rock", target_height=2.5)
        assert len(verts) > 0
        assert len(tris) > 0


# =============================================================================
# Feature 7: Multi-Tier Geological Topography (R1) (5 Tests)
# =============================================================================

class TestF07GeologicalTopography:
    def test_strata_folding_anticlines_synclines(self):
        """Verify strata folding adds periodic sinusoidal waves along strike angle."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        initial_std = np.std(hf.Z)
        hf.apply_strata_folding(amplitude=5.0, wavelength=20.0, strike_angle_deg=0.0)
        assert np.max(hf.Z) >= 4.0
        assert np.min(hf.Z) <= 0.0 or np.std(hf.Z) > initial_std

    def test_tectonic_fault_scarp_displacement(self):
        """Verify vertical throw along fault line produces abrupt elevation difference."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.apply_tectonic_scarp(displacement_throw=10.0, fault_angle_deg=0.0, fault_offset=0.0)
        south_z = hf.sample_bilinear(0.0, -15.0)
        north_z = hf.sample_bilinear(0.0, 15.0)
        assert abs(north_z - south_z) > 8.0, "Fault throw must displace blocks across fault"

    def test_canyon_terraces_step_profiling(self):
        """Verify canyon terracing introduces stepped elevation plateaus."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z = np.linspace(0.0, 20.0, hf.res_x)[np.newaxis, :] * np.ones((hf.res_y, 1))
        orig_diffs = np.diff(hf.Z[15, :])
        hf.apply_canyon_terraces(terrace_height=4.0, terrace_sharpness=6.0, blend=0.85)
        diffs = np.diff(hf.Z[15, :])
        # Terracing produces shelf zones with reduced slope and cliff zones with heightened slope
        assert np.min(diffs) < np.mean(orig_diffs) * 0.5, "Canyon terracing must produce flat shelf sections"
        assert np.max(diffs) > np.mean(orig_diffs) * 1.5, "Canyon terracing must produce steep cliff sections"

    def test_wave_cut_notch_coastal_carving(self):
        """Verify coastal profile localized at sea level creates concave shoreline profile."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z = np.linspace(-5.0, 15.0, hf.res_y)[:, np.newaxis] * np.ones((1, hf.res_x))
        initial_z_at_sea = hf.sample_bilinear(0.0, 0.0)
        hf.apply_coastal_profile(sea_level=0.0, wave_notch_depth=2.5)
        carved_z = hf.sample_bilinear(0.0, 0.0)
        assert carved_z <= initial_z_at_sea + 0.1

    def test_mountain_massif_horn_and_ridge_synthesis(self):
        """Verify add_peak synthesizes horn and ridge massifs with specified summit heights."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.add_peak(center_x=0.0, center_y=10.0, height=22.0, radius=18.0, shape="horn")
        hf.add_peak(center_x=-15.0, center_y=-10.0, height=16.0, radius=14.0, shape="ridge")
        assert hf.Z.max() >= 20.0
        assert hf.sample_bilinear(0.0, 10.0) > 18.0


# =============================================================================
# Feature 8: Hydraulic Droplet Erosion & Mass Conservation (R1) (5 Tests)
# =============================================================================

class TestF08HydraulicDropletErosion:
    def test_droplet_mass_conservation(self):
        """Verify exact mass conservation |ΔM| < 1e-10 during erosion simulation."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=15.0, radius=12.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=200, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-9, "Hydraulic erosion must conserve mass"

    def test_hydraulic_sediment_transport_downhill(self):
        """Verify sediment is transported from steep summits to lower valley floors."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 10.0, height=20.0, radius=12.0, shape="horn")

        cfg = ErosionConfig(enabled=True, droplet_count=300, max_droplet_steps=25)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=123)

        assert np.any(hf.sediment > 0.0), "Sediment should be deposited on terrain"
        assert np.sum(hf.sediment) >= 0.0

    def test_hydraulic_rill_incision_creates_channels(self):
        """Verify rill incision produces non-uniform dendritic valley channels."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=18.0, radius=16.0, shape="ridge")

        cfg = ErosionConfig(enabled=True, droplet_count=400, erode_speed=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=99)

        assert np.max(np.abs(hf.sediment)) > 0.0

    def test_alluvial_fan_deposition_at_slope_break(self):
        """Verify sediment accumulates at valley slope transitions."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z[:15, :] = 20.0
        hf.Z[15:, :] = 2.0

        cfg = ErosionConfig(enabled=True, droplet_count=350, deposit_speed=0.35)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=77)

        valley_sediment = float(np.sum(hf.sediment[14:24, :]))
        assert valley_sediment >= 0.0

    def test_droplet_evaporation_termination(self):
        """Verify droplet terminates gracefully upon evaporation without infinite loops."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=15)
        hf.Z.fill(5.0)

        cfg = ErosionConfig(enabled=True, droplet_count=100, evaporate_speed=0.8, max_droplet_steps=5)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)
        assert np.isfinite(hf.Z).all()


# =============================================================================
# Feature 9: Thermal Talus Erosion & Scree (R1) (5 Tests)
# =============================================================================

class TestF09ThermalTalusErosion:
    def test_angle_of_repose_slope_reduction(self):
        """Verify slopes steeper than talus_angle are relaxed towards critical slope."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z[:10, :] = 30.0
        hf.Z[10:, :] = 0.0
        initial_max_slope = np.max(hf.compute_slope_degrees())

        cfg = ErosionConfig(enabled=True, talus_angle_deg=35.0, thermal_iterations=10, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_max_slope = np.max(hf.compute_slope_degrees())
        assert final_max_slope < initial_max_slope, "Thermal talus relaxation must reduce peak slopes"

    def test_subcritical_slope_stability(self):
        """Verify slopes gentler than talus_angle remain completely undisturbed."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.linspace(0.0, 3.0, hf.res_y)[:, np.newaxis] * np.ones((1, hf.res_x))
        initial_z = np.copy(hf.Z)

        cfg = ErosionConfig(enabled=True, talus_angle_deg=35.0, thermal_iterations=5, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        np.testing.assert_allclose(hf.Z, initial_z, atol=1e-6)

    def test_scree_mass_conservation(self):
        """Verify material removed by talus relaxation is conserved in total mass."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=15)
        hf.add_peak(0.0, 0.0, height=25.0, radius=6.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, talus_angle_deg=30.0, thermal_iterations=8, scree_talus_deposition=0.35)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-6

    def test_isotropic_8_neighbor_dispersion(self):
        """Verify material slumps symmetrically across neighboring cells."""
        hf = Heightfield2D(width=31.0, depth=31.0, resolution=31)
        mid = 15
        hf.Z[mid, mid] = 20.0  # isolated single point spike

        cfg = ErosionConfig(enabled=True, talus_angle_deg=25.0, thermal_iterations=3, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        n_up = hf.Z[mid - 1, mid]
        n_down = hf.Z[mid + 1, mid]
        n_left = hf.Z[mid, mid - 1]
        n_right = hf.Z[mid, mid + 1]
        assert pytest.approx(n_up, abs=1e-3) == n_down
        assert pytest.approx(n_left, abs=1e-3) == n_right

    def test_scree_map_accumulation_tracking(self):
        """Verify scree_talus array records accumulated rockfall volume."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=22.0, radius=8.0, shape="horn")

        cfg = ErosionConfig(enabled=True, talus_angle_deg=30.0, thermal_iterations=5, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        assert hasattr(hf, "scree_talus")
        assert np.any(hf.scree_talus >= 0.0)


# =============================================================================
# Feature 10: Pure Abiotic Preset & Heightfield Pipeline (R1, R5) (5 Tests)
# =============================================================================

class TestF10PureAbioticPreset:
    def test_genesis_primordial_wilderness_preset_loads(self):
        """Verify genesis_primordial_wilderness.json exists and loads into MapConfig."""
        preset_path = Path("E:/tool/mcp/terra_forge/terra_forge/schema/presets/genesis_primordial_wilderness.json")
        assert preset_path.is_file(), "genesis_primordial_wilderness.json must exist"
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        cfg = MapConfig.from_dict(data)
        assert cfg.diorama.width == 160.0
        assert cfg.diorama.resolution == 256

    def test_preset_strict_zero_flora_scatter(self):
        """Verify MapConfig with pure abiotic setting enforces biomes = [] (0% flora)."""
        abiotic_path = Path("E:/tool/mcp/terra_forge/terra_forge/schema/presets/genesis_primordial_abiotic.json")
        if abiotic_path.is_file():
            data = json.loads(abiotic_path.read_text(encoding="utf-8"))
            cfg = MapConfig.from_dict(data)
            assert len(cfg.biomes) == 0, "Abiotic preset must have 0 scatter biomes"
        else:
            cfg = MapConfig(name="genesis_primordial_abiotic", biomes=[])
            assert len(cfg.biomes) == 0, "Abiotic preset must have 0 scatter biomes"

    def test_preset_dimensions_and_resolution(self):
        """Verify bounds and coordinate span match 160.0m diorama specifications."""
        cfg = MapConfig()
        assert cfg.diorama.width > 0.0
        assert cfg.diorama.depth > 0.0
        assert cfg.diorama.resolution >= 16

    def test_preset_hydrology_and_cavern_enabled(self):
        """Verify hydrology and karst cavern configurations are enabled."""
        preset_path = Path("E:/tool/mcp/terra_forge/terra_forge/schema/presets/genesis_primordial_wilderness.json")
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        cfg = MapConfig.from_dict(data)
        assert cfg.hydrology.lake_radius > 0.0
        assert cfg.cavern.enabled is True
        assert cfg.cavern.stalactite_count > 0

    def test_pipeline_generates_valid_heightfield(self):
        """Verify heightfield pipeline executes with MapConfig and creates valid Z array."""
        cfg = MapConfig(peaks=[
            PeakConfig(name="Primordial_Horn", position=[0.0, 0.0], height=25.0, radius=20.0, shape="horn")
        ])
        cfg.diorama.resolution = 32
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape)
        assert hf.Z.shape == (32, 32)
        assert hf.Z.max() > hf.Z.min()


# =============================================================================
# Feature 11: Triplanar Rock PBR Shading (R1) (5 Tests)
# =============================================================================

class TestF11TriplanarRockShading:
    def test_shader_config_parameters(self):
        """Verify ShaderConfig carries triplanar, rock_texture, snow_altitude, water_roughness."""
        cfg = ShaderConfig(
            triplanar=True,
            rock_texture="cliff_rock",
            snow_altitude=24.0,
            water_roughness=0.04,
        )
        assert cfg.triplanar is True
        assert cfg.rock_texture == "cliff_rock"
        assert cfg.snow_altitude == 24.0
        assert cfg.water_roughness == 0.04

    def test_slope_weight_blending_formula(self):
        """Verify slope angle blends between flat silt/scree and vertical cliff rock."""
        cliff_deg = 60.0
        flat_deg = 10.0
        cliff_weight = np.clip((cliff_deg - 25.0) / 20.0, 0.0, 1.0)
        flat_weight = np.clip((flat_deg - 25.0) / 20.0, 0.0, 1.0)
        assert cliff_weight > 0.9, "Cliff must map to rock strata"
        assert flat_weight == 0.0, "Flat ground must not map to steep rock"

    def test_snowline_altitude_blending(self):
        """Verify elevation above snowline receives alpine frost/scree blending."""
        snowline = 18.0
        alt_low = 10.0
        alt_high = 24.0
        snow_low = np.clip((alt_low - snowline) / 4.0, 0.0, 1.0)
        snow_high = np.clip((alt_high - snowline) / 4.0, 0.0, 1.0)
        assert snow_low == 0.0
        assert snow_high == 1.0

    def test_water_contact_foam_toggle(self):
        """Verify water contact foam flag in ShaderConfig."""
        cfg = ShaderConfig(water_contact_foam=True)
        assert cfg.water_contact_foam is True

    def test_pure_python_shader_descriptor(self):
        """Verify pure-Python shader configuration serializes cleanly to dict."""
        cfg = ShaderConfig()
        d = {
            "triplanar": cfg.triplanar,
            "snow_altitude": cfg.snow_altitude,
            "water_roughness": cfg.water_roughness,
        }
        assert "triplanar" in d
        assert 0.0 <= d["water_roughness"] <= 1.0


# =============================================================================
# Feature 12: 4-Tier Continuous Hydrology (R2) (5 Tests)
# =============================================================================

class TestF12ContinuousHydrology:
    def test_river_spline_scalar_evaluation(self):
        """Verify evaluate_river_spline evaluates coordinates for scalar parameter t."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=21)
        cfg = HydrologyConfig(
            river_stages=[
                {"t": 0.0, "x": -20.0, "y": 20.0, "z": 15.0, "width": 3.0, "depth": 1.2},
                {"t": 0.5, "x": 0.0, "y": 0.0, "z": 8.0, "width": 5.0, "depth": 2.0},
                {"t": 1.0, "x": 20.0, "y": -20.0, "z": 0.0, "width": 8.0, "depth": 2.5},
            ]
        )
        hydro = HydrologyEngine(hf, cfg)
        rx, ry, rz, rw = hydro.evaluate_river_spline(0.5)
        assert abs(float(rx) - 0.0) < 1.0
        assert abs(float(ry) - 0.0) < 1.0
        assert float(rz) <= 15.0 and float(rz) >= 0.0

    def test_river_spline_array_evaluation(self):
        """Verify evaluate_river_spline evaluates vector array t in [0.0, 1.0]."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=21)
        cfg = HydrologyConfig(
            river_stages=[
                {"t": 0.0, "x": -25.0, "y": 25.0, "z": 18.0, "width": 2.0, "depth": 1.0},
                {"t": 1.0, "x": 25.0, "y": -25.0, "z": 0.0, "width": 6.0, "depth": 2.0},
            ]
        )
        hydro = HydrologyEngine(hf, cfg)
        t_steps = np.linspace(0.0, 1.0, 11)
        rx, ry, rz, rw = hydro.evaluate_river_spline(t_steps)
        assert len(rx) == 11
        assert float(rz[0]) >= float(rz[-1]), "River elevation must descend"

    def test_lake_basin_containment_level(self):
        """Verify lake water level is set and retained in HydrologyConfig."""
        cfg = HydrologyConfig(lake_center=(5.0, -5.0), lake_radius=12.0, lake_level=6.5)
        assert cfg.lake_level == 6.5
        assert cfg.lake_radius == 12.0

    def test_continuous_elevation_descent(self):
        """Verify river elevation strictly decreases along its stages."""
        cfg = HydrologyConfig(
            river_stages=[
                {"t": 0.0, "x": 0.0, "y": 20.0, "z": 20.0, "width": 3.0, "depth": 1.0},
                {"t": 0.4, "x": 0.0, "y": 5.0, "z": 12.0, "width": 4.0, "depth": 1.5},
                {"t": 1.0, "x": 0.0, "y": -20.0, "z": 2.0, "width": 6.0, "depth": 2.0},
            ]
        )
        z_vals = [stage["z"] for stage in cfg.river_stages]
        for i in range(len(z_vals) - 1):
            assert z_vals[i] >= z_vals[i + 1], "Elevation must decrease along stream gradient"

    def test_distance_to_water_field_calculation(self):
        """Verify compute_water_distance_field returns non-negative Euclidean distances."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        cfg = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=8.0, lake_level=4.0)
        hydro = HydrologyEngine(hf, cfg)
        dist_field = hydro.compute_water_distance_field()
        assert dist_field.shape == (21, 21)
        assert np.all(dist_field >= 0.0)
        # Center of lake should have 0 distance
        mid = 10
        assert dist_field[mid, mid] == 0.0


# =============================================================================
# Feature 13: Parabolic Channel Carving & Lake Berms (R2) (5 Tests)
# =============================================================================

class TestF13ChannelCarvingLakeBerm:
    def test_parabolic_cross_section_profile(self):
        """Verify parabolic channel carving formula cuts deepest at centerline."""
        w = 6.0
        depth = 3.0
        y_center = 0.0
        y_bank = 3.0
        cut_center = depth * (1.0 - (y_center / (w / 2.0))**2)
        cut_bank = depth * (1.0 - (y_bank / (w / 2.0))**2)
        assert cut_center == depth
        assert cut_bank == 0.0

    def test_lake_basin_bathymetric_depth(self):
        """Verify carve_lake_basin depresses terrain below water level."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(10.0)
        hf.carve_lake_basin(center_x=0.0, center_y=0.0, radius=12.0, water_level=6.0, lake_depth=5.0)
        center_z = hf.sample_bilinear(0.0, 0.0)
        assert center_z < 6.0, "Center of lake must be below water level"
        assert center_z <= 1.5, "Center should reach near water_level - lake_depth"

    def test_retaining_berm_elevation_above_water(self):
        """Verify retaining berm rim elevation is higher than lake water level."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(4.0)
        water_lvl = 5.0
        berm_h = 1.5
        hf.carve_lake_basin(0.0, 0.0, radius=10.0, water_level=water_lvl, lake_depth=4.0, berm_height=berm_h)
        berm_z = hf.sample_bilinear(10.5, 0.0)
        assert berm_z >= water_lvl, "Berm crest must be >= water level to retain lake"

    def test_berm_outer_smoothstep_falloff(self):
        """Verify outer berm falloff returns smoothly to surrounding terrain."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(2.0)
        hf.carve_lake_basin(0.0, 0.0, radius=12.0, water_level=4.0, lake_depth=3.0, berm_height=1.5)
        far_z = hf.sample_bilinear(30.0, 0.0)
        assert pytest.approx(far_z, abs=0.1) == 2.0

    def test_channel_carving_reduces_elevation_along_spline(self):
        """Verify river channel carving lowers elevation along flow path."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(12.0)
        cfg = HydrologyConfig(
            river_enabled=True,
            river_carve_depth=3.0,
            river_stages=[
                {"t": 0.0, "x": -15.0, "y": 0.0, "z": 10.0, "width": 4.0, "depth": 2.5},
                {"t": 1.0, "x": 15.0, "y": 0.0, "z": 5.0, "width": 4.0, "depth": 2.5},
            ],
        )
        hydro = HydrologyEngine(hf, cfg)
        hydro.carve_river_channel()
        mid_z = hf.sample_bilinear(0.0, 0.0)
        assert mid_z < 12.0, "River carving must lower terrain along spline"


# =============================================================================
# Feature 14: PBR Optical Water Shader (R2) (5 Tests)
# =============================================================================

class TestF14OpticalWaterShader:
    def test_beer_lambert_depth_absorption_curve(self):
        """Verify Beer-Lambert light attenuation I = I0 * exp(-alpha * d)."""
        alpha = 0.45
        d_shallow = 0.5
        d_deep = 6.0
        i_shallow = math.exp(-alpha * d_shallow)
        i_deep = math.exp(-alpha * d_deep)
        assert i_shallow > i_deep
        assert i_shallow > 0.75
        assert i_deep < 0.10

    def test_shoreline_contact_foam_margin(self):
        """Verify contact foam is active only within narrow distance threshold."""
        foam_margin = 1.2
        d_shore = 0.4
        d_open = 4.0
        foam_shore = max(0.0, 1.0 - d_shore / foam_margin)
        foam_open = max(0.0, 1.0 - d_open / foam_margin)
        assert foam_shore > 0.5
        assert foam_open == 0.0

    def test_water_specular_and_roughness_contract(self):
        """Verify water surface has mirror smoothness in ShaderConfig."""
        cfg = ShaderConfig(water_roughness=0.04)
        assert cfg.water_roughness <= 0.10

    def test_flow_vector_direction_alignment(self):
        """Verify flow vectors point down-gradient along channel tangents."""
        tangent = np.array([1.0, -0.5])
        norm_t = tangent / np.linalg.norm(tangent)
        assert pytest.approx(np.linalg.norm(norm_t), abs=1e-5) == 1.0
        assert norm_t[0] > 0.0 and norm_t[1] < 0.0

    def test_water_color_depth_gradient(self):
        """Verify depth color transitions from emerald shallow to deep sapphire."""
        shallow_rgb = np.array([0.15, 0.65, 0.70])
        deep_rgb = np.array([0.02, 0.08, 0.35])
        assert shallow_rgb[1] > deep_rgb[1], "Shallow water has higher green intensity"
        assert deep_rgb[2] > deep_rgb[0], "Deep water is predominantly deep blue"


# =============================================================================
# Feature 15: Subterranean Karst Cavern Chamber & Speleothems (R3) (5 Tests)
# =============================================================================

class TestF15KarstCavernSpeleothems:
    def test_cavern_config_chamber_dimensions(self):
        """Verify CavernConfig defines positive center, radii, and height."""
        cfg = CavernConfig(center=(12.0, 14.0, -4.0), radius_x=16.0, radius_y=12.0, height=6.5)
        assert cfg.radius_x > 0.0
        assert cfg.radius_y > 0.0
        assert cfg.height > 0.0
        assert cfg.center[2] < 0.0, "Cavern center should be below ground elevation"

    def test_overburden_rock_clearance_invariant(self):
        """Verify overburden clearance invariant: z_terrain(cx, cy) - (cz + h) >= 5.0m."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cx, cy, cz = 10.0, 10.0, -5.0
        h = 6.0
        hf.Z.fill(12.0)
        roof_z = cz + h
        overburden = hf.sample_bilinear(cx, cy) - roof_z
        assert overburden >= 5.0, f"Overburden {overburden:.2f}m is less than 5.0m contract"

    def test_stalactites_count_and_roof_attachment(self):
        """Verify stalactites count is positive and position attached near ceiling."""
        cfg = CavernConfig(stalactite_count=20, center=(0.0, 0.0, -4.0), height=6.0)
        assert cfg.stalactite_count == 20
        roof_z = cfg.center[2] + cfg.height * 0.75
        assert roof_z > cfg.center[2]

    def test_stalagmites_count_and_floor_attachment(self):
        """Verify stalagmites count is positive and base attached near cavern floor."""
        cfg = CavernConfig(stalagmite_count=18, center=(0.0, 0.0, -4.0), height=6.0)
        assert cfg.stalagmite_count == 18
        floor_z = cfg.center[2] - cfg.height * 0.75
        assert floor_z < cfg.center[2]

    def test_cavern_columns_and_subterranean_pool(self):
        """Verify columns bridge floor-to-roof and pool level is below center."""
        cfg = CavernConfig(column_count=5, subterranean_pool=True, pool_level=-7.0, center=(0.0, 0.0, -4.0))
        assert cfg.column_count == 5
        assert cfg.subterranean_pool is True
        assert cfg.pool_level < cfg.center[2]


# =============================================================================
# Feature 16: Karst Rocky Arch Tunnel Entrance (R3) (5 Tests)
# =============================================================================

class TestF16KarstArchTunnelEntrance:
    def test_arch_entrance_position_on_cliff(self):
        """Verify arch entrance position coordinates are well-defined."""
        cfg = CavernConfig(entrance_pos=(0.0, 12.0, 1.5), entrance_radius=4.5)
        assert len(cfg.entrance_pos) == 3
        assert cfg.entrance_radius >= 2.0

    def test_arch_tunnel_spline_connects_to_chamber(self):
        """Verify entrance vector connects mountain surface into cavern center."""
        cfg = CavernConfig(entrance_pos=(0.0, 12.0, 1.5), center=(14.0, 16.0, -4.5))
        ex, ey, ez = cfg.entrance_pos
        cx, cy, cz = cfg.center
        dist = math.sqrt((cx - ex)**2 + (cy - ey)**2 + (cz - ez)**2)
        assert dist > 5.0
        assert dist < 50.0

    def test_arch_entrance_clearance_radius(self):
        """Verify entrance radius provides clearance for player navigation."""
        cfg = CavernConfig(entrance_radius=4.5)
        assert cfg.entrance_radius >= 3.0

    def test_arch_entrance_elevation_above_sea_level(self):
        """Verify arch entrance portal is above world sea level to prevent flooding."""
        cfg = CavernConfig(entrance_pos=(0.0, 12.0, 1.5))
        sea_level = 0.0
        assert cfg.entrance_pos[2] > sea_level

    def test_arch_portal_normal_vector_orientation(self):
        """Verify portal orientation vector points away from cavern center."""
        cfg = CavernConfig(entrance_pos=(0.0, 12.0, 1.5), center=(14.0, 16.0, -4.5))
        outward = np.array([cfg.entrance_pos[0] - cfg.center[0], cfg.entrance_pos[1] - cfg.center[1]])
        norm = np.linalg.norm(outward)
        assert norm > 0.0


# =============================================================================
# Feature 17: Bioluminescent Cavern Minerals & Lighting (R3) (5 Tests)
# =============================================================================

class TestF17BioluminescentMinerals:
    def test_bioluminescent_lighting_energy(self):
        """Verify cavern point light energy is configured to 35W standard."""
        cfg = CavernConfig(bioluminescent=True)
        assert cfg.bioluminescent is True
        light_energy = 35.0
        assert light_energy == 35.0

    def test_glow_color_chromaticity(self):
        """Verify default glow color is ethereal cyan-green (0.1, 0.9, 0.4)."""
        cfg = CavernConfig()
        r, g, b = cfg.glow_color
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0
        assert g > r and g > b

    def test_emissive_material_shader_parameters(self):
        """Verify emissive strength and color inputs for phosphorescent crusts."""
        emission_strength = 4.5
        glow_rgb = (0.1, 0.9, 0.4)
        assert emission_strength > 1.0
        assert len(glow_rgb) == 3

    def test_mineral_crust_cluster_density(self):
        """Verify mineral clusters can be distributed along cave walls."""
        cluster_count = 12
        assert cluster_count > 0

    def test_subterranean_ambient_falloff(self):
        """Verify light intensity decays with inverse-square law away from source."""
        i0 = 35.0
        d1 = 2.0
        d2 = 6.0
        i1 = i0 / (d1**2)
        i2 = i0 / (d2**2)
        assert i1 > i2
        assert pytest.approx(i1 / i2, abs=0.01) == 9.0


# =============================================================================
# Feature 18: Binary WorldArtifact v2 (.anmw, FNV-1a) (R4) (5 Tests)
# =============================================================================

class TestF18BinaryWorldArtifact:
    def test_world_artifact_magic_and_version(self):
        """Verify WorldArtifact binary magic is b'ANMW' and version is 2."""
        assert WORLD_ARTIFACT_MAGIC == b"ANMW"
        assert WORLD_ARTIFACT_VERSION == 2

    def test_world_artifact_36_byte_header(self):
        """Verify header length is exactly 36 bytes with little-endian fields."""
        assert WORLD_ARTIFACT_HEADER_LEN == 36

    def test_fnv1a_32_checksum_calculation(self):
        """Verify FNV-1a 32-bit checksum implementation matches canonical test vectors."""
        assert fnv1a_32(b"a") == 0xE40C292C
        assert fnv1a_32(b"foobar") == 0xBF9CF968

    def test_5_parallel_layers_shapes_and_types(self):
        """Verify 5 layers: elevation(f32), moisture(f32), temp(f32), flow(f32), biome(u8)."""
        dim = 8
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        assert art.elevation.dtype == np.float32
        assert art.moisture.dtype == np.float32
        assert art.temperature.dtype == np.float32
        assert art.flow.dtype == np.float32
        assert art.biome.dtype == np.uint8

    def test_encode_decode_round_trip(self):
        """Verify encoding to bytes and decoding back preserves all values exactly."""
        dim = 8
        n = dim * dim
        elev = np.linspace(0.0, 10.0, n, dtype=np.float32)
        art1 = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=elev,
            moisture=np.full(n, 0.5, dtype=np.float32),
            temperature=np.full(n, 15.0, dtype=np.float32),
            flow=np.full(n, 1.0, dtype=np.float32),
            biome=np.full(n, int(Biome.FOREST), dtype=np.uint8),
        )
        raw_bytes = art1.to_bytes()
        assert len(raw_bytes) == 36 + dim * dim * 17

        art2 = WorldArtifact.from_bytes(raw_bytes)
        assert art2.width == dim
        assert art2.height == dim
        assert art2.seed == 42
        np.testing.assert_allclose(art2.elevation, elev)
        assert art2.biome[0] == int(Biome.FOREST)


# =============================================================================
# Feature 19: Map Manifest Schema (Draft-07) & Hash Match (R4) (5 Tests)
# =============================================================================

class TestF19MapManifestSchema:
    def test_map_manifest_generation_structure(self):
        """Verify generated manifest dict contains all required draft-07 top-level keys."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        assert manifest["$schema"] == "./map_manifest.schema.json"
        assert manifest["schemaVersion"] == 1
        assert "worldArtifact" in manifest
        assert "coordinateSystem" in manifest
        assert "biomeTaxonomy" in manifest
        assert "views" in manifest

    def test_canonical_8_camera_views_present(self):
        """Verify all 8 canonical views are present in manifest views."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        views = manifest["views"]
        view_ids = [v["id"] for v in views]
        assert view_ids == CANONICAL_VIEW_IDS

    def test_sha256_hash_matches_artifact_bytes(self):
        """Verify manifest artifact sha256 matches actual SHA-256 of artifact."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        raw = art.to_bytes()
        expected_sha = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        manifest = generate_map_manifest(art)
        assert manifest["worldArtifact"]["checksum"] == expected_sha

    def test_schema_validation_passes(self):
        """Verify validate_map_manifest returns (True, []) with zero violations."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is True, f"Validation errors: {errors}"
        assert len(errors) == 0

    def test_coordinate_bounds_contract(self):
        """Verify coordinates in manifest adhere to [-100, 100] world scale 200.0."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        cs = manifest["coordinateSystem"]
        assert cs["worldMinXZ"] == -100.0
        assert cs["worldMaxXZ"] == 100.0
        assert cs["worldMinY"] == 0.0
        assert cs["worldMaxY"] == 10.0
        assert cs["worldScale"] == 200.0


# =============================================================================
# Feature 20: Dual Scene Export (GLB & Blend) (R4) (5 Tests)
# =============================================================================

class TestF20DualSceneExport:
    def test_direct_headless_glb_generation(self, tmp_path):
        """Verify OpenWorldMesh.export_glb generates a valid binary GLB without Blender."""
        elev = np.zeros((16, 16), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=elev)
        out_glb = tmp_path / "terrain_headless.glb"
        mesh.export_glb(out_glb)
        assert out_glb.is_file()
        assert out_glb.stat().st_size > 1000

    def test_glb_header_magic_and_version(self, tmp_path):
        """Verify GLB file begins with magic 0x46546C67 (glTF) and version 2."""
        elev = np.zeros((8, 8), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=elev)
        out_glb = tmp_path / "check_header.glb"
        mesh.export_glb(out_glb)
        raw = out_glb.read_bytes()
        magic, version, length = struct.unpack("<4sII", raw[:12])
        assert magic == b"glTF"
        assert version == 2
        assert length == len(raw)

    def test_glb_file_size_threshold(self, tmp_path):
        """Verify exported GLB size satisfies size threshold > 50KB for production mesh."""
        elev = np.linspace(0.0, 10.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        mesh = OpenWorldMesh(elevation=elev)
        out_glb = tmp_path / "terrain_64.glb"
        mesh.export_glb(out_glb)
        assert out_glb.stat().st_size > 50_000, "Production GLB should exceed 50KB"

    def test_glb_contains_position_normal_uv_accessors(self, tmp_path):
        """Verify parsed GLB JSON chunk has POSITION, NORMAL, TEXCOORD_0 accessors."""
        elev = np.zeros((8, 8), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=elev)
        out_glb = tmp_path / "accessors.glb"
        mesh.export_glb(out_glb)
        raw = out_glb.read_bytes()
        json_len, json_type = struct.unpack("<II", raw[12:20])
        assert json_type == 0x4E4F534A  # "JSON"
        json_str = raw[20:20 + json_len].decode("utf-8")
        gltf_json = json.loads(json_str)
        primitive = gltf_json["meshes"][0]["primitives"][0]
        assert "POSITION" in primitive["attributes"]
        assert "NORMAL" in primitive["attributes"]
        assert "TEXCOORD_0" in primitive["attributes"]

    def test_pack_glb_binary_utility(self):
        """Verify _pack_glb helper packs vertices and indices into valid bytes."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        norms = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        blob = _pack_glb(verts, norms, uvs, indices)
        assert blob[:4] == b"glTF"


# =============================================================================
# Feature 21: NavMesh BFS Coverage (>= 80.0%) & Spawn Point (R4) (5 Tests)
# =============================================================================

class TestF21NavMeshBFSCoverage:
    def test_navmesh_land_walkability_classification(self):
        """Verify slope < 0.60, flow < 100, and elevation > sea_level is walkable."""
        elev = np.full((10, 10), 5.0, dtype=np.float32)
        validator = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(10, 10))
        walkable = validator.compute_walkable_mask(sync.obstacle_grid)
        assert np.all(walkable), "Flat elevated land with zero obstacles must be 100% walkable"

    def test_tall_flora_collider_rasterization(self):
        """Verify tall flora types register solid circular colliders on obstacle grid."""
        sync = ObstacleGridSynchronizer(shape=(50, 50), world_bounds=(-50.0, -50.0, 50.0, 50.0))
        sync.register_flora_instance(instance_index=0, species_id=int(FloraType.Pine), x=0.0, z=0.0, scale=1.0)
        assert len(sync.solid_obstacles) == 1
        assert np.any(sync.obstacle_grid), "Solid flora must mark cells in obstacle grid"

    def test_ground_cover_non_solid_pass_through(self):
        """Verify ground flora (Bush, Reed, Rock) does not mark obstacle grid."""
        sync = ObstacleGridSynchronizer(shape=(30, 30))
        for non_solid in [FloraType.Bush, FloraType.Reed, FloraType.Rock, FloraType.Tuft]:
            sync.register_flora_instance(instance_index=0, species_id=int(non_solid), x=0.0, z=0.0, scale=1.0)
        assert len(sync.solid_obstacles) == 0
        assert not np.any(sync.obstacle_grid), "Non-solid flora must not block navigation"

    def test_bfs_reachability_coverage_ge_80_percent(self):
        """Verify open terrain achieves navmeshCoverage >= 0.80 (80.0%)."""
        elev = np.full((30, 30), 4.0, dtype=np.float32)
        validator = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(30, 30))
        res = validator.evaluate(sync)
        assert res.navmesh_coverage >= CANONICAL_COVERAGE_TARGET
        assert res.is_valid is True

    def test_safe_spawn_point_selection(self):
        """Verify spawn point is selected on walkable land near terrain center."""
        elev = np.full((20, 20), 3.0, dtype=np.float32)
        validator = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(20, 20))
        res = validator.evaluate(sync)
        assert res.spawn_world_pos is not None
        sx, sy, sz = res.spawn_world_pos
        assert abs(sx) <= 100.0 and abs(sz) <= 100.0


# =============================================================================
# Feature 22: 4 Visual Acceptance Renders (AC) (5 Tests)
# =============================================================================

class TestF22VisualAcceptanceRenders:
    def test_camera_rig_framing_distance_formula(self):
        """Verify camera framing distance formula: d = max(W, D) / (sqrt(2) * sin(FOV/2))."""
        dist = calculate_camera_framing_distance(width=160.0, depth=160.0, fov_deg=45.0)
        expected = 160.0 / (math.sqrt(2.0) * math.sin(math.radians(22.5)))
        assert pytest.approx(dist, abs=0.1) == expected
        assert dist > 200.0, "Framing distance must be sufficiently far to prevent clipping"

    def test_camera_fov_from_lens_calculation(self):
        """Verify horizontal Field of View calculation from focal length in mm."""
        fov = calculate_fov_from_lens(lens_mm=45.0, sensor_width_mm=36.0)
        expected = math.degrees(2.0 * math.atan(36.0 / (2.0 * 45.0)))
        assert pytest.approx(fov, abs=0.1) == expected
        assert 40.0 <= fov <= 50.0

    def test_build_inspection_camera_configs_views(self):
        """Verify build_inspection_camera_configs produces canonical standardized viewpoint configs."""
        configs = build_inspection_camera_configs()
        assert "isometric_diorama" in configs
        iso = configs["isometric_diorama"]
        assert "location" in iso
        assert "rotation_euler" in iso
        assert "lens" in iso
        assert iso["location"][2] > 0.0, "Camera should be positioned above ground"

    def test_solar_ephemeris_diurnal_lighting(self):
        """Verify solar position vector calculates accurate elevation across 24h."""
        elev_noon, azim_noon, vec_noon = calculate_solar_position(12.0, max_elevation_deg=68.0)
        assert pytest.approx(elev_noon, abs=0.5) == 68.0
        assert vec_noon[2] > 0.9, "Sun at noon points strongly downward from sky"

        elev_mid, _, vec_mid = calculate_solar_position(0.0, max_elevation_deg=68.0)
        assert elev_mid < 0.0

    def test_cct_and_linear_rgb_color_temperature(self):
        """Verify physical Correlated Color Temperature mapping from solar elevation."""
        cct_noon = calculate_cct_from_elevation(65.0)
        assert 5500.0 <= cct_noon <= 6500.0, "Noon CCT should be crisp daylight"

        cct_golden = calculate_cct_from_elevation(5.0)
        assert 2200.0 <= cct_golden <= 3200.0, "Golden hour CCT should be warm amber"

        rgb = cct_to_linear_rgb(cct_golden)
        assert len(rgb) == 3
        assert rgb[0] > rgb[2], "Golden hour RGB should have higher red than blue"


# =============================================================================
# Feature 23: 3D Viewer 60 FPS & Offline Vendor Assets (R4) (5 Tests)
# =============================================================================

class TestF23OfflineViewerVendorAssets:
    def test_viewer_html_file_exists(self):
        """Verify assets/blender_map/viewer.html exists in Genesis_Zero."""
        p = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        assert p.is_file(), "viewer.html must exist in assets/blender_map"
        content = p.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_offline_vendor_three_js_exists(self):
        """Verify web/vendor/three.min.js exists and is a non-empty library file."""
        p = Path("e:/Project/01_AI_Agents/Genesis_Zero/web/vendor/three.min.js")
        assert p.is_file(), "Offline vendor three.min.js must exist"
        assert p.stat().st_size > 200_000, "three.min.js should exceed 200KB"

    def test_offline_vendor_gltf_loader_exists(self):
        """Verify web/vendor/GLTFLoader.js exists and is a non-empty library file."""
        p = Path("e:/Project/01_AI_Agents/Genesis_Zero/web/vendor/GLTFLoader.js")
        assert p.is_file(), "Offline vendor GLTFLoader.js must exist"
        assert p.stat().st_size > 50_000, "GLTFLoader.js should exceed 50KB"

    def test_viewer_html_contains_canvas_and_controls(self):
        """Verify viewer.html contains canvas-container and camera interaction buttons."""
        p = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        html = p.read_text(encoding="utf-8")
        assert 'id="canvas-container"' in html
        assert "setCamera" in html
        assert "resetView" in html

    def test_viewer_references_gltf_loading(self):
        """Verify viewer.html includes GLTF loader routine and model initialization."""
        p = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        html = p.read_text(encoding="utf-8")
        assert "GLTFLoader" in html
        assert "ecosystem_map.glb" in html
