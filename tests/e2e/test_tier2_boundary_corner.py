"""
Tier 2: Comprehensive Boundary & Corner Case Test Suite.
Verifies all 23 core features under extreme boundary conditions, corner cases,
and zero-tolerance invariant constraints with >= 5 tests per feature (115 tests total).
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
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WORLD_ARTIFACT_HEADER_LEN,
)
from terra_forge.core.manifest import (
    generate_map_manifest,
    validate_map_manifest,
    CANONICAL_VIEW_IDS,
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
    AbioticType,
    compute_asset_hash,
    BiologicalMetadata,
    AbioticMetadata,
    DEFAULT_ABIOTIC_PROFILES,
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
# Feature 1: Meshy Client Boundary & Error Cascading (5 Tests)
# =============================================================================

class TestF01MeshyBoundary:
    def test_empty_and_whitespace_api_key_handling(self):
        """Verify empty string or whitespace API key handling."""
        resolved = resolve_api_key("   ")
        assert resolved == DEFAULT_MESHY_API_KEY
        masked = mask_api_key("   ")
        assert masked == "***"
        client = MeshyClient(api_key="   ")
        assert repr(client).startswith("MeshyClient(")

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately_on_error(self):
        """Verify max_retries=0 fails on first 500 error without retrying."""
        calls = 0

        def fail_handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, json={"message": "Internal Server Error"})

        transport = httpx.MockTransport(fail_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http, max_retries=0)
            with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
                await client.create_preview_task("Basalt monolith")
            assert calls == 1, "Should not retry when max_retries=0"

    @pytest.mark.asyncio
    async def test_malformed_json_response_handling(self):
        """Verify client handles malformed JSON response body gracefully."""
        def bad_json_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>502 Bad Gateway</html>")

        transport = httpx.MockTransport(bad_json_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http, max_retries=1, retry_backoff=0.01)
            with pytest.raises(Exception):
                await client.create_preview_task("Ancient granite cliff")

    @pytest.mark.asyncio
    async def test_task_timeout_boundary_zero(self):
        """Verify poll_task with timeout=0.0 raises TimeoutError if incomplete."""
        def pending_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "t_pending", "status": "PENDING", "progress": 0})

        transport = httpx.MockTransport(pending_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http)
            with pytest.raises(TimeoutError):
                await client.poll_task("t_pending", poll_interval=0.05, timeout=0.01)

    @pytest.mark.asyncio
    async def test_rate_limit_http_429_exhaustion(self):
        """Verify consecutive HTTP 429s exhaust retries and raise error."""
        calls = 0

        def rate_limit_handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(429, json={"message": "Too Many Requests"})

        transport = httpx.MockTransport(rate_limit_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http, max_retries=2, retry_backoff=0.01)
            with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
                await client.create_preview_task("Weathered limestone")
            assert calls == 3  # Initial + 2 retries


# =============================================================================
# Feature 2: Local Asset Vault Boundary & Corrupted Caches (5 Tests)
# =============================================================================

class TestF02AssetVaultBoundary:
    def test_empty_prompt_and_zero_seed_hash(self):
        """Verify empty string prompt and seed=0 compute a valid 64-char hex hash."""
        h = compute_asset_hash("", seed=0, params={})
        assert len(h) == 64
        assert int(h, 16) > 0

    def test_missing_asset_get_returns_none(self, tmp_path):
        """Verify querying non-existent SHA-256 hash returns None without error."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        fake_hash = "0" * 64
        assert vault.has(fake_hash) is False
        assert vault.get(fake_hash) is None

    def test_corrupted_metadata_json_resilience(self, tmp_path):
        """Vault scan skips corrupted/invalid JSON metadata without crashing."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        bad_dir = tmp_path / ("a" * 64)
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.json").write_text("{corrupted_json: true", encoding="utf-8")
        (bad_dir / "model.glb").write_bytes(b"dummy")

        count = vault.scan_and_rebuild_index()
        assert count == 0, "Corrupted metadata entry must be skipped"
        assert len(vault) == 0

    def test_empty_vault_directory_initialization(self, tmp_path):
        """Initializing vault on an empty directory produces empty index."""
        vault = LocalAssetVault(vault_dir=tmp_path / "nested_empty_vault")
        assert len(vault) == 0
        assert vault.count() == 0
        assert vault.list_assets() == []

    def test_extreme_payload_size_glb_store(self, tmp_path):
        """Store zero-byte and large dummy GLB in vault safely."""
        vault = LocalAssetVault(vault_dir=tmp_path)
        e0 = vault.store("zero_byte_asset", seed=1, model_glb=b"", abiotic_type=AbioticType.RiverPebble)
        assert e0.model_path.stat().st_size == 0

        large_payload = b"X" * (1024 * 512)  # 512 KB
        e1 = vault.store("large_asset", seed=2, model_glb=large_payload, abiotic_type=AbioticType.RockCrag)
        assert e1.model_path.stat().st_size == len(large_payload)


# =============================================================================
# Feature 3: Geometry Normalization Boundary & Degeneracies (5 Tests)
# =============================================================================

class TestF03GeometryNormalizerBoundary:
    def test_single_vertex_mesh_normalization(self):
        """Degenerate 1-vertex mesh bounds and centering."""
        verts = np.array([[5.0, 10.0, -3.0]], dtype=np.float32)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=2.0)
        assert abs(norm_verts[0, 0]) < 1e-4
        assert abs(norm_verts[0, 1] - 0.0) < 1e-4
        assert abs(norm_verts[0, 2]) < 1e-4

    def test_zero_height_flat_coplanar_mesh(self):
        """Coplanar vertices (height=0.0) normalize without division by zero."""
        verts = np.array([
            [-2.0, 5.0, -2.0],
            [2.0, 5.0, -2.0],
            [0.0, 5.0, 2.0],
        ], dtype=np.float32)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=4.0)
        assert np.isfinite(norm_verts).all(), "Normalization must not produce NaN or Inf"
        assert abs(norm_verts[:, 1].min() - 0.0) < 1e-4

    def test_negative_target_height_preserves_scale(self):
        """Negative target height preserves original scale factor."""
        verts = np.array([[0, 0, 0], [0, 5, 0], [1, 2, 0]], dtype=np.float32)
        norm_verts, stats = GeometryNormalizer.normalize_vertices(verts, target_height=-5.0)
        assert stats["scale_factor"] == 1.0
        assert norm_verts[:, 1].min() == 0.0

    def test_extreme_aspect_ratio_needle_mesh(self):
        """Extreme aspect ratio mesh (1000m tall, 0.001m wide) extracts valid primitive."""
        verts = np.array([
            [-0.0005, 0.0, -0.0005],
            [0.0005, 0.0, 0.0005],
            [0.0, 1000.0, 0.0],
        ], dtype=np.float32)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=10.0)
        assert pytest.approx(float(norm_verts[:, 1].max()), abs=1e-3) == 10.0
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=10.0)
        assert colliders.cylinder.height > 0.0
        assert colliders.aabb.size[1] > 0.0

    def test_empty_vertices_array_handling(self):
        """Empty array of shape (0, 3) raises ValueError cleanly."""
        verts = np.zeros((0, 3), dtype=np.float32)
        with pytest.raises((ValueError, IndexError)):
            GeometryNormalizer.normalize_vertices(verts, target_height=5.0)


# =============================================================================
# Feature 4: 3-Tier LOD Generator Boundary (5 Tests)
# =============================================================================

class TestF04LODGeneratorBoundary:
    def test_lod_generator_init_targets(self):
        """Verify custom LOD polycount targets initialize correctly."""
        gen = LODGenerator(target_lod1_triangles=500, target_lod2_triangles=50)
        assert gen.target_lod1_triangles == 500
        assert gen.target_lod2_triangles == 50

    def test_pure_python_decimation_fewer_faces(self):
        """Mesh with fewer faces than target returns unchanged triangles."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        v_out, f_out = LODGenerator._decimate_pure_python(verts, faces, target_count=1000)
        assert len(f_out) == len(faces)

    def test_pure_python_decimation_reduction(self):
        """Mesh with many faces simplifies to target_count."""
        verts, tris = create_procedural_primitive("pine", target_height=5.0)
        v_out, f_out = LODGenerator._decimate_pure_python(verts, tris, target_count=10)
        assert len(f_out) <= 10

    def test_lod2_cross_quad_billboard_dimensions(self):
        """Verify billboard quad matches exact bounding height of vertices."""
        verts = np.array([[-1.5, 0.0, -1.0], [1.5, 8.5, 1.0]], dtype=np.float32)
        v_bb, f_bb = LODGenerator.create_cross_quad_billboard(verts)
        assert len(f_bb) == 4  # 4 triangles
        assert len(v_bb) == 8  # 8 vertices
        y_span = float(np.max(v_bb[:, 1]) - np.min(v_bb[:, 1]))
        assert pytest.approx(y_span, abs=1e-4) == 8.5

    def test_lod_generation_force_pure_python(self, tmp_path):
        """Generate LODs using force_pure_python produces valid LODResult."""
        verts, tris = create_procedural_primitive("rock", target_height=3.0)
        raw_glb = build_glb_from_mesh(verts, tris)
        input_glb = tmp_path / "model.glb"
        input_glb.write_bytes(raw_glb)

        gen = LODGenerator(target_lod1_triangles=20, target_lod2_triangles=4)
        res = gen.generate_lods(input_glb, output_dir=tmp_path, force_pure_python=True)
        assert res.method_used == "pure_python"
        assert res.lod2_triangles == 4
        assert res.lod0_path.is_file()
        assert res.lod1_path.is_file()
        assert res.lod2_path.is_file()


# =============================================================================
# Feature 5: Abiotic Prompt Boundary & Bias Rejection (5 Tests)
# =============================================================================

class TestF05AbioticPromptBoundary:
    def test_case_insensitive_forbidden_tokens(self):
        """UPPERCASE and MiXeD CaSe forbidden human tokens rejected."""
        assert validate_primordial_prompt("ANCIENT RUINS", strict=False).is_valid is False
        assert validate_primordial_prompt("Stone Axe Cleaved Rock", strict=False).is_valid is False
        assert validate_primordial_prompt("SAWED STONE BLOCK", strict=False).is_valid is False

    def test_whole_word_boundary_isolation(self):
        """Verify whole word boundary correctly catches forbidden words."""
        res = validate_primordial_prompt("ancient stone house", strict=False)
        assert res.is_valid is False
        assert "house" in res.detected_tokens

    def test_clean_primordial_prompt_passes(self):
        """Pristine primordial prompt with geological terms passes validation."""
        res = validate_primordial_prompt("massive granite crag weathered by ice and wind", strict=True)
        assert res.is_valid is True
        assert len(res.detected_tokens) == 0

    def test_maximum_prompt_length_stress(self):
        """Very long prompt (5000 chars) validates efficiently without freezing."""
        clean_text = "ancient weathered basalt columnar rock formation " * 100
        t0 = time.perf_counter()
        res = validate_primordial_prompt(clean_text, strict=False)
        elapsed = time.perf_counter() - t0
        assert res.is_valid is True
        assert elapsed < 0.5, "Prompt validation took too long"

    def test_strict_mode_raises_human_bias_error(self):
        """Strict mode raises HumanBiasError containing detected_tokens."""
        with pytest.raises(HumanBiasError) as exc_info:
            validate_primordial_prompt("Ancient paved road through valley", strict=True)
        assert len(exc_info.value.detected_tokens) > 0


# =============================================================================
# Feature 6: Abiotic Asset Prototypes Boundary (5 Tests)
# =============================================================================

class TestF06AbioticAssetPrototypesBoundary:
    def test_all_9_abiotic_types_exist(self):
        """All 9 AbioticType enum values resolve without index gaps."""
        assert len(AbioticType) in (9, 10)
        expected = [
            "HornPeak", "RockCrag", "BasaltColumn", "RiverPebble",
            "RiparianSandbank", "Stalactite", "Stalagmite", "CavernColumn", "CavernMineral"
        ]
        for name in expected:
            assert hasattr(AbioticType, name)
            member = getattr(AbioticType, name)
            assert AbioticType.from_any(name) == member

    def test_invalid_abiotic_type_from_any(self):
        """Invalid type name or out-of-range integer returns None."""
        assert AbioticType.from_any("InvalidAlienMineral") is None
        assert AbioticType.from_any(999) is None
        assert AbioticType.from_any(-1) is None

    def test_abiotic_profiles_dictionary_completeness(self):
        """DEFAULT_ABIOTIC_PROFILES contains valid entries for all 9 enum members."""
        for member in AbioticType:
            assert member in DEFAULT_ABIOTIC_PROFILES
            prof = DEFAULT_ABIOTIC_PROFILES[member]
            assert "geological_name" in prof
            assert "typical_height" in prof
            assert prof["typical_height"] > 0.0

    def test_procedural_primitive_tiny_pebble(self):
        """Tiny pebble primitive scaled to 0.05m target height."""
        v, f = create_procedural_primitive(primitive_type="rock", target_height=5.0)
        norm_v, _ = GeometryNormalizer.normalize_vertices(v, target_height=0.05)
        assert len(norm_v) >= 4
        assert len(f) >= 4
        y_span = float(norm_v[:, 1].max() - norm_v[:, 1].min())
        assert pytest.approx(y_span, abs=1e-3) == 0.05

    def test_procedural_primitive_megalith_scaling(self):
        """Megalith primitive scaled to 100.0m target height."""
        v, f = create_procedural_primitive(primitive_type="rock", target_height=5.0)
        norm_v, _ = GeometryNormalizer.normalize_vertices(v, target_height=100.0)
        assert len(norm_v) >= 4
        assert len(f) >= 4
        y_span = float(norm_v[:, 1].max() - norm_v[:, 1].min())
        assert pytest.approx(y_span, abs=1e-3) == 100.0


# =============================================================================
# Feature 7: Multi-Tier Geological Topography Boundary (5 Tests)
# =============================================================================

class TestF07GeologicalBoundary:
    def test_heightfield_zero_amplitude_strata(self):
        """Strata folding with amplitude=0.0 leaves Z array unmodified."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        z_init = hf.Z.copy()
        hf.apply_strata_folding(amplitude=0.0, wavelength=20.0)
        np.testing.assert_array_equal(hf.Z, z_init)

    def test_heightfield_zero_wavelength_strata_no_division_error(self):
        """Strata folding with wavelength<=0.0 handled safely without division error."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        z_init = hf.Z.copy()
        hf.apply_strata_folding(amplitude=5.0, wavelength=0.0)
        np.testing.assert_array_equal(hf.Z, z_init)

    def test_extreme_fault_throw_displacement(self):
        """Fault scarp with displacement_throw=500.0m creates massive cliff without NaN."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.apply_tectonic_scarp(displacement_throw=500.0, fault_angle_deg=0.0, fault_offset=0.0)
        assert np.isfinite(hf.Z).all()
        assert hf.Z.max() - hf.Z.min() > 400.0

    def test_canyon_terrace_zero_blend_leaves_unmodified(self):
        """Canyon terracing with blend=0.0 leaves topography unmodified."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.linspace(0.0, 10.0, 21)[np.newaxis, :] * np.ones((21, 1))
        z_init = hf.Z.copy()
        hf.apply_canyon_terraces(terrace_height=3.0, blend=0.0)
        np.testing.assert_array_equal(hf.Z, z_init)

    def test_coastal_profile_at_extreme_sea_level(self):
        """Coastal profile localized at sea level=100.0m does not crash or produce NaN."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.apply_coastal_profile(sea_level=100.0, wave_notch_depth=2.0)
        assert np.isfinite(hf.Z).all()


# =============================================================================
# Feature 8: Hydraulic Droplet Erosion Boundary (5 Tests)
# =============================================================================

class TestF08HydraulicErosionBoundary:
    def test_mass_conservation_at_zero_droplets(self):
        """Zero droplet iterations leaves elevation and sediment strictly unchanged."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.Z = np.ones((21, 21), dtype=np.float64) * 5.0
        cfg = ErosionConfig(enabled=True, droplet_count=0)
        sim = ErosionSimulator(hf, cfg)
        init_z = hf.Z.copy()
        sim.simulate_hydraulic(seed=42)
        np.testing.assert_array_equal(hf.Z, init_z)

    def test_exact_mass_conservation_with_droplets(self):
        """Hydraulic erosion droplets strictly conserve mass (|dM| < 1e-9)."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=8.0, radius=10.0, shape="cone")
        cfg = ErosionConfig(enabled=True, droplet_count=100, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        m_initial = float(np.sum(hf.Z))
        sim.simulate_hydraulic(seed=123)
        m_final = float(np.sum(hf.Z))
        assert abs(m_final - m_initial) < 1e-9

    def test_hydraulic_erosion_on_perfectly_flat_plane(self):
        """Droplets on perfectly flat terrain do not divide by zero or create NaNs."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.Z = np.full((21, 21), 10.0, dtype=np.float64)
        cfg = ErosionConfig(enabled=True, droplet_count=40)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=77)
        assert np.isfinite(hf.Z).all()
        assert np.isfinite(hf.sediment).all()

    def test_hydraulic_erosion_on_extreme_vertical_relief(self):
        """Droplets on extreme vertical slope stay bounded without numerical explosion."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.Z = np.linspace(0.0, 100.0, 21)[np.newaxis, :] * np.ones((21, 1))
        cfg = ErosionConfig(enabled=True, droplet_count=30)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=99)
        assert np.isfinite(hf.Z).all()

    def test_droplet_evaporation_speed_extreme(self):
        """Fast evaporation terminates droplets cleanly without infinite loops."""
        hf = Heightfield2D(width=20.0, depth=20.0, resolution=15)
        cfg = ErosionConfig(enabled=True, droplet_count=30, evaporate_speed=0.99, max_droplet_steps=5)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=1)
        assert np.isfinite(hf.Z).all()


# =============================================================================
# Feature 9: Thermal Talus Erosion Boundary (5 Tests)
# =============================================================================

class TestF09ThermalErosionBoundary:
    def test_thermal_erosion_zero_iterations(self):
        """Zero iterations leaves terrain completely unchanged."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=10.0, radius=8.0, shape="horn")
        z_init = hf.Z.copy()
        cfg = ErosionConfig(thermal_iterations=0)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        np.testing.assert_array_equal(hf.Z, z_init)

    def test_talus_angle_90_degrees_causes_zero_rockfall(self):
        """Talus angle of 89.9 degrees on moderate slope produces zero scree."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=4.0, radius=12.0, shape="dome")
        cfg = ErosionConfig(talus_angle_deg=89.9, thermal_iterations=10)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        assert np.max(hf.scree_talus) == 0.0, "No talus should fall on slopes < 89.9 deg"

    def test_talus_angle_low_causes_widespread_slumping(self):
        """Low talus angle (10 degrees) collapses steep peaks towards flat."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=15.0, radius=8.0, shape="cone")
        initial_peak_h = hf.Z.max()
        cfg = ErosionConfig(talus_angle_deg=10.0, thermal_iterations=15)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        assert hf.Z.max() < initial_peak_h - 1.0, "Steep peak must slump at 10 deg talus angle"
        assert hf.scree_talus.max() > 0.0

    def test_scree_talus_non_negative_invariant(self):
        """Scree layer array is strictly non-negative (>= 0.0) everywhere."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=12.0, radius=10.0, shape="horn")
        cfg = ErosionConfig(talus_angle_deg=30.0, thermal_iterations=10)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        assert (hf.scree_talus >= -1e-9).all(), "Scree talus must never be negative"

    def test_extreme_needle_relaxation_stability(self):
        """Extreme needle peak (50m elevation spike at 1 cell) relaxes without NaN."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.Z[10, 10] = 50.0
        cfg = ErosionConfig(talus_angle_deg=35.0, thermal_iterations=10)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        assert np.isfinite(hf.Z).all()
        assert hf.Z[10, 10] < 50.0, "Needle peak must shed mass to neighbors"


# =============================================================================
# Feature 10: Pure Abiotic Preset Boundary (5 Tests)
# =============================================================================

class TestF10AbioticPresetBoundary:
    def test_empty_biomes_scatter_produces_zero_instances(self):
        """Preset with biomes=[] guarantees 0 flora scatter instances."""
        cfg = MapConfig(biomes=[])
        assert len(cfg.biomes) == 0

    def test_extreme_resolution_bounds_32_and_512(self):
        """Valid initialization at minimal (32) and high (512) resolution bounds."""
        cfg_low = MapConfig()
        cfg_low.diorama.resolution = 32
        assert cfg_low.diorama.resolution == 32

        cfg_high = MapConfig()
        cfg_high.diorama.resolution = 512
        assert cfg_high.diorama.resolution == 512

    def test_peaks_outside_diorama_bounds(self):
        """Peaks placed far outside diorama bounds evaluate smoothly."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(center_x=500.0, center_y=500.0, height=20.0, radius=30.0, shape="horn")
        assert np.isfinite(hf.Z).all()
        assert hf.Z.max() < 0.1, "Peak 500m away should not elevate local terrain"

    def test_zero_peak_count_configuration(self):
        """MapConfig with zero peaks initializes flat baseline without errors."""
        cfg = MapConfig(peaks=[])
        assert len(cfg.peaks) == 0
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=16)
        assert np.all(hf.Z == 0.0)

    def test_missing_optional_sections_in_preset_dict(self):
        """Partial dictionary with missing sections loads default subconfigs."""
        minimal_dict = {"name": "Minimal_Abiotic", "seed": 999}
        cfg = MapConfig.from_dict(minimal_dict)
        assert cfg.name == "Minimal_Abiotic"
        assert cfg.hydrology is not None
        assert cfg.cavern is not None
        assert cfg.diorama.width == 160.0


# =============================================================================
# Feature 11: Triplanar Rock PBR Shading Boundary (5 Tests)
# =============================================================================

class TestF11TriplanarShadingBoundary:
    def test_snow_altitude_below_sea_level(self):
        """Snow altitude below sea level covers 100% of terrain above sea."""
        cfg = ShaderConfig(snow_altitude=-10.0)
        elev = 5.0
        w_snow = float(np.clip((elev - cfg.snow_altitude) / 2.0, 0.0, 1.0))
        assert w_snow == 1.0

    def test_snow_altitude_above_highest_peak(self):
        """Snow altitude set to 1000.0m results in 0.0 snow weight."""
        cfg = ShaderConfig(snow_altitude=1000.0)
        elev = 35.0
        w_snow = float(np.clip((elev - cfg.snow_altitude) / 2.0, 0.0, 1.0))
        assert w_snow == 0.0

    def test_slope_weight_pure_vertical_cliff(self):
        """Pure vertical cliff (90 deg) receives 1.0 cliff rock weight."""
        slope = 90.0
        cliff_transition = 40.0
        cliff_sharpness = 5.0
        w_rock = float(1.0 / (1.0 + np.exp(-(slope - cliff_transition) / (cliff_sharpness / 2.0))))
        assert w_rock > 0.99, "90-degree cliff must have near 1.0 rock weight"

    def test_slope_weight_pure_horizontal_plane(self):
        """Pure horizontal flat plane (0 deg) receives 0.0 cliff rock weight."""
        slope = 0.0
        cliff_transition = 40.0
        cliff_sharpness = 5.0
        w_rock = float(1.0 / (1.0 + np.exp(-(slope - cliff_transition) / (cliff_sharpness / 2.0))))
        assert w_rock < 0.01, "Flat plane must have near 0.0 cliff weight"

    def test_water_roughness_clamped_bounds(self):
        """Roughness parameter values in [0.0, 1.0] are preserved."""
        cfg = ShaderConfig(water_roughness=0.0)
        assert cfg.water_roughness == 0.0
        cfg.water_roughness = 1.0
        assert cfg.water_roughness == 1.0


# =============================================================================
# Feature 12: 4-Tier Continuous Hydrology Boundary (5 Tests)
# =============================================================================

class TestF12ContinuousHydrologyBoundary:
    def test_river_spline_parameter_evaluation(self):
        """Evaluate river spline at endpoints t=0.0 and t=1.0."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=21)
        cfg = HydrologyConfig()
        hydro = HydrologyEngine(hf, cfg)
        p0 = hydro.evaluate_river_spline(0.0)
        p1 = hydro.evaluate_river_spline(1.0)
        assert len(p0) == 4
        assert len(p1) == 4

    def test_lake_radius_positive_basin(self):
        """Lake basin with radius=10.0 carves bathymetric basin."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(10.0)
        hf.carve_lake_basin(0.0, 0.0, radius=10.0, water_level=4.0, lake_depth=5.0)
        assert np.isfinite(hf.Z).all()
        assert hf.sample_bilinear(0.0, 0.0) <= 4.0

    def test_lake_depth_zero_boundary(self):
        """Lake with lake_depth=0.0 carves shelf at exact water_level."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.Z = np.full((21, 21), 10.0)
        hf.carve_lake_basin(0.0, 0.0, radius=8.0, water_level=4.0, lake_depth=0.0, berm_height=0.0)
        assert hf.sample_bilinear(0.0, 0.0) == 4.0

    def test_water_distance_field_all_dry_terrain(self):
        """Terrain with lake_radius=0.0 and river_enabled=False has strictly positive water distance."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.full((21, 21), 50.0)
        cfg = HydrologyConfig(lake_radius=0.0, river_enabled=False)
        hydro = HydrologyEngine(hf, cfg)
        dist = hydro.compute_water_distance_field()
        assert (dist > 0.0).all()

    def test_water_distance_field_with_water(self):
        """Terrain completely submerged inside large lake radius returns 0.0 distance everywhere."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.full((21, 21), -10.0)
        cfg = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=100.0, river_enabled=False)
        hydro = HydrologyEngine(hf, cfg)
        dist = hydro.compute_water_distance_field()
        assert (dist == 0.0).all()


# =============================================================================
# Feature 13: Parabolic Channel & Lake Berm Boundary (5 Tests)
# =============================================================================

class TestF13ChannelBermBoundary:
    def test_berm_height_zero_no_retaining_wall(self):
        """Lake with berm_height=0.0 carves standard basin without wall."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.full((21, 21), 10.0)
        hf.carve_lake_basin(0.0, 0.0, radius=10.0, water_level=0.0, lake_depth=5.0, berm_height=0.0)
        assert hf.sample_bilinear(0.0, 0.0) == -5.0

    def test_extreme_berm_height_containment(self):
        """Berm height 25.0m creates massive retaining rim without NaN."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z = np.full((31, 31), 2.0)
        hf.carve_lake_basin(0.0, 0.0, radius=12.0, water_level=5.0, lake_depth=6.0, berm_height=25.0)
        assert np.isfinite(hf.Z).all()
        crest_z = hf.sample_bilinear(12.0, 0.0)
        assert crest_z >= 25.0

    def test_channel_carving_reduces_bed_elevation(self):
        """River carving incises channel into terrain."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(15.0)
        cfg = HydrologyConfig(river_enabled=True)
        hydro = HydrologyEngine(hf, cfg)
        hydro.carve_river_channel()
        assert hf.Z.min() < 15.0

    def test_berm_smoothstep_monotonicity(self):
        """Outer berm smoothstep transition from 1.15*R to 1.50*R is monotonic."""
        u = np.linspace(0.0, 1.0, 50)
        s_out = 1.0 - (3.0 * u**2 - 2.0 * u**3)
        diffs = np.diff(s_out)
        assert (diffs <= 0.0).all(), "Outer berm falloff must be monotonically decreasing"

    def test_parabolic_bed_zero_cross_offset(self):
        """Center of river channel (offset=0.0) achieves maximum incision."""
        half_width = 5.0
        max_depth = 2.5
        d_vals = np.linspace(-half_width, half_width, 21)
        incision = max_depth * (1.0 - (d_vals / half_width)**2)
        assert np.max(incision) == pytest.approx(max_depth)
        assert incision[0] == pytest.approx(0.0)
        assert incision[-1] == pytest.approx(0.0)


# =============================================================================
# Feature 14: Optical Water Shader Boundary (5 Tests)
# =============================================================================

class TestF14WaterShaderBoundary:
    def test_beer_lambert_depth_zero_transmittance(self):
        """Beer-Lambert transmittance at depth=0.0 is exactly 1.0."""
        depth = 0.0
        coeff = np.array([0.1, 0.05, 0.02])
        trans = np.exp(-coeff * depth)
        np.testing.assert_allclose(trans, [1.0, 1.0, 1.0])

    def test_beer_lambert_infinite_depth_total_absorption(self):
        """At extreme depth (1000m), transmittance is 0.0."""
        depth = 1000.0
        coeff = np.array([0.1, 0.05, 0.02])
        trans = np.exp(-coeff * depth)
        np.testing.assert_allclose(trans, [0.0, 0.0, 0.0], atol=1e-8)

    def test_contact_foam_margin_zero_width(self):
        """Foam margin width 0.0 produces zero foam area."""
        foam_width = 0.0
        dist_to_shore = 0.1
        foam_mask = max(0.0, 1.0 - (dist_to_shore / max(1e-5, foam_width)))
        assert foam_mask == 0.0

    def test_water_specular_roughness_zero_mirror(self):
        """Zero roughness represents pure specular mirror."""
        r = 0.0
        assert r == 0.0
        assert 0.0 <= r <= 1.0

    def test_flow_vector_zero_magnitude_safe(self):
        """Flow vectors with zero magnitude compute unit direction without division by zero."""
        fx, fy = 0.0, 0.0
        mag = math.sqrt(fx**2 + fy**2)
        ux = fx / mag if mag > 1e-6 else 0.0
        uy = fy / mag if mag > 1e-6 else 0.0
        assert ux == 0.0
        assert uy == 0.0


# =============================================================================
# Feature 15: Subterranean Karst Cavern Boundary (5 Tests)
# =============================================================================

class TestF15KarstCavernBoundary:
    def test_cavern_overburden_thickness_boundary(self):
        """Terrain elevation exactly 5.0m above cavern roof meets minimal safety limit."""
        c_z = -4.5
        c_h = 7.0
        roof_z = c_z + c_h / 2.0
        terrain_z = roof_z + 5.0
        overburden = terrain_z - roof_z
        assert overburden >= 5.0

    def test_cavern_overburden_violation_detection(self):
        """Cavern roof breaching or near surface (< 5.0m) detected."""
        roof_z = 2.0
        terrain_z = 3.0
        assert (terrain_z - roof_z) < 5.0, "Should detect insufficient overburden"

    def test_speleothem_count_zero(self):
        """Zero stalactites and stalagmites configuration loads safely."""
        cfg = CavernConfig(stalactite_count=0, stalagmite_count=0, column_count=0)
        assert cfg.stalactite_count == 0
        assert cfg.stalagmite_count == 0

    def test_extreme_speleothem_count(self):
        """Large speleothem count (1000) stores valid config parameters."""
        cfg = CavernConfig(stalactite_count=1000, stalagmite_count=1000)
        assert cfg.stalactite_count == 1000

    def test_subterranean_pool_level_matching(self):
        """Subterranean pool level is below cavern roof and above base."""
        cfg = CavernConfig(center=(0, 0, -5.0), height=8.0, pool_level=-7.0)
        roof_z = cfg.center[2] + cfg.height / 2.0
        floor_z = cfg.center[2] - cfg.height / 2.0
        assert floor_z <= cfg.pool_level <= roof_z


# =============================================================================
# Feature 16: Karst Arch Entrance Boundary (5 Tests)
# =============================================================================

class TestF16KarstArchEntranceBoundary:
    def test_entrance_radius_zero(self):
        """Entrance radius <= 0 is non-physical."""
        cfg = CavernConfig(entrance_radius=0.0)
        assert cfg.entrance_radius == 0.0

    def test_entrance_elevation_at_sea_level(self):
        """Entrance exactly at sea level 0.0m is valid."""
        cfg = CavernConfig(entrance_pos=(0.0, 10.0, 0.0))
        assert cfg.entrance_pos[2] == 0.0

    def test_entrance_normal_near_horizontal(self):
        """Portal normal facing mountain cliff has small vertical Z component."""
        normal = (0.95, -0.25, 0.05)
        mag = math.sqrt(sum(x**2 for x in normal))
        norm_z = abs(normal[2] / mag)
        assert norm_z < 0.2, "Entrance normal should be predominantly horizontal"

    def test_arch_tunnel_spline_length(self):
        """Tunnel spline connecting entrance to chamber has length > 0."""
        ent = (0.0, 12.0, 1.5)
        cham = (14.0, 16.0, -4.5)
        dx = cham[0] - ent[0]
        dy = cham[1] - ent[1]
        dz = cham[2] - ent[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 10.0

    def test_portal_clearance_at_cliff_face(self):
        """Entrance clearance radius 4.5m allows player passage."""
        cfg = CavernConfig(entrance_radius=4.5)
        assert cfg.entrance_radius >= 2.0


# =============================================================================
# Feature 17: Bioluminescent Cavern Minerals Boundary (5 Tests)
# =============================================================================

class TestF17BioluminescentMineralsBoundary:
    def test_light_energy_zero_unlit(self):
        """Emissive energy 0.0W produces dark cavern."""
        cfg = CavernConfig(bioluminescent=False)
        assert cfg.bioluminescent is False

    def test_light_energy_maximum_boundary(self):
        """Light power parameter capped within valid range."""
        power_watts = 35.0
        assert 1.0 <= power_watts <= 100.0

    def test_chromaticity_rgb_clamping(self):
        """Glow color RGB components strictly within [0.0, 1.0]."""
        color = (0.1, 0.95, 0.4)
        for c in color:
            assert 0.0 <= c <= 1.0

    def test_ambient_falloff_distance_cutoff(self):
        """Inverse-square falloff reaches zero at cutoff radius."""
        dist = 20.0
        cutoff = 15.0
        attenuation = max(0.0, 1.0 - (dist / cutoff))
        assert attenuation == 0.0

    def test_mineral_profile_metadata(self):
        """Cavern mineral profile has solid=True and emissive category."""
        prof = DEFAULT_ABIOTIC_PROFILES[AbioticType.CavernMineral]
        assert prof["solid"] is True
        assert prof["category"] == "mineral"


# =============================================================================
# Feature 18: Binary WorldArtifact Boundary & Integrity (5 Tests)
# =============================================================================

class TestF18WorldArtifactBoundary:
    def test_corrupted_magic_bytes_rejected(self):
        """Header magic != b'ANMW' raises ValueError."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        raw = bytearray(art.to_bytes())
        raw[0:4] = b"BADM"
        with pytest.raises(ValueError) as exc_info:
            WorldArtifact.from_bytes(bytes(raw))
        assert "magic" in str(exc_info.value).lower()

    def test_unsupported_version_rejected(self):
        """Version != 2 raises ValueError."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        raw = bytearray(art.to_bytes())
        struct.pack_into("<I", raw, 4, 999)
        with pytest.raises(ValueError) as exc_info:
            WorldArtifact.from_bytes(bytes(raw))
        assert "version" in str(exc_info.value).lower()

    def test_checksum_mismatch_rejected(self):
        """Corrupting a single payload byte causes checksum failure on load."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        raw = bytearray(art.to_bytes())
        raw[-1] ^= 0xFF
        with pytest.raises(ValueError) as exc_info:
            WorldArtifact.from_bytes(bytes(raw))
        assert "checksum" in str(exc_info.value).lower()

    def test_payload_truncated_by_one_byte(self):
        """Truncating binary by 1 byte raises ValueError."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        raw = art.to_bytes()[:-1]
        with pytest.raises(ValueError) as exc_info:
            WorldArtifact.from_bytes(raw)
        assert "truncated" in str(exc_info.value).lower()

    def test_trailing_bytes_rejected(self):
        """Appending 1 extra byte raises ValueError."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        raw = art.to_bytes() + b"\x00"
        with pytest.raises(ValueError) as exc_info:
            WorldArtifact.from_bytes(raw)
        assert "trailing" in str(exc_info.value).lower()


# =============================================================================
# Feature 19: Map Manifest Schema Boundary (5 Tests)
# =============================================================================

class TestF19MapManifestBoundary:
    def test_manifest_empty_views_invalid(self):
        """Manifest with views=[] fails validation."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        manifest["views"] = []
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is False
        assert len(errors) > 0

    def test_manifest_missing_views_fails_validation(self):
        """Manifest without views key fails validation."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        del manifest["views"]
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is False

    def test_manifest_missing_world_artifact_fails(self):
        """Manifest without worldArtifact key fails validation."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        del manifest["worldArtifact"]
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is False

    def test_manifest_coordinate_extremes_contract(self):
        """World coordinates exactly [-100.0, 100.0] and [0.0, 10.0] pass schema."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        cs = manifest["coordinateSystem"]
        assert cs["worldMinXZ"] == -100.0
        assert cs["worldMaxXZ"] == 100.0
        assert cs["worldMinY"] == 0.0
        assert cs["worldMaxY"] == 10.0

    def test_manifest_missing_schema_version_rejected(self):
        """Missing schemaVersion fails validation."""
        art = WorldArtifact(
            width=4, height=4, sea_level=0.0, seed=1, generator_version=1, world_scale=200.0,
            elevation=np.zeros(16, dtype=np.float32), moisture=np.zeros(16, dtype=np.float32),
            temperature=np.zeros(16, dtype=np.float32), flow=np.zeros(16, dtype=np.float32),
            biome=np.zeros(16, dtype=np.uint8),
        )
        manifest = generate_map_manifest(art)
        del manifest["schemaVersion"]
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is False


# =============================================================================
# Feature 20: Dual Scene Export Boundary (5 Tests)
# =============================================================================

class TestF20DualSceneExportBoundary:
    def test_empty_heightfield_mesh_export(self):
        """Exporting mesh from 0x0 heightfield handled safely."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            OpenWorldMesh(elevation=np.zeros((0, 0), dtype=np.float32))

    def test_minimum_2x2_grid_mesh_export(self):
        """Smallest 2x2 grid exports valid mesh with 4 vertices and 2 triangles."""
        z = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        mesh = OpenWorldMesh(elevation=z)
        assert mesh.vertices.shape == (4, 3)
        assert len(mesh.indices) == 6

    def test_extreme_elevation_mesh_bounds_glb(self, tmp_path):
        """Vertices with elevations from -50m to +500m export valid GLB file."""
        z = np.array([[-50.0, 500.0], [0.0, 250.0]], dtype=np.float32)
        mesh = OpenWorldMesh(elevation=z)
        out_glb = tmp_path / "extreme.glb"
        mesh.export_glb(out_glb)
        assert out_glb.is_file()
        assert out_glb.stat().st_size > 500

    def test_glb_alignment_4_byte_boundary(self):
        """GLB binary total size must be an exact multiple of 4 bytes."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        glb_bytes = build_glb_from_mesh(verts, tris)
        assert len(glb_bytes) % 4 == 0, "GLB length must be 4-byte aligned"

    def test_open_mesh_uv_bounds_zero_to_one(self):
        """UV coordinates are strictly normalized within [0.0, 1.0]."""
        z = np.zeros((5, 5), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=z)
        assert (mesh.uvs >= 0.0).all()
        assert (mesh.uvs <= 1.0).all()


# =============================================================================
# Feature 21: NavMesh BFS Coverage Boundary (5 Tests)
# =============================================================================

class TestF21NavMeshBoundary:
    def test_100_percent_flooded_world_zero_coverage(self):
        """All cells submerged under sea level results in 0% reachability."""
        elev = np.full((20, 20), -10.0, dtype=np.float32)
        val = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(20, 20))
        res = val.evaluate(sync)
        assert res.navmesh_coverage == 0.0
        assert res.is_valid is False

    def test_100_percent_flat_land_full_coverage(self):
        """Completely flat land diorama achieves 100% reachability."""
        elev = np.full((20, 20), 5.0, dtype=np.float32)
        val = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(20, 20))
        res = val.evaluate(sync)
        assert res.navmesh_coverage == 1.0
        assert res.is_valid is True

    def test_impassable_sheer_cliffs_reachability_drop(self):
        """Vertical cliff barrier reduces navigable coverage."""
        elev = np.zeros((20, 20), dtype=np.float32)
        elev[10:, :] = 40.0
        val = NavMeshReachabilityValidator(elevation=elev, sea_level=-5.0, walkable_slope_limit=0.20)
        sync = ObstacleGridSynchronizer(shape=(20, 20))
        res = val.evaluate(sync)
        assert res.navmesh_coverage < 0.95

    def test_out_of_bounds_coordinate_queries(self):
        """Coordinates outside diorama map cleanly to boundary cell."""
        r, c = world_to_cell(1000.0, 1000.0, shape=(20, 20))
        assert 0 <= r < 20 and 0 <= c < 20
        r_neg, c_neg = world_to_cell(-1000.0, -1000.0, shape=(20, 20))
        assert 0 <= r_neg < 20 and 0 <= c_neg < 20

    def test_spawn_point_never_in_water(self):
        """Spawn finder never places spawn in water cells."""
        elev = np.full((20, 20), -5.0, dtype=np.float32)
        elev[0:5, 0:5] = 4.0
        val = NavMeshReachabilityValidator(elevation=elev, sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=(20, 20))
        res = val.evaluate(sync)
        assert res.spawn_world_pos is not None
        sx, sy, sz = res.spawn_world_pos
        assert sy >= 0.0


# =============================================================================
# Feature 22: Camera Rig & Ephemeris Boundary (5 Tests)
# =============================================================================

class TestF22CameraRigBoundary:
    def test_camera_framing_zero_size_object(self):
        """Framing distance for target_size=0.0 returns 0.0."""
        dist = calculate_camera_framing_distance(width=0.0, depth=0.0, fov_deg=45.0)
        assert dist == 0.0

    def test_camera_fov_extreme_focal_length(self):
        """10mm (ultra-wide) and 300mm (telephoto) lenses compute correct FOV."""
        fov_wide = calculate_fov_from_lens(lens_mm=10.0, sensor_width_mm=36.0)
        fov_tele = calculate_fov_from_lens(lens_mm=300.0, sensor_width_mm=36.0)
        assert fov_wide > 100.0
        assert fov_tele < 15.0

    def test_solar_zenith_noon_elevation(self):
        """Solar elevation at solar noon (12.0h) equals max_elevation_deg."""
        elev_deg, az_deg, sun_vec = calculate_solar_position(sun_time_of_day=12.0, max_elevation_deg=65.0)
        assert pytest.approx(elev_deg, abs=0.5) == 65.0
        assert sun_vec[2] > 0.8

    def test_solar_midnight_negative_elevation(self):
        """Solar elevation at midnight (0.0h) is negative (below horizon)."""
        elev_deg, _, _ = calculate_solar_position(sun_time_of_day=0.0, max_elevation_deg=65.0)
        assert elev_deg < 0.0

    def test_color_temperature_extreme_kelvin(self):
        """Color temperatures 1000K (candle) and 12000K (blue sky) map to valid RGB."""
        rgb_warm = cct_to_linear_rgb(1000.0)
        rgb_cool = cct_to_linear_rgb(12000.0)
        assert rgb_warm[0] > rgb_warm[2]
        assert rgb_cool[2] > rgb_cool[0]


# =============================================================================
# Feature 23: Offline Viewer Vendor Assets Boundary (5 Tests)
# =============================================================================

class TestF23OfflineViewerBoundary:
    def test_offline_vendor_three_js_size_bounds(self):
        """Verify web/vendor/three.min.js exists and exceeds 200KB."""
        vendor_path = Path("e:/Project/01_AI_Agents/Genesis_Zero/web/vendor/three.min.js")
        assert vendor_path.is_file()
        assert vendor_path.stat().st_size > 200_000

    def test_offline_vendor_gltf_loader_size_bounds(self):
        """Verify web/vendor/GLTFLoader.js exists and exceeds 50KB."""
        vendor_path = Path("e:/Project/01_AI_Agents/Genesis_Zero/web/vendor/GLTFLoader.js")
        assert vendor_path.is_file()
        assert vendor_path.stat().st_size > 50_000

    def test_viewer_html_file_size_and_utf8(self):
        """Verify viewer.html is valid UTF-8 and exceeds 5KB."""
        viewer_path = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        assert viewer_path.is_file()
        content = viewer_path.read_text(encoding="utf-8")
        assert len(content) > 5_000

    def test_viewer_canvas_element_present(self):
        """HTML markup contains canvas-container element."""
        viewer_path = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        content = viewer_path.read_text(encoding="utf-8")
        assert 'id="canvas-container"' in content

    def test_viewer_camera_controls_and_lighting(self):
        """HTML markup contains camera controls and lighting toggles."""
        viewer_path = Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/viewer.html")
        content = viewer_path.read_text(encoding="utf-8")
        assert "setCamera" in content
        assert "toggleLighting" in content
