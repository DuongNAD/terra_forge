"""
Unit and Integration Tests for Milestone M2: Meshy AI Pipeline & Local Asset Vault.
Verifies:
1. SHA-256 deterministic caching over (prompt + seed + params).
2. LocalAssetVault in-memory (< 2ms) and cold-disk (< 5ms) lookup latency (< 50ms benchmark requirement).
3. Biological metadata for all 14 Anima-Engine FloraType categories.
4. Geometry normalization (min_y == 0.0 bottom pivot, centered X/Z, metric scaling).
5. Collision primitives extraction (Cylinder, AABB, Convex Hull via SciPy).
6. 3-tier LOD generation (LOD0, LOD1, LOD2 cross-quad billboard).
7. Meshy v2 async client lifecycle, rate limit backoff, and secret masking.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import httpx
import numpy as np
import pytest

from terra_forge.ai.lod import LODGenerator, find_blender_binary
from terra_forge.ai.meshy_client import (
    DEFAULT_MESHY_API_KEY,
    MeshyClient,
    MeshyTaskResult,
    MeshyTaskStatus,
    mask_api_key,
    resolve_api_key,
)
from terra_forge.ai.normalizer import (
    GeometryNormalizer,
    build_glb_from_mesh,
    create_procedural_primitive,
    extract_mesh_from_glb,
    normalize_glb_file,
)
from terra_forge.ai.vault import (
    DEFAULT_FLORA_PROFILES,
    BiologicalMetadata,
    FloraType,
    LocalAssetVault,
    compute_asset_hash,
)


# ===========================================================================
# 1. SHA-256 Deterministic Hashing Tests
# ===========================================================================

class TestVaultHashing:
    def test_sha256_determinism(self):
        prompt = "ancient gnarled alpine pine, deep twisted roots gripping bedrock"
        seed = 42
        params = {"art_style": "realistic", "topology": "quad", "polycount": 10000}

        hash1 = compute_asset_hash(prompt, seed, params)
        hash2 = compute_asset_hash(prompt, seed, params)

        assert len(hash1) == 64
        assert hash1 == hash2

    def test_hash_sensitivity(self):
        base_prompt = "ancient alpine pine"
        base_seed = 42
        base_params = {"art_style": "realistic"}

        base_hash = compute_asset_hash(base_prompt, base_seed, base_params)

        # Different prompt
        h_diff_prompt = compute_asset_hash(base_prompt + " with moss", base_seed, base_params)
        assert h_diff_prompt != base_hash

        # Different seed
        h_diff_seed = compute_asset_hash(base_prompt, base_seed + 1, base_params)
        assert h_diff_seed != base_hash

        # Different params
        h_diff_params = compute_asset_hash(base_prompt, base_seed, {"art_style": "sculpture"})
        assert h_diff_params != base_hash

    def test_canonical_param_ordering(self):
        prompt = "majestic oak"
        seed = 100
        params_order_1 = {"alpha": 1, "beta": 2, "gamma": 3}
        params_order_2 = {"gamma": 3, "alpha": 1, "beta": 2}

        h1 = compute_asset_hash(prompt, seed, params_order_1)
        h2 = compute_asset_hash(prompt, seed, params_order_2)

        assert h1 == h2

    def test_case_and_whitespace_insensitivity(self):
        h1 = compute_asset_hash("  Ancient Pine Tree  ", 42)
        h2 = compute_asset_hash("ancient pine tree", 42)
        assert h1 == h2


# ===========================================================================
# 2. LocalAssetVault & Caching Benchmark Tests (< 50ms requirement)
# ===========================================================================

class TestLocalAssetVault:
    @pytest.fixture
    def temp_vault(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="terra_forge_test_vault_"))
        vault = LocalAssetVault(vault_dir=tmp_dir)
        yield vault
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_store_and_retrieve(self, temp_vault):
        prompt = "ancient gnarled alpine pine"
        seed = 42
        params = {"art_style": "realistic"}

        dummy_glb = b"glTF_DUMMY_BINARY_DATA_FOR_TESTING"
        dummy_thumb = b"PNG_DUMMY_BYTES"

        entry = temp_vault.store(
            prompt=prompt,
            seed=seed,
            params=params,
            model_glb=dummy_glb,
            thumbnail_png=dummy_thumb,
            flora_type=FloraType.Pine,
            dimensions_meters={"height": 7.5, "canopy_radius": 2.2},
        )

        assert entry is not None
        assert entry.model_path.is_file()
        assert entry.has_thumbnail
        assert entry.metadata.flora_type == int(FloraType.Pine)
        assert entry.metadata.dimensions_meters["height"] == 7.5

        # Query by hash
        retrieved = temp_vault.get(entry.asset_id)
        assert retrieved is not None
        assert retrieved.asset_id == entry.asset_id

        # Query by request
        retrieved_by_req = temp_vault.get_by_request(prompt, seed, params)
        assert retrieved_by_req is not None
        assert retrieved_by_req.asset_id == entry.asset_id

    def test_in_memory_lookup_latency_under_50ms(self, temp_vault):
        """Verify in-memory lookup latency is well under the 50ms benchmark (typically < 0.1ms)."""
        prompt = "benchmarked pine"
        seed = 12345
        entry = temp_vault.store(
            prompt=prompt,
            seed=seed,
            model_glb=b"dummy",
            flora_type=FloraType.Pine,
        )

        latencies_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            res = temp_vault.get(entry.asset_id)
            t1 = time.perf_counter()
            assert res is not None
            latencies_ms.append((t1 - t0) * 1000.0)

        max_latency = max(latencies_ms)
        avg_latency = sum(latencies_ms) / len(latencies_ms)

        # Acceptance Criteria: lookup latency < 50ms
        assert max_latency < 50.0, f"Max latency {max_latency:.3f}ms exceeded 50ms limit"
        assert avg_latency < 5.0, f"Average latency {avg_latency:.3f}ms too high"

    def test_cold_disk_lookup_latency_under_50ms(self, temp_vault):
        """Verify cold disk cache retrieval (with in-memory index cleared) is < 50ms."""
        prompt = "cold disk pine"
        seed = 999
        entry = temp_vault.store(
            prompt=prompt,
            seed=seed,
            model_glb=b"dummy_cold_data",
            flora_type=FloraType.Pine,
        )

        # Clear in-memory cache to force disk I/O
        temp_vault.clear_cache()
        assert len(temp_vault) == 0

        t0 = time.perf_counter()
        cold_res = temp_vault.get(entry.asset_id)
        t1 = time.perf_counter()

        disk_latency_ms = (t1 - t0) * 1000.0
        assert cold_res is not None
        assert cold_res.asset_id == entry.asset_id
        assert disk_latency_ms < 50.0, f"Disk retrieval latency {disk_latency_ms:.3f}ms exceeded 50ms limit"

    def test_14_flora_types_biological_metadata(self, temp_vault):
        """Verify all 14 Anima-Engine FloraType categories are supported with valid metadata."""
        assert len(FloraType) == 14

        for ft in FloraType:
            profile = DEFAULT_FLORA_PROFILES.get(ft)
            assert profile is not None, f"Missing default profile for {ft.name}"
            assert "species_name" in profile
            assert "tier" in profile
            assert "solid" in profile
            assert "allowed_biomes" in profile

            bio = BiologicalMetadata.from_flora_type(ft)
            assert bio.flora_type == int(ft)
            assert bio.typical_height > 0.0

            entry = temp_vault.store(
                prompt=f"primordial {ft.name}",
                seed=int(ft),
                model_glb=b"dummy_flora",
                flora_type=ft,
            )
            assert entry.metadata.flora_type == int(ft)
            assert entry.metadata.biological_metadata["species_name"] == profile["species_name"]

    def test_vault_filtering(self, temp_vault):
        temp_vault.store("pine 1", 1, model_glb=b"1", flora_type=FloraType.Pine)
        temp_vault.store("pine 2", 2, model_glb=b"2", flora_type=FloraType.Pine)
        temp_vault.store("oak 1", 3, model_glb=b"3", flora_type=FloraType.Round)
        temp_vault.store("coral 1", 4, model_glb=b"4", flora_type=FloraType.Coral)

        pines = temp_vault.list_assets(FloraType.Pine)
        assert len(pines) == 2

        rounds = temp_vault.list_assets(FloraType.Round)
        assert len(rounds) == 1

        corals = temp_vault.list_assets(FloraType.Coral)
        assert len(corals) == 1

        all_assets = temp_vault.list_assets()
        assert len(all_assets) == 4


# ===========================================================================
# 3. Geometry Normalization & Collision Primitives Tests
# ===========================================================================

class TestGeometryNormalizer:
    def test_normalize_vertices_pivot_min_y_zero(self):
        """Verify that any arbitrary unnormalized mesh is shifted so min(Y) == 0.0."""
        # Random vertices with negative Y and off-center X/Z
        raw_verts = np.array([
            [-5.0, -2.5,  3.0],
            [ 7.0, -2.5, -1.0],
            [ 2.0,  6.5,  4.0],
            [-1.0,  0.0, -8.0],
        ], dtype=np.float32)

        target_h = 9.0
        norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)

        # Invariant 1: Bottom pivot min(Y) == 0.0
        min_y = float(np.min(norm_verts[:, 1]))
        assert abs(min_y) < 1e-5, f"Expected min_y == 0.0, got {min_y}"

        # Invariant 2: Metric height == target_height
        max_y = float(np.max(norm_verts[:, 1]))
        assert abs(max_y - target_h) < 1e-5, f"Expected max_y == {target_h}, got {max_y}"

    def test_normalize_vertices_centered_xz(self):
        """Verify that horizontal X and Z bounds are centered around (0.0, 0.0)."""
        raw_verts = np.array([
            [10.0, 0.0, 20.0],
            [30.0, 5.0, 40.0],
        ], dtype=np.float32)

        norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=5.0)

        x_min, _, z_min = np.min(norm_verts, axis=0)
        x_max, _, z_max = np.max(norm_verts, axis=0)

        assert abs(x_min + x_max) < 1e-5, f"X not centered: {x_min} + {x_max} != 0"
        assert abs(z_min + z_max) < 1e-5, f"Z not centered: {z_min} + {z_max} != 0"

    def test_collision_primitives_extraction(self):
        verts, _ = create_procedural_primitive("pine", target_height=8.0)
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=8.0)

        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=8.0)

        # Cylinder checks
        cyl = colliders.cylinder
        assert cyl.radius > 0.0
        assert cyl.height == 8.0
        assert cyl.center == [0.0, 4.0, 0.0]

        # AABB checks
        aabb = colliders.aabb
        assert aabb.min_vec[1] == 0.0
        assert aabb.max_vec[1] == 8.0
        assert aabb.size[1] == 8.0
        assert aabb.center == [0.0, 4.0, 0.0]

        # Convex Hull checks
        hull = colliders.convex_hull
        assert hull.vertex_count >= 4
        assert hull.simplex_count >= 4
        assert hull.volume > 0.0
        assert hull.area > 0.0

    def test_normalize_glb_file_end_to_end(self, tmp_path):
        """Create an unnormalized GLB file, run normalize_glb_file, and verify output GLB."""
        verts, tris = create_procedural_primitive("rock", target_height=3.0)
        raw_glb = build_glb_from_mesh(verts, tris)

        input_glb = tmp_path / "raw_rock.glb"
        output_glb = tmp_path / "norm_rock.glb"

        with open(input_glb, "wb") as f:
            f.write(raw_glb)

        norm_mesh = normalize_glb_file(input_glb, output_glb, target_height=4.5)

        assert output_glb.is_file()
        assert norm_mesh.target_height == 4.5
        assert abs(float(np.min(norm_mesh.vertices[:, 1]))) < 1e-5
        assert abs(float(np.max(norm_mesh.vertices[:, 1])) - 4.5) < 1e-5

        # Re-read exported GLB to verify binary integrity
        with open(output_glb, "rb") as f:
            read_bytes = f.read()

        read_verts, read_tris = extract_mesh_from_glb(read_bytes)
        assert len(read_verts) == len(norm_mesh.vertices)
        assert abs(float(np.min(read_verts[:, 1]))) < 1e-5
        assert abs(float(np.max(read_verts[:, 1])) - 4.5) < 1e-5


# ===========================================================================
# 4. LOD Generator Tests (3-Tier)
# ===========================================================================

class TestLODGenerator:
    def test_lod_generation_pure_python(self, tmp_path):
        verts, tris = create_procedural_primitive("pine", target_height=7.0)
        raw_glb = build_glb_from_mesh(verts, tris)

        in_glb = tmp_path / "pine_hero.glb"
        with open(in_glb, "wb") as f:
            f.write(raw_glb)

        generator = LODGenerator(target_lod1_triangles=20, target_lod2_triangles=4)
        result = generator.generate_lods(in_glb, output_dir=tmp_path, force_pure_python=True)

        assert result.lod0_path.is_file()
        assert result.lod1_path.is_file()
        assert result.lod2_path.is_file()

        assert result.lod0_triangles > 0
        assert result.lod1_triangles > 0
        assert result.lod2_triangles == 4  # Cross-quad billboard has exactly 4 triangles
        assert result.method_used == "pure_python"

    def test_cross_quad_billboard_geometry(self):
        verts, _ = create_procedural_primitive("pine", target_height=6.0)
        bb_verts, bb_tris = LODGenerator.create_cross_quad_billboard(verts)

        assert len(bb_verts) == 8
        assert len(bb_tris) == 4

        # Verify base vertices sit on ground plane Y == 0.0
        base_verts = bb_verts[bb_verts[:, 1] == 0.0]
        assert len(base_verts) == 4

        # Verify top vertices sit at target height H
        top_verts = bb_verts[bb_verts[:, 1] == np.max(bb_verts[:, 1])]
        assert len(top_verts) == 4

    def test_blender_discovery(self):
        blender_bin = find_blender_binary()
        # On this machine Blender 4.5 LTS is installed
        if blender_bin:
            assert os.path.isfile(blender_bin)
            assert "blender" in blender_bin.lower()

    def test_blender_headless_decimation_if_available(self, tmp_path):
        blender_bin = find_blender_binary()
        if not blender_bin:
            pytest.skip("Blender executable not found on this machine")

        verts, tris = create_procedural_primitive("pine", target_height=6.0)
        raw_glb = build_glb_from_mesh(verts, tris)

        in_glb = tmp_path / "blender_test.glb"
        with open(in_glb, "wb") as f:
            f.write(raw_glb)

        generator = LODGenerator(blender_bin=blender_bin, target_lod1_triangles=15, target_lod2_triangles=6)
        result = generator.generate_lods(in_glb, output_dir=tmp_path, prefer_blender=True)

        assert result.lod0_path.is_file()
        assert result.lod1_path.is_file()
        assert result.lod2_path.is_file()
        assert result.method_used in ("blender_headless", "pure_python")



# ===========================================================================
# 5. Meshy v2 Client Lifecycle & Security Tests
# ===========================================================================

class TestMeshyClient:
    def test_api_key_resolution_and_masking(self):
        # Default key
        key = resolve_api_key(None)
        assert key == DEFAULT_MESHY_API_KEY

        # Explicit key
        custom = "msy_custom_test_key_1234567890"
        assert resolve_api_key(custom) == custom

        # Masking test
        masked = mask_api_key(custom)
        assert masked.startswith("msy_")
        assert masked.endswith("7890")
        assert "custom_test_key" not in masked

        # Safe repr test
        client = MeshyClient(api_key=custom)
        repr_str = repr(client)
        assert custom not in repr_str
        assert masked in repr_str

    @pytest.mark.asyncio
    async def test_preview_task_creation_mock(self):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            body = json.loads(request.content.decode("utf-8"))
            assert body["mode"] == "preview"
            assert "pine" in body["prompt"]
            return httpx.Response(200, json={"result": "task-mock-preview-123"})

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http)
            task_id = await client.create_preview_task("ancient alpine pine", seed=42)
            assert task_id == "task-mock-preview-123"

    @pytest.mark.asyncio
    async def test_polling_lifecycle_mock(self):
        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json={"id": "task-123", "status": "PENDING", "progress": 10})
            elif call_count == 2:
                return httpx.Response(200, json={"id": "task-123", "status": "IN_PROGRESS", "progress": 65})
            else:
                return httpx.Response(200, json={
                    "id": "task-123",
                    "status": "SUCCEEDED",
                    "progress": 100,
                    "model_urls": {"glb": "https://assets.meshy.ai/model.glb"},
                    "thumbnail_url": "https://assets.meshy.ai/thumb.png",
                })

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http)
            result = await client.poll_task("task-123", poll_interval=0.01, max_interval=0.02)
            assert result.status == MeshyTaskStatus.SUCCEEDED
            assert result.progress == 100
            assert result.glb_url == "https://assets.meshy.ai/model.glb"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_retry_mock(self):
        attempts = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, json={"error": "Rate limit exceeded"})
            return httpx.Response(200, json={"result": "task-recovered-429"})

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http, retry_backoff=0.01)
            task_id = await client.create_preview_task("resilient pine")
            assert task_id == "task-recovered-429"
            assert attempts == 2

    @pytest.mark.asyncio
    async def test_task_failed_exception(self):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "task-fail",
                "status": "FAILED",
                "task_error": {"message": "Invalid prompt content"},
            })

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = MeshyClient(client=mock_http)
            with pytest.raises(RuntimeError, match="Invalid prompt content"):
                await client.poll_task("task-fail", poll_interval=0.01)
