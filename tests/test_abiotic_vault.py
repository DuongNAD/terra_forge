"""
Comprehensive Unit and Integration Tests for Milestone M1: Abiotic Generative Pipeline & Asset Vault.

Verifies:
1. AbioticType enum taxonomy (all 9 types: HornPeak, RockCrag, BasaltColumn, RiverPebble,
   RiparianSandbank, Stalactite, Stalagmite, CavernColumn, CavernMineral).
2. AbioticPromptEngine & ABIOTIC_NEGATIVE_PROMPT rejecting flora, fauna, and human structures.
3. Procedural mesh generation for all 9 abiotic types with clean manifold geometry.
4. GeometryNormalizer invariants (min_y == 0.0 bottom pivot, metric scaling, centered X/Z,
   collision primitives: cylinder, aabb, convex_hull).
5. LocalAssetVault storage, indexing, and filtering of abiotic assets.
6. Vault retrieval latency benchmarks (< 50ms for in-memory and cold-disk).
7. 3-tier LOD generation via LODGenerator.
8. End-to-end fetch_abiotic_asset round-trip caching.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from terra_forge.ai.lod import LODGenerator
from terra_forge.ai.normalizer import GeometryNormalizer, extract_mesh_from_glb
from terra_forge.ai.vault import (
    DEFAULT_ABIOTIC_PROFILES,
    AbioticMetadata,
    AbioticType,
    AssetEntry,
    AssetMetadata,
    LocalAssetVault,
    compute_asset_hash,
)
from terra_forge.blender.asset_fetcher import (
    AbioticAssetResult,
    create_abiotic_mesh,
    fetch_abiotic_asset,
    resolve_prototype_object,
)
from terra_forge.ecology.primordial import (
    ABIOTIC_NEGATIVE_PROMPT,
    ABIOTIC_POSITIVE_TEMPLATES,
    FORBIDDEN_ABIOTIC_TOKENS,
    FORBIDDEN_BIOLOGICAL_TOKENS,
    FORBIDDEN_HUMAN_TOKENS,
    AbioticBiasError,
    AbioticPromptEngine,
    ValidationResult,
    validate_abiotic_prompt,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_vault():
    """Provides an isolated clean LocalAssetVault instance in a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="tf_abiotic_vault_test_")
    vault = LocalAssetVault(vault_dir=temp_dir)
    yield vault
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def prompt_engine():
    """Provides a default AbioticPromptEngine instance."""
    return AbioticPromptEngine()


# =============================================================================
# 1. AbioticType Enum Taxonomy & Profiles
# =============================================================================

class TestAbioticTypeTaxonomy:
    """Test AbioticType enum values, names, resolution, and profiles."""

    def test_enum_canonical_values(self):
        """Verify all abiotic enum values match the authoritative specification."""
        assert AbioticType.HornPeak == 0
        assert AbioticType.RockCrag == 1
        assert AbioticType.BasaltColumn == 2
        assert AbioticType.RiverPebble == 3
        assert AbioticType.RiparianSandbank == 4
        assert AbioticType.Stalactite == 5
        assert AbioticType.Stalagmite == 6
        assert AbioticType.CavernColumn == 7
        assert AbioticType.CavernMineral == 8
        assert AbioticType.KarstArch == 9
        assert len(AbioticType) in (9, 10)

    def test_from_any_resolution(self):
        """Verify AbioticType.from_any handles enums, ints, and various string formats."""
        # Direct enum
        assert AbioticType.from_any(AbioticType.BasaltColumn) == AbioticType.BasaltColumn
        # Integer
        assert AbioticType.from_any(2) == AbioticType.BasaltColumn
        assert AbioticType.from_any(8) == AbioticType.CavernMineral
        assert AbioticType.from_any(99) is None
        # String exact
        assert AbioticType.from_any("BasaltColumn") == AbioticType.BasaltColumn
        # String case-insensitive & formatted
        assert AbioticType.from_any("basaltcolumn") == AbioticType.BasaltColumn
        assert AbioticType.from_any("basalt_column") == AbioticType.BasaltColumn
        assert AbioticType.from_any("horn_peak") == AbioticType.HornPeak
        assert AbioticType.from_any("river_pebble") == AbioticType.RiverPebble
        assert AbioticType.from_any("riparian_sandbank") == AbioticType.RiparianSandbank
        assert AbioticType.from_any("cavern_mineral") == AbioticType.CavernMineral
        # Numeric string
        assert AbioticType.from_any("0") == AbioticType.HornPeak
        assert AbioticType.from_any("invalid_xyz") is None

    def test_default_abiotic_profiles_complete(self):
        """Verify all 9 abiotic types have geological profiles defined."""
        for at in AbioticType:
            assert at in DEFAULT_ABIOTIC_PROFILES, f"Missing profile for {at.name}"
            prof = DEFAULT_ABIOTIC_PROFILES[at]
            assert "geological_name" in prof
            assert "common_name" in prof
            assert "category" in prof
            assert "typical_height" in prof
            assert "typical_radius" in prof
            assert "rock_type" in prof
            assert "solid" in prof
            assert prof["typical_height"] > 0.0
            assert prof["typical_radius"] > 0.0

    def test_abiotic_metadata_roundtrip(self):
        """Verify AbioticMetadata dataclass serialization and deserialization."""
        meta = AbioticMetadata.from_abiotic_type(AbioticType.Stalactite, overrides={"typical_height": 4.2})
        assert meta.abiotic_type == int(AbioticType.Stalactite)
        assert meta.typical_height == 4.2
        assert meta.category == "speleothem"
        assert meta.solid is True

        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["abiotic_type"] == 5
        assert d["typical_height"] == 4.2


# =============================================================================
# 2. Abiotic Prompt Engine & Negative Filter (0% Flora, 0% Fauna)
# =============================================================================

class TestAbioticPromptEngineAndFilter:
    """Test AbioticPromptEngine, ABIOTIC_NEGATIVE_PROMPT, and validation filter."""

    def test_negative_prompt_contains_mandatory_exclusions(self, prompt_engine):
        """Verify ABIOTIC_NEGATIVE_PROMPT explicitly rejects biological and human elements."""
        neg = prompt_engine.get_negative_prompt().lower()
        # Biological exclusions
        assert "tree" in neg
        assert "plant" in neg
        assert "leaf" in neg or "leaves" in neg
        assert "flower" in neg
        assert "grass" in neg
        assert "moss" in neg
        assert "lichen" in neg
        assert "shrub" in neg or "bush" in neg
        assert "fungus" in neg or "mushroom" in neg
        assert "animal" in neg
        assert "fauna" in neg
        assert "creature" in neg
        assert "bird" in neg
        assert "fish" in neg
        # Human/artifact exclusions
        assert "human" in neg
        assert "building" in neg
        assert "architecture" in neg
        assert "road" in neg
        assert "bridge" in neg
        assert "ruins" in neg
        assert "lumber" in neg or "wood plank" in neg
        assert "saw cuts" in neg

    @pytest.mark.parametrize("forbidden_token", [
        "tree", "trees", "plant", "plants", "flora", "leaf", "leaves",
        "flower", "flowers", "grass", "moss", "lichen", "shrub", "bush",
        "fungus", "mushroom", "animal", "fauna", "creature", "bird", "fish",
        "saw", "cuts", "lumber", "plank", "road", "bridge", "ruins",
        "building", "house", "cabin", "paved path", "column base", "stone wall",
    ])
    def test_forbidden_tokens_strictly_rejected(self, forbidden_token):
        """Verify validate_abiotic_prompt strictly raises AbioticBiasError on forbidden words."""
        test_prompt = f"majestic geological formation with {forbidden_token} on the surface"
        with pytest.raises(AbioticBiasError) as exc_info:
            validate_abiotic_prompt(test_prompt, strict=True)
        assert exc_info.value.detected_tokens
        assert any(forbidden_token in t for t in exc_info.value.detected_tokens)

    def test_non_strict_validation_result(self):
        """Verify non-strict validation returns ValidationResult(is_valid=False)."""
        res = validate_abiotic_prompt("granite cliff with a green tree branch", strict=False)
        assert res.is_valid is False
        assert bool(res) is False
        assert len(res.detected_tokens) >= 1

    def test_all_9_positive_templates_pass_validation(self, prompt_engine):
        """Verify all positive templates for the 9 abiotic types pass validation."""
        for at in AbioticType:
            prompt = prompt_engine.generate_prompt(at, seed=123)
            assert isinstance(prompt, str)
            assert len(prompt) > 30
            # Strict validation must pass without raising
            res = prompt_engine.validate(prompt, strict=True)
            assert res.is_valid is True
            assert bool(res) is True
            assert len(res.detected_tokens) == 0

    def test_custom_style_modifiers(self, prompt_engine):
        """Verify valid geological modifiers pass, while biological modifiers fail."""
        clean_prompt = prompt_engine.generate_prompt(
            AbioticType.HornPeak,
            style_modifiers=["frost-weathered slate strata", "raw mineral deposit"],
        )
        assert "slate strata" in clean_prompt
        assert "raw mineral deposit" in clean_prompt

        with pytest.raises(AbioticBiasError):
            prompt_engine.generate_prompt(
                AbioticType.HornPeak,
                style_modifiers=["surrounded by alpine pine trees"],
            )

    @pytest.mark.parametrize("evasion_prompt,expected_token", [
        ("column/base of the cliff", "column/base"),
        ("column.base of the cliff", "column.base"),
        ("stone/wall along river", "stone/wall"),
        ("mossy granite cliff", "mossy"),
        ("woody debris in ravine", "woody"),
        ("algae bloom in lake", "algae"),
        ("dense jungle foliage", "jungle"),
        ("coral reef shoreline", "coral"),
        ("spider web on rocks", "spider"),
        ("ancient fallen log", "log"),
        ("p.l.a.n.t growth in crevice", "plant"),
        ("t r e e on the peak", "tree"),
    ])
    def test_evasion_and_expanded_tokens_rejected(self, evasion_prompt, expected_token):
        """Verify adversarial bypass attempts and expanded tokens are strictly rejected."""
        with pytest.raises(AbioticBiasError) as exc_info:
            validate_abiotic_prompt(evasion_prompt, strict=True)
        assert any(expected_token in t for t in exc_info.value.detected_tokens)


# =============================================================================
# 3. Procedural Mesh Generation for all 9 Abiotic Types
# =============================================================================

class TestAbioticProceduralGeometry:
    """Test procedural 3D mesh generation for all 9 abiotic types."""

    @pytest.mark.parametrize("abiotic_type", list(AbioticType))
    def test_create_abiotic_mesh_clean_topology(self, abiotic_type):
        """Verify all 9 abiotic types generate non-empty, clean manifold meshes."""
        target_h = 6.0
        verts, tris = create_abiotic_mesh(abiotic_type, target_height=target_h)

        assert isinstance(verts, np.ndarray)
        assert isinstance(tris, np.ndarray)
        assert verts.dtype == np.float32
        assert tris.dtype == np.uint32
        assert len(verts) >= 8, f"{abiotic_type.name} had too few vertices: {len(verts)}"
        assert len(tris) >= 6, f"{abiotic_type.name} had too few triangles: {len(tris)}"

        # Verify triangle indices are valid
        assert np.max(tris) < len(verts), f"Index out of range for {abiotic_type.name}"
        assert np.min(tris) >= 0

        # No NaNs or Infs
        assert not np.isnan(verts).any(), f"NaNs in vertices for {abiotic_type.name}"
        assert not np.isinf(verts).any(), f"Infs in vertices for {abiotic_type.name}"

        # Geometric non-degeneracy: all triangles must have 3 distinct vertex indices
        # AND non-zero spatial surface area (> 1e-6)
        for i, tri in enumerate(tris):
            assert len(set(tri)) == 3, f"Degenerate index triangle at index {i} in {abiotic_type.name}"
            p0 = verts[tri[0]]
            p1 = verts[tri[1]]
            p2 = verts[tri[2]]
            cross_prod = np.cross(p1 - p0, p2 - p0)
            area = 0.5 * float(np.linalg.norm(cross_prod))
            assert area > 1e-6, (
                f"Degenerate zero-area triangle at index {i} in {abiotic_type.name}: "
                f"indices={tri.tolist()}, area={area:.8e}, "
                f"p0={p0.tolist()}, p1={p1.tolist()}, p2={p2.tolist()}"
            )

    def test_create_abiotic_mesh_target_height_robustness(self):
        """Verify create_abiotic_mesh handles None, negative, zero, and infinite heights cleanly."""
        for at in AbioticType:
            # None fallback
            v_none, _ = create_abiotic_mesh(at, target_height=None)
            assert len(v_none) > 0
            
            # Non-positive fallback
            v_zero, _ = create_abiotic_mesh(at, target_height=0.0)
            assert len(v_zero) > 0
            v_neg, _ = create_abiotic_mesh(at, target_height=-5.0)
            assert len(v_neg) > 0
            
            # Infinite and NaN fallback without NaNs
            v_inf, _ = create_abiotic_mesh(at, target_height=float('inf'))
            assert not np.isnan(v_inf).any() and not np.isinf(v_inf).any()
            
            v_nan, _ = create_abiotic_mesh(at, target_height=float('nan'))
            assert not np.isnan(v_nan).any() and not np.isinf(v_nan).any()

    def test_create_abiotic_mesh_seed_determinism(self):
        """Verify create_abiotic_mesh produces deterministic rotation with seed."""
        v1, _ = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=42)
        v2, _ = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=42)
        assert np.allclose(v1, v2)
        v3, _ = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=99)
        assert not np.allclose(v1, v3)

    def test_basalt_columns_polygonal_structure(self):
        """Verify basalt column cluster generates multiple prism columns."""
        verts, tris = create_abiotic_mesh(AbioticType.BasaltColumn, target_height=5.0)
        # Cluster of 7 columns should have substantial vertex and triangle count
        assert len(verts) >= 70
        assert len(tris) >= 100

    def test_speleothems_geometry(self):
        """Verify speleothem meshes: stalactite, stalagmite, and cavern column."""
        # Stalactite: conical tapering dripstone
        st_verts, st_tris = create_abiotic_mesh(AbioticType.Stalactite, target_height=4.0)
        assert len(st_verts) >= 50
        assert len(st_tris) >= 80

        # Stalagmite: floor mound
        sm_verts, sm_tris = create_abiotic_mesh(AbioticType.Stalagmite, target_height=3.0)
        assert len(sm_verts) >= 50
        assert len(sm_tris) >= 80

        # Cavern column: fused pillar
        col_verts, col_tris = create_abiotic_mesh(AbioticType.CavernColumn, target_height=8.0)
        assert len(col_verts) >= 100
        assert len(col_tris) >= 150

    def test_cavern_mineral_crystal_druse(self):
        """Verify cavern mineral generates faceted druse crystals on substrate slab."""
        verts, tris = create_abiotic_mesh(AbioticType.CavernMineral, target_height=2.0)
        assert len(verts) >= 80
        assert len(tris) >= 100

    def test_river_pebble_oblate_cluster(self):
        """Verify river pebble generates multiple flattened oblate pebbles."""
        verts, tris = create_abiotic_mesh(AbioticType.RiverPebble, target_height=0.4)
        assert len(verts) >= 200
        assert len(tris) >= 300


# =============================================================================
# 4. Geometry Normalization & Colliders Invariants
# =============================================================================

class TestAbioticNormalizationAndColliders:
    """Verify GeometryNormalizer invariants on all 9 abiotic types."""

    @pytest.mark.parametrize("abiotic_type", list(AbioticType))
    def test_normalization_invariants_for_all_types(self, abiotic_type):
        """
        Verify:
        1. Bottom pivot min_y == 0.0
        2. Metric height == target_height
        3. Horizontally centered (abs(min_x + max_x) < 1e-4, abs(min_z + max_z) < 1e-4)
        4. Colliders present: cylinder, aabb, convex_hull
        """
        target_h = 7.5
        raw_verts, tris = create_abiotic_mesh(abiotic_type, target_height=target_h)

        norm_verts, stats = GeometryNormalizer.normalize_vertices(raw_verts, target_height=target_h)
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h)

        # Invariant 1: Bottom pivot min_y == 0.0
        min_y = float(np.min(norm_verts[:, 1]))
        assert abs(min_y) < 1e-5, f"Expected min_y == 0.0 for {abiotic_type.name}, got {min_y}"

        # Invariant 2: Metric height == target_height
        max_y = float(np.max(norm_verts[:, 1]))
        assert abs(max_y - target_h) < 1e-5, f"Expected max_y == {target_h} for {abiotic_type.name}, got {max_y}"

        # Invariant 3: Centered X and Z
        min_x = float(np.min(norm_verts[:, 0]))
        max_x = float(np.max(norm_verts[:, 0]))
        min_z = float(np.min(norm_verts[:, 2]))
        max_z = float(np.max(norm_verts[:, 2]))
        assert abs(min_x + max_x) < 1e-4, f"X not centered for {abiotic_type.name}: {min_x}, {max_x}"
        assert abs(min_z + max_z) < 1e-4, f"Z not centered for {abiotic_type.name}: {min_z}, {max_z}"

        # Invariant 4: Colliders present and valid
        assert colliders.cylinder is not None
        assert colliders.cylinder.radius > 0.0
        assert colliders.cylinder.height == target_h
        assert colliders.cylinder.center == [0.0, target_h / 2.0, 0.0]

        assert colliders.aabb is not None
        assert colliders.aabb.min_vec[1] == 0.0
        assert colliders.aabb.max_vec[1] == target_h
        assert colliders.aabb.size[0] > 0.0
        assert colliders.aabb.size[2] > 0.0

        assert colliders.convex_hull is not None
        assert colliders.convex_hull.vertex_count >= 4
        assert colliders.convex_hull.simplex_count >= 4
        assert colliders.convex_hull.volume > 0.0
        assert colliders.convex_hull.area > 0.0

    def test_extract_collision_primitives_small_asset_clamping(self):
        """Verify cylinder radius on small assets never exceeds bounding extent."""
        for h in [0.001, 0.01, 0.05, 0.1]:
            verts, tris = create_abiotic_mesh(AbioticType.RiverPebble, target_height=h)
            norm_v, _ = GeometryNormalizer.normalize_vertices(verts, target_height=h)
            colliders = GeometryNormalizer.extract_collision_primitives(norm_v, target_height=h)
            max_ext = max(abs(colliders.aabb.min_vec[0]), abs(colliders.aabb.max_vec[0]),
                          abs(colliders.aabb.min_vec[2]), abs(colliders.aabb.max_vec[2]))
            assert colliders.cylinder.radius <= max_ext + 1e-4
            assert colliders.cylinder.radius > 0.0


# =============================================================================
# 5. Local Asset Vault Storage & Performance Benchmarks
# =============================================================================

class TestAbioticVaultStorageAndPerformance:
    """Test LocalAssetVault storing, indexing, filtering, and retrieval latency."""

    def test_abiotic_asset_storage_and_metadata(self, temp_vault):
        """Verify storing an abiotic asset records abiotic_type and abiotic_metadata."""
        target_h = 6.0
        res = fetch_abiotic_asset(AbioticType.BasaltColumn, target_height=target_h, vault=temp_vault, save_to_vault=True)

        assert res.abiotic_type == AbioticType.BasaltColumn
        assert res.model_path is not None
        assert res.model_path.is_file()
        assert len(temp_vault) == 1

        # Check metadata file
        meta_file = res.model_path.parent / "metadata.json"
        assert meta_file.is_file()
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["abiotic_type"] == int(AbioticType.BasaltColumn)
        assert data["abiotic_metadata"] is not None
        assert data["abiotic_metadata"]["category"] == "igneous"
        assert data["collision_primitives"]["cylinder"]["radius"] > 0.0

    def test_abiotic_vault_filtering(self, temp_vault):
        """Verify LocalAssetVault filters by abiotic_type and separates flora and abiotic assets."""
        fetch_abiotic_asset(AbioticType.HornPeak, vault=temp_vault, save_to_vault=True)
        fetch_abiotic_asset(AbioticType.BasaltColumn, vault=temp_vault, save_to_vault=True)
        fetch_abiotic_asset(AbioticType.CavernMineral, vault=temp_vault, save_to_vault=True)

        assert len(temp_vault) == 3
        # Filter by specific abiotic type
        horn_assets = temp_vault.list_assets(abiotic_type=AbioticType.HornPeak)
        assert len(horn_assets) == 1
        assert horn_assets[0].metadata.abiotic_type == int(AbioticType.HornPeak)

        basalt_assets = temp_vault.list_assets(abiotic_type=AbioticType.BasaltColumn)
        assert len(basalt_assets) == 1

        # Flora filter should return empty for abiotic assets
        assert len(temp_vault.list_assets(flora_type=0)) == 0

    def test_in_memory_lookup_latency_under_50ms(self, temp_vault):
        """Benchmark: In-memory lookup latency must be well under 50ms requirement."""
        res = fetch_abiotic_asset(AbioticType.RockCrag, vault=temp_vault, save_to_vault=True)
        asset_id = res.model_path.parent.name

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            entry = temp_vault.get(asset_id)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

        avg_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))

        # Must be well below 50ms requirement (typically < 0.1ms)
        assert avg_latency < 5.0, f"Average in-memory latency too high: {avg_latency:.4f}ms"
        assert max_latency < 50.0, f"Max in-memory latency exceeded 50ms: {max_latency:.4f}ms"

    def test_cold_disk_lookup_latency_under_50ms(self, temp_vault):
        """Benchmark: Cold disk retrieval (in-memory index cleared) must be under 50ms."""
        res = fetch_abiotic_asset(AbioticType.Stalactite, vault=temp_vault, save_to_vault=True)
        asset_id = res.model_path.parent.name

        latencies = []
        for _ in range(10):
            temp_vault.clear_cache()
            assert len(temp_vault._index) == 0

            t0 = time.perf_counter()
            entry = temp_vault.get(asset_id)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)
            assert entry is not None

        avg_cold_latency = float(np.mean(latencies))
        assert avg_cold_latency < 50.0, f"Cold disk latency exceeded 50ms: {avg_cold_latency:.4f}ms"

    def test_lod_generator_creates_3_tiers(self, temp_vault):
        """Verify LODGenerator produces LOD0, LOD1, and LOD2 GLB files."""
        res = fetch_abiotic_asset(AbioticType.RiverPebble, target_height=0.5, vault=temp_vault, save_to_vault=True)
        lod_gen = LODGenerator()

        lod_res = lod_gen.generate_lods(res.model_path, output_dir=res.model_path.parent)
        assert lod_res.lod0_path.is_file()
        assert lod_res.lod1_path.is_file()
        assert lod_res.lod2_path.is_file()

        # Check GLB magic bytes on all tiers
        for p in [lod_res.lod0_path, lod_res.lod1_path, lod_res.lod2_path]:
            with open(p, "rb") as f:
                header = f.read(4)
            assert header == b"glTF", f"Invalid glTF header in {p}"


# =============================================================================
# 6. End-to-End fetch_abiotic_asset Round-Trip
# =============================================================================

class TestFetchAbioticAssetIntegration:
    """Test fetch_abiotic_asset caching and prototype resolution."""

    def test_fetch_abiotic_asset_vault_roundtrip(self, temp_vault):
        """Verify cold fetch procedurally creates & stores, while second fetch hits cache."""
        at = AbioticType.CavernColumn
        target_h = 8.0

        # Pass 1: Cold fetch -> creates procedural normalized asset and stores into vault
        assert len(temp_vault) == 0
        res1 = fetch_abiotic_asset(at, target_height=target_h, vault=temp_vault, save_to_vault=True)

        assert res1.source == "procedural_normalized"
        assert res1.is_normalized is True
        assert res1.target_height == target_h
        assert res1.model_path is not None
        assert res1.model_path.is_file()
        assert len(temp_vault) == 1

        # Pass 2: Cache hit -> loads from vault
        res2 = fetch_abiotic_asset(at, target_height=target_h, vault=temp_vault)
        assert res2.source == "vault"
        assert res2.model_path == res1.model_path
        assert res2.is_normalized is True
        assert res2.target_height == target_h

    def test_resolve_prototype_object_abiotic(self, temp_vault):
        """Verify resolve_prototype_object handles abiotic prototype names."""
        for at in AbioticType:
            obj = resolve_prototype_object(at.name.lower(), vault=temp_vault)
            assert obj is not None
            assert hasattr(obj, "data")
            assert hasattr(obj, "name")
            assert "Proto_" in obj.name
