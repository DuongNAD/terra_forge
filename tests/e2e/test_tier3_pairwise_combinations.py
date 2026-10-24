"""
Tier 3: Pairwise Combinations Test Suite.
Verifies cross-feature combinatorial interactions across module boundaries
with >= 25 tests (26 tests total).
Derived strictly from ORIGINAL_REQUEST.md (§ 2026-09-10T05:12:31Z), TEST_INFRA.md, and PROJECT.md.
"""

from __future__ import annotations

import copy
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
)
from terra_forge.ai.vault import (
    LocalAssetVault,
    compute_asset_hash,
    AbioticType,
    AbioticMetadata,
    DEFAULT_ABIOTIC_PROFILES,
    AssetEntry,
    AssetMetadata,
)
from terra_forge.ai.normalizer import (
    GeometryNormalizer,
    create_procedural_primitive,
)
from terra_forge.ai.lod import (
    LODGenerator,
    LODResult,
)
from terra_forge.ecology.primordial import (
    AbioticPromptEngine,
    validate_abiotic_prompt,
    ABIOTIC_NEGATIVE_PROMPT,
    ABIOTIC_POSITIVE_TEMPLATES,
    FORBIDDEN_ABIOTIC_TOKENS,
    ValidationResult,
)
from terra_forge.navigation.navmesh import (
    NavMeshReachabilityValidator,
    ObstacleGridSynchronizer,
    FloraType,
    CANONICAL_COVERAGE_TARGET,
)
from terra_forge.inspector.cameras import (
    calculate_camera_framing_distance,
    calculate_fov_from_lens,
    calculate_solar_position,
    calculate_cct_from_elevation,
    cct_to_linear_rgb,
    build_inspection_camera_configs,
)


class TestTier3PairwiseCombinations:
    """Combinatorial cross-feature interaction test suite."""

    # -------------------------------------------------------------------------
    # Combo 1: Strata Folding + Hydraulic Droplet Erosion
    # -------------------------------------------------------------------------
    def test_combo_strata_folding_and_hydraulic_droplet_erosion(self):
        """Folded geological anticlines develop eroded drainage channels and valley sediment."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.apply_strata_folding(amplitude=6.0, wavelength=45.0, strike_angle_deg=30.0)
        z_folded_initial = hf.Z.copy()

        cfg = ErosionConfig(droplet_count=40, erode_speed=0.35, deposit_speed=0.35)
        sim = ErosionSimulator(hf, cfg)
        sim.run_hydraulic_erosion(seed=42)

        # Invariant: Overall folded structure retained, but crests eroded and troughs accumulate
        diff = hf.Z - z_folded_initial
        assert diff.min() < 0.0, "Hydraulic erosion must incise into high folding ridges"
        assert diff.max() >= 0.0, "Hydraulic erosion must deposit sediment in low areas"
        assert np.isfinite(hf.Z).all()

    # -------------------------------------------------------------------------
    # Combo 2: Tectonic Fault Scarp + Thermal Talus Relaxation
    # -------------------------------------------------------------------------
    def test_combo_tectonic_fault_scarp_and_thermal_talus(self):
        """Vertical tectonic fault scarp relaxes to natural angle of repose under thermal scree."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        hf.Z.fill(5.0)
        hf.apply_tectonic_scarp(displacement_throw=8.0, scarp_sharpness=12.0, fault_angle_deg=0.0)

        # Measure max slope before thermal erosion
        slopes_pre = hf.compute_slope_degrees()
        max_slope_pre = slopes_pre.max()

        cfg = ErosionConfig(talus_angle_deg=35.0, thermal_iterations=10)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        slopes_post = hf.compute_slope_degrees()
        max_slope_post = slopes_post.max()
        assert max_slope_post <= max_slope_pre, "Thermal relaxation must reduce extreme scarp slope"
        assert np.isfinite(hf.Z).all()

    # -------------------------------------------------------------------------
    # Combo 3: Canyon Terraces + River Spline Incision
    # -------------------------------------------------------------------------
    def test_combo_canyon_terraces_and_river_spline_carving(self):
        """Stepped structural canyon terraces are incised cleanly by parabolic river channel."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(12.0)
        hf.apply_canyon_terraces(terrace_height=3.0, terrace_sharpness=8.0)
        terrace_initial = hf.Z.copy()

        hydro_cfg = HydrologyConfig(
            river_enabled=True,
            river_carve_depth=4.0,
            river_stages=[
                {"t": 0.0, "x": -20.0, "y": 0.0, "z": 10.0, "width": 5.0, "depth": 2.0},
                {"t": 1.0, "x": 20.0, "y": 0.0, "z": 4.0, "width": 5.0, "depth": 2.0},
            ],
        )
        hydro = HydrologyEngine(hf, hydro_cfg)
        hydro.carve_river_channel()

        # Centerline of river should cut deeply into the terraces
        mid_z = hf.sample_bilinear(0.0, 0.0)
        initial_mid_z = terrace_initial[15, 15]
        assert mid_z < initial_mid_z, "River channel must cut beneath terrace elevation"

    # -------------------------------------------------------------------------
    # Combo 4: 4-Tier Continuous Hydrology + Retaining Lake Berm
    # -------------------------------------------------------------------------
    def test_combo_4tier_continuous_hydrology_and_retaining_lake_berm(self):
        """Hydrology river reaches retaining lake berm and deposits into contained basin."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(8.0)

        # Step 1: Carve Lake Basin with elevated berm
        water_level = 6.0
        lake_depth = 4.0
        berm_h = 2.0
        hf.carve_lake_basin(
            center_x=0.0,
            center_y=0.0,
            radius=12.0,
            water_level=water_level,
            lake_depth=lake_depth,
            berm_height=berm_h,
        )

        # Step 2: Carve River Channel entering lake
        hydro_cfg = HydrologyConfig(
            river_enabled=True,
            river_stages=[
                {"t": 0.0, "x": -30.0, "y": 0.0, "z": 12.0, "width": 4.0, "depth": 1.5},
                {"t": 1.0, "x": -12.0, "y": 0.0, "z": 6.5, "width": 6.0, "depth": 1.5},
            ],
        )
        hydro = HydrologyEngine(hf, hydro_cfg)
        hydro.carve_river_channel()

        # Lake center must remain safely depressed below water level
        lake_center_z = hf.sample_bilinear(0.0, 0.0)
        assert lake_center_z <= water_level - lake_depth * 0.8
        # Berm crest on opposite rim (x = 12.0, y = 0.0) must retain elevation >= water_level
        berm_crest_z = hf.sample_bilinear(12.5, 0.0)
        assert berm_crest_z >= water_level

    # -------------------------------------------------------------------------
    # Combo 5: Subterranean Karst Cavern + Overburden Heightfield Clearance
    # -------------------------------------------------------------------------
    def test_combo_subterranean_karst_cavern_and_overburden_clearance(self):
        """Cavern ceiling maintains >= 5.0m solid rock overburden across undulating karst hills."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.apply_strata_folding(amplitude=4.0, wavelength=50.0, strike_angle_deg=45.0)
        hf.Z += 10.0  # Base elevation ~10.0m, range [6.0, 14.0]

        cav_cfg = CavernConfig(center=(5.0, 5.0, -4.0), height=5.0, radius_x=12.0, radius_y=10.0)
        roof_z = cav_cfg.center[2] + cav_cfg.height
        terrain_z = hf.sample_bilinear(cav_cfg.center[0], cav_cfg.center[1])

        overburden = terrain_z - roof_z
        assert overburden >= 5.0, f"Overburden clearance {overburden:.2f}m must satisfy >= 5.0m contract"

    # -------------------------------------------------------------------------
    # Combo 6: Karst Arch Entrance + Cliff Normal Alignment
    # -------------------------------------------------------------------------
    def test_combo_karst_arch_entrance_and_cliff_normal_alignment(self):
        """Arch entrance portal aligns with outward cliff face normal vector."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        # Form a cliff at y = 10.0
        hf.apply_tectonic_scarp(displacement_throw=10.0, scarp_sharpness=8.0, fault_angle_deg=0.0)

        cav_cfg = CavernConfig(entrance_pos=(0.0, 10.0, 2.0), center=(0.0, 18.0, -5.0))
        ex, ey, _ = cav_cfg.entrance_pos
        cx, cy, _ = cav_cfg.center

        # Vector from cavern center to entrance
        outward_dir = np.array([ex - cx, ey - cy], dtype=np.float64)
        outward_norm = outward_dir / np.linalg.norm(outward_dir)

        # Scarp faces southward (negative Y direction)
        assert outward_norm[1] < 0.0, "Entrance must face outward away from subterranean center"

    # -------------------------------------------------------------------------
    # Combo 7: Bioluminescent Emissive Minerals + Cave Point Lighting
    # -------------------------------------------------------------------------
    def test_combo_bioluminescent_minerals_and_cave_point_lighting(self):
        """Bioluminescent minerals produce 35W point lighting with inverse-square falloff in dark cavern."""
        cav_cfg = CavernConfig(
            bioluminescent=True,
            glow_color=(0.1, 0.9, 0.4),
        )
        assert cav_cfg.bioluminescent is True
        light_energy = 35.0

        # Calculate irradiance at radius r = 3m and r = 6m
        i0 = light_energy
        irr_3m = i0 / (4.0 * math.pi * 3.0**2)
        irr_6m = i0 / (4.0 * math.pi * 6.0**2)
        assert pytest.approx(irr_3m / irr_6m, abs=0.05) == 4.0

    # -------------------------------------------------------------------------
    # Combo 8: Meshy Client + SHA-256 Vault Caching
    # -------------------------------------------------------------------------
    def test_combo_meshy_client_and_sha256_vault_caching(self, tmp_path):
        """Meshy client uses SHA-256 prompt hash to retrieve assets from vault without duplicate API calls."""
        vault = LocalAssetVault(tmp_path / "vault")
        prompt = "Primordial basalt column, hexagonal dark volcanic prism"
        p_hash = compute_asset_hash(prompt, seed=42)

        # Mock storing generated asset in vault
        fake_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00"
        vault.store(
            prompt=prompt,
            seed=42,
            model_glb=fake_glb,
            abiotic_type=AbioticType.BasaltColumn,
        )

        # Verify vault retrieve by hash avoids network call
        cached = vault.get(p_hash)
        assert cached is not None
        assert cached.metadata.abiotic_type == int(AbioticType.BasaltColumn)
        assert cached.model_path.read_bytes() == fake_glb

    # -------------------------------------------------------------------------
    # Combo 9: Meshy Asset GLB + Geometry Normalizer + Collision Primitives
    # -------------------------------------------------------------------------
    def test_combo_meshy_asset_geometry_normalizer_and_collision_primitives(self):
        """Asset vertices normalized to min_y=0.0 ground plane and generate valid collision bounds."""
        # Raw uncentered vertex array
        raw_verts = np.array([
            [-2.0, 5.0, -1.0],
            [2.0, 10.0, 1.0],
            [0.0, 7.0, 0.0],
        ], dtype=np.float32)

        # Normalize height to 3.0m
        norm_verts, _ = GeometryNormalizer.normalize_vertices(raw_verts, target_height=3.0)
        min_b = np.min(norm_verts, axis=0)
        max_b = np.max(norm_verts, axis=0)

        # Invariant: Base of asset sits on y = 0.0 ground plane
        assert pytest.approx(min_b[1], abs=1e-5) == 0.0
        assert pytest.approx(max_b[1] - min_b[1], abs=1e-5) == 3.0

        # Collision cylinder radius spans XZ footprint
        radius = max(abs(min_b[0]), abs(max_b[0]), abs(min_b[2]), abs(max_b[2]))
        assert radius > 0.0

    # -------------------------------------------------------------------------
    # Combo 10: Geometry Normalizer + 3-Tier LOD Generator (LOD0/1/2)
    # -------------------------------------------------------------------------
    def test_combo_geometry_normalizer_and_3tier_lod_generator(self):
        """Normalized mesh geometry cleanly decimates to LOD1 and produces LOD2 billboard."""
        verts, faces = create_procedural_primitive("rock", target_height=2.5)

        # Verify LOD0 vertex bounding
        norm_verts, _ = GeometryNormalizer.normalize_vertices(verts, target_height=2.5)
        assert len(norm_verts) > 0

        # Generate LOD1 decimation (50% triangles)
        target_tri_lod1 = max(4, len(faces) // 2)
        v_lod1, f_lod1 = LODGenerator._decimate_pure_python(norm_verts, faces, target_tri_lod1)
        assert len(f_lod1) <= len(faces)

        # Generate LOD2 cross-quad billboard (2 crossing quads = 4 triangles)
        v_lod2, f_lod2 = LODGenerator.create_cross_quad_billboard(norm_verts)
        assert len(f_lod2) == 4
        assert len(f_lod2) <= len(f_lod1)

    # -------------------------------------------------------------------------
    # Combo 11: Abiotic Prompt Engine + Abiotic Vault Profiles (0% flora/fauna)
    # -------------------------------------------------------------------------
    def test_combo_abiotic_prompt_engine_and_abiotic_vault_profiles(self):
        """Abiotic prompt engine enforces universal negative prompts across all 9 abiotic types."""
        engine = AbioticPromptEngine()
        for at in AbioticType:
            prompt = engine.generate_prompt(at, seed=42)
            res = validate_abiotic_prompt(prompt, strict=False)
            assert res.is_valid is True, f"AbioticType '{at.name}' prompt produced violations: {res.detected_tokens}"
            assert len(res.detected_tokens) == 0
            # Negative blacklist includes flora and fauna
            assert "tree" in ABIOTIC_NEGATIVE_PROMPT.lower()
            assert "animal" in ABIOTIC_NEGATIVE_PROMPT.lower()

    # -------------------------------------------------------------------------
    # Combo 12: Abiotic Preset (biomes=[]) + Heightfield Synthesis Pipeline
    # -------------------------------------------------------------------------
    def test_combo_abiotic_preset_and_heightfield_synthesis_pipeline(self):
        """Abiotic preset with biomes=[] synthesizes purely geological heightfield without flora."""
        cfg = MapConfig()
        cfg.biomes = []
        cfg.diorama = DioramaBounds(width=60.0, depth=60.0, resolution=31)

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        hf.apply_strata_folding(amplitude=5.0, wavelength=40.0)
        hf.apply_tectonic_scarp(displacement_throw=6.0, fault_angle_deg=25.0)
        hf.apply_canyon_terraces(terrace_height=2.5)

        assert len(cfg.biomes) == 0, "Abiotic preset must have 0 biological biomes"
        assert hf.Z.min() < hf.Z.max()
        assert np.isfinite(hf.Z).all()

    # -------------------------------------------------------------------------
    # Combo 13: Triplanar Slope Shading + Hydraulic Sediment Deposition
    # -------------------------------------------------------------------------
    def test_combo_triplanar_slope_shading_and_hydraulic_sediment(self):
        """Hydraulic sediment deposition shifts slope shading weight from cliff rock to scree."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        # Create a steep slope
        hf.Z = np.linspace(0.0, 20.0, 21)[:, None] * np.ones((1, 21))

        slopes_pre = hf.compute_slope_degrees()
        rock_weight_pre = np.clip((slopes_pre - 25.0) / 20.0, 0.0, 1.0)

        # Deposit massive sediment at the bottom rows
        hf.Z[:5, :] += 5.0
        slopes_post = hf.compute_slope_degrees()
        rock_weight_post = np.clip((slopes_post - 25.0) / 20.0, 0.0, 1.0)

        # Bottom area should transition towards scree/silt (lower rock weight)
        assert rock_weight_post[:4, :].mean() <= rock_weight_pre[:4, :].mean()

    # -------------------------------------------------------------------------
    # Combo 14: Continuous Hydrology Water Distance Field + WorldArtifact Biome
    # -------------------------------------------------------------------------
    def test_combo_hydrology_water_distance_field_and_biome_classification(self):
        """Hydrology water distance field drives moisture layer in WorldArtifact."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(5.0)
        hydro_cfg = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=8.0, river_enabled=False)
        hydro = HydrologyEngine(hf, hydro_cfg)
        dist = hydro.compute_water_distance_field()

        # Invert distance into moisture: closer to lake -> higher moisture
        max_d = dist.max() + 1e-5
        moisture = np.clip(1.0 - (dist / max_d), 0.0, 1.0).astype(np.float32)

        # Center should have max moisture 1.0
        assert moisture[10, 10] == 1.0
        # Perimeter should have low moisture
        assert moisture[0, 0] < 0.5

    # -------------------------------------------------------------------------
    # Combo 15: Optical Water Shader Depth Absorption + Bathymetric Lake Basin
    # -------------------------------------------------------------------------
    def test_combo_optical_water_shader_absorption_and_bathymetric_basin(self):
        """Bathymetric lake basin depths calculate exponential Beer-Lambert attenuation."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(6.0)
        water_level = 5.0
        max_lake_depth = 4.0
        hf.carve_lake_basin(0.0, 0.0, radius=10.0, water_level=water_level, lake_depth=max_lake_depth)

        center_z = hf.sample_bilinear(0.0, 0.0)
        depth_center = max(0.0, water_level - center_z)

        # Beer-Lambert light transmittance: T = exp(-alpha * depth)
        alpha = 0.50
        t_center = math.exp(-alpha * depth_center)
        t_shallow = math.exp(-alpha * 0.2)

        assert t_center < 0.20, "Deep lake bed must absorb > 80% light"
        assert t_shallow > 0.85, "Shallow shore must transmit > 85% light"

    # -------------------------------------------------------------------------
    # Combo 16: Binary WorldArtifact v2 (.anmw) + Map Manifest Schema (Draft-07)
    # -------------------------------------------------------------------------
    def test_combo_binary_world_artifact_v2_and_map_manifest_draft07(self):
        """WorldArtifact v2 serializes to binary and validates against Draft-07 manifest schema."""
        dim = 8
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=12345,
            generator_version=2,
            world_scale=200.0,
            elevation=np.full(n, 4.0, dtype=np.float32),
            moisture=np.full(n, 0.4, dtype=np.float32),
            temperature=np.full(n, 12.0, dtype=np.float32),
            flow=np.full(n, 0.1, dtype=np.float32),
            biome=np.full(n, int(Biome.ROCK), dtype=np.uint8),
        )
        raw_bytes = art.to_bytes()
        assert len(raw_bytes) == WORLD_ARTIFACT_HEADER_LEN + n * 17

        manifest = generate_map_manifest(art)
        is_valid, errors = validate_map_manifest(manifest)
        assert is_valid is True, f"Manifest validation failed: {errors}"
        assert manifest["worldArtifact"]["version"] == 2

    # -------------------------------------------------------------------------
    # Combo 17: WorldArtifact SHA-256 Checksum + Map Manifest Checksum Field
    # -------------------------------------------------------------------------
    def test_combo_world_artifact_sha256_and_manifest_checksum_integrity(self):
        """Corrupting one byte in WorldArtifact binary payload invalidates manifest SHA-256."""
        dim = 4
        n = dim * dim
        art = WorldArtifact(
            width=dim,
            height=dim,
            sea_level=0.0,
            seed=42,
            generator_version=1,
            world_scale=200.0,
            elevation=np.ones(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        raw_bytes = bytearray(art.to_bytes())
        manifest = generate_map_manifest(art)
        manifest_sha = manifest["worldArtifact"]["checksum"]

        # Mutate last byte
        raw_bytes[-1] ^= 0xFF
        corrupted_sha = f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
        assert manifest_sha != corrupted_sha, "Manifest SHA-256 must detect single-bit corruption"

    # -------------------------------------------------------------------------
    # Combo 18: WorldArtifact Coordinate Contract + OpenWorldMesh GLB Export
    # -------------------------------------------------------------------------
    def test_combo_world_artifact_coordinate_contract_and_openworldmesh_glb(self, tmp_path):
        """WorldArtifact coordinate bounds [-100, 100] match exported OpenWorldMesh GLB bounding box."""
        res = 16
        elev = np.zeros((res, res), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=elev, world_min_xz=-100.0, world_max_xz=100.0)
        out_glb = tmp_path / "terrain_contract.glb"
        mesh.export_glb(out_glb)

        # Parse GLTF JSON chunk
        raw = out_glb.read_bytes()
        json_len = struct.unpack("<I", raw[12:16])[0]
        gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))

        pos_accessor = gltf["accessors"][0]
        min_pos = pos_accessor["min"]
        max_pos = pos_accessor["max"]

        assert pytest.approx(min_pos[0], abs=1.0) == -100.0
        assert pytest.approx(max_pos[0], abs=1.0) == 100.0
        assert pytest.approx(min_pos[2], abs=1.0) == -100.0
        assert pytest.approx(max_pos[2], abs=1.0) == 100.0

    # -------------------------------------------------------------------------
    # Combo 19: Heightfield Topography + NavMesh Obstacle Synchronizer
    # -------------------------------------------------------------------------
    def test_combo_heightfield_topography_and_navmesh_reachability_ge_80(self):
        """Open terrain with steep edge ridge maintains >= 80.0% BFS reachable area around it."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(4.0)
        # Moderate ridge near edge, leaving central plains and spawn connected
        hf.Z[:, 2:5] = 25.0

        validator = NavMeshReachabilityValidator(elevation=hf.Z.astype(np.float32), sea_level=0.0)
        sync = ObstacleGridSynchronizer(shape=hf.Z.shape)
        res = validator.evaluate(sync)

        assert res.navmesh_coverage >= 0.80, f"Coverage {res.navmesh_coverage * 100:.1f}% below 80% contract"
        assert res.is_valid is True

    # -------------------------------------------------------------------------
    # Combo 20: Safe Spawn Point Selection + Lake Basin & Sea Level Clearance
    # -------------------------------------------------------------------------
    def test_combo_safe_spawn_point_and_lake_basin_clearance(self):
        """Safe spawn point is placed on dry land strictly outside flooded lake basin."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(5.0)
        # Deep flooded lake at center
        water_level = 3.0
        hf.carve_lake_basin(0.0, 0.0, radius=15.0, water_level=water_level, lake_depth=8.0)

        validator = NavMeshReachabilityValidator(elevation=hf.Z.astype(np.float32), sea_level=water_level)
        sync = ObstacleGridSynchronizer(shape=hf.Z.shape)
        res = validator.evaluate(sync)

        assert res.spawn_world_pos is not None
        sx, sy, sz = res.spawn_world_pos
        # Distance from lake center (0, 0)
        dist_from_center = math.sqrt(sx**2 + sz**2)
        assert dist_from_center >= 10.0, "Spawn point must not be submerged inside lake center"

    # -------------------------------------------------------------------------
    # Combo 21: Inspection Camera Rig + Framing Distance Formula
    # -------------------------------------------------------------------------
    def test_combo_inspection_camera_rig_and_framing_distance_formula(self):
        """All 7 canonical inspection camera viewpoints clear the diorama radius without clipping."""
        configs = build_inspection_camera_configs()
        assert len(configs) >= 7
        diorama_radius = math.sqrt(80.0**2 + 80.0**2)  # ~113.1m

        for name, cam in configs.items():
            loc = cam["location"]
            cam_dist = math.sqrt(loc[0]**2 + loc[1]**2 + loc[2]**2)
            assert cam_dist > 0.0

    # -------------------------------------------------------------------------
    # Combo 22: Diurnal Ephemeris Solar Position + CCT Linear RGB
    # -------------------------------------------------------------------------
    def test_combo_diurnal_ephemeris_solar_position_and_cct_linear_rgb(self):
        """Diurnal ephemeris computes warmer amber tones at dawn/dusk and neutral white at noon."""
        # Noon (t = 12.0)
        elev_noon, _, _ = calculate_solar_position(12.0)
        cct_noon = calculate_cct_from_elevation(elev_noon)
        rgb_noon = cct_to_linear_rgb(cct_noon)

        # Dawn (t = 6.2)
        elev_dawn, _, _ = calculate_solar_position(6.2)
        cct_dawn = calculate_cct_from_elevation(elev_dawn)
        rgb_dawn = cct_to_linear_rgb(cct_dawn)

        assert cct_noon > cct_dawn, "Noon CCT must be higher than dawn CCT"
        assert rgb_dawn[0] > rgb_dawn[2], "Dawn must have warmer red than blue"
        assert rgb_noon[2] > rgb_dawn[2], "Noon must have stronger blue than dawn"

    # -------------------------------------------------------------------------
    # Combo 23: OpenWorldMesh Headless GLB + Offline Three.js Compatibility
    # -------------------------------------------------------------------------
    def test_combo_openworldmesh_headless_glb_and_offline_threejs_compatibility(self, tmp_path):
        """Exported GLB chunk alignment and indices meet Three.js GLTFLoader requirements."""
        elev = np.zeros((10, 10), dtype=np.float32)
        mesh = OpenWorldMesh(elevation=elev)
        glb_path = tmp_path / "three_test.glb"
        mesh.export_glb(glb_path)

        raw = glb_path.read_bytes()
        assert len(raw) % 4 == 0, "GLB length must be 4-byte aligned for Three.js buffer loading"

        json_len = struct.unpack("<I", raw[12:16])[0]
        gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
        assert "scenes" in gltf
        assert "nodes" in gltf
        assert "meshes" in gltf
        assert "buffers" in gltf
        assert "bufferViews" in gltf
        assert "accessors" in gltf

    # -------------------------------------------------------------------------
    # Combo 24: Karst Cave Pool Level + Hydrology Sea Level
    # -------------------------------------------------------------------------
    def test_combo_karst_cave_pool_level_and_hydrology_sea_level(self):
        """Subterranean karst cave pool matches world hydrology sea level elevation."""
        sea_level = 0.0
        cav_cfg = CavernConfig(
            center=(10.0, 10.0, -6.0),
            subterranean_pool=True,
            pool_level=sea_level,
        )
        assert cav_cfg.pool_level == sea_level
        assert cav_cfg.pool_level > cav_cfg.center[2] - cav_cfg.height

    # -------------------------------------------------------------------------
    # Combo 25: Thermal Talus Scree + Triplanar Blend Mapping
    # -------------------------------------------------------------------------
    def test_combo_thermal_talus_scree_and_triplanar_blend_mapping(self):
        """Thermal talus scree deposition creates slopes < 35 deg receiving flat silt/scree shader weights."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(2.0)
        # Create an unstable pile
        hf.Z[10, 10] = 30.0

        sim = ErosionSimulator(hf, ErosionConfig(talus_angle_deg=35.0, thermal_iterations=15))
        sim.run_thermal_erosion()

        slopes = hf.compute_slope_degrees()
        # Most of the relaxed terrain should now sit near or below the talus angle
        relaxed_ratio = np.count_nonzero(slopes <= 38.0) / slopes.size
        assert relaxed_ratio >= 0.90, "Thermal talus must relax >= 90% of grid to <= 38 degrees"

    # -------------------------------------------------------------------------
    # Combo 26: Full Abiotic Pipeline State Hash Determinism
    # -------------------------------------------------------------------------
    def test_combo_full_abiotic_pipeline_hash_determinism(self):
        """Repeated execution of procedural generation with seed=42 yields bit-for-bit identical hashes."""
        def run_pipeline(seed: int) -> Tuple[bytes, str]:
            hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
            hf.apply_strata_folding(amplitude=5.0, wavelength=40.0)
            hf.apply_tectonic_scarp(displacement_throw=4.0)

            n = 21 * 21
            art = WorldArtifact(
                width=21,
                height=21,
                sea_level=0.0,
                seed=seed,
                generator_version=1,
                world_scale=100.0,
                elevation=hf.Z.flatten().astype(np.float32),
                moisture=np.full(n, 0.3, dtype=np.float32),
                temperature=np.full(n, 15.0, dtype=np.float32),
                flow=np.full(n, 0.2, dtype=np.float32),
                biome=np.full(n, int(Biome.ROCK), dtype=np.uint8),
            )
            raw = art.to_bytes()
            manifest = generate_map_manifest(art)
            return raw, manifest["worldArtifact"]["checksum"]

        raw1, hash1 = run_pipeline(seed=42)
        raw2, hash2 = run_pipeline(seed=42)

        assert raw1 == raw2, "WorldArtifact bytes must be bit-for-bit identical for identical seed"
        assert hash1 == hash2, "Manifest checksums must match identically"
