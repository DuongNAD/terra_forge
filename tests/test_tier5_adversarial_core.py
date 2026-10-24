"""
Tier 5 Adversarial Coverage Hardening Suite — Core Engine Modules.

Targeted Modules:
1. terra_forge.core.heightfield (Heightfield2D: extreme resolutions, boundary interpolation, peaks, basins, strata, terraces, fault scarps, coastal profiles)
2. terra_forge.core.erosion (ErosionSimulator: momentum droplet mass conservation |ΔM| < 1e-10, small grid margin collapse, 8-neighbor isotropic thermal talus relaxation)
3. terra_forge.core.hydrology (HydrologyEngine, RiverSplineScalar: arithmetic contracts, zero-division guards, Hermite tangents, flow velocity, distance fields)
4. terra_forge.core.world_artifact (WorldArtifact, Biome: corrupted headers, zero cells, truncation, bit-flip detection, seed peeking, 22 canonical biomes, downsampling)
5. terra_forge.core.manifest (generate_map_manifest, validate_map_manifest, sync_manifest_model_asset: Draft-07 schema stress, GLB synchronization, camera FOVs)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    ErosionConfig,
    HydrologyConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    PeakConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar
from terra_forge.core.world_artifact import (
    WorldArtifact,
    Biome,
    LegacyBiome,
    BIOME_COUNT,
    LEGACY_BIOME_COUNT,
    BIOME_NAMES_EN,
    BIOME_NAMES_VI,
    BIOME_RGB,
    BIOME_EMOJI,
    CANONICAL_GRID_DIM,
    CANONICAL_TOTAL_SIZE_256,
    CANONICAL_WORLD_SCALE,
    HEADER_STRUCT,
    WORLD_ARTIFACT_HEADER_LEN,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    classify_biome,
    classify_biomes_vectorized,
    fnv1a_32,
    map_biome_backend_to_frontend,
    map_biome_frontend_to_backend,
    map_biome_array_to_backend,
    map_biome_array_to_frontend,
)
from terra_forge.core.manifest import (
    CANONICAL_VIEW_CAMERAS,
    CANONICAL_VIEW_IDS,
    CANONICAL_VIEW_LABELS,
    generate_map_manifest,
    load_map_manifest,
    save_map_manifest,
    sync_manifest_model_asset,
    validate_map_manifest,
)


# =============================================================================
# 1. Heightfield2D Adversarial White-Box Hardening
# =============================================================================

class TestHeightfieldAdversarialCore:
    """Stress tests Heightfield2D against extreme resolutions, boundaries, and synthesis math."""

    def test_minimal_2x2_resolution_and_aspect_ratio_stability(self):
        """Minimal 2x2 heightfield must support coordinate conversions, gradients, and normals."""
        hf = Heightfield2D(width=20.0, depth=10.0, resolution=2)
        assert hf.res_x == 2 and hf.res_y == 2
        assert pytest.approx(hf.dx) == 20.0
        assert pytest.approx(hf.dy) == 10.0
        assert hf.Z.shape == (2, 2)

        # Gradients & normals on 2x2 grid
        hf.Z[0, 0] = 1.0
        hf.Z[1, 1] = 5.0
        dz_dx, dz_dy = hf.compute_gradients()
        assert dz_dx.shape == (2, 2) and dz_dy.shape == (2, 2)
        assert not np.isnan(dz_dx).any()
        assert not np.isnan(dz_dy).any()

        normals = hf.compute_normals()
        assert normals.shape == (2, 2, 3)
        magnitudes = np.linalg.norm(normals, axis=-1)
        assert np.allclose(magnitudes, 1.0, atol=1e-7)

        slopes = hf.compute_slope_degrees()
        assert slopes.shape == (2, 2)
        assert np.all(slopes >= 0.0) and np.all(slopes <= 90.0)

    def test_sample_bilinear_exact_bounds_and_extrapolations(self):
        """sample_bilinear must interpolate linear ramps exactly and clamp out-of-bound coords."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=5)
        # Construct exact planar surface: Z = 2*X - 3*Y + 10
        hf.Z = 2.0 * hf.X - 3.0 * hf.Y + 10.0

        # Exact interior interpolation
        for qx in [-30.0, -12.5, 0.0, 17.5, 40.0]:
            for qy in [-40.0, -5.0, 10.0, 35.0]:
                expected = 2.0 * qx - 3.0 * qy + 10.0
                actual = hf.sample_bilinear(qx, qy)
                assert pytest.approx(actual, abs=1e-5) == expected

        # Exact boundary corners
        z_min_corner = hf.sample_bilinear(-50.0, -50.0)
        assert pytest.approx(z_min_corner, abs=1e-5) == hf.Z[0, 0]

        # Extreme out-of-bound clamping (within 1e-2 of grid corner due to 1.0001 clipping boundary epsilon)
        z_far_neg = hf.sample_bilinear(-9999.0, -9999.0)
        assert pytest.approx(z_far_neg, abs=1e-2) == hf.Z[0, 0]
        z_far_pos = hf.sample_bilinear(9999.0, 9999.0)
        assert pytest.approx(z_far_pos, abs=1e-2) == hf.Z[-1, -1]

    def test_normal_vectors_unit_magnitude_and_slope_invariants(self):
        """Normal vectors must be unit length and invariant to global elevation shifts."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        # Subzero rugged topography
        hf.Z = -100.0 + 15.0 * np.sin(0.1 * hf.X) * np.cos(0.1 * hf.Y)

        normals = hf.compute_normals()
        assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-7)

        slopes1 = hf.compute_slope_degrees()
        # Shift entire terrain up by 500m
        hf.Z += 500.0
        slopes2 = hf.compute_slope_degrees()
        assert np.allclose(slopes1, slopes2, atol=1e-6), "Slopes must be invariant to vertical translation"

    def test_add_peak_all_shapes_and_adversarial_parameters(self):
        """add_peak must synthesize all 5 shapes without NaN/Inf, even with extreme parameters."""
        shapes = ["horn", "ridge", "plateau", "dome", "cone"]
        for sh in shapes:
            hf = Heightfield2D(width=80.0, depth=80.0, resolution=25)
            # Center peak
            hf.add_peak(center_x=0.0, center_y=0.0, height=25.0, radius=30.0, shape=sh, ridge_angle_deg=45.0)
            assert not np.isnan(hf.Z).any(), f"Shape {sh} produced NaN"
            assert not np.isinf(hf.Z).any(), f"Shape {sh} produced Inf"
            assert np.max(hf.Z) > 0.0, f"Shape {sh} failed to elevate terrain"

        # Edge case: peak outside terrain boundary
        hf_out = Heightfield2D(width=50.0, depth=50.0, resolution=21)
        hf_out.add_peak(center_x=500.0, center_y=500.0, height=50.0, radius=20.0, shape="horn")
        assert np.all(hf_out.Z == 0.0), "Peak placed far outside domain should have zero influence"

        # Edge case: unknown shape graceful pass
        hf_unk = Heightfield2D(width=50.0, depth=50.0, resolution=11)
        hf_unk.add_peak(center_x=0.0, center_y=0.0, height=10.0, radius=10.0, shape="unknown_shape")
        assert np.all(hf_unk.Z == 0.0)

    def test_lake_basin_carving_berm_and_freeboard_smoothstep(self):
        """carve_lake_basin must carve below water level and construct protective retaining berms."""
        # Case A: Elevated water level with retaining berm
        hf_berm = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        hf_berm.Z.fill(5.0)  # flat 5m plain
        hf_berm.carve_lake_basin(center_x=0.0, center_y=0.0, radius=20.0, water_level=8.0, lake_depth=6.0, berm_height=2.0)

        # Center basin must reach deep bottom: water_level - lake_depth = 2.0
        assert pytest.approx(hf_berm.Z[25, 25], abs=0.1) == 2.0
        # Berm crest must reach water_level + berm_height = 10.0
        assert np.max(hf_berm.Z) >= 9.9
        assert not np.isnan(hf_berm.Z).any()

        # Case B: Standard sea-level basin (water_level=0.0, berm_height=0.0)
        hf_sea = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        hf_sea.Z.fill(5.0)
        hf_sea.carve_lake_basin(center_x=0.0, center_y=0.0, radius=20.0, water_level=0.0, lake_depth=8.0, berm_height=0.0)
        # Deepest point at center must be -8.0
        assert pytest.approx(hf_sea.Z[25, 25], abs=0.1) == -8.0

    def test_add_perlin_octaves_edge_cases_and_determinism(self):
        """Procedural noise must be deterministic for identical seeds and handle zero octaves."""
        hf1 = Heightfield2D(width=50.0, depth=50.0, resolution=32)
        hf2 = Heightfield2D(width=50.0, depth=50.0, resolution=32)

        # Zero octaves: no modification
        hf1.add_perlin_octaves(seed=42, octaves=0)
        assert np.all(hf1.Z == 0.0)

        # Deterministic parity
        hf1.add_perlin_octaves(seed=1337, octaves=4, persistence=0.5, scale=25.0, amplitude=5.0)
        hf2.add_perlin_octaves(seed=1337, octaves=4, persistence=0.5, scale=25.0, amplitude=5.0)
        assert np.array_equal(hf1.Z, hf2.Z)

        # Seed divergence
        hf3 = Heightfield2D(width=50.0, depth=50.0, resolution=32)
        hf3.add_perlin_octaves(seed=9999, octaves=4, persistence=0.5, scale=25.0, amplitude=5.0)
        assert not np.array_equal(hf1.Z, hf3.Z)

    def test_strata_folding_inverted_plunging_and_domain_warp(self):
        """Strata folding must handle inverted amplitudes, zero returns, and plunging hinges."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=41)

        # Zero amplitude or negative wavelength: early return
        z_orig = hf.Z.copy()
        hf.apply_strata_folding(amplitude=0.0, wavelength=40.0)
        assert np.array_equal(hf.Z, z_orig)
        hf.apply_strata_folding(amplitude=10.0, wavelength=-10.0)
        assert np.array_equal(hf.Z, z_orig)

        # Active folding with plunging hinges and domain warping
        hf.apply_strata_folding(
            amplitude=-8.0,  # inverted
            wavelength=35.0,
            strike_angle_deg=120.0,
            plunge_wavelength=80.0,
            harmonic_weight=0.3,
            domain_warp_amp=5.0,
            domain_warp_freq=0.05,
        )
        assert not np.isnan(hf.Z).any()
        assert np.min(hf.Z) < -5.0 and np.max(hf.Z) > 5.0

    def test_canyon_terraces_subzero_elevations_and_steepness(self):
        """apply_canyon_terraces must discretize subzero elevations without zero-crossing discontinuity."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        # Linear slope spanning negative to positive elevations: [-30, +30]
        hf.Z = np.linspace(-30.0, 30.0, 41)[:, np.newaxis] * np.ones((1, 41))

        hf.apply_canyon_terraces(terrace_height=6.0, terrace_sharpness=20.0, blend=1.0)
        assert not np.isnan(hf.Z).any()

        # Check kwargs parameter aliases
        hf2 = Heightfield2D(width=80.0, depth=80.0, resolution=21)
        hf2.Z = np.linspace(0.0, 20.0, 21)[:, np.newaxis] * np.ones((1, 21))
        hf2.apply_canyon_terraces(step_height=4.0, sharpness=10.0, blend_weight=0.9)
        assert not np.isnan(hf2.Z).any()

    def test_tectonic_scarp_finite_fault_tapering_and_strikes(self):
        """apply_tectonic_scarp must taper displacement for finite faults and handle all strike angles."""
        for ang in [-90.0, 0.0, 45.0, 90.0, 180.0]:
            hf = Heightfield2D(width=100.0, depth=100.0, resolution=31)
            hf.apply_tectonic_scarp(
                fault_angle_deg=ang,
                displacement_throw=15.0,
                scarp_sharpness=8.0,
                fault_offset=0.0,
                fault_length=40.0,  # finite fault length
            )
            assert not np.isnan(hf.Z).any()
            # Throw should be bounded by displacement_throw
            assert np.max(hf.Z) <= 15.0 and np.min(hf.Z) >= -15.0

        # Zero displacement: no modification
        hf_zero = Heightfield2D(width=40.0, depth=40.0, resolution=11)
        hf_zero.apply_tectonic_scarp(displacement_throw=0.0)
        assert np.all(hf_zero.Z == 0.0)

    def test_coastal_profile_deep_bathymetry_and_wave_notches(self):
        """apply_coastal_profile must carve Dean's profile offshore and undercut steep cliff faces."""
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=51)
        # Create a cliff face cutting across sea level
        hf.Z = 10.0 * np.tanh(0.1 * hf.Y)

        hf.apply_coastal_profile(
            sea_level=0.0,
            wave_notch_depth=2.5,
            wave_notch_envelope=1.5,
            beach_slope_coefficient=0.15,
            berm_elevation=1.5,
            beach_width=15.0,
            shelf_transition_dist=30.0,
            deep_water_depth=15.0,
            coastline_angle_deg=0.0,
            coastline_offset=0.0,
        )
        assert not np.isnan(hf.Z).any()
        # Offshore (Y > 0) should drop towards -15.0
        assert np.min(hf.Z) <= -10.0
        # Supratidal sandy berm (Y near 0 inland) elevated above sea level
        assert np.any(hf.Z > 0.0)

    def test_apply_geological_features_legacy_aliases_and_fallbacks(self):
        """apply_geological_features must seamlessly accept MapConfig with legacy alias names."""
        cfg = MapConfig()
        cfg.tectonic.enabled = True
        # Set legacy attribute names
        cfg.tectonic.displacement = 12.0
        cfg.tectonic.angle = 30.0

        cfg.strata.enabled = True
        cfg.strata.terrace_enabled = True
        cfg.strata.step_height = 5.0
        cfg.strata.steepness = 8.0
        cfg.strata.blend = 0.8

        cfg.coastal.enabled = True
        cfg.coastal.coastline_angle_deg = 45.0

        hf = Heightfield2D(width=80.0, depth=80.0, resolution=31)
        hf.Z.fill(5.0)
        hf.apply_geological_features(cfg)

        assert not np.isnan(hf.Z).any()
        assert not np.array_equal(hf.Z, np.full((31, 31), 5.0))


# =============================================================================
# 2. Erosion Engine Adversarial White-Box Hardening
# =============================================================================

class TestErosionSimulatorAdversarialCore:
    """Stress tests ErosionSimulator against droplet parameters, mass conservation, and grid bounds."""

    @pytest.mark.parametrize("num_droplets", [0, 1, 25, 250])
    @pytest.mark.parametrize("grid_shape", [(32, 32), (16, 64), (64, 16)])
    def test_hydraulic_exact_mass_conservation_parametric_sweep(self, num_droplets: int, grid_shape: Tuple[int, int]):
        """Hydraulic erosion must preserve exact total mass (|ΔM| < 1e-10) across arbitrary shapes."""
        ny, nx = grid_shape
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=max(nx, ny))
        # Resize grid if non-square
        if nx != ny:
            hf.res_x = nx
            hf.res_y = ny
            hf.dx = hf.width / (nx - 1)
            hf.dy = hf.depth / (ny - 1)
            hf.Z = np.zeros((ny, nx), dtype=np.float64)
            hf.sediment = np.zeros((ny, nx), dtype=np.float64)
            hf.scree_talus = np.zeros((ny, nx), dtype=np.float64)
            hf.water_depth = np.zeros((ny, nx), dtype=np.float64)

        # Synthetic rugged topography
        hf.Z = 15.0 * np.exp(-((hf.dx * np.arange(nx) - 40.0)**2 + (hf.dy * np.arange(ny)[:, np.newaxis] - 40.0)**2) / 400.0)
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=num_droplets)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-10, (
            f"Mass drift detected: initial {initial_mass}, final {final_mass}, delta {final_mass - initial_mass}"
        )
        assert not np.isnan(hf.Z).any()

    def test_hydraulic_small_grid_margin_collapse_graceful_return(self):
        """Extremely small heightfield where low_x >= high_x must return immediately without error."""
        hf = Heightfield2D(width=10.0, depth=10.0, resolution=4)
        hf.Z.fill(5.0)
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=100)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=123)

        assert np.array_equal(hf.Z, np.full((4, 4), 5.0))
        assert float(np.sum(hf.Z)) == initial_mass

    def test_hydraulic_extreme_inertia_and_evaporation_speeds(self):
        """Droplet simulation must survive extreme inertia (0.0 and 0.99) and extreme evaporation."""
        for inertia, evap in [(0.0, 0.99), (0.99, 0.01)]:
            hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
            hf.Z = np.linspace(20.0, 0.0, 25)[:, np.newaxis] * np.ones((1, 25))
            initial_mass = float(np.sum(hf.Z))

            cfg = ErosionConfig(
                enabled=True,
                droplet_count=50,
                droplet_inertia=inertia,
                evaporate_speed=evap,
            )
            sim = ErosionSimulator(hf, cfg)
            sim.simulate_hydraulic(seed=777)

            final_mass = float(np.sum(hf.Z))
            assert abs(final_mass - initial_mass) < 1e-10
            assert not np.isnan(hf.Z).any()

    def test_thermal_talus_exact_mass_conservation_and_scree_tracking(self):
        """Thermal talus relaxation must strictly conserve mass and accumulate debris in scree_talus."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        # Create a sheer vertical cliff wall in the center: left = 30m, right = 0m
        hf.Z[:, :15] = 30.0
        hf.Z[:, 15:] = 0.0
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=10,
            talus_angle_deg=30.0,  # 30-degree repose angle
            scree_talus_deposition=0.8,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-10, "Thermal talus mass must be conserved"
        assert not np.isnan(hf.Z).any()

        # Scree accumulation layer must capture deposited rockfall volume
        assert hasattr(hf, "scree_talus") and hf.scree_talus is not None
        assert np.max(hf.scree_talus) > 0.0, "Talus debris must accumulate at cliff base"
        assert np.all(hf.scree_talus >= 0.0)

    def test_thermal_talus_flat_terrain_and_angle_boundaries(self):
        """Thermal erosion on flat terrain or near-vertical repose angle must terminate cleanly."""
        # 1. Flat terrain: zero slope -> zero iterations of slumping
        hf_flat = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf_flat.Z.fill(10.0)
        cfg_flat = ErosionConfig(enabled=True, thermal_iterations=5, talus_angle_deg=35.0)
        sim_flat = ErosionSimulator(hf_flat, cfg_flat)
        sim_flat.run_thermal_erosion()
        assert np.all(hf_flat.Z == 10.0)

        # 2. Angle of repose 89 degrees: steep cliff does not slump
        hf_steep = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf_steep.Z[:, :10] = 20.0
        hf_steep.Z[:, 10:] = 0.0
        cfg_steep = ErosionConfig(enabled=True, thermal_iterations=5, talus_angle_deg=89.0)
        sim_steep = ErosionSimulator(hf_steep, cfg_steep)
        sim_steep.run_thermal_erosion()
        assert np.array_equal(hf_steep.Z[:, :10], np.full((21, 10), 20.0))

    def test_thermal_talus_non_square_grid_isotropy(self):
        """Thermal talus on non-square heightfields must execute isotropic relaxation without error."""
        hf = Heightfield2D(width=100.0, depth=40.0, resolution=25)
        # Custom non-square shape
        hf.res_x = 41
        hf.res_y = 17
        hf.dx = hf.width / (hf.res_x - 1)
        hf.dy = hf.depth / (hf.res_y - 1)
        hf.Z = np.zeros((17, 41), dtype=np.float64)
        hf.scree_talus = np.zeros((17, 41), dtype=np.float64)
        hf.Z[:8, :] = 25.0

        initial_mass = float(np.sum(hf.Z))
        cfg = ErosionConfig(enabled=True, thermal_iterations=5, talus_angle_deg=25.0)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        assert abs(float(np.sum(hf.Z)) - initial_mass) < 1e-10
        assert not np.isnan(hf.Z).any()


# =============================================================================
# 3. Hydrology Engine Adversarial White-Box Hardening
# =============================================================================

class TestHydrologyEngineAdversarialCore:
    """Stress tests HydrologyEngine and RiverSplineScalar arithmetic and spline math."""

    def test_river_spline_scalar_complete_arithmetic_and_indexing_contract(self):
        """RiverSplineScalar must behave as float while supporting 1D index [0], [-1] and ops."""
        s1 = RiverSplineScalar(12.5)
        s2 = RiverSplineScalar(2.5)

        # Indexing & properties
        assert s1[0] == 12.5
        assert s1[-1] == 12.5
        assert len(s1) == 1
        assert s1.shape == (1,)
        assert s1.ndim == 1
        assert s1.item() == 12.5

        # Out-of-range indexing
        with pytest.raises(IndexError):
            _ = s1[1]
        with pytest.raises(IndexError):
            _ = s1[-2]

        # Arithmetic operations
        assert pytest.approx(float(s1 + s2)) == 15.0
        assert pytest.approx(float(s1 - s2)) == 10.0
        assert pytest.approx(float(s1 * s2)) == 31.25
        assert pytest.approx(float(s1 / s2)) == 5.0
        assert pytest.approx(float(-s1)) == -12.5

        # Reverse operations with regular floats/ints
        assert pytest.approx(float(10.0 + s1)) == 22.5
        assert pytest.approx(float(20.0 - s1)) == 7.5
        assert pytest.approx(float(2.0 * s1)) == 25.0
        assert pytest.approx(float(25.0 / s1)) == 2.0

        # Type preservation
        assert isinstance(s1 + 5.0, RiverSplineScalar)
        assert isinstance(s1 - 5.0, RiverSplineScalar)
        assert isinstance(s1 * 2.0, RiverSplineScalar)
        assert isinstance(s1 / 2.0, RiverSplineScalar)

    def test_river_spline_scalar_zero_division_guards(self):
        """Division by zero on RiverSplineScalar must raise ZeroDivisionError."""
        s = RiverSplineScalar(10.0)
        with pytest.raises(ZeroDivisionError):
            _ = s / 0.0
        with pytest.raises(ZeroDivisionError):
            _ = 10.0 / RiverSplineScalar(0.0)

    def test_stage_normalization_mixed_keys_and_defaults(self):
        """_normalize_stages must handle empty lists, uppercase/lowercase keys, and single points."""
        # Empty list
        assert HydrologyEngine._normalize_stages([]) == []

        # Single point waypoint list (cannot form segment)
        assert HydrologyEngine._normalize_stages([{"x": 0.0, "y": 0.0}]) == []

        # Multi-point waypoints with mixed casing
        raw = [
            {"X": -50.0, "Y": 20.0, "Z": 10.0, "width": 3.0},
            {"x": 0.0, "y": 0.0, "z": 5.0, "width": 6.0},
            {"X": 40.0, "y": -30.0, "Z": 0.0, "width": 8.0},
        ]
        norm = HydrologyEngine._normalize_stages(raw)
        assert len(norm) == 2
        assert norm[0]["start"] == (-50.0, 20.0, 10.0, 3.0)
        assert norm[0]["end"] == (0.0, 0.0, 5.0, 6.0)
        assert norm[1]["start"] == (0.0, 0.0, 5.0, 6.0)
        assert norm[1]["end"] == (40.0, -30.0, 0.0, 8.0)

    def test_spline_evaluation_out_of_bounds_and_hermite_tangents(self):
        """Spline evaluation must handle out-of-domain t and Hermite cubic tangents."""
        cfg = HydrologyConfig(
            river_enabled=True,
            river_stages=[
                {
                    "name": "Hermite Gorge",
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "start": (-40.0, -40.0, 15.0, 4.0),
                    "end": (40.0, 40.0, 0.0, 8.0),
                    "tangent_in": (10.0, 0.0, -5.0),
                    "tangent_out": (0.0, 10.0, -2.0),
                }
            ],
        )
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=21)
        hydro = HydrologyEngine(hf, cfg)

        # Boundary values
        rx0, ry0, rz0, rw0 = hydro.evaluate_river_spline(0.0)
        assert pytest.approx(float(rx0)) == -40.0
        assert pytest.approx(float(rz0)) == 15.0

        rx1, ry1, rz1, rw1 = hydro.evaluate_river_spline(1.0)
        assert pytest.approx(float(rx1)) == 40.0
        assert pytest.approx(float(rz1)) == 0.0

        # Out-of-bounds t (< 0 or > 1) returns 0.0 gracefully
        rx_neg, _, _, _ = hydro.evaluate_river_spline(-0.5)
        assert float(rx_neg) == 0.0
        rx_pos, _, _, _ = hydro.evaluate_river_spline(1.5)
        assert float(rx_pos) == 0.0

    def test_spline_evaluation_scalar_array_parity(self):
        """evaluate_river_spline must yield identical values between scalar and array queries."""
        cfg = HydrologyConfig(river_enabled=True)
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=31)
        hydro = HydrologyEngine(hf, cfg)

        t_samples = np.linspace(0.0, 1.0, 25)
        arr_rx, arr_ry, arr_rz, arr_rw = hydro.evaluate_river_spline(t_samples)

        for i, t in enumerate(t_samples):
            s_rx, s_ry, s_rz, s_rw = hydro.evaluate_river_spline(float(t))
            assert pytest.approx(float(s_rx), abs=1e-5) == arr_rx[i]
            assert pytest.approx(float(s_ry), abs=1e-5) == arr_ry[i]
            assert pytest.approx(float(s_rz), abs=1e-5) == arr_rz[i]
            assert pytest.approx(float(s_rw), abs=1e-5) == arr_rw[i]

    def test_flow_velocity_flat_gradients_and_vector_field(self):
        """evaluate_flow_velocity and compute_flow_velocity_field must handle flat sections."""
        cfg = HydrologyConfig(river_enabled=True)
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=31)
        hydro = HydrologyEngine(hf, cfg)

        # Flow velocity on central lake transit (flat gradient: -tz == 0)
        # Should not produce NaN or zero division
        vx, vy, vz = hydro.evaluate_flow_velocity(0.5)
        assert not math.isnan(float(vx))
        assert not math.isnan(float(vy))
        assert not math.isnan(float(vz))

        # 2D velocity vector field across heightfield
        vxf, vyf = hydro.compute_flow_velocity_field()
        assert vxf.shape == (31, 31) and vyf.shape == (31, 31)
        assert not np.isnan(vxf).any()
        assert not np.isnan(vyf).any()

    def test_channel_carving_and_water_depth_rasterization_boundaries(self):
        """Channel carving and water depth rasterization must modify river bed and stamp water."""
        cfg = HydrologyConfig(
            river_enabled=True,
            lake_radius=15.0,
            lake_center=(0.0, 0.0),
            lake_level=4.0,
            river_carve_depth=2.0,
        )
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(6.0)  # above lake level

        hydro = HydrologyEngine(hf, cfg)
        hydro.carve_river_channel()
        assert np.min(hf.Z) < 6.0, "River channel incision must lower elevations along channel"

        depth_map = hydro.rasterize_water_depth()
        assert depth_map.shape == (41, 41)
        assert not np.isnan(depth_map).any()
        assert np.all(depth_map >= 0.0)

    def test_water_distance_field_monotonicity_and_dry_defaults(self):
        """compute_water_distance_field must return 999.0 when dry and increase away from water."""
        # 1. Completely dry world
        cfg_dry = HydrologyConfig(river_enabled=False, lake_radius=0.0)
        hf_dry = Heightfield2D(width=50.0, depth=50.0, resolution=21)
        hydro_dry = HydrologyEngine(hf_dry, cfg_dry)
        dist_dry = hydro_dry.compute_water_distance_field()
        assert np.all(dist_dry == 999.0)

        # 2. Central lake only: distance increases monotonically with radius
        cfg_lake = HydrologyConfig(river_enabled=False, lake_radius=10.0, lake_center=(0.0, 0.0))
        hf_lake = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hydro_lake = HydrologyEngine(hf_lake, cfg_lake)
        dist_lake = hydro_lake.compute_water_distance_field()

        # Center point inside lake must have distance 0.0
        assert dist_lake[15, 15] == 0.0
        # Perimeter distance increases radially
        assert dist_lake[0, 0] > dist_lake[10, 10] > 0.0


# =============================================================================
# 4. WorldArtifact Binary & Biome Taxonomy Adversarial Hardening
# =============================================================================

class TestWorldArtifactAdversarialCore:
    """Stress tests WorldArtifact binary codec, FNV-1a checksums, and 22-biome classification."""

    def test_world_artifact_corrupted_header_rejections(self):
        """from_bytes must reject any input shorter than 36 bytes."""
        short_buffers = [b"", b"ANM", b"ANMW" + b"\x00" * 20]
        for buf in short_buffers:
            with pytest.raises(ValueError, match="too short for header"):
                WorldArtifact.from_bytes(buf)

    def test_world_artifact_magic_and_version_rejections(self):
        """from_bytes must reject bad magic or unsupported versions."""
        # Bad magic
        bad_magic_header = struct.pack("<4sIIIfIIfI", b"BADM", 2, 256, 256, 0.0, 1337, 2, 200.0, 0)
        with pytest.raises(ValueError, match="bad artifact magic"):
            WorldArtifact.from_bytes(bad_magic_header + b"\x00" * (256 * 256 * 17))

        # Unsupported version (e.g. version 1 or 3)
        for bad_ver in [1, 3, 99]:
            bad_ver_header = struct.pack("<4sIIIfIIfI", b"ANMW", bad_ver, 256, 256, 0.0, 1337, 2, 200.0, 0)
            with pytest.raises(ValueError, match="unsupported artifact version"):
                WorldArtifact.from_bytes(bad_ver_header + b"\x00" * (256 * 256 * 17))

    def test_world_artifact_zero_cells_and_dimension_rejections(self):
        """from_bytes must reject artifacts with zero cells."""
        zero_w = struct.pack("<4sIIIfIIfI", b"ANMW", 2, 0, 256, 0.0, 1337, 2, 200.0, 0)
        with pytest.raises(ValueError, match="zero cells"):
            WorldArtifact.from_bytes(zero_w)

        zero_h = struct.pack("<4sIIIfIIfI", b"ANMW", 2, 256, 0, 0.0, 1337, 2, 200.0, 0)
        with pytest.raises(ValueError, match="zero cells"):
            WorldArtifact.from_bytes(zero_h)

    def test_world_artifact_truncation_and_trailing_byte_rejections(self):
        """from_bytes must reject truncated payloads and unexpected trailing bytes."""
        n = 4 * 4
        elev = np.zeros(n, dtype="<f4")
        moist = np.zeros(n, dtype="<f4")
        temp = np.zeros(n, dtype="<f4")
        flow = np.zeros(n, dtype="<f4")
        biome = np.zeros(n, dtype=np.uint8)
        art = WorldArtifact(4, 4, 0.0, 1337, 2, 200.0, elev, moist, temp, flow, biome)
        valid_bytes = art.to_bytes()

        # Truncation by 1 byte
        with pytest.raises(ValueError, match="artifact truncated"):
            WorldArtifact.from_bytes(valid_bytes[:-1])

        # Trailing 1 byte
        with pytest.raises(ValueError, match="trailing byte"):
            WorldArtifact.from_bytes(valid_bytes + b"\x00")

    def test_world_artifact_payload_checksum_bit_flip_tampering(self):
        """A single flipped bit anywhere in the payload must trigger a checksum mismatch error."""
        n = 8 * 8
        elev = np.zeros(n, dtype="<f4")
        moist = np.zeros(n, dtype="<f4")
        temp = np.zeros(n, dtype="<f4")
        flow = np.zeros(n, dtype="<f4")
        biome = np.zeros(n, dtype=np.uint8)
        art = WorldArtifact(8, 8, 0.0, 1337, 2, 200.0, elev, moist, temp, flow, biome)
        raw = bytearray(art.to_bytes())

        # Flip a bit in elevation layer (offset 40)
        raw[40] ^= 0x01
        with pytest.raises(ValueError, match="artifact checksum mismatch"):
            WorldArtifact.from_bytes(bytes(raw))

        # Flip a bit in biome layer (offset 36 + n*16 + 2)
        raw = bytearray(art.to_bytes())
        raw[36 + n * 16 + 2] ^= 0x01
        with pytest.raises(ValueError, match="artifact checksum mismatch"):
            WorldArtifact.from_bytes(bytes(raw))

    def test_world_artifact_peek_seed_fast_path_and_validation(self):
        """peek_seed must extract world seed at offset 20 without full payload parse."""
        n = 2 * 2
        elev = np.zeros(n, dtype="<f4")
        moist = np.zeros(n, dtype="<f4")
        temp = np.zeros(n, dtype="<f4")
        flow = np.zeros(n, dtype="<f4")
        biome = np.zeros(n, dtype=np.uint8)
        art = WorldArtifact(2, 2, 0.0, 987654321, 2, 200.0, elev, moist, temp, flow, biome)
        data = art.to_bytes()

        assert WorldArtifact.peek_seed(data) == 987654321

        # Rejection of invalid header in peek_seed
        with pytest.raises(ValueError, match="too short for header"):
            WorldArtifact.peek_seed(b"SHORT")

    def test_biome_classification_full_22_canonical_reachability(self):
        """classify_biome must synthesize all target biomes under specific physical conditions."""
        # 1. OCEAN
        assert classify_biome(elev=0.1, temp=0.5, moist=0.5, flow=0.0, slope=0.0, sea_level=0.3) == Biome.OCEAN
        # 2. BEACH
        assert classify_biome(elev=0.31, temp=0.5, moist=0.5, flow=0.0, slope=0.1, coast=0.8, sea_level=0.3) == Biome.BEACH
        # 3. ROCK (coastal cliff)
        assert classify_biome(elev=0.31, temp=0.5, moist=0.5, flow=0.0, slope=0.6, coast=0.8, sea_level=0.3) == Biome.ROCK
        # 4. MANGROVE
        assert classify_biome(elev=0.31, temp=0.7, moist=0.7, flow=0.0, slope=0.1, coast=0.8, sea_level=0.3) == Biome.MANGROVE
        # 5. GLACIER
        assert classify_biome(elev=0.6, temp=0.02, moist=0.45, flow=0.0, slope=0.1, sea_level=0.3) == Biome.GLACIER
        # 6. SNOW
        assert classify_biome(elev=0.6, temp=0.10, moist=0.45, flow=0.0, slope=0.1, sea_level=0.3) == Biome.SNOW
        # 7. RIVER
        assert classify_biome(elev=0.5, temp=0.5, moist=0.5, flow=0.8, slope=0.1, sea_level=0.3) == Biome.RIVER
        # 8. ALPINE
        assert classify_biome(elev=0.6, temp=0.25, moist=0.4, flow=0.0, slope=0.1, sea_level=0.3) == Biome.ALPINE
        # 9. BOG
        assert classify_biome(elev=0.35, temp=0.25, moist=0.8, flow=0.25, slope=0.1, sea_level=0.3) == Biome.BOG
        # 10. SWAMP
        assert classify_biome(elev=0.35, temp=0.5, moist=0.8, flow=0.25, slope=0.1, sea_level=0.3) == Biome.SWAMP
        # 11. TAIGA
        assert classify_biome(elev=0.4, temp=0.22, moist=0.63, flow=0.0, slope=0.0, sea_level=0.3) == Biome.TAIGA
        # 12. TUNDRA
        assert classify_biome(elev=0.4, temp=0.22, moist=0.40, flow=0.0, slope=0.0, sea_level=0.3) == Biome.TUNDRA
        # 13. STEPPE
        assert classify_biome(elev=0.4, temp=0.35, moist=0.10, flow=0.0, slope=0.1, sea_level=0.3) == Biome.STEPPE
        # 14. SHRUBLAND
        assert classify_biome(elev=0.4, temp=0.35, moist=0.25, flow=0.0, slope=0.1, sea_level=0.3) == Biome.SHRUBLAND
        # 15. DESERT
        assert classify_biome(elev=0.4, temp=0.50, moist=0.10, flow=0.0, slope=0.1, sea_level=0.3) == Biome.DESERT
        # 16. GRASSLAND
        assert classify_biome(elev=0.4, temp=0.50, moist=0.35, flow=0.0, slope=0.1, sea_level=0.3) == Biome.GRASSLAND
        # 17. FOREST
        assert classify_biome(elev=0.4, temp=0.50, moist=0.65, flow=0.0, slope=0.1, sea_level=0.3) == Biome.FOREST
        # 18. BADLANDS
        assert classify_biome(elev=0.4, temp=0.65, moist=0.10, flow=0.0, slope=0.30, sea_level=0.3) == Biome.BADLANDS
        # 19. CHAPARRAL
        assert classify_biome(elev=0.4, temp=0.65, moist=0.30, flow=0.0, slope=0.1, sea_level=0.3) == Biome.CHAPARRAL
        # 20. JUNGLE
        assert classify_biome(elev=0.4, temp=0.65, moist=0.80, flow=0.0, slope=0.1, sea_level=0.3) == Biome.JUNGLE
        # 21. SAVANNA
        assert classify_biome(elev=0.4, temp=0.80, moist=0.30, flow=0.0, slope=0.1, sea_level=0.3) == Biome.SAVANNA

    def test_biome_taxonomy_metadata_and_rgb_invariants(self):
        """Taxonomy metadata tables must strictly contain 22 entries and valid RGB tuples."""
        assert len(BIOME_NAMES_EN) == 22
        assert len(BIOME_NAMES_VI) == 22
        assert len(BIOME_RGB) == 22
        assert len(BIOME_EMOJI) == 22

        for r, g, b in BIOME_RGB:
            assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_biome_bidirectional_mapping_and_out_of_bounds_clamping(self):
        """map_biome functions must map all 22 and 11 biomes and handle out-of-bounds indices."""
        for b in range(22):
            leg = map_biome_frontend_to_backend(b)
            assert 0 <= leg < 11

        for lb in range(11):
            can = map_biome_backend_to_frontend(lb)
            assert 0 <= can < 22

        # Out-of-bounds indices fallback to GRASSLAND
        assert map_biome_frontend_to_backend(-1) == LegacyBiome.GRASSLAND
        assert map_biome_frontend_to_backend(999) == LegacyBiome.GRASSLAND
        assert map_biome_backend_to_frontend(-1) == Biome.GRASSLAND
        assert map_biome_backend_to_frontend(999) == Biome.GRASSLAND

        # Vectorized array projection
        arr_in = np.array([-5, 0, 7, 21, 100], dtype=np.int32)
        arr_back = map_biome_array_to_backend(arr_in)
        assert len(arr_back) == 5
        assert np.all(arr_back >= 0) and np.all(arr_back < 11)

    def test_from_heightfield_flat_and_zero_flow_synthesis(self):
        """from_heightfield must handle completely flat terrain without division by zero."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=16)
        hf.Z.fill(0.0)  # perfectly flat
        art = WorldArtifact.from_heightfield(hf, seed=42)

        assert art.width == 16 and art.height == 16
        assert np.all(art.elevation == 0.5)  # fallback default
        assert not np.isnan(art.elevation).any()
        assert not np.isnan(art.moisture).any()
        assert not np.isnan(art.temperature).any()

    def test_to_terrain_map_downsampling_and_upsampling_parity(self):
        """to_terrain_map must correctly downsample/upsample into standard simulation dictionary."""
        n = 16 * 16
        elev = np.linspace(0.0, 1.0, n, dtype="<f4")
        moist = np.full(n, 0.5, dtype="<f4")
        temp = np.full(n, 0.5, dtype="<f4")
        flow = np.full(n, 0.2, dtype="<f4")
        biome = np.full(n, int(Biome.FOREST), dtype=np.uint8)
        art = WorldArtifact(16, 16, 0.0, 1337, 2, 200.0, elev, moist, temp, flow, biome)

        # Downsample to 8x4
        t_map = art.to_terrain_map(target_w=8, target_h=4)
        assert t_map["width"] == 8 and t_map["height"] == 4
        assert len(t_map["elevations"]) == 32
        assert len(t_map["flows"]) == 32
        assert len(t_map["biomes"]) == 32
        assert t_map["biomes"][0] == map_biome_frontend_to_backend(int(Biome.FOREST))


# =============================================================================
# 5. Map Manifest Generator & Validator Adversarial Hardening
# =============================================================================

class TestMapManifestAdversarialCore:
    """Stress tests generate_map_manifest and validate_map_manifest against malformed structures."""

    def test_manifest_generator_input_type_rejections(self):
        """generate_map_manifest must reject invalid artifact input types."""
        with pytest.raises(TypeError, match="unsupported artifact input type"):
            generate_map_manifest(12345)  # int not supported
        with pytest.raises(TypeError, match="unsupported artifact input type"):
            generate_map_manifest({"invalid": "dict"})

    def test_manifest_generator_from_object_bytes_and_path(self):
        """generate_map_manifest must accept WorldArtifact, raw bytes, or file path identically."""
        n = 4 * 4
        elev = np.zeros(n, dtype="<f4")
        moist = np.zeros(n, dtype="<f4")
        temp = np.zeros(n, dtype="<f4")
        flow = np.zeros(n, dtype="<f4")
        biome = np.zeros(n, dtype=np.uint8)
        art = WorldArtifact(4, 4, 0.0, 1337, 2, 200.0, elev, moist, temp, flow, biome)
        art_bytes = art.to_bytes()

        m1 = generate_map_manifest(art, captured_override=False)
        m2 = generate_map_manifest(art_bytes, captured_override=False)

        assert m1["worldArtifact"]["checksum"] == m2["worldArtifact"]["checksum"]
        assert m1["worldArtifact"]["bytes"] == m2["worldArtifact"]["bytes"]

        valid, errors = validate_map_manifest(m1)
        assert valid is True, f"Generated manifest must be valid: {errors}"

    def test_manifest_generator_captured_override_and_image_dir(self):
        """captured_override and view_images_dir must set captured booleans and hashes correctly."""
        n = 4 * 4
        art = WorldArtifact(4, 4, 0.0, 1337, 2, 200.0, np.zeros(n, dtype="<f4"), np.zeros(n, dtype="<f4"), np.zeros(n, dtype="<f4"), np.zeros(n, dtype="<f4"), np.zeros(n, dtype=np.uint8))

        # Explicit captured_override=True
        m_true = generate_map_manifest(art, captured_override=True)
        for v in m_true["views"]:
            assert v["captured"] is True

        # Test view_images_dir verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_dir = Path(tmp_dir)
            sample_png = img_dir / "overview.png"
            sample_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            m_dir = generate_map_manifest(art, view_images_dir=img_dir)
            v_overview = next(v for v in m_dir["views"] if v["id"] == "overview")
            assert v_overview["captured"] is True
            assert v_overview["bytes"] == len(sample_png.read_bytes())
            assert v_overview["checksum"].startswith("sha256:")

    def test_manifest_validator_schema_version_rejections(self):
        """validate_map_manifest must reject missing or unsupported schemaVersion."""
        valid_manifest = {
            "schemaVersion": 1,
            "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0, "bytes": 100},
            "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
            "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
            "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
        }

        # Missing schemaVersion
        m_missing = copy.deepcopy(valid_manifest)
        del m_missing["schemaVersion"]
        val, err = validate_map_manifest(m_missing)
        assert val is False and any("missing required field: schemaVersion" in e for e in err)

        # Wrong schemaVersion
        m_wrong = copy.deepcopy(valid_manifest)
        m_wrong["schemaVersion"] = 2
        val, err = validate_map_manifest(m_wrong)
        assert val is False and any("schemaVersion must be 1" in e for e in err)

    def test_manifest_validator_world_artifact_block_rejections(self):
        """validate_map_manifest must reject malformed worldArtifact blocks."""
        base = {
            "schemaVersion": 1,
            "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
            "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
            "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
        }

        # Bad magic
        m_bad_magic = copy.deepcopy(base)
        m_bad_magic["worldArtifact"] = {"path": "a.anmw", "magic": "BADM", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0}
        val, err = validate_map_manifest(m_bad_magic)
        assert val is False and any('magic must be the literal "ANMW"' in e for e in err)

        # Negative width
        m_neg_w = copy.deepcopy(base)
        m_neg_w["worldArtifact"] = {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": -5, "height": 256, "seaLevel": 0.0}
        val, err = validate_map_manifest(m_neg_w)
        assert val is False and any("width must be an integer >= 1" in e for e in err)

    def test_manifest_validator_coordinate_system_bounds_rejections(self):
        """validate_map_manifest must reject non-canonical coordinate bounds."""
        base = {
            "schemaVersion": 1,
            "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
            "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
            "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
        }

        for field, bad_val in [("worldMinXZ", -50), ("worldMaxXZ", 200), ("worldMinY", -1), ("worldMaxY", 20)]:
            m = copy.deepcopy(base)
            m["coordinateSystem"] = {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256}
            m["coordinateSystem"][field] = bad_val
            val, err = validate_map_manifest(m)
            assert val is False, f"Expected validation failure for {field}={bad_val}"

    def test_manifest_validator_biome_taxonomy_counts_rejections(self):
        """validate_map_manifest must enforce canonicalCount=22 and legacyCount=11."""
        base = {
            "schemaVersion": 1,
            "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
            "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
            "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
        }

        m_bad_canon = copy.deepcopy(base)
        m_bad_canon["biomeTaxonomy"] = {"canonicalCount": 20, "legacyCount": 11}
        val, _ = validate_map_manifest(m_bad_canon)
        assert val is False

        m_bad_leg = copy.deepcopy(base)
        m_bad_leg["biomeTaxonomy"] = {"canonicalCount": 22, "legacyCount": 9}
        val, _ = validate_map_manifest(m_bad_leg)
        assert val is False

    def test_manifest_validator_views_and_camera_rejections(self):
        """validate_map_manifest must reject non-canonical view IDs or illegal camera tuples."""
        base = {
            "schemaVersion": 1,
            "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
            "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
            "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
        }

        # Non-canonical view id
        m_bad_id = copy.deepcopy(base)
        m_bad_id["views"] = [{"id": "random_unknown_view", "imagePath": "map-views/random.png", "camera": {"position": [0, 0, 0], "target": [0, 0, 0]}}]
        val, err = validate_map_manifest(m_bad_id)
        assert val is False and any("not a canonical view id" in e for e in err)

        # 2D position instead of 3D
        m_2d_pos = copy.deepcopy(base)
        m_2d_pos["views"] = [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 0], "target": [0, 0, 0]}}]
        val, err = validate_map_manifest(m_2d_pos)
        assert val is False and any("position must be a [x, y, z] number triple" in e for e in err)

    def test_manifest_validator_model_asset_block_rejections(self):
        """validate_map_manifest must reject malformed modelAsset blocks (e.g. negative bytes)."""
        valid_manifest = {
            "schemaVersion": 1,
            "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
            "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
            "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
            "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
            "modelAsset": {"path": "m.glb", "format": "gltf-binary", "bytes": -500},
        }
        val, err = validate_map_manifest(valid_manifest)
        assert val is False and any("modelAsset.bytes must be a positive integer" in e for e in err)

    def test_manifest_sync_model_asset_and_fov_enforcement(self):
        """sync_manifest_model_asset must update hash and enforce fov: 45.0 across views."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            man_path = tmp_path / "map_manifest.json"
            glb_path = tmp_path / "test_model.glb"
            glb_path.write_bytes(b"glTF\x02\x00\x00\x00" + b"\x00" * 200)

            initial_manifest = {
                "schemaVersion": 1,
                "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
                "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
                "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
                "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
            }
            save_map_manifest(initial_manifest, man_path)

            synced = sync_manifest_model_asset(man_path, glb_path)
            assert "modelAsset" in synced
            assert synced["modelAsset"]["bytes"] == len(glb_path.read_bytes())
            assert synced["modelAsset"]["checksum"].startswith("sha256:")
            assert synced["views"][0]["camera"]["fov"] == 45.0

    def test_manifest_persistence_save_load_validation(self):
        """save_map_manifest and load_map_manifest must reject invalid files and preserve valid ones."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "manifest.json"

            # Saving invalid manifest raises ValueError
            with pytest.raises(ValueError, match="refusing to save invalid map manifest"):
                save_map_manifest({"invalid": "manifest"}, p)

            # Saving valid manifest succeeds
            valid_m = {
                "schemaVersion": 1,
                "worldArtifact": {"path": "a.anmw", "magic": "ANMW", "version": 2, "width": 256, "height": 256, "seaLevel": 0.0},
                "coordinateSystem": {"worldMinXZ": -100, "worldMaxXZ": 100, "worldMinY": 0, "worldMaxY": 10, "gridDim": 256},
                "biomeTaxonomy": {"canonicalCount": 22, "legacyCount": 11},
                "views": [{"id": "overview", "imagePath": "map-views/overview.png", "camera": {"position": [0, 1, 2], "target": [0, 0, 0]}}],
            }
            save_map_manifest(valid_m, p)
            loaded = load_map_manifest(p)
            assert loaded["schemaVersion"] == 1
