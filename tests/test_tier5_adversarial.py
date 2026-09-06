"""
Tier 5 Adversarial Coverage Hardening Suite for TerraForge Engine.

Targeted Modules:
- terra_forge.core.erosion (hydraulic momentum erosion, 8-neighbor isotropic thermal talus)
- terra_forge.core.heightfield (strata folding, terraces, tectonic scarps, coastal profiles)
- terra_forge.core.hydrology (river splines, water distance field, velocity fields)
- terra_forge.core.biomes (microclimate, multi-tier vegetation, Poisson-disk sampling)
- terra_forge.inspector.validator (2-manifold invariants, lake berm freeboard, basin containment)
- terra_forge.inspector.cameras (solar ephemeris, blackbody CCT, camera framing math)

Adversarial Stress Dimensions:
1. High aspect-ratio heightfields (16x256, 256x16, non-square physical extents)
2. Zero, inverted, grazing, and arbitrary insolation vectors
3. Subzero polar extremes, hyper-thermal deserts, temperature inversions, extreme moisture bounds
4. Zero-droplet, single-droplet, and extreme droplet parameter erosion runs
5. High-density Poisson-disk packing saturation limits and oversized exclusion radii
6. Diorama slabs with non-square resolutions and high aspect ratios (Euler chi=2)
7. Disconnected multiple lake bodies, dry river channels, and wall breach detection
8. Presets mutated with extreme geological, hydrological, and diurnal parameters
"""

from __future__ import annotations
import copy
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pytest

from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar
from terra_forge.core.biomes import (
    BiomeEngine,
    bridson_poisson_disk_sampling,
    EcologicalTier,
    PROTOTYPE_TAXONOMY,
)
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.inspector.validator import QualityValidator
from terra_forge.inspector.cameras import (
    calculate_solar_position,
    calculate_cct_from_elevation,
    cct_to_linear_rgb,
    calculate_optical_air_mass,
    calculate_sun_intensity,
    calculate_fov_from_lens,
    calculate_camera_framing_distance,
    build_inspection_camera_configs,
    setup_sun_lighting,
    setup_inspection_cameras,
)
from terra_forge.inspector.vision_rig import (
    build_2x2_vision_grid,
    resolve_snapshot_key,
)
from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    ErosionConfig,
    HydrologyConfig,
    BiomeScatterConfig,
    PeakConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    CavernConfig,
    AtmosphereConfig,
    PRESET_CONFIGS,
)


def create_non_square_heightfield(width: float, depth: float, res_x: int, res_y: int) -> Heightfield2D:
    """Helper to synthesize heightfields with distinct horizontal and vertical grid resolutions."""
    hf = Heightfield2D(width=width, depth=depth, resolution=res_x)
    hf.res_x = int(res_x)
    hf.res_y = int(res_y)
    hf.dx = hf.width / (hf.res_x - 1)
    hf.dy = hf.depth / (hf.res_y - 1)
    hf.x_coords = np.linspace(-hf.width / 2.0, hf.width / 2.0, hf.res_x, dtype=np.float64)
    hf.y_coords = np.linspace(-hf.depth / 2.0, hf.depth / 2.0, hf.res_y, dtype=np.float64)
    hf.X, hf.Y = np.meshgrid(hf.x_coords, hf.y_coords)
    hf.Z = np.zeros((hf.res_y, hf.res_x), dtype=np.float64)
    hf.sediment = np.zeros((hf.res_y, hf.res_x), dtype=np.float64)
    hf.scree_talus = np.zeros((hf.res_y, hf.res_x), dtype=np.float64)
    hf.water_depth = np.zeros((hf.res_y, hf.res_x), dtype=np.float64)
    return hf


# =============================================================================
# Dimension 1: High Aspect-Ratio Heightfields
# =============================================================================

class TestAdversarialAspectRatioHeightfield:
    """Adversarial testing of heightfields under extreme aspect ratios."""

    def test_high_aspect_ratio_16x256_normals_and_gradients(self):
        """Stress-test a narrow corridor heightfield (16m x 256m, 16x256 grid, aspect ratio 1:16)."""
        hf = create_non_square_heightfield(width=16.0, depth=256.0, res_x=16, res_y=256)
        assert hf.Z.shape == (256, 16)
        assert hf.dx == pytest.approx(16.0 / 15.0)
        assert hf.dy == pytest.approx(256.0 / 255.0)

        # Add longitudinal ridge and sharp peaks along the spine
        hf.add_peak(0.0, 0.0, height=35.0, radius=8.0, shape="horn")
        hf.add_peak(0.0, 80.0, height=25.0, radius=6.0, shape="ridge", ridge_angle_deg=90.0)

        dz_dx, dz_dy = hf.compute_gradients()
        assert dz_dx.shape == (256, 16)
        assert dz_dy.shape == (256, 16)
        assert np.all(np.isfinite(dz_dx))
        assert np.all(np.isfinite(dz_dy))

        normals = hf.compute_normals()
        assert normals.shape == (256, 16, 3)
        norms = np.linalg.norm(normals, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)

        slopes = hf.compute_slope_degrees()
        assert slopes.shape == (256, 16)
        assert np.all(slopes >= 0.0)
        assert np.all(slopes <= 90.0)

    def test_high_aspect_ratio_256x16_geological_strata_and_scarps(self):
        """Stress-test wide horizontal corridor (256m x 16m, 256x16 grid, aspect ratio 16:1)."""
        hf = create_non_square_heightfield(width=256.0, depth=16.0, res_x=256, res_y=16)

        # Apply strata folding across cross-strike orientation
        hf.apply_strata_folding(
            amplitude=8.0,
            wavelength=30.0,
            strike_angle_deg=90.0,
            plunge_wavelength=60.0,
            domain_warp_amp=4.0,
        )
        assert np.all(np.isfinite(hf.Z))

        # Apply canyon terracing on thin dimension
        hf.apply_canyon_terraces(terrace_height=4.0, terrace_sharpness=8.0, blend=0.9)
        assert np.all(np.isfinite(hf.Z))

        # Apply tectonic fault scarp
        hf.apply_tectonic_scarp(fault_angle_deg=0.0, displacement_throw=12.0, scarp_sharpness=12.0)
        assert np.all(np.isfinite(hf.Z))

        # Continuous bilinear sampling bounds check
        test_samples = [(-128.0, -8.0), (128.0, 8.0), (0.0, 0.0), (-64.0, 5.0), (100.0, -3.0)]
        for x, y in test_samples:
            z_val = hf.sample_bilinear(x, y)
            assert math.isfinite(z_val)

    def test_extreme_anisotropic_physical_extent_1x100(self):
        """Stress-test extreme physical dimensions: width=2.0m, depth=200.0m with resolution=64."""
        hf = Heightfield2D(width=2.0, depth=200.0, resolution=64)
        assert hf.dx == pytest.approx(2.0 / 63.0)
        assert hf.dy == pytest.approx(200.0 / 63.0)

        hf.add_peak(0.0, 0.0, height=20.0, radius=1.0, shape="cone")
        normals = hf.compute_normals()
        norms = np.linalg.norm(normals, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_coastal_profile_on_high_aspect_ratio_grid(self):
        """Verify coastal Dean's profile and wave-cut notch on non-square heightfield (32x128)."""
        hf = create_non_square_heightfield(width=40.0, depth=160.0, res_x=32, res_y=128)
        hf.Z[:] = 5.0

        hf.apply_coastal_profile(
            sea_level=0.0,
            wave_notch_depth=2.0,
            wave_notch_envelope=1.5,
            beach_slope_coefficient=0.15,
            berm_elevation=1.0,
            beach_width=10.0,
            coastline_angle_deg=0.0,
            coastline_offset=0.0,
        )

        assert np.all(np.isfinite(hf.Z))
        # Offshore seabed should be submerged below sea level
        assert np.any(hf.Z < 0.0)


# =============================================================================
# Dimension 2: Zero and Inverted Insolation Vectors
# =============================================================================

class TestAdversarialInsolationVectors:
    """Adversarial testing of solar insolation with degenerate and inverted vectors."""

    def test_zero_insolation_vector_graceful_fallback(self):
        """Verify zero vector [0, 0, 0] does not trigger ZeroDivisionError or produce NaNs."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf.Z[:] = 2.0
        normals = hf.compute_normals()

        # Pass pure zero vector
        zero_vec = np.array([0.0, 0.0, 0.0])
        insolation = BiomeEngine.compute_insolation(normals, sun_vector=zero_vec)

        assert insolation.shape == (20, 20)
        assert np.all(np.isfinite(insolation))
        assert np.all(insolation >= 0.0)
        assert np.all(insolation <= 1.0)
        # Fallback vector [0, 0, 1] on flat ground [0, 0, 1] yields 1.0 insolation
        assert np.allclose(insolation, 1.0)

    def test_inverted_downward_insolation_vector(self):
        """Verify downward pointing sun vector [0, 0, -1] is treated as illumination ray direction."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf.Z[:] = 5.0
        normals = hf.compute_normals()  # All [0, 0, 1]

        downward_vec = np.array([0.0, 0.0, -1.0])
        insolation = BiomeEngine.compute_insolation(normals, sun_vector=downward_vec)

        # For downward vector, dot(n, L) = -1.0, but code applies -dot when L_z < 0 -> 1.0
        assert np.allclose(insolation, 1.0)

    def test_inverted_arbitrary_downward_angles(self):
        """Verify arbitrary downward pointing sun rays always produce valid [0, 1] values."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf.add_peak(0.0, 0.0, height=15.0, radius=15.0, shape="horn")
        normals = hf.compute_normals()

        test_vectors = [
            np.array([1.0, 0.0, -1.0]),
            np.array([-0.5, -0.5, -2.0]),
            np.array([0.0, 1.0, -0.01]),
        ]

        for vec in test_vectors:
            insolation = BiomeEngine.compute_insolation(normals, sun_vector=vec)
            assert np.all(np.isfinite(insolation))
            assert np.all(insolation >= 0.0)
            assert np.all(insolation <= 1.0)

    def test_horizontal_grazing_insolation_vectors(self):
        """Verify horizontal grazing light (L_z == 0.0) produces 0.0 insolation on flat ground."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=15)
        hf.Z[:] = 0.0
        normals = hf.compute_normals()  # Upward [0, 0, 1]

        for grazing_vec in [np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([-1.0, 1.0, 0.0])]:
            insolation = BiomeEngine.compute_insolation(normals, sun_vector=grazing_vec)
            assert np.allclose(insolation, 0.0)

    def test_insolation_with_tuple_of_gradients(self):
        """Verify compute_insolation accepts both (nx, ny, nz) tuple and (H, W, 3) ndarray."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=15)
        hf.add_peak(0.0, 0.0, height=10.0, radius=10.0)
        normals = hf.compute_normals()
        normals_tuple = (normals[:, :, 0], normals[:, :, 1], normals[:, :, 2])

        sun_vec = np.array([0.5, 0.5, 0.707])
        iso_arr = BiomeEngine.compute_insolation(normals, sun_vec)
        iso_tup = BiomeEngine.compute_insolation(normals_tuple, sun_vec)

        assert np.allclose(iso_arr, iso_tup)


# =============================================================================
# Dimension 3: Negative and Extreme Temperatures and Moisture
# =============================================================================

class TestAdversarialMicroclimateTemperaturesAndMoisture:
    """Adversarial stress testing of microclimate equations under extreme boundary conditions."""

    def test_extreme_subzero_polar_temperatures(self):
        """Verify temperature calculations at Antarctic extremes (-60°C base, 5000m peak)."""
        elevation = np.array([[0.0, 1000.0], [3000.0, 5000.0]])
        temps = BiomeEngine.compute_temperature(
            elevation,
            sea_level=0.0,
            lapse_rate=0.0098,  # Dry adiabatic lapse rate
            base_temp=-60.0,
        )

        assert temps[0, 0] == pytest.approx(-60.0)
        assert temps[0, 1] == pytest.approx(-69.8)
        assert temps[1, 0] == pytest.approx(-89.4)
        assert temps[1, 1] == pytest.approx(-109.0)
        assert np.all(np.isfinite(temps))

    def test_temperature_inversion_negative_lapse_rate(self):
        """Verify thermal atmospheric inversion (negative lapse rate where high altitude is warmer)."""
        elevation = np.linspace(0.0, 2000.0, 10)
        temps = BiomeEngine.compute_temperature(
            elevation,
            sea_level=0.0,
            lapse_rate=-0.004,  # High ground warmer
            base_temp=-10.0,
        )

        assert temps[-1] > temps[0]
        assert temps[-1] == pytest.approx(-10.0 - (-0.004 * 2000.0))

    def test_extreme_soil_moisture_hyper_arid_and_deluge(self):
        """Verify soil moisture index stays strictly clamped within [0.0, 1.0] under extreme precipitation/drought."""
        water_dist = np.array([[0.0, 10.0], [500.0, 1e6]])
        slopes = np.array([[0.0, 30.0], [60.0, 89.0]])
        insolation = np.array([[0.0, 0.5], [0.8, 1.0]])

        # Extreme drought: base_rain = -5.0
        arid_moist = BiomeEngine.compute_moisture(
            water_dist, slopes, insolation, sigma_moisture=15.0, base_rain=-5.0
        )
        assert np.all(arid_moist >= 0.0)
        assert np.all(arid_moist <= 1.0)
        assert arid_moist[1, 1] == 0.0  # Completely dry desert

        # Deluge: base_rain = 5.0
        deluge_moist = BiomeEngine.compute_moisture(
            water_dist, slopes, insolation, sigma_moisture=15.0, base_rain=5.0
        )
        assert np.all(deluge_moist >= 0.0)
        assert np.all(deluge_moist <= 1.0)
        assert deluge_moist[0, 0] == 1.0  # Saturated

    def test_negative_sigma_moisture_division_by_zero_safety(self):
        """Verify negative or zero sigma_moisture does not cause ZeroDivisionError."""
        water_dist = np.full((10, 10), 20.0)
        # Should be protected by max(1.0, sigma_moisture)
        moist = BiomeEngine.compute_moisture(water_dist, sigma_moisture=-10.0)
        assert np.all(np.isfinite(moist))
        assert np.all(moist >= 0.0)
        assert np.all(moist <= 1.0)

    def test_inverted_niche_boundaries_returns_zero_suitability(self):
        """Verify inverted niche thresholds (min > max) yield 0 suitability without unhandled exceptions."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf.Z[:] = 10.0
        dist_to_water = np.full((20, 20), 5.0)

        engine = BiomeEngine(hf, dist_to_water)
        cfg = BiomeScatterConfig(
            name="InvertedNicheTest",
            prototype="pine_tall",
            min_altitude=5.0,
            max_altitude=25.0,
            min_slope_deg=0.0,
            max_slope_deg=45.0,
        )

        suit = engine.evaluate_microclimate_suitability(
            cfg,
            min_temp=40.0,
            max_temp=10.0,  # Inverted: 40 > 10
            min_moisture=0.8,
            max_moisture=0.2,  # Inverted: 0.8 > 0.2
        )
        assert np.all(suit == 0.0)


# =============================================================================
# Dimension 4: Zero-Droplet and Single-Droplet Erosion Runs
# =============================================================================

class TestAdversarialErosionDroplets:
    """Adversarial stress testing of hydraulic and thermal erosion simulators."""

    def test_zero_droplet_hydraulic_erosion_mass_conservation(self):
        """Verify droplet_count=0 leaves heightfield exactly unmodified with ΔM == 0.0."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=20.0, radius=12.0)
        initial_z = hf.Z.copy()
        initial_mass = float(np.sum(initial_z))

        cfg = ErosionConfig(enabled=True, droplet_count=0)
        sim = ErosionSimulator(hf, cfg)
        out_hf, sediment, water = sim.simulate_hydraulic(seed=42)

        assert np.array_equal(out_hf.Z, initial_z)
        assert float(np.sum(out_hf.Z)) == pytest.approx(initial_mass)
        assert np.all(sediment == 0.0)
        assert np.all(water == 0.0)

    def test_disabled_erosion_simulator(self):
        """Verify enabled=False exits immediately without side effects."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=25)
        hf.Z[:] = 10.0
        initial_z = hf.Z.copy()

        cfg = ErosionConfig(enabled=False, droplet_count=1000)
        sim = ErosionSimulator(hf, cfg)
        out_hf, sediment, water = sim.simulate_hydraulic(seed=123)

        assert np.array_equal(out_hf.Z, initial_z)
        assert np.all(sediment == 0.0)

    def test_single_droplet_hydraulic_erosion_mass_conservation(self):
        """Verify exactly 1 droplet runs to completion and satisfies exact mass conservation (|ΔM| < 1e-9)."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0)
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=1,
            max_droplet_steps=64,
            erode_speed=0.5,
            deposit_speed=0.5,
        )
        sim = ErosionSimulator(hf, cfg)
        out_hf, sediment, water = sim.simulate_hydraulic(seed=999)

        final_mass = float(np.sum(out_hf.Z))
        assert abs(final_mass - initial_mass) < 1e-9, (
            f"Mass conservation violated with 1 droplet: initial={initial_mass}, final={final_mass}"
        )
        assert np.sum(water) > 0.0  # Single droplet left flux trace

    def test_extreme_inertia_zero_and_one(self):
        """Verify droplet inertia extremes (0.0: pure gradient follower, 1.0: pure ballistic momentum)."""
        for inertia in [0.0, 1.0]:
            hf = Heightfield2D(width=40.0, depth=40.0, resolution=25)
            hf.add_peak(0.0, 0.0, height=20.0, radius=12.0)
            initial_mass = float(np.sum(hf.Z))

            cfg = ErosionConfig(
                enabled=True,
                droplet_count=20,
                droplet_inertia=inertia,
                max_droplet_steps=32,
            )
            sim = ErosionSimulator(hf, cfg)
            sim.simulate_hydraulic(seed=42)

            final_mass = float(np.sum(hf.Z))
            assert abs(final_mass - initial_mass) < 1e-9

    def test_extreme_thermal_talus_angles(self):
        """Verify 8-neighbor thermal talus at steep (89°) vs shallow (1°) angle of repose."""
        # 1. 89°: almost vertical cliff is stable, 0 volume slumps
        hf_steep = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf_steep.add_peak(0.0, 0.0, height=15.0, radius=10.0)
        initial_steep = hf_steep.Z.copy()

        cfg_steep = ErosionConfig(
            enabled=True,
            thermal_iterations=5,
            talus_angle_deg=89.0,
            scree_talus_deposition=0.5,
        )
        sim_steep = ErosionSimulator(hf_steep, cfg_steep)
        sim_steep.run_thermal_erosion()
        assert np.allclose(hf_steep.Z, initial_steep)

        # 2. 1°: extreme collapse; all slopes flatten down, mass strictly conserved
        hf_flat = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf_flat.add_peak(0.0, 0.0, height=15.0, radius=10.0)
        initial_mass = float(np.sum(hf_flat.Z))

        cfg_flat = ErosionConfig(
            enabled=True,
            thermal_iterations=5,
            talus_angle_deg=1.0,
            scree_talus_deposition=0.5,
        )
        sim_flat = ErosionSimulator(hf_flat, cfg_flat)
        sim_flat.run_thermal_erosion()

        final_mass = float(np.sum(hf_flat.Z))
        assert abs(final_mass - initial_mass) < 1e-9
        assert np.sum(hf_flat.scree_talus) > 0.0  # Scree recorded


# =============================================================================
# Dimension 5: High-Density Poisson-Disk Packing Limits
# =============================================================================

class TestAdversarialPoissonDiskPackingLimits:
    """Adversarial stress testing of spatial-grid Poisson-disk blue noise clustering."""

    def test_hyper_dense_poisson_saturation_termination(self):
        """Verify Bridson algorithm terminates cleanly when domain reaches maximum geometric packing."""
        width, depth = 30.0, 30.0
        r_min = 2.5
        # Target density asks for 1000 points in 900 m2 (which can physically hold at most ~165 disks)
        points = bridson_poisson_disk_sampling(width=width, depth=depth, r_min=r_min, k=20, seed=42)

        assert len(points) > 0
        assert len(points) < 200  # Physically bounded by disk packing limit

        # Verify pairwise distance invariant: dist >= r_min - 1e-5
        pts_arr = np.array(points)
        for i in range(len(pts_arr)):
            dists = np.linalg.norm(pts_arr[i + 1:] - pts_arr[i], axis=1)
            if len(dists) > 0:
                min_d = np.min(dists)
                assert min_d >= r_min - 1e-5, f"Poisson-disk violation: min dist {min_d} < {r_min}"

    def test_poisson_disk_radius_larger_than_domain(self):
        """Verify passing r_min >= max(width, depth) places at most 1 point without infinite loops."""
        width, depth = 20.0, 20.0
        r_min = 35.0  # Larger than diorama
        points = bridson_poisson_disk_sampling(width=width, depth=depth, r_min=r_min, seed=101)

        assert len(points) <= 1
        if len(points) == 1:
            px, py = points[0]
            assert -width / 2.0 <= px <= width / 2.0
            assert -depth / 2.0 <= py <= depth / 2.0

    def test_poisson_scatter_zero_suitability_map(self):
        """Verify passing all-zero suitability map returns empty list immediately."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        dist_to_water = np.full((20, 20), 10.0)
        engine = BiomeEngine(hf, dist_to_water)

        zero_map = np.zeros((20, 20))
        pts = engine.poisson_scatter(zero_map, min_radius=2.0, max_count=50)
        assert pts == []

    def test_poisson_scatter_strict_boundary_clamping(self):
        """Verify generated scatter points never exceed [-W/2, W/2] and [-D/2, D/2]."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z[:] = 10.0
        dist_to_water = np.full((25, 25), 10.0)
        engine = BiomeEngine(hf, dist_to_water)

        cfg = BiomeScatterConfig(
            name="BoundaryTest",
            prototype="pine_tall",
            min_altitude=0.0,
            max_altitude=50.0,
            density=0.5,
            poisson_dist_min=2.0,
            seed=777,
        )

        instances = engine.generate_poisson_scatter(cfg, max_samples=300)
        assert len(instances) > 0

        for x, y, z, scale, rot_z in instances:
            assert -25.0 <= x <= 25.0, f"X coordinate leaked outside bounds: {x}"
            assert -25.0 <= y <= 25.0, f"Y coordinate leaked outside bounds: {y}"
            assert math.isfinite(z)
            assert scale > 0.0
            assert 0.0 <= rot_z <= 2.0 * math.pi


# =============================================================================
# Dimension 6: Diorama Slabs with Non-Square Resolutions and High Aspect Ratios
# =============================================================================

class TestAdversarialDioramaSlabNonSquareAndHighAspectRatio:
    """Adversarial stress testing of diorama 2-manifold topological invariants on non-square grids."""

    @pytest.mark.parametrize(
        "res_x, res_y, w, d",
        [
            (16, 256, 16.0, 256.0),   # 1:16 aspect ratio
            (256, 16, 256.0, 16.0),   # 16:1 aspect ratio
            (17, 65, 30.0, 120.0),    # Odd non-square
            (65, 17, 120.0, 30.0),    # Transposed odd non-square
            (4, 32, 10.0, 80.0),      # Minimal grid corridor
            (32, 4, 80.0, 10.0),      # Minimal grid strip
        ],
    )
    def test_non_square_diorama_slab_topological_invariants(self, res_x, res_y, w, d):
        """
        Verify that DioramaSlabBuilder produces 100% watertight 2-manifold diorama solids
        satisfying Euler-Poincare characteristic chi = V - E + F == 2 across all non-square grids.
        """
        hf = create_non_square_heightfield(width=w, depth=d, res_x=res_x, res_y=res_y)
        # Add undulating surface
        hf.Z = 4.0 + 2.0 * np.sin(0.1 * hf.X) * np.cos(0.1 * hf.Y)

        bounds = DioramaBounds(width=w, depth=d, z_base=-10.0, resolution=max(res_x, res_y))
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        metrics = topo["metrics"]

        assert topo["passed"] is True, f"Topology validation failed for {res_x}x{res_y}: {topo['errors']}"
        assert topo["is_watertight"] is True
        assert metrics["boundary_edges"] == 0, f"Open boundary edges: {metrics['boundary_edges']}"
        assert metrics["euler_characteristic"] == 2, f"Euler chi={metrics['euler_characteristic']} != 2"
        assert metrics["non_manifold_edges"] == 0
        assert metrics["non_manifold_vertices"] == 0
        assert metrics["orientation_conflicts"] == 0

    def test_minimal_3x3_diorama_slab(self):
        """Verify minimal 3x3 grid diorama slab topology."""
        hf = Heightfield2D(width=10.0, depth=10.0, resolution=3)
        hf.Z[:] = 2.0
        bounds = DioramaBounds(width=10.0, depth=10.0, z_base=-5.0, resolution=3)
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        assert topo["is_watertight"] is True
        assert topo["euler_characteristic"] == 2

    def test_deep_subterranean_baseplate_vertex_colors(self):
        """Verify deep subterranean baseplate (z_base = -150.0m) calculates vertex colors without overflow."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=16)
        hf.Z[:] = 30.0
        bounds = DioramaBounds(width=40.0, depth=40.0, z_base=-150.0, resolution=16)
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        colors = mesh_data["vertex_colors"]
        assert len(colors) == len(mesh_data["vertices"])
        for r, g, b, a in colors:
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0
            assert a == 1.0


# =============================================================================
# Dimension 7: Multiple Disconnected Lake Bodies and Dry River Channels
# =============================================================================

class TestAdversarialHydrologyMultipleLakesAndDryChannels:
    """Adversarial stress testing of hydrology under dry channels and disconnected multi-lake setups."""

    def test_dry_river_channel_disabled_hydrology(self):
        """Verify river_enabled=False produces zero flow velocity and leaves terrain untouched."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
        hf.Z[:] = 10.0
        initial_z = hf.Z.copy()

        cfg = HydrologyConfig(
            river_enabled=False,
            lake_radius=10.0,
            lake_center=(0.0, 0.0),
            lake_level=5.0,
        )
        engine = HydrologyEngine(hf, cfg)

        # Carving should do nothing
        engine.carve_river_channel()
        assert np.array_equal(hf.Z, initial_z)

        # Velocity field should be all zero
        vx, vy = engine.compute_flow_velocity_field()
        assert np.all(vx == 0.0)
        assert np.all(vy == 0.0)

        # Water distance field should be distance to lake only
        dist = engine.compute_water_distance_field()
        center_dist = np.sqrt(hf.X**2 + hf.Y**2)
        expected_dist = np.maximum(0.0, center_dist - 10.0)
        assert np.allclose(dist, expected_dist)

    def test_empty_river_stages_evaluation(self):
        """Verify river_stages=[] evaluates safely without IndexError."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=15)
        cfg = HydrologyConfig(river_enabled=True, river_stages=[])
        engine = HydrologyEngine(hf, cfg)

        # Scalar evaluation
        res_scalar = engine.evaluate_river_spline(0.5)
        assert isinstance(res_scalar[0], RiverSplineScalar)
        assert float(res_scalar[0]) == 0.0

        # Vector evaluation
        t_arr = np.array([0.0, 0.5, 1.0])
        rx, ry, rz, rw = engine.evaluate_river_spline(t_arr)
        assert rx.shape == (3,)
        assert np.all(rx == 0.0)

    def test_multiple_disconnected_lakes_carving(self):
        """Verify carving multiple isolated lakes at different elevations across heightfield."""
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=64)
        hf.Z[:] = 15.0

        # Lake 1: High alpine tarn (NW)
        hf.carve_lake_basin(
            center_x=-35.0,
            center_y=35.0,
            radius=15.0,
            water_level=12.0,
            lake_depth=6.0,
            berm_height=1.5,
        )
        # Lake 2: Valley lake (SE)
        hf.carve_lake_basin(
            center_x=30.0,
            center_y=-30.0,
            radius=20.0,
            water_level=6.0,
            lake_depth=8.0,
            berm_height=1.2,
        )

        assert np.all(np.isfinite(hf.Z))
        # Center of Lake 1 should be carved to water_level - lake_depth = 6.0
        z_tarn = hf.sample_bilinear(-35.0, 35.0)
        assert z_tarn == pytest.approx(6.0, abs=0.5)

        # Center of Lake 2 should be carved to 6.0 - 8.0 = -2.0
        z_valley = hf.sample_bilinear(30.0, -30.0)
        assert z_valley == pytest.approx(-2.0, abs=0.5)

    def test_water_basin_wall_breach_detection(self):
        """Verify QualityValidator correctly flags when a lake intersects exterior diorama wall."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=32)
        hf.Z[:] = 10.0

        # Half width is 40.0. Place lake center at x=30.0 with radius=15.0 -> breaches wall at x=40 by 5.0m
        map_cfg = MapConfig(
            name="BreachTest",
            diorama=DioramaBounds(width=80.0, depth=80.0, z_base=-10.0),
            hydrology=HydrologyConfig(
                lake_center=(30.0, 0.0),
                lake_radius=15.0,
                lake_level=8.0,
            ),
        )

        val = QualityValidator(map_cfg, hf)
        res = val.validate_water_basin_containment()

        assert res["passed"] is False
        assert any("breach" in err.lower() or "wall clearance" in err.lower() for err in res["errors"])

    def test_lake_berm_freeboard_violation_detection(self):
        """Verify QualityValidator flags insufficient retaining berm freeboard (< 0.50m)."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=32)
        # Flat ground at 4.2m
        hf.Z[:] = 4.2

        # Lake level is 4.0m, meaning surrounding terrain is only 0.20m above water (below 0.50m threshold)
        map_cfg = MapConfig(
            name="BermTest",
            diorama=DioramaBounds(width=80.0, depth=80.0, z_base=-10.0),
            hydrology=HydrologyConfig(
                lake_center=(0.0, 0.0),
                lake_radius=12.0,
                lake_level=4.0,
                lake_depth=5.0,
                river_enabled=False,
            ),
        )

        val = QualityValidator(map_cfg, hf)
        res = val.validate_lake_berm(min_freeboard_threshold=0.50)

        assert res["passed"] is False
        assert any("freeboard" in err.lower() for err in res["errors"])


# =============================================================================
# Dimension 8: Presets Mutated with Extreme Parameters
# =============================================================================

class TestAdversarialPresetMutations:
    """Adversarial stress testing of full preset generation pipelines with mutated parameters."""

    def test_alpine_preset_mutated_extreme_horns(self):
        """Mutate Alpine preset with towering 120m horn, deep cirque, and verify watertight diorama."""
        cfg = copy.deepcopy(PRESET_CONFIGS["alpine_peak_glacial_lake"])
        # Extreme heights
        cfg.peaks[0].height = 120.0
        cfg.peaks[0].radius = 45.0
        cfg.erosion.droplet_count = 100  # Fast execution for CI
        cfg.erosion.thermal_iterations = 8

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=33)
        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(seed=42)
        ero.run_thermal_erosion()

        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()
        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])

        assert topo["is_watertight"] is True
        assert topo["metrics"]["euler_characteristic"] == 2

    def test_canyon_preset_mutated_extreme_terraces(self):
        """Mutate Canyon preset with massive 25m step terraces and aggressive sharpness."""
        cfg = copy.deepcopy(PRESET_CONFIGS["canyon_semiarid_plateau"])
        cfg.strata.terrace_height = 25.0
        cfg.strata.terrace_sharpness = 30.0
        cfg.strata.terrace_blend = 1.0

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=33)
        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape)

        hf.apply_canyon_terraces(
            terrace_height=cfg.strata.terrace_height,
            terrace_sharpness=cfg.strata.terrace_sharpness,
            blend=cfg.strata.terrace_blend,
        )

        assert np.all(np.isfinite(hf.Z))
        # Topography should have discrete steps
        slopes = hf.compute_slope_degrees()
        assert np.all(np.isfinite(slopes))

    def test_tropical_preset_mutated_extreme_sea_level(self):
        """Mutate Tropical Island preset with elevated sea level (+20m) and intense wave notch."""
        cfg = copy.deepcopy(PRESET_CONFIGS["tropical_island_coastal_karst"])
        cfg.coastal.sea_level = 20.0
        cfg.coastal.wave_notch_depth = 6.0
        cfg.coastal.wave_notch_envelope = 3.0

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=33)
        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape)

        hf.apply_coastal_profile(
            sea_level=cfg.coastal.sea_level,
            wave_notch_depth=cfg.coastal.wave_notch_depth,
            wave_notch_envelope=cfg.coastal.wave_notch_envelope,
        )
        assert np.all(np.isfinite(hf.Z))

    def test_diurnal_lighting_extreme_temporal_mutations(self):
        """Stress-test diurnal solar equations with out-of-range hours (-720.0, 1e6, fractional)."""
        extreme_hours = [-720.0, -24.5, 0.0, 12.0, 24.0, 100.5, 1e6]

        for t in extreme_hours:
            elev, azim, vec = calculate_solar_position(t)
            assert -90.0 <= elev <= 90.0
            assert 0.0 <= azim < 360.0
            assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

            cct = calculate_cct_from_elevation(elev)
            assert 2000.0 <= cct <= 10000.0

            r, g, b = cct_to_linear_rgb(cct)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

            intensity = calculate_sun_intensity(elev)
            assert intensity >= 0.0
            if elev <= 0.0:
                assert intensity == 0.0

    def test_camera_framing_distance_extreme_aspect_ratios(self):
        """Verify automated camera framing math d = max(w, d) / (sqrt(2) * sin(FOV/2)) under extreme aspect ratios."""
        for w, d in [(10.0, 500.0), (500.0, 10.0), (1.0, 1000.0)]:
            fov = calculate_fov_from_lens(lens_mm=50.0)
            dist = calculate_camera_framing_distance(width=w, depth=d, fov_deg=fov)
            assert math.isfinite(dist)
            assert dist > max(w, d)  # Distance must exceed max span to frame diorama

            configs = build_inspection_camera_configs(bounds=DioramaBounds(width=w, depth=d))
            assert "isometric_diorama" in configs
            assert "eye_level_first_person" in configs
            assert "closeup_macro" in configs


# =============================================================================
# Dimension 9: Ecological Stratification and Creature Impacts
# =============================================================================

class TestAdversarialEcologicalStratificationAndCreatureImpacts:
    """Adversarial stress testing of multi-tier ecology and creature environmental impacts."""

    def test_creature_game_trails_generation_and_rasterization(self):
        """Verify animal game trail simulation and continuous proximity rasterization."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=40)
        hf.add_peak(0.0, 0.0, height=20.0, radius=25.0)
        dist_to_water = np.sqrt(hf.X**2 + hf.Y**2)

        engine = BiomeEngine(hf, dist_to_water)
        trails = engine.generate_game_trails(count=4, seed=42)

        assert len(trails) > 0
        for trail in trails:
            assert trail.ndim == 2
            assert trail.shape[1] == 2
            assert len(trail) >= 4
            # Trail points must stay inside diorama bounds
            assert np.all(trail[:, 0] >= -40.0)
            assert np.all(trail[:, 0] <= 40.0)
            assert np.all(trail[:, 1] >= -40.0)
            assert np.all(trail[:, 1] <= 40.0)

        # Rasterize trails into continuous mask
        trail_mask = engine.compute_game_trail_mask(trails=trails, trail_width=2.5)
        assert trail_mask.shape == (40, 40)
        assert np.all(np.isfinite(trail_mask))
        assert np.all(trail_mask >= 0.0)
        assert np.all(trail_mask <= 1.0)
        assert np.any(trail_mask > 0.0)

        # Churned/disturbed soil patches
        soil_mask = engine.compute_disturbed_soil_mask(trail_mask=trail_mask, churn_patches=6, seed=99)
        assert soil_mask.shape == (40, 40)
        assert np.all(np.isfinite(soil_mask))
        assert np.all(soil_mask >= 0.0)
        assert np.all(soil_mask <= 1.0)

    def test_underwater_ecosystem_mask_bathymetry(self):
        """Verify underwater ecosystem mask isolates submerged marine/lake zones within depth/slope bounds."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=30)
        # Gradient from -20m to +20m
        hf.Z = np.linspace(-20.0, 20.0, 30)[:, np.newaxis] * np.ones((1, 30))

        engine = BiomeEngine(hf, dist_to_water=np.zeros((30, 30)), sea_level=0.0)
        underwater_mask = engine.compute_underwater_ecosystem_mask(max_depth=15.0, max_slope_deg=45.0)

        assert underwater_mask.shape == (30, 30)
        # Positive elevations (> 0.2m) must NEVER be classified as underwater
        above_water = hf.Z > 0.2
        assert np.all(underwater_mask[above_water] == 0.0)

        # Extreme deep trenches (> 15m deep) must be excluded by max_depth
        too_deep = hf.Z < -15.0
        assert np.all(underwater_mask[too_deep] == 0.0)

    def test_stratified_biome_configs_taxonomical_tiers(self):
        """Verify that create_stratified_biome_configs generates valid configs for all 6 ecological tiers."""
        configs = BiomeEngine.create_stratified_biome_configs(
            include_underwater=True,
            include_decay=True,
            include_creature_trails=True,
        )

        tiers_found = set()
        for cfg in configs:
            assert isinstance(cfg, BiomeScatterConfig)
            assert cfg.prototype in PROTOTYPE_TAXONOMY
            tier = PROTOTYPE_TAXONOMY[cfg.prototype]["tier"]
            tiers_found.add(tier)
            assert cfg.density > 0.0
            assert cfg.poisson_dist_min > 0.0

        expected_tiers = {
            EcologicalTier.CANOPY,
            EcologicalTier.UNDERSTORY,
            EcologicalTier.GROUND_FLORA,
            EcologicalTier.DECAY_WINDTHROW,
            EcologicalTier.BIOLOGICAL_IMPACT,
            EcologicalTier.UNDERWATER,
        }
        assert tiers_found == expected_tiers

    def test_poisson_scatter_on_high_aspect_ratio_corridor(self):
        """Verify Poisson scatter instances on a narrow corridor heightfield (16m x 160m)."""
        hf = create_non_square_heightfield(width=16.0, depth=160.0, res_x=16, res_y=160)
        hf.Z[:] = 8.0
        dist_to_water = np.full((160, 16), 15.0)
        engine = BiomeEngine(hf, dist_to_water)

        cfg = BiomeScatterConfig(
            name="CorridorScatter",
            prototype="shrub",
            min_altitude=0.0,
            max_altitude=20.0,
            density=0.1,
            poisson_dist_min=2.0,
            seed=42,
        )

        instances = engine.generate_poisson_scatter(cfg, max_samples=100)
        assert len(instances) > 0
        for x, y, z, scale, rot_z in instances:
            assert -8.0 <= x <= 8.0
            assert -80.0 <= y <= 80.0
            assert math.isfinite(z)


# =============================================================================
# Dimension 10: Topological 2-Manifold Validator Invariants
# =============================================================================

class TestAdversarialValidatorTopologicalInvariants:
    """Adversarial stress testing of QualityValidator topological and geometric checks."""

    def test_validator_catches_degenerate_mesh_too_few_vertices_or_faces(self):
        """Verify validator flags degenerate meshes with V < 4 or F < 4."""
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        faces = [[0, 1]]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert any("degenerate mesh" in e.lower() for e in res["errors"])

    def test_validator_catches_open_boundary_edges(self):
        """Verify validator flags open boundary edges on non-closed meshes with V>=4, F>=4."""
        # 4 triangular sides of an open pyramid without bottom baseplate (5 vertices, 4 faces)
        vertices = [
            (0.0, 0.0, 0.0),   # 0
            (1.0, 0.0, 0.0),   # 1
            (1.0, 1.0, 0.0),   # 2
            (0.0, 1.0, 0.0),   # 3
            (0.5, 0.5, 1.0),   # 4: apex
        ]
        faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["is_watertight"] is False
        assert res["boundary_edges"] == 4
        assert any("watertight violation" in e.lower() for e in res["errors"])

    def test_validator_catches_non_manifold_edge_three_incident_faces(self):
        """Verify validator flags non-manifold edges with >=3 incident faces."""
        # Closed tetrahedron (4 faces) plus a 5th face attached to edge (1, 2)
        vertices = [
            (0.0, 0.0, 0.0),  # 0
            (1.0, 0.0, 0.0),  # 1
            (0.5, 0.86, 0.0), # 2
            (0.5, 0.28, 0.8), # 3
            (2.0, 1.0, 0.0),  # 4
        ]
        faces = [
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [0, 2, 1],
            [1, 2, 4],        # 5th face incident to edge (1, 2)
        ]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["non_manifold_edges"] >= 1
        assert any("non-manifold edge" in e.lower() for e in res["errors"])

    def test_validator_catches_conflicting_face_orientations(self):
        """Verify validator flags inconsistent normal winding on closed mesh."""
        # Tetrahedron with one reversed face causing identical edge traversal
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.86, 0.0),
            (0.5, 0.28, 0.8),
        ]
        # Normal tetrahedron: [0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]
        # Reverse [0, 2, 1] to [0, 1, 2] -> edge (0, 1) traversed in same direction
        faces = [
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [0, 1, 2],
        ]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["orientation_conflicts"] >= 1
        assert any("normal winding violation" in e.lower() for e in res["errors"])

    def test_validator_catches_non_manifold_pinch_point_vertex(self):
        """Verify validator flags pinch-point non-manifold vertices (two cones/tetrahedra touching at 1 apex)."""
        # Tetrahedron 1: vertices 0, 1, 2, 3 (apex is 0)
        # Tetrahedron 2: vertices 0, 4, 5, 6 (apex is 0, pinch point!)
        vertices = [
            (0.0, 0.0, 0.0),   # 0: shared pinch apex
            (1.0, 0.0, 1.0),   # 1
            (0.0, 1.0, 1.0),   # 2
            (-1.0, 0.0, 1.0),  # 3
            (1.0, 0.0, -1.0),  # 4
            (0.0, 1.0, -1.0),  # 5
            (-1.0, 0.0, -1.0), # 6
        ]
        faces = [
            # Tet 1
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2],
            # Tet 2
            [0, 5, 4], [0, 6, 5], [0, 4, 6], [4, 5, 6],
        ]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["non_manifold_vertices"] >= 1
        assert res["metrics"]["pinch_point_vertices"] >= 1

    def test_validator_catches_degenerate_faces_and_out_of_bounds_indices(self):
        """Verify validator flags degenerate faces (< 3 verts, duplicates, out-of-bounds)."""
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        faces = [
            [0, 1],            # < 3 vertices
            [0, 1, 1],         # Duplicate vertex
            [0, 1, 999],       # Out of bounds index
            [0, 1, 2, 3],      # Valid quad
        ]

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert len(res["errors"]) >= 3

    def test_cavern_clearance_roof_breach_detection(self):
        """Verify validate_cavern_clearance detects roof punching through surface."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        # Terrain at 10.0m
        hf.Z[:] = 10.0

        # Cavern center z = 6.0, height = 6.0 -> roof apex = 6.0 + 3.0 = 9.0m.
        # Surface is 10.0m, rock overburden is only 1.0m (below 5.0m threshold)
        map_cfg = MapConfig(
            name="CavernTest",
            cavern=CavernConfig(
                enabled=True,
                center=(0.0, 0.0, 6.0),
                height=6.0,
                min_roof_clearance=5.0,
            ),
        )

        val = QualityValidator(map_cfg, hf)
        res_fail = val.validate_cavern_clearance()
        assert res_fail["passed"] is False
        assert any("cavern clearance violation" in e.lower() for e in res_fail["errors"])

        # Elevate terrain to 20.0m -> rock overburden is 11.0m (passes)
        hf.Z[:] = 20.0
        res_pass = val.validate_cavern_clearance()
        assert res_pass["passed"] is True

    def test_validate_all_unified_pipeline(self):
        """Verify validate_all executes all topology, hydrological, and cavern checks."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z[:] = 15.0
        map_cfg = MapConfig(
            name="UnifiedValidationTest",
            diorama=DioramaBounds(width=40.0, depth=40.0, z_base=-10.0, resolution=21),
            hydrology=HydrologyConfig(lake_radius=0.0, river_enabled=False),
            cavern=CavernConfig(enabled=False),
        )

        val = QualityValidator(map_cfg, hf)
        results = val.validate_all()

        assert results["passed"] is True
        assert results["metrics"]["is_watertight"] is True
        assert results["metrics"]["euler_characteristic"] == 2


# =============================================================================
# Dimension 11: Hydrology Flow Velocity and Arbitrary Spline Dynamics
# =============================================================================

class TestAdversarialHydrologyVelocityAndSplines:
    """Adversarial testing of hydrology flow velocity equations and dynamic spline tangents."""

    def test_evaluate_river_spline_hermite_tangents(self):
        """Verify parametric spline evaluation with custom incoming/outgoing Hermite tangents."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
        cfg = HydrologyConfig(
            river_enabled=True,
            river_stages=[
                {
                    "name": "HermiteStage",
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "start": [-20.0, -20.0, 15.0, 3.0],
                    "end": [20.0, 20.0, 2.0, 5.0],
                    "tangent_in": [10.0, 0.0, -5.0],
                    "tangent_out": [0.0, 10.0, -2.0],
                }
            ],
        )

        engine = HydrologyEngine(hf, cfg)
        t_samples = np.linspace(0.0, 1.0, 50)
        rx, ry, rz, rw = engine.evaluate_river_spline(t_samples)

        assert rx.shape == (50,)
        assert np.all(np.isfinite(rx))
        assert np.all(np.isfinite(ry))
        assert np.all(np.isfinite(rz))
        assert np.all(rw >= 3.0)
        assert rx[0] == pytest.approx(-20.0)
        assert rx[-1] == pytest.approx(20.0)

    def test_evaluate_river_spline_meander_oscillations(self):
        """Verify meander amplitude and cycle oscillations along river normal vector."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
        cfg = HydrologyConfig(
            river_enabled=True,
            river_meander_amplitude=2.0,
            river_stages=[
                {
                    "name": "OscillatingStage",
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "start": [0.0, -25.0, 10.0, 4.0],
                    "end": [0.0, 25.0, 2.0, 4.0],
                    "meander_amp": 3.0,
                    "meander_cycles": 2.0,
                }
            ],
        )

        engine = HydrologyEngine(hf, cfg)
        t_samples = np.linspace(0.0, 1.0, 100)
        rx, ry, rz, rw = engine.evaluate_river_spline(t_samples)

        # Baseline trajectory is along Y axis (X = 0). Normal is along X.
        # Check that X exhibits lateral sinusoidal oscillation
        assert np.max(np.abs(rx)) > 0.5
        assert np.min(rx) < -0.5

    def test_evaluate_flow_velocity_acceleration_on_drop(self):
        """Verify downstream gradient (-dz/dt) accelerates flow velocity over waterfalls."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
        # Stage with massive vertical drop (cascade) vs stage with zero drop (flat lake)
        cfg = HydrologyConfig(
            river_enabled=True,
            river_stages=[
                {
                    "name": "Cascade",
                    "t_start": 0.0,
                    "t_end": 0.5,
                    "start": [-20.0, 0.0, 30.0, 3.0],
                    "end": [-10.0, 0.0, 2.0, 3.0],  # 28m drop in t=0.5
                },
                {
                    "name": "FlatPool",
                    "t_start": 0.5,
                    "t_end": 1.0,
                    "start": [-10.0, 0.0, 2.0, 3.0],
                    "end": [10.0, 0.0, 2.0, 3.0],   # 0m drop in t=0.5
                },
            ],
        )

        engine = HydrologyEngine(hf, cfg)
        # Sample on cascade (t=0.25) and on flat pool (t=0.75)
        vx_drop, vy_drop, vz_drop = engine.evaluate_flow_velocity(0.25)
        vx_flat, vy_flat, vz_flat = engine.evaluate_flow_velocity(0.75)

        speed_drop = math.sqrt(float(vx_drop)**2 + float(vy_drop)**2 + float(vz_drop)**2)
        speed_flat = math.sqrt(float(vx_flat)**2 + float(vy_flat)**2 + float(vz_flat)**2)

        # Drop must induce higher flow velocity
        assert speed_drop > speed_flat

    def test_compute_water_distance_field_both_lake_and_river(self):
        """Verify water distance field finds the minimum Euclidean distance between river and lake."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=40)
        cfg = HydrologyConfig(
            river_enabled=True,
            lake_radius=10.0,
            lake_center=(20.0, 20.0),
            river_stages=[
                {
                    "name": "StraightRiver",
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "start": [-30.0, -20.0, 10.0, 2.0],
                    "end": [-30.0, 20.0, 5.0, 2.0],
                }
            ],
        )

        engine = HydrologyEngine(hf, cfg)
        dist = engine.compute_water_distance_field()

        assert dist.shape == (40, 40)
        assert np.all(dist >= 0.0)

        # Near lake center (20, 20), distance should be 0.0
        gx, gy = hf.world_to_grid(20.0, 20.0)
        assert dist[int(gy), int(gx)] == 0.0

        # Near river at (-30, 0), distance should be 0.0
        rgx, rgy = hf.world_to_grid(-30.0, 0.0)
        assert dist[int(rgy), int(rgx)] == 0.0


# =============================================================================
# Dimension 12: Erosion Kinematic and Boundary Extremes
# =============================================================================

class TestAdversarialErosionKinematicExtremes:
    """Adversarial stress testing of hydraulic droplet kinematics and extreme parameters."""

    def test_zero_and_single_step_droplets(self):
        """Verify max_droplet_steps=0 and max_droplet_steps=1 conserve mass (|ΔM| < 1e-9)."""
        for steps in [0, 1]:
            hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
            hf.add_peak(0.0, 0.0, height=18.0, radius=12.0)
            initial_mass = float(np.sum(hf.Z))

            cfg = ErosionConfig(enabled=True, droplet_count=10, max_droplet_steps=steps)
            sim = ErosionSimulator(hf, cfg)
            sim.simulate_hydraulic(seed=42)

            final_mass = float(np.sum(hf.Z))
            assert abs(final_mass - initial_mass) < 1e-9

    def test_instant_evaporation_speed_unity(self):
        """Verify evaporate_speed=1.0 (instant evaporation) does not fail or divide by zero."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=20)
        hf.add_peak(0.0, 0.0, height=15.0, radius=10.0)
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=25, evaporate_speed=1.0)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=77)

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-9

    def test_hydraulic_and_thermal_erosion_on_non_square_grid_16x128(self):
        """Verify hydraulic and thermal erosion run cleanly on anisotropic grid (16x128)."""
        hf = create_non_square_heightfield(width=20.0, depth=160.0, res_x=16, res_y=128)
        hf.add_peak(0.0, 0.0, height=25.0, radius=8.0)
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=50,
            thermal_iterations=5,
            talus_angle_deg=35.0,
            scree_talus_deposition=0.4,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_hydraulic_erosion(seed=42)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-9
        assert np.all(np.isfinite(hf.Z))
        assert np.sum(hf.scree_talus) >= 0.0


# =============================================================================
# Dimension 13: Atmospheric Lighting and Vision Rig Staging
# =============================================================================

class TestAdversarialAtmosphericLightingAndVisionGrid:
    """Adversarial stress testing of solar ephemeris lighting info and vision matrix composition."""

    def test_setup_sun_lighting_diurnal_regimes_pure_python(self):
        """Verify setup_sun_lighting generates physically consistent photometric outputs across diurnal hours."""
        # 1. Midnight
        info_midnight = setup_sun_lighting(sun_time_of_day=0.0)
        assert info_midnight["elevation_deg"] < 0.0
        assert info_midnight["sun_intensity"] == 0.0
        assert info_midnight["cct_kelvin"] == pytest.approx(9000.0)
        assert info_midnight["sky_strength"] == pytest.approx(0.30)

        # 2. Dawn / Horizon (6.0h)
        info_dawn = setup_sun_lighting(sun_time_of_day=6.0)
        assert info_dawn["elevation_deg"] == pytest.approx(0.0, abs=1e-3)
        assert info_dawn["cct_kelvin"] == pytest.approx(2200.0, abs=1.0)
        assert info_dawn["sun_intensity"] == 0.0
        assert info_dawn["sky_strength"] == pytest.approx(0.30)

        # 3. Golden Hour (6.5h, elevation approx 8.8°)
        info_golden = setup_sun_lighting(sun_time_of_day=6.5)
        assert 0.0 < info_golden["elevation_deg"] < 10.0
        assert 2200.0 <= info_golden["cct_kelvin"] <= 3200.0
        assert info_golden["sky_strength"] == pytest.approx(0.75)

        # 4. High Noon (12.0h)
        info_noon = setup_sun_lighting(sun_time_of_day=12.0)
        assert info_noon["elevation_deg"] > 60.0
        assert info_noon["cct_kelvin"] >= 5800.0
        assert info_noon["sun_intensity"] > 4.0
        assert info_noon["sky_strength"] == pytest.approx(0.95)

    def test_setup_inspection_cameras_with_heightfield_elevation_offset(self):
        """Verify setup_inspection_cameras calculates eye-level elevation matching terrain + 1.7m."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=25)
        # Elevate terrain at eye-level pedestal point (-0.22 * width, -0.22 * depth)
        ped_x, ped_y = -22.0, -22.0
        hf.Z[:] = 14.5

        map_cfg = MapConfig(
            name="CameraRigTest",
            diorama=DioramaBounds(width=100.0, depth=100.0, z_base=-10.0),
        )

        cameras = setup_inspection_cameras(config=map_cfg, hf=hf)
        assert "isometric_diorama" in cameras
        assert "eye_level_first_person" in cameras
        assert "closeup_macro" in cameras
        assert "topdown" in cameras

        eye_cam = cameras["eye_level_first_person"]
        expected_cam_z = 14.5 + 1.7
        assert eye_cam["location"][2] == pytest.approx(expected_cam_z, abs=0.1)

    def test_resolve_snapshot_key_legacy_aliases(self):
        """Verify resolve_snapshot_key safely maps modern and legacy viewpoint names."""
        paths = {
            "isometric_diorama": "/tmp/iso.png",
            "eye_level_first_person": "/tmp/eye.png",
            "closeup_macro": "/tmp/close.png",
        }

        # Direct lookups
        assert resolve_snapshot_key("isometric_diorama", paths) == "/tmp/iso.png"
        assert resolve_snapshot_key("eye_level_first_person", paths) == "/tmp/eye.png"

        # Legacy alias lookups
        assert resolve_snapshot_key("iso_sw", paths) == "/tmp/iso.png"
        assert resolve_snapshot_key("iso_ne", paths) == "/tmp/eye.png"
        assert resolve_snapshot_key("closeup_water", paths) == "/tmp/close.png"

        # Missing key returns None
        assert resolve_snapshot_key("nonexistent_camera", paths) is None

    def test_build_2x2_vision_grid_stitching(self, tmp_path):
        """Verify 2x2 vision grid composer generates a composite image of size (2*W, 2*H)."""
        from PIL import Image
        # Create 4 synthetic snapshot images
        img_paths = {}
        for key in ["isometric_diorama", "eye_level_first_person", "closeup_macro", "topdown"]:
            p = str(tmp_path / f"{key}.png")
            im = Image.new("RGB", (320, 240), color=(100, 150, 200))
            im.save(p)
            img_paths[key] = p

        out_grid = str(tmp_path / "vision_grid_test.png")
        res_path = build_2x2_vision_grid(img_paths, out_grid)

        assert res_path is not None
        assert Path(res_path).exists()
        grid_result = Image.open(res_path)
        assert grid_result.size == (640, 480)

    def test_kasten_young_optical_air_mass_subhorizon_clamping(self):
        """Verify Kasten-Young optical air mass clamps to 38.0 for all sub-horizon elevations."""
        for elev in [-90.0, -45.0, -10.0, -0.01, 0.0]:
            m = calculate_optical_air_mass(elev)
            assert m == 38.0

        # At zenith
        m_zenith = calculate_optical_air_mass(90.0)
        assert m_zenith == pytest.approx(1.0, abs=0.01)

    def test_geological_features_orchestration_and_all_peak_shapes(self):
        """Verify apply_geological_features orchestrates strata, tectonics, terraces, and coastal profiles."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=33)

        # Test all 5 procedural peak shapes
        for shape in ["horn", "ridge", "plateau", "dome", "cone"]:
            hf.add_peak(0.0, 0.0, height=10.0, radius=15.0, shape=shape)
            assert np.all(np.isfinite(hf.Z))

        # Add fractal Perlin octaves
        hf.add_perlin_octaves(seed=42, octaves=3, persistence=0.5, scale=30.0, amplitude=3.0)
        assert np.all(np.isfinite(hf.Z))

        map_cfg = MapConfig(
            name="GeologyOrchestrationTest",
            diorama=DioramaBounds(width=80.0, depth=80.0),
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=4.0,
                wavelength=30.0,
                terrace_enabled=True,
                terrace_height=5.0,
                terrace_sharpness=6.0,
            ),
            tectonic=TectonicScarpConfig(
                enabled=True,
                fault_angle_deg=45.0,
                displacement_throw=10.0,
                scarp_sharpness=8.0,
                fault_length=40.0,
            ),
            coastal=CoastalProfileConfig(
                enabled=True,
                sea_level=0.0,
                wave_notch_depth=1.5,
                wave_notch_envelope=1.0,
                coastline_angle_deg=0.0,
            ),
        )

        hf.apply_geological_features(map_cfg)
        assert np.all(np.isfinite(hf.Z))

    def test_river_channel_carving_and_named_stage_presets(self):
        """Verify carve_river_channel incises realistic trenches along multi-stage river splines."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=40)
        hf.Z[:] = 20.0

        cfg = HydrologyConfig(
            river_enabled=True,
            lake_level=6.0,
            sea_level=0.0,
            river_carve_depth=3.0,
            river_stages=[
                {
                    "name": "Mountain Cascades",
                    "t_start": 0.0,
                    "t_end": 0.35,
                    "start": [-30.0, 40.0, 25.0, 3.0],
                    "end": [-10.0, 15.0, 10.0, 4.0],
                },
                {
                    "name": "Valley Meander",
                    "t_start": 0.35,
                    "t_end": 0.70,
                    "start": [-10.0, 15.0, 10.0, 4.0],
                    "end": [10.0, -10.0, 6.0, 6.0],
                },
                {
                    "name": "Central Lake Transit",
                    "t_start": 0.70,
                    "t_end": 0.85,
                    "is_lake_transit": True,
                    "start": [10.0, -10.0, 6.0, 6.0],
                    "end": [20.0, -25.0, 6.0, 7.0],
                },
                {
                    "name": "Outlet Gorge & Bay",
                    "t_start": 0.85,
                    "t_end": 1.0,
                    "is_outlet": True,
                    "start": [20.0, -25.0, 6.0, 7.0],
                    "end": [35.0, -45.0, 0.0, 8.0],
                },
            ],
        )

        engine = HydrologyEngine(hf, cfg)
        engine.carve_river_channel(num_samples=80)

        # River channel must be carved strictly below initial 20.0m elevation
        assert np.min(hf.Z) < 18.0

        # Flow velocity field along the channel
        vx_field, vy_field = engine.compute_flow_velocity_field(v_base=3.0)
        speed_field = np.sqrt(vx_field**2 + vy_field**2)
        assert np.max(speed_field) > 0.0

    def test_microclimate_convenience_methods(self):
        """Verify get_temperature_field, get_insolation_field, get_moisture_field and compute_microclimate."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=20.0, radius=20.0)
        dist_to_water = np.full((25, 25), 10.0)

        engine = BiomeEngine(hf, dist_to_water, sea_level=0.0)
        climate = engine.compute_microclimate()

        assert "temperature" in climate
        assert "insolation" in climate
        assert "moisture" in climate
        assert "slopes" in climate

        assert climate["temperature"].shape == (25, 25)
        assert climate["insolation"].shape == (25, 25)
        assert climate["moisture"].shape == (25, 25)
        assert climate["slopes"].shape == (25, 25)

        assert np.all(np.isfinite(climate["temperature"]))
        assert np.all(np.isfinite(climate["insolation"]))
        assert np.all(np.isfinite(climate["moisture"]))



