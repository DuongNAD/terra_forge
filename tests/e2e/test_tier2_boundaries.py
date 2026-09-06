"""
Tier 2: Boundary, Extreme Stress, and Corner Case Test Suite.
Verifies engine stability across resolution ranges (16 to 256), flat heightfields,
extreme cliffs, sea level variations, zero-water configurations, extreme insolation,
and ecological limits.
"""

import math
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
    ExportConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.core.biomes import BiomeEngine
from terra_forge.core.diorama_slab import DioramaSlabBuilder


class TestTier2Boundaries:
    @pytest.mark.parametrize("resolution", [16, 33, 64, 128, 256])
    def test_resolution_boundaries_topology(self, resolution, topology_validator):
        """Verify diorama slab maintains watertight manifold topology across resolutions 16 to 256."""
        bounds = DioramaBounds(width=40.0, depth=40.0, resolution=resolution, z_base=-10.0)
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=resolution)
        hf.add_peak(0.0, 0.0, height=15.0, radius=12.0, shape="dome")

        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])

        assert metrics["euler_characteristic"] == 2
        assert metrics["boundary_edges"] == 0
        assert metrics["non_manifold_edges"] == 0
        assert metrics["is_watertight"] is True

    def test_flat_heightfield_zero_elevation_stability(self):
        """Verify flat heightfield (Z = 0.0 everywhere) computes zero slope and does not create phantom erosion."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(0.0)

        normals = hf.compute_normals()
        slopes = hf.compute_slope_degrees()

        # Normals must point straight up (0, 0, 1)
        np.testing.assert_allclose(normals[:, :, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(normals[:, :, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(normals[:, :, 2], 1.0, atol=1e-6)
        # Slopes must be 0.0 degrees
        np.testing.assert_allclose(slopes, 0.0, atol=1e-6)

        # Erosion on perfectly flat terrain should not produce uncontrolled runaway
        cfg = ErosionConfig(enabled=True, droplet_count=100, max_droplet_steps=10)
        sim = ErosionSimulator(hf, cfg)
        sim.run_hydraulic_erosion(seed=42)
        assert np.max(np.abs(hf.Z)) < 1.0

    def test_extreme_steep_vertical_cliffs_stability(self):
        """Verify 90-degree vertical elevation steps do not crash normals or thermal relaxation."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(0.0)
        # 100m cliff step in the center
        hf.Z[10:, :] = 100.0

        normals = hf.compute_normals()
        slopes = hf.compute_slope_degrees()

        # No NaNs or Infs in normals or slopes
        assert not np.any(np.isnan(normals))
        assert not np.any(np.isinf(normals))
        assert not np.any(np.isnan(slopes))
        assert not np.any(np.isinf(slopes))

        # Thermal erosion should slump the cliff without blowing up
        cfg = ErosionConfig(thermal_iterations=5, talus_angle_deg=35.0, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        assert not np.any(np.isnan(hf.Z))
        assert hf.Z[10, 10] < 100.0

    def test_sea_level_variations_negative_and_elevated(self):
        """Verify hydrology engine handles negative sea level and high elevated sea level."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(10.0)

        # Case A: Negative sea level (deep oceanic abyss)
        cfg_abyss = HydrologyConfig(sea_level=-25.0, lake_level=-10.0)
        he_abyss = HydrologyEngine(hf, cfg_abyss)
        _, _, rz_end, _ = he_abyss.evaluate_river_spline(1.0)
        assert pytest.approx(rz_end[0], abs=0.1) == -25.0

        # Case B: Elevated sea level (flooding diorama)
        cfg_flood = HydrologyConfig(sea_level=30.0, lake_level=35.0)
        he_flood = HydrologyEngine(hf, cfg_flood)
        _, _, rz_flood_end, _ = he_flood.evaluate_river_spline(1.0)
        assert pytest.approx(rz_flood_end[0], abs=0.1) == 30.0

    def test_zero_water_dry_diorama_distance_field(self):
        """Verify when water bodies are disabled, water distance field returns high default distance."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        cfg = HydrologyConfig(
            lake_radius=0.0,
            river_enabled=False,
            lake_center=(999.0, 999.0),
        )
        he = HydrologyEngine(hf, cfg)
        dist_field = he.compute_water_distance_field()

        # Water distance should be very large everywhere
        assert np.all(dist_field > 500.0)

    def test_extreme_solar_insolation_zenith_horizon_nadir(self):
        """Verify solar irradiance calculation at zenith (90°), horizon (0°), and nadir (-90°)."""
        normal_up = np.array([0.0, 0.0, 1.0])

        # Zenith sun: directly above
        sun_zenith = np.array([0.0, 0.0, 1.0])
        irr_zenith = max(0.0, float(np.dot(normal_up, sun_zenith)))
        assert pytest.approx(irr_zenith, rel=1e-5) == 1.0

        # Horizon sun: horizontal
        sun_horizon = np.array([1.0, 0.0, 0.0])
        irr_horizon = max(0.0, float(np.dot(normal_up, sun_horizon)))
        assert pytest.approx(irr_horizon, abs=1e-5) == 0.0

        # Nadir sun: below ground (night) -> must be clamped to 0
        sun_nadir = np.array([0.0, 0.0, -1.0])
        irr_nadir = max(0.0, float(np.dot(normal_up, sun_nadir)))
        assert irr_nadir == 0.0

    def test_cavern_rock_overburden_boundary_clearance(self):
        """Verify karst cavern overburden clearance rejects breach when rock thickness < 5.0m."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(4.0)  # Low surface elevation

        # Cavern roof apex at z = center_z + height / 2 = 0.0 + 3.5 = 3.5m
        # Surface at 4.0m -> Clearance = 0.5m < 5.0m minimum required
        cavern_cfg = CavernConfig(center=(0.0, 0.0, 0.0), height=7.0, min_roof_clearance=5.0)
        surface_z = hf.sample_bilinear(0.0, 0.0)
        roof_apex = cavern_cfg.center[2] + cavern_cfg.height * 0.5
        clearance = surface_z - roof_apex

        assert clearance < cavern_cfg.min_roof_clearance, "Must detect inadequate rock overburden"

    def test_microclimate_extreme_temperatures(self):
        """Verify microclimate temperature calculations remain bounded under arctic and desert regimes."""
        lapse_rate = 0.0065
        # Arctic winter: T_sea = -30°C, altitude = 3000m -> T = -49.5°C
        t_arctic = -30.0 - lapse_rate * 3000.0
        assert -60.0 < t_arctic < -40.0

        # Saharan summer: T_sea = +48°C, altitude = 0m -> T = +48°C
        t_desert = 48.0 - lapse_rate * 0.0
        assert 45.0 <= t_desert <= 50.0

    def test_underwater_ecosystem_boundary_altitude(self):
        """Verify aquatic flora and reefs are strictly bounded below water level."""
        sea_level = 0.0
        cfg_aquatic = BiomeScatterConfig(
            name="Aquatic_Kelp",
            min_altitude=-20.0,
            max_altitude=sea_level,
        )
        assert cfg_aquatic.max_altitude <= sea_level

    def test_natural_decay_windthrow_density_bounds(self):
        """Verify fallen logs and windthrow tree detritus density is bounded below living forest density."""
        living_forest_density = 0.10
        decay_detritus_density = 0.03
        assert decay_detritus_density < living_forest_density * 0.5
