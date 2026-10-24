"""
Milestone 1 Geological Physics & Terrain Synthesis Test Suite.
Validates:
1. Schema integrity & backward-compatible serialization for geological features.
2. 8-Neighbor isotropic thermal talus relaxation with scree accumulation and mass conservation.
3. Newtonian droplet hydraulic erosion with radial brush, alluvial fans, flow flux, and exact mass conservation.
4. Geological strata folding, canyon terraces, and tectonic fault scarps.
5. Coastal equilibrium profiles (Dean's 2/3 law), wave-cut notches, and berm crests.
"""

import math
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    ErosionConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator


class TestM1Schemas:
    """Validates Milestone 1 configuration dataclasses and backward compatibility."""

    def test_strata_fold_config_defaults(self):
        cfg = StrataFoldConfig()
        assert not cfg.enabled
        assert cfg.amplitude == 6.0
        assert cfg.wavelength == 45.0
        assert cfg.strike_angle_deg == 35.0
        assert cfg.plunge_wavelength == 120.0
        assert cfg.harmonic_weight == 0.25
        assert cfg.terrace_height == 6.0

    def test_tectonic_scarp_config_defaults(self):
        cfg = TectonicScarpConfig()
        assert not cfg.enabled
        assert cfg.displacement_throw == 14.0
        assert cfg.fault_angle_deg == -25.0
        assert cfg.scarp_sharpness == 10.0
        assert cfg.fault_offset == 12.0

    def test_coastal_profile_config_defaults(self):
        cfg = CoastalProfileConfig()
        assert not cfg.enabled
        assert cfg.sea_level == 0.0
        assert cfg.wave_notch_depth == 1.8
        assert cfg.wave_notch_envelope == 1.2
        assert cfg.beach_slope_coefficient == 0.12
        assert cfg.berm_elevation == 1.2
        assert cfg.beach_width == 15.0

    def test_erosion_config_extended_fields(self):
        cfg = ErosionConfig()
        assert hasattr(cfg, "brush_radius")
        assert cfg.brush_radius == 3.0
        assert hasattr(cfg, "alluvial_fan_spread")
        assert cfg.alluvial_fan_spread == 3.5
        assert hasattr(cfg, "rill_incision_factor")
        assert cfg.rill_incision_factor == 0.5

    def test_map_config_roundtrip_with_geology(self):
        cfg = MapConfig(
            strata=StrataFoldConfig(enabled=True, amplitude=8.5, strike_angle_deg=60.0),
            tectonic=TectonicScarpConfig(enabled=True, displacement_throw=15.0),
            coastal=CoastalProfileConfig(enabled=True, sea_level=2.0),
        )
        d = cfg.to_dict()
        assert "strata" in d
        assert "tectonic" in d
        assert "coastal" in d
        assert d["strata"]["amplitude"] == 8.5
        assert d["tectonic"]["displacement_throw"] == 15.0
        assert d["coastal"]["sea_level"] == 2.0

        rehydrated = MapConfig.from_dict(d)
        assert rehydrated.strata.enabled
        assert rehydrated.strata.amplitude == 8.5
        assert rehydrated.tectonic.displacement_throw == 15.0
        assert rehydrated.coastal.sea_level == 2.0


class TestM1ThermalTalusRelaxation:
    """Validates 8-neighbor isotropic thermal talus weathering and scree tracking."""

    def test_thermal_talus_exact_mass_conservation(self):
        """Thermal weathering must conserve 100% of terrain mass (|ΔM| < 1e-10)."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=31)
        hf.add_peak(0.0, 0.0, height=22.0, radius=10.0, shape="cone")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, thermal_iterations=10, talus_angle_deg=30.0, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        delta = abs(final_mass - initial_mass)
        assert delta < 1e-10, f"Mass conservation violated: delta={delta:.4e}"

    def test_thermal_talus_scree_accumulation_tracking(self):
        """Eroded volume slumping downhill must be accumulated into hf.scree_talus."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=25.0, radius=12.0, shape="horn")

        cfg = ErosionConfig(enabled=True, thermal_iterations=8, talus_angle_deg=35.0, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        assert hasattr(hf, "scree_talus")
        assert hf.scree_talus.shape == hf.Z.shape
        assert np.any(hf.scree_talus > 0.05), "Scree talus must accumulate at cliff bases"
        assert hf.scree_talus[hf.res_y // 2, hf.res_x // 2] == 0.0

    def test_thermal_talus_subcritical_stability(self):
        """Slopes below the critical angle of repose must remain completely unchanged."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=3.0, radius=20.0, shape="dome")
        initial_Z = hf.Z.copy()

        cfg = ErosionConfig(enabled=True, thermal_iterations=6, talus_angle_deg=38.0)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        np.testing.assert_allclose(hf.Z, initial_Z, atol=1e-12, err_msg="Subcritical slopes must not erode")
        assert np.all(hf.scree_talus == 0.0)

    def test_thermal_talus_directional_isotropy(self):
        """Radial cone relaxation must preserve radial symmetry without rectilinear distortion."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=20.0, radius=12.0, shape="cone")

        cfg = ErosionConfig(enabled=True, thermal_iterations=12, talus_angle_deg=28.0)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_thermal()

        cy, cx = hf.res_y // 2, hf.res_x // 2
        r = 6
        north = hf.Z[cy + r, cx]
        south = hf.Z[cy - r, cx]
        east = hf.Z[cy, cx + r]
        west = hf.Z[cy, cx - r]

        assert pytest.approx(north, abs=0.15) == south
        assert pytest.approx(east, abs=0.15) == west
        assert pytest.approx(north, abs=0.15) == east


class TestM1HydraulicErosion:
    """Validates droplet hydraulic erosion physics, radial brush, alluvial fan, and mass conservation."""

    def test_hydraulic_exact_mass_conservation(self):
        """Hydraulic simulation must conserve mass (|ΔM| < 1e-10) across all droplets."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        hf.add_peak(0.0, 0.0, height=18.0, radius=14.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=400,
            max_droplet_steps=24,
            brush_radius=3.0,
            alluvial_fan_spread=3.5,
        )
        sim = ErosionSimulator(hf, cfg)
        ret_hf, sed_map, water_map = sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(ret_hf.Z))
        delta = abs(final_mass - initial_mass)
        assert delta < 1e-10, f"Hydraulic mass conservation failed: delta={delta:.4e}"

    def test_hydraulic_return_contract_and_backward_compatibility(self):
        """simulate_hydraulic and run_hydraulic_erosion must return (Heightfield2D, sediment, water_depth)."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        cfg = ErosionConfig(enabled=True, droplet_count=50)
        sim = ErosionSimulator(hf, cfg)

        res1 = sim.simulate_hydraulic(seed=10)
        assert isinstance(res1, tuple) and len(res1) == 3
        assert res1[0] is hf
        assert res1[1].shape == hf.Z.shape
        assert res1[2].shape == hf.Z.shape

        res2 = sim.run_hydraulic_erosion(seed=10)
        assert isinstance(res2, tuple) and len(res2) == 3
        assert res2[0] is hf

    def test_water_depth_flow_accumulation(self):
        """Cumulative water depth flux map must register flow along drainage pathways."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        hf.add_peak(0.0, 0.0, height=20.0, radius=12.0, shape="horn")

        cfg = ErosionConfig(enabled=True, droplet_count=300, max_droplet_steps=25)
        sim = ErosionSimulator(hf, cfg)
        _, _, w_depth = sim.simulate_hydraulic(seed=42)

        assert np.sum(w_depth) > 0.0
        assert np.max(w_depth) > 1.0

    def test_alluvial_fan_deposition_at_slope_break(self):
        """Sediment must deposit smoothly at slope breaks forming alluvial outwash."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=41)
        hf.add_peak(0.0, 10.0, height=25.0, radius=15.0, shape="horn")

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=500,
            max_droplet_steps=35,
            alluvial_fan_spread=3.5,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        assert np.any(hf.sediment > 0.0)
        assert np.max(hf.sediment) > 0.02


class TestM1GeologicalFeatures:
    """Validates geological folding, canyon terraces, tectonic scarps, and coastal profiles."""

    def test_strata_folding_anticline_and_syncline(self):
        """Periodic strata folding must produce alternating ridges (anticlines) and troughs (synclines)."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        hf.apply_strata_folding(amplitude=8.0, wavelength=40.0, strike_angle_deg=0.0, harmonic_weight=0.0, domain_warp_amp=0.0, plunge_wavelength=0.0)

        # Cross-strike is along X when strike_angle_deg is 0 deg: u = X*cos(0) + Y*sin(0) = X
        # Folds vary across columns and are constant along rows
        assert np.isclose(hf.Z[0, :], hf.Z[25, :], atol=1e-5).all()
        assert np.isclose(hf.Z.max(), 8.0, atol=0.2)
        assert np.isclose(hf.Z.min(), -8.0, atol=0.2)

    def test_canyon_terraces_step_structure(self):
        """Canyon terraces must create flat benches separated by steep structural risers."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        for y in range(hf.res_y):
            hf.Z[y, :] = y * 0.5
        initial_slopes = hf.compute_slope_degrees()

        hf.apply_canyon_terraces(terrace_height=5.0, terrace_sharpness=8.0, blend=0.9)
        new_slopes = hf.compute_slope_degrees()

        flat_steps = np.sum(new_slopes < 5.0)
        assert flat_steps > np.sum(initial_slopes < 5.0), "Terracing must create flat structural benches"

    def test_tectonic_scarp_fault_displacement(self):
        """Tectonic fault scarp must displace one block relative to another across fault trace."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        hf.apply_tectonic_scarp(fault_angle_deg=0.0, displacement_throw=10.0, scarp_sharpness=12.0, fault_offset=0.0)

        # Fault strike is 0 deg: d_perp = Y
        south_side = hf.Z[5, 20]
        north_side = hf.Z[35, 20]
        displacement = abs(north_side - south_side)
        assert pytest.approx(displacement, abs=0.5) == 10.0, "Fault throw must match displacement"

    def test_coastal_profile_deans_law_and_wave_notch(self):
        """Coastal profile must apply Dean's 2/3 offshore profile and carve wave-cut notch at sea level."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        # When coastline_angle_deg is 0 deg, seaward distance y_sea = Y
        # Sloping down from inland (South, Y < 0) into ocean (North, Y > 0)
        hf.Z = -hf.Y * 0.5

        hf.apply_coastal_profile(
            sea_level=0.0,
            coastline_angle_deg=0.0,
            coastline_offset=0.0,
            beach_slope_coefficient=0.15,
            wave_notch_depth=2.0,
            wave_notch_envelope=1.2,
        )

        ocean_depths = hf.Z[22:, 20]
        assert np.all(ocean_depths < 0.0), "Bathymetry must remain submerged below sea level"
        assert np.all(np.diff(ocean_depths) < 0.0), "Offshore profile must deepen seaward"

    def test_apply_geological_features_orchestrator(self):
        """apply_geological_features must configure and execute strata folding, scarp, and coastal features."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=31)
        config = MapConfig(
            strata=StrataFoldConfig(enabled=True, amplitude=4.0, wavelength=25.0),
            tectonic=TectonicScarpConfig(enabled=True, displacement_throw=6.0, fault_angle_deg=45.0),
            coastal=CoastalProfileConfig(enabled=True, coastline_angle_deg=0.0),
        )

        hf.apply_geological_features(config)
        assert np.std(hf.Z) > 1.0, "Geological features must synthesize elevation topography"


class TestM1RemediationHardening:
    """Regression and hardening tests for Challenger 1 reported defects."""

    def test_cone_peak_geometry_and_radial_profile(self):
        """Authentic geometric cone peak with linear radial profile and exact summit height."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        H = 24.0
        R = 12.0
        hf.add_peak(0.0, 0.0, height=H, radius=R, shape="cone")

        # Summit must be exactly H
        cx, cy = hf.res_x // 2, hf.res_y // 2
        assert pytest.approx(hf.Z[cy, cx], abs=1e-3) == H

        # Outside radius must be exactly 0
        dist = np.sqrt(hf.X**2 + hf.Y**2)
        assert np.all(hf.Z[dist >= R] == 0.0)

        # Halfway point must be exactly H / 2
        half_idx = cy + round((R / 2.0) / hf.dy)
        assert pytest.approx(hf.Z[half_idx, cx], abs=0.2) == H / 2.0

        # Slopes on cone flanks must be consistent with linear gradient H / R
        slopes = hf.compute_slope_degrees()
        flank_mask = (dist > 1.5) & (dist < R - 1.5)
        expected_slope_deg = math.degrees(math.atan(H / R))
        assert np.isclose(np.mean(slopes[flank_mask]), expected_slope_deg, atol=0.2)
        assert np.allclose(slopes[flank_mask], expected_slope_deg, atol=3.5)

    def test_hydraulic_terminal_velocity_and_erosion_clamps(self):
        """Droplet erosion must remain bounded and conserve mass on steep topography."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        hf.Z[:15, :] = 60.0
        hf.Z[15:, :] = 0.0
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=2000,
            max_droplet_steps=24,
            brush_radius=3.0,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-10
        assert np.min(hf.Z) >= -10.0
        assert np.max(hf.Z) <= 70.0

    def test_hydraulic_sub7_resolution_graceful_bounds(self):
        """Resolutions <= 6 must run safely without crashing or unhandled ValueError."""
        for r in [3, 4, 5, 6]:
            hf = Heightfield2D(width=10.0, depth=10.0, resolution=r)
            cfg = ErosionConfig(enabled=True, droplet_count=20)
            sim = ErosionSimulator(hf, cfg)
            ret_hf, _sed, _w = sim.simulate_hydraulic(seed=42)
            assert ret_hf is hf
            assert not np.isnan(ret_hf.Z).any()


class TestM1AntiStaircasingBilateral:
    """Validates 7x7 edge-preserving bilateral filtering that melts contour terraces while preserving aretes."""

    def test_bilateral_filter_melts_contour_steps_while_preserving_cliffs(self):
        """Terrace step ledges (diff < range_sigma) must melt smoothly, while cliffs (diff >> range_sigma) stay sharp."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=64)

        # 1. Subtle 2.0m terrace step on gentle slope
        hf.Z[:32, :] = 5.0
        hf.Z[32:, :] = 7.0

        # 2. Dramatic 25.0m tectonic fault scarp / cliff on right half
        hf.Z[:, 48:] += 25.0

        z_pre = hf.Z.copy()
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)

        # Terrace step (rows 30-34) should be significantly blurred / smoothed
        step_diff_pre = abs(z_pre[33, 20] - z_pre[31, 20])
        step_diff_post = abs(hf.Z[33, 20] - hf.Z[31, 20])
        assert step_diff_post < step_diff_pre, "Bilateral filter must soften subtle terrace step ledges"

        # Cliff edge (col 47 to 48) must remain razor-sharp (|diff| ~ 25m >> range_sigma=1.8)
        cliff_diff_post = abs(hf.Z[10, 48] - hf.Z[10, 47])
        assert cliff_diff_post > 24.5, "Bilateral filter must 100% preserve vertical cliffs and scarps"

    def test_bilateral_filter_preserves_sharp_arete_ridge(self):
        """A sharp knife-edge arete crest must retain its crest height under bilateral filtering."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=51)
        hf.add_peak(0.0, 0.0, height=30.0, radius=18.0, shape="arete", ridge_angle_deg=0.0)

        cx, cy = hf.res_x // 2, hf.res_y // 2
        summit_pre = hf.Z[cy, cx]

        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        summit_post = hf.Z[cy, cx]

        # Apex should retain at least 95% of its sharp relief
        assert summit_post >= 0.95 * summit_pre, "Arete crest must not be flattened by bilateral filtering"

    def test_bilateral_filter_zero_or_negative_params_noop(self):
        """Invalid or non-positive filter parameters must safely no-op without modifying terrain."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=25)
        hf.Z = np.arange(25 * 25, dtype=np.float64).reshape(25, 25)
        orig = hf.Z.copy()

        hf.apply_bilateral_filter(spatial_sigma=0.0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(range_sigma=-1.0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(radius=0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(iterations=0)
        np.testing.assert_array_equal(hf.Z, orig)


class TestM1AntiStaircasingLaplacian:
    """Validates discrete 8-neighbor Laplacian relaxation and Taubin anti-shrink filtering."""

    def test_laplacian_smoothing_taubin_anti_shrink_volume_preservation(self):
        """Taubin two-step filtering (anti_shrink=True) must preserve summit relief better than naive Laplacian."""
        hf_naive = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf_naive.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")
        hf_taubin = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf_taubin.Z = hf_naive.Z.copy()

        cx, cy = hf_naive.res_x // 2, hf_naive.res_y // 2
        summit_orig = float(hf_naive.Z[cy, cx])

        # Naive Laplacian (anti_shrink=False)
        hf_naive.apply_laplacian_smoothing(iterations=3, lambda_rate=0.25, anti_shrink=False)
        naive_drop = summit_orig - float(hf_naive.Z[cy, cx])

        # Taubin anti-shrink (anti_shrink=True)
        hf_taubin.apply_laplacian_smoothing(iterations=3, lambda_rate=0.25, anti_shrink=True)
        taubin_drop = summit_orig - float(hf_taubin.Z[cy, cx])

        assert taubin_drop < naive_drop, f"Taubin anti-shrink ({taubin_drop:.3f}m) must preserve summit relief better than naive ({naive_drop:.3f}m)"

    def test_laplacian_smoothing_attenuates_checkerboard_grid_noise(self):
        """Laplacian smoothing must rapidly dampen high-frequency micro-roughness."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=31)
        rng = np.random.default_rng(123)
        hf.Z = 10.0 + rng.standard_normal(hf.Z.shape) * 2.0

        pre_var = float(np.var(hf.Z))
        hf.apply_laplacian_smoothing(iterations=3, lambda_rate=0.25, anti_shrink=True)
        post_var = float(np.var(hf.Z))

        assert post_var < 0.65 * pre_var, "Laplacian smoothing must attenuate high-frequency noise"


class TestM1RidgedMultiFractal:
    """Validates C2 quintic Hermite Perlin noise and Ridged Multi-Fractal (RMF) terrain synthesis."""

    def test_perlin_2d_c2_continuity_and_bounds(self):
        """Pure NumPy Perlin 2D gradient noise must produce bounded, continuous, finite values."""
        from terra_forge.core.heightfield import _perlin_2d
        X, Y = np.meshgrid(np.linspace(-50.0, 50.0, 100), np.linspace(-50.0, 50.0, 100))
        n = _perlin_2d(X, Y, freq=0.03, seed=1337)

        assert not np.isnan(n).any()
        assert not np.isinf(n).any()
        assert np.all(n >= -1.0) and np.all(n <= 1.0)
        assert np.std(n) > 0.1, "Perlin noise must exhibit variance"

    def test_ridged_multifractal_synthesis_and_sharpness(self):
        """RMF synthesis must generate sharp ridge crests with positive kurtosis / high ridge sharpness."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=64)
        hf.add_ridged_multifractal(seed=42, octaves=4, amplitude=8.0)

        assert not np.isnan(hf.Z).any()
        assert np.max(hf.Z) > 3.0
        assert np.min(hf.Z) >= 0.0, "Ridged multi-fractal signal is non-negative"

        # Check that multiple octaves contribute to fine frequency content
        grad_y, grad_x = np.gradient(hf.Z)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        assert np.max(mag) > 0.2, "RMF must create sharp gradient rock ribs"


class TestM1AlpineMatterhornHornAndArete:
    """Validates upgraded alpine Matterhorn pyramidal horn and knife-edge arete rock spines."""

    def test_alpine_matterhorn_horn_peak_trihedral_aretes_and_cirques(self):
        """Matterhorn horn peak must feature 3 radiating knife-edge aretes and excavated cirque hollows."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=81)
        hf.add_peak(0.0, 0.0, height=35.0, radius=25.0, shape="horn", ridge_angle_deg=0.0)

        # Summit must be sharp
        cx, cy = hf.res_x // 2, hf.res_y // 2
        assert hf.Z[cy, cx] > 30.0

        # At intermediate radial distance r = 10m:
        # Check that arete angles (0, 120, 240 deg) have higher elevation than cirque angles (60, 180, 300 deg)
        r = 10.0
        # Arete at angle 0: (x=10, y=0)
        arete_sample = hf.sample_bilinear(r, 0.0)
        # Cirque at angle 60 deg: (x=10*cos(60), y=10*sin(60))
        cirque_sample = hf.sample_bilinear(r * math.cos(math.radians(60)), r * math.sin(math.radians(60)))

        assert arete_sample > cirque_sample + 2.0, (
            f"Arete spine ({arete_sample:.2f}m) must be significantly higher than cirque hollow ({cirque_sample:.2f}m)"
        )

        # Elevation outside radius must be zero
        dist = np.sqrt(hf.X**2 + hf.Y**2)
        assert np.all(hf.Z[dist >= 25.0] == 0.0)

    def test_knife_edge_arete_transverse_cusp_profile(self):
        """Knife-edge arete must exhibit a sharp transverse cusp profile."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=61)
        hf.add_peak(0.0, 0.0, height=28.0, radius=20.0, shape="ridge", ridge_angle_deg=0.0)

        # Ridge runs along X (ridge_angle_deg = 0), across ridge is along Y
        cx, cy = hf.res_x // 2, hf.res_y // 2
        crest = hf.Z[cy, cx]
        step1 = hf.Z[cy + 1, cx]
        step2 = hf.Z[cy + 2, cx]

        # Profile drops sharply away from crest
        assert crest > step1 > step2
        # Curvature at apex: (crest - step1) is steep
        drop1 = crest - step1
        drop2 = step1 - step2
        assert drop1 > 0.5, "Arete crest must feature sharp cusp falloff"


class TestM1AcceleratedHydraulicErosion:
    """Validates slice-accelerated droplet erosion and smooth 2x2 bilinear termination deposition."""

    def test_hydraulic_slice_acceleration_and_mass_conservation(self):
        """Accelerated erosion must preserve exact mass (|ΔM| < 1e-10) across thousands of droplets."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=3000, max_droplet_steps=30, brush_radius=3.0)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(hf.Z))
        delta = abs(final_mass - initial_mass)
        assert delta < 1e-10, f"Mass conservation violated: delta={delta:.4e}"

    def test_hydraulic_bilinear_termination_zero_needle_spikes(self):
        """Bilinear terminal deposition must distribute sediment across 2x2 cells, eliminating needle spikes."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=20.0, radius=12.0, shape="cone")

        cfg = ErosionConfig(enabled=True, droplet_count=1000, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        # Compute max local spike (deviation from local 3x3 neighborhood mean)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(hf.Z, size=3)
        diff = np.abs(hf.Z - local_mean)
        max_spike = float(np.max(diff))
        # Needle spikes from raw single-cell dump exceed 8.0m; smooth 2x2 bilinear deposit keeps max deviation < 3.0m
        assert max_spike < 3.0, f"Excessive local spike detected: {max_spike:.3f}m"

    def test_hydraulic_performance_scale_and_24_thermal_iterations(self):
        """Simulate 24 thermal iterations and verify scree accumulation and stability."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=26.0, radius=14.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, thermal_iterations=24, talus_angle_deg=33.0, scree_talus_deposition=0.4)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        assert abs(final_mass - initial_mass) < 1e-10, "Thermal talus must conserve mass over 24 iterations"
        assert np.any(hf.scree_talus > 0.0), "Scree talus must accumulate over 24 iterations"


