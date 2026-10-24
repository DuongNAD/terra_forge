"""
Challenger M1-1 Adversarial Stress Suite: Anti-Staircasing & Slope Continuity.
Focuses on empirical challenge of:
1. Slope continuity and elimination of stepped contour terraces (no bimodal slope spikes).
2. Bilateral filter behaviour on synthetic stepped ramps vs sharp tectonic cliffs.
3. Preservation of Matterhorn horn peak summit relief, trihedral radiating aretes, and cirques.
4. Edge cases: extreme resolutions (tiny, rectangular, large), zero/negative parameters, negative baseline elevations, and large amplitudes.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import MapConfig, ErosionConfig
from terra_forge.core.erosion import ErosionSimulator


class TestChallengerSlopeContinuityAndAntiStaircasing:
    """Adversarially validates slope continuity and absence of staircasing / stepped terraces."""

    def test_zero_stepped_terraces_on_mountain_flanks(self):
        """
        Synthesize alpine Matterhorn horn peak and test slope distribution.
        Terraced terrain creates bimodal slope spikes (excessive flat treads and steep risers).
        Clean abiotic pipeline must exhibit unimodal, continuous slope without flat shelf plateaus.
        """
        # 1. Pipeline with Anti-Staircasing (Bilateral Stage A + Laplacian Stage B)
        hf_clean = Heightfield2D(width=160.0, depth=160.0, resolution=256)
        hf_clean.add_peak(-25.0, 40.0, height=34.0, radius=26.0, shape="horn", ridge_angle_deg=0.0)
        hf_clean.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        hf_clean.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)

        slopes_clean = hf_clean.compute_slope_degrees()

        # Focus on mountain flanks (radial distance between 3m and 24m from peak summit)
        dx = hf_clean.X - (-25.0)
        dy = hf_clean.Y - 40.0
        r = np.sqrt(dx**2 + dy**2)
        flank_mask = (r > 3.0) & (r < 24.0)

        clean_flank_slopes = slopes_clean[flank_mask]

        # Invariant 1: Mountain flank must have exactly 0 low-slope benches (< 8 degrees)
        flat_ratio_clean = float(np.mean(clean_flank_slopes < 8.0))
        assert flat_ratio_clean == 0.0, f"Excessive flat terrace shelves on flank: {flat_ratio_clean * 100:.2f}%"

        # 2. Compare against deliberately terraced terrain
        hf_terr = Heightfield2D(width=160.0, depth=160.0, resolution=256)
        hf_terr.add_peak(-25.0, 40.0, height=34.0, radius=26.0, shape="horn", ridge_angle_deg=0.0)
        hf_terr.apply_canyon_terraces(terrace_height=5.0, terrace_sharpness=6.0, blend=0.85)
        slopes_terr = hf_terr.compute_slope_degrees()
        terr_flank_slopes = slopes_terr[flank_mask]

        flat_ratio_terr = float(np.mean(terr_flank_slopes < 8.0))
        # Terraced terrain has massive flat bench ratio (~15%)
        assert flat_ratio_terr > 0.10, f"Expected terraced terrain to exhibit high flat bench ratio, got {flat_ratio_terr:.3f}"

        # Invariant 2: Bimodality test - slope histogram of clean terrain must be smooth
        hist_clean, _ = np.histogram(clean_flank_slopes, bins=15, range=(0, 60))
        hist_terr, _ = np.histogram(terr_flank_slopes, bins=15, range=(0, 60))

        # Terraced terrain has pronounced spike in low slope bins [0, 8) deg (over 600 cells)
        low_slope_terr_cells = int(hist_terr[0] + hist_terr[1])
        low_slope_clean_cells = int(hist_clean[0] + hist_clean[1])
        assert low_slope_terr_cells > 500, "Terraced terrain must have staircase spike in near-zero slopes"
        assert low_slope_clean_cells == 0, "Anti-staircasing terrain must have 0 near-zero slope spike"

    def test_radial_transect_monotonic_slope_continuity(self):
        """
        Along radial transects down the horn mountain, elevation must descend without
        repeating stair-step plateaus (where dZ/dr ~ 0 for multiple consecutive cells).
        """
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=121)
        hf.add_peak(0.0, 0.0, height=32.0, radius=25.0, shape="horn", ridge_angle_deg=0.0)
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)

        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        r_samples = np.linspace(2.0, 23.0, 40)

        for ang in angles:
            rad = math.radians(ang)
            elevations = [hf.sample_bilinear(r * math.cos(rad), r * math.sin(rad)) for r in r_samples]
            elevations = np.array(elevations)

            # Gradient along radial descent: dZ / dr should be negative
            dz_dr = np.diff(elevations) / np.diff(r_samples)

            # Check that there are no horizontal flat benches (where |dZ/dr| < 0.05 on the flank)
            near_zero_gradient = np.abs(dz_dr) < 0.05
            consecutive_flat = 0
            max_consecutive_flat = 0
            for is_flat in near_zero_gradient:
                if is_flat:
                    consecutive_flat += 1
                    max_consecutive_flat = max(max_consecutive_flat, consecutive_flat)
                else:
                    consecutive_flat = 0

            assert max_consecutive_flat <= 1, (
                f"Stepped plateau detected along radial angle {ang} deg: "
                f"{max_consecutive_flat} consecutive flat samples"
            )

    def test_hydraulic_erosion_produces_zero_isolated_needle_spikes(self):
        """
        Adversarially test hydraulic droplet erosion with thousands of droplets.
        Bilinear termination deposition must prevent single-cell needle spikes (> 3.0m departure from 3x3 mean).
        """
        from scipy.ndimage import uniform_filter

        hf = Heightfield2D(width=60.0, depth=60.0, resolution=61)
        hf.add_peak(0.0, 0.0, height=28.0, radius=18.0, shape="horn")

        z_pre = hf.Z.copy()

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=4000,
            max_droplet_steps=40,
            brush_radius=3.0,
            scree_talus_deposition=0.4,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        # Measure local spike in erosion delta field (delta_Z = Z_post - Z_pre)
        # to ensure droplet deposition does not create single-cell needle spikes (> 2.5m)
        delta_z = hf.Z - z_pre
        delta_mean = uniform_filter(delta_z, size=3)
        diff = np.abs(delta_z - delta_mean)
        max_spike = float(np.max(diff))

        assert max_spike < 2.5, f"Isolated needle spike detected post-erosion: {max_spike:.3f}m"


class TestChallengerBilateralFilterSteppedRampAndCliffs:
    """Adversarially validates bilateral filter: melts terraces while preserving cliffs."""

    def test_bilateral_filter_melts_synthetic_stepped_ramp(self):
        """
        Construct a synthetic stepped ramp (step delta = 1.2m, within range_sigma = 1.8m).
        The bilateral filter must smooth/attenuate the step riser gradient by at least 40%.
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=61)
        # Staircase ramp: step every 6 grid units
        x_idx = np.arange(61)
        step_profile = np.floor(x_idx / 6.0) * 1.2 + 0.05 * x_idx
        hf.Z = np.tile(step_profile, (61, 1)).astype(np.float64)

        z_pre = hf.Z.copy()
        grad_pre = np.max(np.abs(np.diff(z_pre, axis=1)))

        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)

        z_post = hf.Z.copy()
        grad_post = np.max(np.abs(np.diff(z_post, axis=1)))

        reduction = (grad_pre - grad_post) / grad_pre
        assert reduction >= 0.40, f"Terrace step riser not sufficiently melted: reduction={reduction * 100:.1f}%"

    def test_bilateral_filter_preserves_sharp_tectonic_cliffs(self):
        """
        Construct a massive tectonic fault scarp (throw = 22.0m >> range_sigma = 1.8m).
        The bilateral filter must preserve >= 99% of the cliff vertical throw and sharpness.
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=64)
        hf.Z[:, :32] = 5.0
        hf.Z[:, 32:] = 27.0  # 22.0m vertical step cliff

        cliff_pre = float(hf.Z[30, 32] - hf.Z[30, 31])
        assert cliff_pre == 22.0

        # Apply bilateral filter for 1, 2, and 3 iterations
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=3)

        cliff_post = float(hf.Z[30, 32] - hf.Z[30, 31])
        retention = cliff_post / cliff_pre

        assert retention >= 0.99, f"Tectonic cliff degraded by bilateral filter: retention={retention * 100:.2f}%"

    def test_bilateral_filter_range_sigma_cutoff_response(self):
        """
        Test step responses across varying step heights:
        - Steps < range_sigma (e.g. 0.5m, 1.0m) must be substantially attenuated.
        - Steps >> range_sigma (e.g. 10.0m, 20.0m) must remain sharp.
        """
        step_heights = [0.5, 1.0, 1.5, 3.0, 8.0, 15.0]
        attenuations = []

        for sh in step_heights:
            hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
            hf.Z[:, :15] = 10.0
            hf.Z[:, 15:] = 10.0 + sh

            grad_pre = float(hf.Z[15, 15] - hf.Z[15, 14])
            hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
            grad_post = float(hf.Z[15, 15] - hf.Z[15, 14])

            rel_atten = (grad_pre - grad_post) / grad_pre
            attenuations.append(rel_atten)

        # Small step (0.5m) must have high attenuation (> 30%)
        assert attenuations[0] > 0.30, f"Subtle step not attenuated: {attenuations[0]:.2f}"
        # Massive step (15.0m) must have near-zero attenuation (< 1%)
        assert attenuations[-1] < 0.01, f"Massive cliff over-smoothed: {attenuations[-1]:.2f}"
        # Attenuation must monotonically decrease as step height grows
        for i in range(len(attenuations) - 1):
            assert attenuations[i] >= attenuations[i + 1] - 0.02, "Filter response is not monotonic with step height"

    def test_bilateral_filter_multi_iteration_cliff_stability(self):
        """
        Multiple iterations (5 iterations) of bilateral filter on sharp cliff must not degrade cliff.
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        hf.Z[:, :15] = 0.0
        hf.Z[:, 15:] = 25.0

        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=5)
        step = float(hf.Z[15, 15] - hf.Z[15, 14])
        assert step / 25.0 >= 0.98, f"5 iterations of bilateral filter degraded cliff: step={step:.2f}"


class TestChallengerMatterhornApexAndAretePreservation:
    """Adversarially validates that horn summit and knife-edge aretes survive anti-staircasing."""

    def test_matterhorn_apex_relief_preservation(self):
        """
        Matterhorn horn peak apex must retain >= 95% of summit elevation after
        the complete two-stage anti-staircasing pipeline (Bilateral + Taubin Laplacian).
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=81)
        hf.add_peak(0.0, 0.0, height=34.0, radius=26.0, shape="horn", ridge_angle_deg=0.0)

        cx, cy = hf.res_x // 2, hf.res_y // 2
        summit_orig = float(hf.Z[cy, cx])

        # Stage A: Bilateral
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        summit_stage_a = float(hf.Z[cy, cx])
        assert summit_stage_a / summit_orig >= 0.98, "Stage A bilateral filter degraded summit"

        # Stage B: Taubin Laplacian
        hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)
        summit_stage_b = float(hf.Z[cy, cx])
        retention = summit_stage_b / summit_orig

        assert retention >= 0.95, f"Summit relief eroded below 95%: retention={retention * 100:.2f}%"

    def test_matterhorn_trihedral_arete_spine_and_cirque_relief(self):
        """
        Matterhorn trihedral aretes (at 0, 120, 240 deg) must maintain sharp contrast
        against glacial cirque hollows (at 60, 180, 300 deg) after two-stage filtering.
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=81)
        hf.add_peak(0.0, 0.0, height=34.0, radius=26.0, shape="horn", ridge_angle_deg=0.0)

        # Measure pre-filtering arete vs cirque relief at r = 10m
        r = 10.0
        z_arete_0_pre = hf.sample_bilinear(r * math.cos(0.0), r * math.sin(0.0))
        z_cirque_60_pre = hf.sample_bilinear(r * math.cos(math.radians(60)), r * math.sin(math.radians(60)))
        relief_pre = z_arete_0_pre - z_cirque_60_pre

        # Run Stage A + Stage B anti-staircasing
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)

        z_arete_0_post = hf.sample_bilinear(r * math.cos(0.0), r * math.sin(0.0))
        z_cirque_60_post = hf.sample_bilinear(r * math.cos(math.radians(60)), r * math.sin(math.radians(60)))
        relief_post = z_arete_0_post - z_cirque_60_post

        relief_retention = relief_post / relief_pre
        assert relief_retention >= 0.95, f"Arete-cirque relief degraded: retention={relief_retention * 100:.2f}%"
        assert relief_post >= 5.0, f"Absolute arete relief insufficient: {relief_post:.2f}m"

    def test_knife_edge_arete_transverse_cusp_sharpness(self):
        """
        Transverse profile across knife-edge arete must maintain sharp cusp (apex curvature).
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=61)
        hf.add_peak(0.0, 0.0, height=28.0, radius=20.0, shape="ridge", ridge_angle_deg=0.0)

        cx, cy = hf.res_x // 2, hf.res_y // 2
        # Ridge runs along X axis, across is along Y axis
        # Measure second derivative across ridge: d^2 Z / dy^2
        pre_d2z = hf.Z[cy + 1, cx] - 2.0 * hf.Z[cy, cx] + hf.Z[cy - 1, cx]

        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)

        post_d2z = hf.Z[cy + 1, cx] - 2.0 * hf.Z[cy, cx] + hf.Z[cy - 1, cx]

        # Second derivative must remain strongly negative (convex sharp ridge apex)
        assert post_d2z < -0.3, f"Arete apex lost sharp convexity: d2z={post_d2z:.3f}"
        assert abs(post_d2z) >= 0.70 * abs(pre_d2z), "Arete cusp severely flattened"


class TestChallengerEdgeCasesExtremeInputs:
    """Adversarially tests boundary conditions, extreme resolutions, negative/zero inputs, and massive amplitudes."""

    def test_extreme_and_degenerate_resolutions(self):
        """
        Filter must gracefully execute on degenerate resolutions:
        - resolution=1 raises ZeroDivisionError in Heightfield2D (due to dx = width / (res - 1)).
        - resolutions in [2, 3, 5, 7] (where grid <= 7x7 filter window) must execute cleanly without NaNs.
        - rectangular non-square grids (33x65, 65x33) must run cleanly.
        - large resolution (512x512) must run cleanly.
        """
        # 1. Degenerate resolution=1 raises ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            Heightfield2D(width=10.0, depth=10.0, resolution=1)

        # 2. Valid small resolutions <= 7x7 filter window
        small_resolutions = [2, 3, 5, 7]
        for res in small_resolutions:
            hf = Heightfield2D(width=10.0, depth=10.0, resolution=res)
            hf.Z = np.ones((res, res), dtype=np.float64) * 8.0

            # Bilateral filter with radius=3 (window 7x7) on grid <= 7x7
            hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
            assert not np.isnan(hf.Z).any()
            assert not np.isinf(hf.Z).any()
            np.testing.assert_allclose(hf.Z, 8.0, atol=1e-10)

            # Laplacian smoothing
            hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)
            assert not np.isnan(hf.Z).any()
            assert not np.isinf(hf.Z).any()
            np.testing.assert_allclose(hf.Z, 8.0, atol=1e-10)

        # 3. Rectangular / non-square grids
        for (ny, nx) in [(65, 33), (33, 65)]:
            hf = Heightfield2D(width=30.0, depth=60.0, resolution=33)
            hf.res_y = ny
            hf.res_x = nx
            hf.Z = np.ones((ny, nx), dtype=np.float64) * 15.0
            hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
            hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)
            assert hf.Z.shape == (ny, nx)
            assert not np.isnan(hf.Z).any()
            np.testing.assert_allclose(hf.Z, 15.0, atol=1e-10)

        # 4. Large resolution (512x512)
        hf_large = Heightfield2D(width=160.0, depth=160.0, resolution=512)
        hf_large.Z[256, 256] = 50.0
        hf_large.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        assert not np.isnan(hf_large.Z).any()
        assert not np.isinf(hf_large.Z).any()

    def test_zero_and_negative_parameters_safe_noop(self):
        """
        Non-positive or invalid filter parameters must safely no-op without altering terrain.
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z = np.arange(21 * 21, dtype=np.float64).reshape(21, 21)
        orig = hf.Z.copy()

        # Bilateral no-op cases
        hf.apply_bilateral_filter(spatial_sigma=0.0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(spatial_sigma=-2.5)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(range_sigma=0.0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(range_sigma=-1.8)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(radius=0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(radius=-3)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(iterations=0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_bilateral_filter(iterations=-2)
        np.testing.assert_array_equal(hf.Z, orig)

        # Laplacian no-op cases
        hf.apply_laplacian_smoothing(iterations=0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_laplacian_smoothing(iterations=-1)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_laplacian_smoothing(lambda_rate=0.0)
        np.testing.assert_array_equal(hf.Z, orig)

        hf.apply_laplacian_smoothing(lambda_rate=-0.2)
        np.testing.assert_array_equal(hf.Z, orig)

    def test_negative_baseline_elevation_correctly_handled(self):
        """
        Terrain with negative elevations (e.g. bathymetry or submarine trench in [-60.0, -10.0])
        must be filtered correctly without clamping to 0 or blowing up.
        """
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=31)
        hf.Z = -30.0 + np.sin(0.2 * hf.X) * np.cos(0.2 * hf.Y) * 10.0
        assert np.all(hf.Z < 0.0), "Sanity check: all elevations must be negative"

        z_pre = hf.Z.copy()
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=1)
        hf.apply_laplacian_smoothing(iterations=1, lambda_rate=0.20, anti_shrink=True)

        assert not np.isnan(hf.Z).any()
        assert not np.isinf(hf.Z).any()
        assert np.all(hf.Z < -15.0), "Negative elevations must not be artificially clamped to 0.0"
        assert abs(np.mean(hf.Z) - np.mean(z_pre)) < 0.2, "Mean elevation must be preserved"

    def test_massive_amplitudes_and_steep_gradients(self):
        """
        Peak with extreme elevation (Z = 5000.0m on small radius = 10.0m, slope > 89.8 deg).
        Must run without floating-point overflow, underflow exceptions, or NaNs.
        """
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        hf.add_peak(0.0, 0.0, height=5000.0, radius=10.0, shape="horn")

        assert np.max(hf.Z) > 4500.0
        hf.apply_bilateral_filter(spatial_sigma=2.5, range_sigma=1.8, radius=3, iterations=2)
        hf.apply_laplacian_smoothing(iterations=2, lambda_rate=0.20, anti_shrink=True)

        assert not np.isnan(hf.Z).any(), "NaN detected in extreme amplitude test"
        assert not np.isinf(hf.Z).any(), "Inf detected in extreme amplitude test"
        assert np.max(hf.Z) > 4000.0, "Massive peak was completely obliterated"
