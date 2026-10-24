"""
Milestone 2 Empirical Challenge Test Suite:
1. Hydraulic Erosion Strict Mass Conservation (|ΔM| < 1e-10) across multi-seed (42, 1337, 2026, 777, 99999)
   and droplet counts (1k, 5k, 10k, 20k, 30k).
2. Thermal Talus Relaxation on steep cliffs (>60 deg) relaxing to angle of repose plus tolerance,
   with exact scree_talus vs deposited volume matching and mass conservation.
3. Strata Folding alternating anticline / syncline relief and Tectonic Fault Scarp displacement throw
   across arbitrary rotated fault angles (0, 45, 90, -30, 135 deg).
4. Genesis Primordial Abiotic preset full pipeline simulation integrity and mass conservation.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    ErosionConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    PeakConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator


class TestEmpiricalHydraulicMassConservation:
    """Empirical challenge 1: Strict hydraulic erosion mass conservation (|ΔM| < 1e-10)."""

    @pytest.mark.parametrize(
        "seed,droplet_count",
        [
            (42, 1000),
            (1337, 5000),
            (2026, 10000),
            (777, 20000),
            (99999, 30000),
        ],
    )
    def test_hydraulic_strict_mass_conservation_multi_seed_droplets(self, seed: int, droplet_count: int):
        """Run droplet erosion across seeds and droplet counts (1k to 30k); assert |ΔM| < 1e-10."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        hf.add_peak(-10.0, 15.0, height=28.0, radius=20.0, shape="horn")
        hf.add_peak(15.0, -10.0, height=18.0, radius=16.0, shape="plateau")

        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=droplet_count,
            droplet_inertia=0.06,
            sediment_capacity_factor=4.5,
            min_sediment_capacity=0.01,
            erode_speed=0.32,
            deposit_speed=0.32,
            gravity=4.0,
            max_droplet_steps=64,
            brush_radius=3.0,
            alluvial_fan_spread=3.5,
            rill_incision_factor=0.5,
        )

        sim = ErosionSimulator(hf, cfg)
        ret_hf, sed_map, w_depth = sim.simulate_hydraulic(seed=seed)

        final_mass = float(np.sum(ret_hf.Z))
        delta_m = abs(final_mass - initial_mass)

        # Assert mass conservation to machine precision bound < 1e-10
        assert delta_m < 1e-10, (
            f"Hydraulic mass conservation failed for seed={seed}, droplets={droplet_count}: "
            f"|ΔM| = {delta_m:.4e} >= 1e-10"
        )
        # Verify physical integrity: no NaN, no Inf, active deposition
        assert not np.isnan(ret_hf.Z).any()
        assert not np.isinf(ret_hf.Z).any()
        assert np.any(sed_map > 0.0), f"No sediment deposited for seed={seed}, droplets={droplet_count}"
        assert np.any(w_depth > 0.0), f"No water flux recorded for seed={seed}, droplets={droplet_count}"

    def test_hydraulic_mass_conservation_folded_terraced_terrain(self):
        """Mass conservation holds on complex folded and canyon-terraced bedrock."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        hf.apply_strata_folding(amplitude=5.0, wavelength=25.0, strike_angle_deg=45.0)
        hf.apply_canyon_terraces(terrace_height=4.0, terrace_sharpness=6.0, blend=0.8)

        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=15000,
            droplet_inertia=0.07,
            brush_radius=2.5,
            alluvial_fan_spread=3.0,
        )
        sim = ErosionSimulator(hf, cfg)
        ret_hf, _, _ = sim.simulate_hydraulic(seed=8888)

        final_mass = float(np.sum(ret_hf.Z))
        delta_m = abs(final_mass - initial_mass)
        assert delta_m < 1e-10, f"Folded bedrock hydraulic erosion leaked mass: |ΔM| = {delta_m:.4e}"


class TestEmpiricalThermalTalusRelaxation:
    """Empirical challenge 2: Thermal talus slope relaxation & exact scree volume conservation."""

    @pytest.mark.parametrize("talus_angle", [33.0, 35.0])
    def test_steep_cliff_relaxation_bounds_to_angle_of_repose(self, talus_angle: float):
        """
        Steep cliff (>60 deg) must relax down to angle of repose plus tolerance.
        Initial slope is ~63.4 deg (central difference on 4m step with dx=1m).
        """
        hf = Heightfield2D(width=24.0, depth=24.0, resolution=25)
        # dx = 1.0. Height difference of 4.0 across step gives central slope arctan(4/2) = 63.43 deg > 60 deg
        hf.Z[:12, :] = 4.0
        hf.Z[12:, :] = 0.0

        slopes_before = hf.compute_slope_degrees()
        initial_max_slope = float(np.max(slopes_before))
        assert initial_max_slope > 60.0, f"Initial cliff slope {initial_max_slope:.2f} deg must be > 60 deg"

        z_init = hf.Z.copy()
        initial_mass = float(np.sum(z_init))

        # Run thermal relaxation with sufficient iterations to achieve asymptotic repose
        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=80,
            talus_angle_deg=talus_angle,
            scree_talus_deposition=1.0,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        slopes_after = hf.compute_slope_degrees()
        final_max_slope = float(np.max(slopes_after))

        # Slope tolerance: relaxed slope must not exceed angle of repose plus 0.5 degrees numerical tolerance
        tolerance_deg = 0.5
        assert final_max_slope <= talus_angle + tolerance_deg, (
            f"Talus slope {final_max_slope:.3f} deg exceeded angle of repose {talus_angle:.1f} deg + {tolerance_deg} deg"
        )
        assert final_max_slope >= talus_angle - 1.0, (
            f"Talus slope {final_max_slope:.3f} deg under-relaxed unexpectedly below repose"
        )

        # Global mass conservation
        final_mass = float(np.sum(hf.Z))
        delta_m = abs(final_mass - initial_mass)
        assert delta_m < 1e-10, f"Thermal relaxation mass conservation violated: |ΔM| = {delta_m:.4e}"

    def test_scree_talus_matches_deposited_volume_exactly(self):
        """
        Assert scree_talus volume on the depositional apron matches deposited volume exactly.
        On every receiving cell (where Z_final > Z_initial), scree_talus == (Z_final - Z_initial).
        """
        hf = Heightfield2D(width=24.0, depth=24.0, resolution=25)
        hf.Z[:12, :] = 4.0
        hf.Z[12:, :] = 0.0

        z_init = hf.Z.copy()
        talus_deg = 33.0
        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=60,
            talus_angle_deg=talus_deg,
            scree_talus_deposition=1.0,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        z_final = hf.Z.copy()

        # Deposited cells
        deposited_mask = z_final > z_init
        assert np.any(deposited_mask), "Must have cells that received deposited material"

        scree_on_dep = float(np.sum(hf.scree_talus[deposited_mask]))
        volume_deposited = float(np.sum(z_final[deposited_mask] - z_init[deposited_mask]))

        # Exact match assertion: |scree - deposited_volume| < 1e-10
        diff_dep = abs(scree_on_dep - volume_deposited)
        assert diff_dep < 1e-10, (
            f"Scree talus does not match deposited volume: scree={scree_on_dep:.8f}, "
            f"deposited={volume_deposited:.8f}, diff={diff_dep:.4e}"
        )

        # Total scree across whole grid equals total eroded bedrock volume
        eroded_mask = z_final < z_init
        scree_total = float(np.sum(hf.scree_talus))
        bedrock_loss_total = float(np.sum(z_init[eroded_mask] - z_final[eroded_mask]))
        assert abs(scree_total - (bedrock_loss_total + float(np.sum(hf.scree_talus[eroded_mask])))) < 1e-10


class TestEmpiricalStrataAndFaultScarpGeometry:
    """Empirical challenge 3: Strata folding alternating relief and fault scarp rotated throws."""

    def test_strata_folding_alternating_anticline_syncline_relief(self):
        """Verify strata folding synthesizes periodic alternating anticlines (peaks) and synclines (troughs)."""
        hf = Heightfield2D(width=160.0, depth=160.0, resolution=161)
        amp = 5.0
        wlen = 40.0
        strike_deg = 0.0  # Folds run along Y, alternating along X

        hf.apply_strata_folding(
            amplitude=amp,
            wavelength=wlen,
            strike_angle_deg=strike_deg,
            plunge_wavelength=0.0,
            harmonic_weight=0.0,
            domain_warp_amp=0.0,
        )

        mid_y = hf.res_y // 2
        x_profile = hf.Z[mid_y, :]

        maxima_indices = []
        minima_indices = []
        for i in range(1, len(x_profile) - 1):
            if x_profile[i] > x_profile[i - 1] and x_profile[i] > x_profile[i + 1]:
                if x_profile[i] > 0.8 * amp:
                    maxima_indices.append(i)
            elif x_profile[i] < x_profile[i - 1] and x_profile[i] < x_profile[i + 1]:
                if x_profile[i] < -0.8 * amp:
                    minima_indices.append(i)

        assert len(maxima_indices) >= 3, f"Expected >= 3 anticlines, found {len(maxima_indices)}"
        assert len(minima_indices) >= 3, f"Expected >= 3 synclines, found {len(minima_indices)}"

        # Verify alternating order: maxima and minima must interleave
        all_extrema = sorted([(idx, "anticline") for idx in maxima_indices] + [(idx, "syncline") for idx in minima_indices])
        for k in range(len(all_extrema) - 1):
            assert all_extrema[k][1] != all_extrema[k + 1][1], (
                f"Extrema at {all_extrema[k]} and {all_extrema[k+1]} do not alternate anticline/syncline"
            )

        # Distance between successive crests must equal wavelength (40.0m +/- dx)
        for k in range(len(maxima_indices) - 1):
            dist_m = abs(hf.x_coords[maxima_indices[k + 1]] - hf.x_coords[maxima_indices[k]])
            assert pytest.approx(dist_m, abs=hf.dx * 1.5) == wlen, (
                f"Anticline spacing {dist_m:.2f}m != wavelength {wlen}m"
            )

    @pytest.mark.parametrize("fault_angle", [0.0, 45.0, 90.0, -30.0, 135.0])
    def test_fault_scarp_displacement_throw_across_rotations(self, fault_angle: float):
        """
        Verify fault scarp displacement throw matches configuration across rotated fault angles.
        Samples +d and -d along the fault normal (-sin theta, cos theta).
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=81)
        throw = 6.0
        sharpness = 12.0

        hf.apply_tectonic_scarp(
            fault_angle_deg=fault_angle,
            displacement_throw=throw,
            scarp_sharpness=sharpness,
            fault_offset=0.0,
        )

        rad = math.radians(fault_angle)
        nx = -math.sin(rad)
        ny = math.cos(rad)

        d_sample = 15.0
        z_pos = hf.sample_bilinear(d_sample * nx, d_sample * ny)
        z_neg = hf.sample_bilinear(-d_sample * nx, -d_sample * ny)

        measured_throw = z_pos - z_neg
        assert pytest.approx(measured_throw, abs=0.05) == throw, (
            f"Fault throw {measured_throw:.4f} != expected {throw:.4f} at angle {fault_angle} deg"
        )
        assert pytest.approx(z_pos, abs=0.05) == 0.5 * throw
        assert pytest.approx(z_neg, abs=0.05) == -0.5 * throw


class TestEmpiricalPresetAbioticSimulation:
    """Empirical challenge 4: Full pipeline validation on genesis_primordial_abiotic preset."""

    def test_full_preset_pipeline_erosion_mass_conservation(self):
        """Run full preset geological pipeline and assert mass conservation on combined erosion."""
        preset_p = Path(__file__).resolve().parent.parent / "terra_forge" / "schema" / "presets" / "genesis_primordial_abiotic.json"
        with open(preset_p, "r", encoding="utf-8") as f:
            cfg = MapConfig.model_validate(json.load(f))

        assert len(cfg.biomes) == 0, "Abiotic preset must have 0 biomes"

        hf = Heightfield2D(
            width=cfg.diorama.width,
            depth=cfg.diorama.depth,
            resolution=cfg.diorama.resolution,
        )
        hf.add_perlin_octaves(seed=cfg.seed, octaves=3, amplitude=2.5)

        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)

        hf.apply_geological_features(cfg)

        hf.carve_lake_basin(
            cfg.hydrology.lake_center[0],
            cfg.hydrology.lake_center[1],
            cfg.hydrology.lake_radius,
            cfg.hydrology.lake_level,
            cfg.hydrology.lake_depth,
        )

        mass_pre_erosion = float(np.sum(hf.Z))

        ero_cfg = ErosionConfig(
            enabled=True,
            droplet_count=5000,
            droplet_inertia=cfg.erosion.droplet_inertia,
            brush_radius=cfg.erosion.brush_radius,
            alluvial_fan_spread=cfg.erosion.alluvial_fan_spread,
            thermal_iterations=6,
            talus_angle_deg=cfg.erosion.talus_angle_deg,
            scree_talus_deposition=cfg.erosion.scree_talus_deposition,
        )
        sim = ErosionSimulator(hf, ero_cfg)
        sim.run_hydraulic_erosion(cfg.seed)
        sim.run_thermal_erosion()

        mass_post_erosion = float(np.sum(hf.Z))
        delta_m = abs(mass_post_erosion - mass_pre_erosion)

        assert delta_m < 1e-10, f"Full pipeline erosion mass conservation failed: |ΔM| = {delta_m:.4e}"
        assert np.max(hf.scree_talus) > 0.05, "Scree talus must accumulate on preset terrain"
        assert np.max(hf.sediment) > 0.05, "Sediment must accumulate on preset terrain"
