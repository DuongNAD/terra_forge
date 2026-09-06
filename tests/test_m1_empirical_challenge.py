"""
Standalone Empirical Challenge & Stress Suite for Milestone 1 Geology & Physics.
Tests:
1. Hydraulic droplet erosion mass conservation under extreme stress (50,000 droplets).
2. Terrain elevation boundedness and stability (detects exponential velocity/erosion runaway).
3. Thermal erosion mass conservation across 50 iterations on steep cones (|Delta M| < 1e-10).
4. Scree talus distribution (hf.scree_talus >= 0.0 everywhere, non-zero at cliff bases).
5. Robustness under extreme parameter sweeps and boundary resolutions.
6. Validation of add_peak shape handling (detects missing 'cone' shape silently failing).
"""

import math
import time
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    ErosionConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    MapConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator


class TestEmpiricalHydraulicStress:
    """Stress testing hydraulic droplet erosion under 50,000 droplets and extreme topography."""

    @pytest.mark.parametrize(
        "terrain_type",
        ["flat_plane", "vertical_cliff", "random_noise", "steep_horn"],
    )
    def test_hydraulic_50k_droplets_surface_mass_conservation(self, terrain_type):
        """Verify surface elevation mass conservation |sum(Z_fin) - sum(Z_init)| < 1e-9 across 50k droplets."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        if terrain_type == "flat_plane":
            hf.Z.fill(20.0)
        elif terrain_type == "vertical_cliff":
            hf.Z[:15, :] = 50.0
            hf.Z[15:, :] = 0.0
        elif terrain_type == "random_noise":
            rng = np.random.default_rng(12345)
            hf.Z = rng.uniform(0.0, 40.0, (31, 31))
        elif terrain_type == "steep_horn":
            hf.add_peak(0.0, 0.0, height=30.0, radius=12.0, shape="horn")

        z_initial = float(np.sum(hf.Z))
        cfg = ErosionConfig(enabled=True, droplet_count=50000, max_droplet_steps=24)
        sim = ErosionSimulator(hf, cfg)

        t0 = time.perf_counter()
        sim.simulate_hydraulic(seed=42)
        elapsed = time.perf_counter() - t0

        z_final = float(np.sum(hf.Z))
        delta_mass = abs(z_final - z_initial)
        z_min = float(np.min(hf.Z))
        z_max = float(np.max(hf.Z))

        # Check for unconstrained runaway explosion
        assert -1000.0 <= z_min, f"Runaway pit carving on {terrain_type}: min_Z={z_min:.2e}"
        assert z_max <= 1000.0, f"Runaway sediment spike on {terrain_type}: max_Z={z_max:.2e}"

        # Exact physical mass conservation of the surface elevation
        assert delta_mass < 1e-9, f"Mass conservation violated on {terrain_type}: {delta_mass:.4e}"

        # No NaNs or Infs
        assert not np.isnan(hf.Z).any()
        assert not np.isinf(hf.Z).any()
        assert not np.isnan(hf.sediment).any()
        assert not np.isnan(hf.water_depth).any()

    def test_hydraulic_flat_plane_mass_conservation(self):
        """On a flat plane, sediment is 0, so both |Z_fin - Z_init| < 1e-9 and prompt formula hold strictly."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=31)
        hf.Z.fill(25.0)
        z_initial = float(np.sum(hf.Z))

        cfg = ErosionConfig(enabled=True, droplet_count=50000)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        z_final = float(np.sum(hf.Z))
        sed_final = float(np.sum(hf.sediment))
        delta_mass = abs(z_final - z_initial)
        prompt_formula_delta = abs(z_final + sed_final - z_initial)

        assert delta_mass < 1e-9
        assert prompt_formula_delta < 1e-9
        assert sed_final == 0.0

    def test_hydraulic_resolution_boundary_sub7(self):
        """Verify resolutions <= 6 run safely without crashing or ValueError."""
        cfg = ErosionConfig(enabled=True, droplet_count=10)
        hf = Heightfield2D(width=10.0, depth=10.0, resolution=5)
        sim = ErosionSimulator(hf, cfg)
        ret_hf, _sed, _w = sim.simulate_hydraulic(seed=42)
        assert ret_hf is hf
        assert not np.isnan(ret_hf.Z).any()


class TestEmpiricalThermalStress:
    """Stress testing thermal weathering across 50 iterations on steep topography."""

    def test_thermal_50_iterations_cone_mass_conservation(self):
        """Thermal weathering must conserve 100% mass (|sum(Z_fin) - sum(Z_init)| < 1e-10) over 50 iterations."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=35.0, radius=10.0, shape="cone")
        z_initial = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=50,
            talus_angle_deg=30.0,
            scree_talus_deposition=0.5,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        z_final = float(np.sum(hf.Z))
        delta_mass = abs(z_final - z_initial)

        assert delta_mass < 1e-10, f"Thermal mass conservation violated: {delta_mass:.4e}"
        assert not np.isnan(hf.Z).any()
        assert not np.isinf(hf.Z).any()

    def test_scree_talus_accumulation_at_cliff_bases(self):
        """Verify hf.scree_talus >= 0.0 everywhere with non-zero scree accumulation at cliff bases."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=35.0, radius=10.0, shape="cone")

        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=50,
            talus_angle_deg=30.0,
            scree_talus_deposition=0.5,
        )
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        # 1. Non-negative everywhere
        assert np.all(hf.scree_talus >= 0.0), "Scree talus must be >= 0.0 everywhere"
        # 2. Maximum accumulation occurs at cliff bases
        assert np.max(hf.scree_talus) > 1.0, "Significant scree must accumulate"
        # 3. Peak summit has zero scree (source area, not deposition)
        cx, cy = hf.res_x // 2, hf.res_y // 2
        assert hf.scree_talus[cy, cx] == 0.0, "Summit should not accumulate scree"
        # 4. Ring around the base has positive scree
        base_scree = [
            hf.scree_talus[cy + 10, cx],
            hf.scree_talus[cy - 10, cx],
            hf.scree_talus[cy, cx + 10],
            hf.scree_talus[cy, cx - 10],
        ]
        assert all(s > 0.05 for s in base_scree), "Cliff base perimeter must contain non-zero scree"

    def test_add_peak_cone_shape(self):
        """Verify add_peak synthesizes authentic geometric cone shape."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=20.0, radius=10.0, shape="cone")
        assert np.isclose(np.max(hf.Z), 20.0, atol=0.1)
        assert np.min(hf.Z) >= 0.0


class TestExtremeParameterSweeps:
    """Evaluates numerical stability and exception safety across extreme parameter ranges."""

    @pytest.mark.parametrize(
        "param_name,config_override",
        [
            ("zero_droplets", ErosionConfig(enabled=True, droplet_count=0)),
            ("single_droplet", ErosionConfig(enabled=True, droplet_count=1)),
            ("zero_steps", ErosionConfig(enabled=True, droplet_count=50, max_droplet_steps=0)),
            ("high_gravity", ErosionConfig(enabled=True, droplet_count=50, gravity=100.0)),
            ("zero_gravity", ErosionConfig(enabled=True, droplet_count=50, gravity=0.0)),
            ("zero_erode_speed", ErosionConfig(enabled=True, droplet_count=50, erode_speed=0.0)),
            ("max_erode_speed", ErosionConfig(enabled=True, droplet_count=50, erode_speed=1.0)),
            ("zero_deposit_speed", ErosionConfig(enabled=True, droplet_count=50, deposit_speed=0.0)),
            ("max_deposit_speed", ErosionConfig(enabled=True, droplet_count=50, deposit_speed=1.0)),
            ("zero_evap", ErosionConfig(enabled=True, droplet_count=50, evaporate_speed=0.0)),
            ("max_evap", ErosionConfig(enabled=True, droplet_count=50, evaporate_speed=1.0)),
            ("huge_brush", ErosionConfig(enabled=True, droplet_count=50, brush_radius=50.0)),
            ("zero_brush", ErosionConfig(enabled=True, droplet_count=50, brush_radius=0.0)),
            ("huge_fan", ErosionConfig(enabled=True, droplet_count=50, alluvial_fan_spread=50.0)),
            ("zero_thermal_iter", ErosionConfig(enabled=True, thermal_iterations=0)),
            ("high_thermal_iter", ErosionConfig(enabled=True, thermal_iterations=100)),
            ("talus_1_deg", ErosionConfig(enabled=True, thermal_iterations=5, talus_angle_deg=1.0)),
            ("talus_89_deg", ErosionConfig(enabled=True, thermal_iterations=5, talus_angle_deg=89.0)),
            ("zero_scree_dep", ErosionConfig(enabled=True, thermal_iterations=5, scree_talus_deposition=0.0)),
            ("full_scree_dep", ErosionConfig(enabled=True, thermal_iterations=5, scree_talus_deposition=1.0)),
        ],
    )
    def test_extreme_erosion_parameters(self, param_name, config_override):
        """Verify no NaN, Inf, or unhandled exceptions under extreme parameter values."""
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=25)
        hf.add_peak(0.0, 0.0, height=20.0, radius=10.0, shape="horn")

        sim = ErosionSimulator(hf, config_override)
        sim.simulate_hydraulic(seed=42)
        sim.run_thermal_erosion()

        assert not np.isnan(hf.Z).any(), f"NaN in Z for {param_name}"
        assert not np.isinf(hf.Z).any(), f"Inf in Z for {param_name}"
        assert not np.isnan(hf.sediment).any(), f"NaN in sediment for {param_name}"
        assert not np.isnan(hf.scree_talus).any(), f"NaN in scree_talus for {param_name}"
        assert not np.isnan(hf.water_depth).any(), f"NaN in water_depth for {param_name}"

    def test_geological_synthesis_extreme_parameters(self):
        """Verify geological features (strata, scarp, canyon, coastal) under extreme parameters."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=35)
        hf.Z.fill(10.0)

        cfg = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=50.0,
                wavelength=10.0,
                strike_angle_deg=180.0,
                harmonic_weight=0.9,
            ),
            tectonic=TectonicScarpConfig(
                enabled=True,
                displacement_throw=100.0,
                fault_angle_deg=90.0,
                scarp_sharpness=50.0,
            ),
            coastal=CoastalProfileConfig(
                enabled=True,
                sea_level=0.0,
                wave_notch_depth=10.0,
                beach_slope_coefficient=0.8,
            ),
        )

        hf.apply_geological_features(cfg)
        hf.apply_canyon_terraces(terrace_height=15.0, terrace_sharpness=20.0, blend=1.0)

        assert not np.isnan(hf.Z).any()
        assert not np.isinf(hf.Z).any()
        assert hf.Z.max() > 20.0
        assert hf.Z.min() < 0.0
