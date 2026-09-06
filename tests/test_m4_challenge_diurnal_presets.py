"""
Milestone 4 Empirical Challenge & Adversarial Stress Suite:
Diurnal Lighting, Atmospheric Optics & 4 Natural Presets Geomorphology.

Verified Criteria:
1. Diurnal lighting sweep across 24 hours:
   - Solar elevation matches sinusoidal curve: elev(t) = max_elev * sin((t - 6) / 12 * pi).
   - Solar azimuth sweeps full 360° clockwise from North (15°/hr).
   - Correlated Color Temperature (CCT) transitions smoothly:
     * 2200K at dawn/dusk (0° elevation) to 3200K at 10°
     * 5800K to 6500K at high midday (>45° elevation)
     * 7000K - 9000K cool ambient during twilight and night (<0° elevation)
   - Tanner-Helland Linear RGB values are strictly non-negative, finite, and in [0.0, 1.0].
2. Optical air mass and solar intensity:
   - Kasten-Young air mass m=1.0 at zenith, m=38.0 at horizon.
   - Intensity I -> 0.0 when sun drops below horizon (elev <= 0.0°).
   - Monotonic extinction above horizon.
3. The 4 natural preset files:
   - alpine_peak_glacial_lake.json
   - temperate_forest_river_basin.json
   - tropical_island_coastal_karst.json
   - canyon_semiarid_plateau.json
   - Validates via MapConfig.from_dict, MapConfig.model_validate, and pydantic.TypeAdapter.
4. Procedural heightfield generation & watertight diorama slab mesh building:
   - Every diorama mesh is 100% watertight:
     * boundary_edges == 0
     * euler_characteristic == 2 (chi = V - E + F)
     * non_manifold_edges == 0
     * non_manifold_vertices == 0
     * orientation_conflicts == 0
5. Adversarial edge cases:
   - Extreme diurnal hours (negative, >24, fractional).
   - Extreme CCT ranges (1000K to 40000K).
   - Resolution invariance for diorama watertightness (res=21, 33, 45).
"""

import json
import math
import numpy as np
import pytest
import pydantic

from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    PRESET_CONFIGS,
)
from terra_forge.cli import PRESETS_DIR, load_preset_or_file
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
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

M4_PRESET_NAMES = [
    "alpine_peak_glacial_lake",
    "temperate_forest_river_basin",
    "tropical_island_coastal_karst",
    "canyon_semiarid_plateau",
]


class TestDiurnalLightingSweep:
    """Empirical verification of 24-hour diurnal solar ephemeris and directional tracking."""

    def test_solar_elevation_sinusoidal_sweep_24h(self):
        """Verify solar elevation precisely matches the sinusoidal diurnal equation:
        elev(t) = max_elev * sin((t - 6) / 12 * pi) across 2400 discrete time steps."""
        max_elev = 68.0
        time_steps = np.linspace(0.0, 24.0, 2401)

        for t in time_steps:
            elev_deg, azim_deg, sun_vec = calculate_solar_position(t, max_elevation_deg=max_elev)

            t_norm = t % 24.0
            expected_elev = max_elev * math.sin((t_norm - 6.0) / 12.0 * math.pi)
            expected_elev = max(-90.0, min(90.0, expected_elev))

            assert elev_deg == pytest.approx(expected_elev, abs=1e-5), (
                f"Solar elevation mismatch at hour {t:.2f}: got {elev_deg}, expected {expected_elev}"
            )
            assert -90.0 <= elev_deg <= 90.0

    def test_solar_azimuth_sweep_360_degrees(self):
        """Verify solar azimuth sweeps full 360° clockwise from North (15°/hr) across 24 hours."""
        for hour in range(25):
            t = float(hour)
            _, azim_deg, _ = calculate_solar_position(t)
            expected_azim = (t / 24.0 * 360.0) % 360.0

            assert azim_deg == pytest.approx(expected_azim, abs=1e-4), (
                f"Azimuth mismatch at hour {t}: got {azim_deg}, expected {expected_azim}"
            )

        _, azim_midnight, _ = calculate_solar_position(0.0)
        _, azim_dawn, _ = calculate_solar_position(6.0)
        _, azim_noon, _ = calculate_solar_position(12.0)
        _, azim_dusk, _ = calculate_solar_position(18.0)

        assert azim_midnight == pytest.approx(0.0, abs=1e-4)   # North
        assert azim_dawn == pytest.approx(90.0, abs=1e-4)     # East
        assert azim_noon == pytest.approx(180.0, abs=1e-4)    # South
        assert azim_dusk == pytest.approx(270.0, abs=1e-4)    # West

    def test_solar_direction_unit_vector_normalization(self):
        """Verify solar direction vector (lx, ly, lz) has exact unit magnitude (|v| == 1.0)
        and follows hemispheric orientation."""
        time_samples = np.linspace(0.0, 24.0, 481)
        for t in time_samples:
            elev_deg, azim_deg, vec = calculate_solar_position(t)
            lx, ly, lz = vec

            mag = math.sqrt(lx * lx + ly * ly + lz * lz)
            assert mag == pytest.approx(1.0, abs=1e-6), (
                f"Solar direction vector is not normalized at hour {t}: norm={mag}"
            )

            if elev_deg > 0.0:
                assert lz > 0.0, f"Sun elevation positive ({elev_deg:.2f}°) but lz <= 0 at hour {t}"
            elif elev_deg < 0.0:
                assert lz < 0.0, f"Sun elevation negative ({elev_deg:.2f}°) but lz >= 0 at hour {t}"

    def test_diurnal_sweep_extreme_and_negative_hours(self):
        """Verify calculate_solar_position safely handles out-of-range hours via modulo 24."""
        for t in [-48.0, -18.0, -6.0, 24.0, 30.0, 72.0, 100.5]:
            elev, azim, vec = calculate_solar_position(t)
            assert math.isfinite(elev)
            assert math.isfinite(azim)
            assert 0.0 <= azim < 360.0
            assert -90.0 <= elev <= 90.0
            mag = math.sqrt(sum(v * v for v in vec))
            assert mag == pytest.approx(1.0, abs=1e-6)


class TestCCTAndPhotometricColor:
    """Empirical verification of blackbody CCT curves and linear photometric RGB generation."""

    def test_cct_continuity_across_elevation_regimes(self):
        """Verify CCT function is continuous across boundary thresholds (-6°, 0°, 10°, 45°)."""
        boundaries = [-6.0, 0.0, 10.0, 45.0]
        eps = 1e-5

        for b in boundaries:
            cct_left = calculate_cct_from_elevation(b - eps)
            cct_center = calculate_cct_from_elevation(b)
            cct_right = calculate_cct_from_elevation(b + eps)

            assert abs(cct_left - cct_center) < 0.5, (
                f"CCT jump discontinuity at left of {b}°: |{cct_left} - {cct_center}| >= 0.5"
            )
            assert abs(cct_right - cct_center) < 0.5, (
                f"CCT jump discontinuity at right of {b}°: |{cct_right} - {cct_center}| >= 0.5"
            )

    def test_cct_transition_regimes(self):
        """Verify CCT transitions through specified physical domains:
        - 2200K at horizon (0.0°) to 3200K at 10° (golden hour)
        - 5800K at 45° to 6500K at 90° (crisp white midday daylight)
        - >= 7000K to 9000K during twilight/night (< 0.0°)."""
        assert calculate_cct_from_elevation(0.0) == pytest.approx(2200.0, abs=1e-3)
        assert calculate_cct_from_elevation(10.0) == pytest.approx(3200.0, abs=1e-3)

        assert calculate_cct_from_elevation(45.0) == pytest.approx(5800.0, abs=1e-3)
        assert calculate_cct_from_elevation(90.0) == pytest.approx(6500.0, abs=1e-3)

        assert calculate_cct_from_elevation(-6.0) == pytest.approx(9000.0, abs=1e-3)
        assert calculate_cct_from_elevation(-30.0) == pytest.approx(9000.0, abs=1e-3)
        assert calculate_cct_from_elevation(-90.0) == pytest.approx(9000.0, abs=1e-3)

        for elev in np.linspace(-6.0, 0.0, 25):
            cct = calculate_cct_from_elevation(elev)
            assert 2200.0 <= cct <= 9000.0

    def test_cct_to_linear_rgb_bounds_and_finiteness(self):
        """Verify Tanner-Helland linear RGB values are strictly non-negative, finite,
        and clamped in [0.0, 1.0] across fine temperature sweep (1000K to 40000K)."""
        temp_sweep = np.linspace(1000.0, 40000.0, 3901)

        for temp in temp_sweep:
            r, g, b = cct_to_linear_rgb(temp)

            assert math.isfinite(r) and math.isfinite(g) and math.isfinite(b), (
                f"Non-finite RGB at {temp}K: ({r}, {g}, {b})"
            )
            assert 0.0 <= r <= 1.0, f"Red out of [0, 1] at {temp}K: {r}"
            assert 0.0 <= g <= 1.0, f"Green out of [0, 1] at {temp}K: {g}"
            assert 0.0 <= b <= 1.0, f"Blue out of [0, 1] at {temp}K: {b}"

    def test_cct_to_linear_rgb_spectral_ratio_physics(self):
        """Verify physical spectral behavior:
        - Warm amber / red dominant at low temperatures (<= 3000K)
        - Crisp neutral white daylight at 5800K - 6500K
        - Cool blue dominant at high temperatures (>= 8000K)."""
        r_warm, g_warm, b_warm = cct_to_linear_rgb(2000.0)
        assert r_warm > g_warm > b_warm, f"Warm spectrum violation: {r_warm}, {g_warm}, {b_warm}"

        r_day, g_day, b_day = cct_to_linear_rgb(6500.0)
        assert abs(r_day - b_day) < 0.25, f"Daylight white balance skewed: R={r_day}, B={b_day}"

        r_cool, g_cool, b_cool = cct_to_linear_rgb(9000.0)
        assert b_cool > r_cool, f"Cool ambient blue violation: R={r_cool}, B={b_cool}"


class TestAtmosphericAirMassAndIntensity:
    """Empirical verification of Kasten-Young air mass and Bouguer-Lambert-Beer optical extinction."""

    def test_kasten_young_air_mass_zenith_to_horizon(self):
        """Verify optical air mass m(elevation):
        - m = 1.00 at zenith (90°) [0.9997 via Kasten-Young formula]
        - m strictly decreases as elevation increases towards zenith
        - m = 38.0 at horizon (0.0°)
        - m is bounded and positive for all elevations."""
        elevations = np.linspace(0.0, 90.0, 901)
        prev_m = 999.0

        for elev in elevations:
            m = calculate_optical_air_mass(elev)
            assert math.isfinite(m)
            assert 0.99 <= m <= 39.0
            if elev > 0.0:
                assert m < prev_m, f"Air mass not strictly decreasing at elev={elev}: {m} >= {prev_m}"
            prev_m = m

        m_zenith = calculate_optical_air_mass(90.0)
        m_horizon = calculate_optical_air_mass(0.0)
        assert m_zenith == pytest.approx(1.0, abs=0.01)
        assert m_horizon == pytest.approx(38.0, abs=0.1)

    def test_sun_intensity_strictly_zero_below_horizon(self):
        """Verify direct solar intensity I -> 0.0 strictly when sun drops below horizon (elev <= 0.0°)."""
        negative_elevations = [-90.0, -68.0, -45.0, -18.0, -6.0, -1.0, -0.01, -1e-6, 0.0]

        for elev in negative_elevations:
            i_val = calculate_sun_intensity(elev, i0=5.5, tau_r=0.10)
            assert i_val == 0.0, f"Intensity not zero at sub-horizon elevation {elev}°: got {i_val}"

    def test_sun_intensity_monotonic_above_horizon(self):
        """Verify direct solar intensity I increases monotonically with solar altitude above horizon."""
        elevations = np.linspace(0.1, 90.0, 900)
        prev_i = -1.0

        for elev in elevations:
            i_val = calculate_sun_intensity(elev, i0=5.5, tau_r=0.10)
            assert i_val > prev_i, f"Intensity not strictly monotonic at elev={elev}°: {i_val} <= {prev_i}"
            assert i_val <= 5.5
            prev_i = i_val


class TestPresetFilesSchemaAndPydantic:
    """Empirical verification of the 4 natural preset files with MapConfig.from_dict and pydantic."""

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_json_file_parsing(self, preset_stem):
        """Verify JSON file exists and contains all required top-level configuration blocks."""
        preset_path = PRESETS_DIR / f"{preset_stem}.json"
        assert preset_path.exists(), f"Preset file missing: {preset_path}"

        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "name", "theme", "seed", "diorama", "peaks",
            "strata", "erosion", "hydrology", "biomes",
            "shaders", "atmosphere", "export"
        ]
        for key in required_keys:
            assert key in data, f"Preset '{preset_stem}' missing required key '{key}'"

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_validation_with_from_dict(self, preset_stem):
        """Verify preset parses into MapConfig via MapConfig.from_dict."""
        preset_path = PRESETS_DIR / f"{preset_stem}.json"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = MapConfig.from_dict(data)
        assert isinstance(cfg, MapConfig)
        assert cfg.diorama.width == 160.0
        assert cfg.diorama.depth == 160.0
        assert len(cfg.peaks) >= 3
        assert len(cfg.biomes) == 4
        assert cfg.atmosphere.enabled is True

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_validation_with_pydantic_type_adapter(self, preset_stem):
        """Verify preset strictly validates against Pydantic TypeAdapter."""
        preset_path = PRESETS_DIR / f"{preset_stem}.json"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        adapter = pydantic.TypeAdapter(MapConfig)
        cfg = adapter.validate_python(data)
        assert isinstance(cfg, MapConfig)
        assert cfg.diorama.resolution >= 33
        assert cfg.atmosphere.max_elevation_deg == 68.0

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_validation_with_model_validate(self, preset_stem):
        """Verify preset validates via MapConfig.model_validate."""
        data = load_preset_or_file(preset_stem)
        cfg = MapConfig.model_validate(data)
        assert isinstance(cfg, MapConfig)
        assert cfg.name is not None
        assert cfg.export.glb_path is not None


class TestPresetHeightfieldAndWatertightDiorama:
    """Empirical verification of heightfield generation and 100% watertight diorama slab mesh building."""

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_procedural_generation_and_watertight_diorama(self, preset_stem):
        """Generate heightfield, apply erosion/hydrology, construct diorama slab mesh,
        and assert 100% watertight 2-manifold topology:
        - boundary_edges == 0
        - euler_characteristic == 2 (chi = V - E + F)
        - non_manifold_edges == 0
        - non_manifold_vertices == 0
        - orientation_conflicts == 0."""
        preset_path = PRESETS_DIR / f"{preset_stem}.json"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MapConfig.model_validate(data)

        # Fast and discrete resolution
        res = 25
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=res)

        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        if cfg.hydrology.lake_radius > 0 and abs(cfg.hydrology.lake_center[0]) < cfg.diorama.width:
            hf.carve_lake_basin(
                cfg.hydrology.lake_center[0],
                cfg.hydrology.lake_center[1],
                cfg.hydrology.lake_radius,
                cfg.hydrology.lake_level,
                cfg.hydrology.lake_depth,
                berm_height=1.5,
            )

        if cfg.erosion.enabled:
            ero = ErosionSimulator(hf, cfg.erosion)
            ero.run_hydraulic_erosion(seed=cfg.seed)
            ero.run_thermal_erosion()

        if cfg.hydrology.river_enabled:
            hydro = HydrologyEngine(hf, cfg.hydrology)
            hydro.carve_river_channel()

        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        metrics = topo["metrics"]

        assert topo["passed"] is True, f"Topology check failed for {preset_stem}: {topo['errors']}"
        assert topo["is_watertight"] is True, f"Mesh not watertight for {preset_stem}"
        assert metrics["boundary_edges"] == 0, f"Found {metrics['boundary_edges']} boundary edges"
        assert metrics["euler_characteristic"] == 2, f"Euler chi={metrics['euler_characteristic']} != 2"
        assert metrics["non_manifold_edges"] == 0, f"Found {metrics['non_manifold_edges']} non-manifold edges"
        assert metrics["non_manifold_vertices"] == 0, f"Found {metrics['non_manifold_vertices']} non-manifold verts"
        assert metrics["pinch_point_vertices"] == 0, f"Found {metrics['pinch_point_vertices']} pinch points"
        assert metrics["orientation_conflicts"] == 0, f"Found {metrics['orientation_conflicts']} winding conflicts"

    def test_watertightness_resolution_invariance(self):
        """Stress-test diorama slab topological invariants across varying grid resolutions (res=17, 33, 41)."""
        cfg = PRESET_CONFIGS["alpine_peak_glacial_lake"]

        for res in [17, 33, 41]:
            hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=res)
            for p in cfg.peaks:
                hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

            slab = DioramaSlabBuilder(hf, cfg.diorama)
            mesh_data = slab.build_mesh_data()
            topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
            metrics = topo["metrics"]

            assert topo["is_watertight"] is True
            assert metrics["boundary_edges"] == 0
            assert metrics["euler_characteristic"] == 2
            assert metrics["non_manifold_edges"] == 0
            assert metrics["non_manifold_vertices"] == 0


class TestHydrologicalAndGeologicalInvariants:
    """Empirical verification of lake berm freeboard and exterior basin containment."""

    @pytest.mark.parametrize("preset_stem", ["alpine_peak_glacial_lake", "temperate_forest_river_basin"])
    def test_lake_berm_freeboard_margin(self, preset_stem):
        """Verify lake retaining berm has minimum vertical freeboard margin >= 0.50m."""
        cfg = PRESET_CONFIGS[preset_stem]
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=33)

        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        hf.carve_lake_basin(
            cfg.hydrology.lake_center[0],
            cfg.hydrology.lake_center[1],
            cfg.hydrology.lake_radius,
            cfg.hydrology.lake_level,
            cfg.hydrology.lake_depth,
            berm_height=1.5,
        )

        validator = QualityValidator(cfg, hf)
        berm_res = validator.validate_lake_berm(min_freeboard_threshold=0.50)

        assert berm_res["passed"] is True, f"Lake berm freeboard violation: {berm_res['errors']}"
        assert berm_res["metrics"]["lake_min_freeboard_berm"] >= 0.50

    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_water_basin_exterior_wall_containment(self, preset_stem):
        """Verify water basin volume does not penetrate exterior walls of diorama."""
        cfg = PRESET_CONFIGS[preset_stem]
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=33)

        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        validator = QualityValidator(cfg, hf)
        contain_res = validator.validate_water_basin_containment()

        assert contain_res["passed"] is True, f"Water basin leak violation: {contain_res['errors']}"
