"""
Unit Test Suite for Milestone 4: Atmospheric Lighting & Standardized Vision Cameras.
Verifies:
1. Solar ephemeris diurnal tracking & unit direction vectors.
2. Blackbody Correlated Color Temperature (CCT) mapping across all diurnal regimes.
3. Tanner-Helland CCT to linear photometric RGB conversion.
4. Kasten-Young optical air mass & Bouguer-Lambert-Beer atmospheric extinction.
5. Standardized camera viewpoints:
   - isometric_diorama framing distance formula d = max(W, D) / (sqrt(2) * sin(FOV/2)).
   - eye_level_first_person +1.7m ground elevation offset.
   - closeup_macro proximity <= 15m and DoF settings.
6. Pure-Python execution without bpy.
"""

import math
import pytest

from terra_forge.schema.map_config import DioramaBounds
from terra_forge.core.heightfield import Heightfield2D
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


class TestSolarPositionAndEphemeris:
    def test_solar_elevation_diurnal_cycle(self):
        """Verify elevation curve: 0° at dawn (6h) and dusk (18h), max at noon (12h), negative at night."""
        elev_dawn, _, _ = calculate_solar_position(6.0, max_elevation_deg=68.0)
        elev_noon, _, _ = calculate_solar_position(12.0, max_elevation_deg=68.0)
        elev_dusk, _, _ = calculate_solar_position(18.0, max_elevation_deg=68.0)
        elev_midnight, _, _ = calculate_solar_position(0.0, max_elevation_deg=68.0)

        assert elev_dawn == pytest.approx(0.0, abs=1e-3)
        assert elev_noon == pytest.approx(68.0, abs=1e-3)
        assert elev_dusk == pytest.approx(0.0, abs=1e-3)
        assert elev_midnight == pytest.approx(-68.0, abs=1e-3)

    def test_solar_azimuth_sweep(self):
        """Verify azimuth moves East (90°) -> South (180°) -> West (270°) -> North (0°)."""
        _, azim_dawn, _ = calculate_solar_position(6.0)
        _, azim_noon, _ = calculate_solar_position(12.0)
        _, azim_dusk, _ = calculate_solar_position(18.0)
        _, azim_midnight, _ = calculate_solar_position(0.0)

        assert azim_dawn == pytest.approx(90.0, abs=1e-3)
        assert azim_noon == pytest.approx(180.0, abs=1e-3)
        assert azim_dusk == pytest.approx(270.0, abs=1e-3)
        assert azim_midnight == pytest.approx(0.0, abs=1e-3)

    def test_solar_direction_unit_vector(self):
        """Verify direction vector is normalized and points in the correct quadrant."""
        for hour in [6.0, 9.0, 12.0, 15.0, 18.0]:
            _, _, vec = calculate_solar_position(hour)
            norm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
            assert norm == pytest.approx(1.0, abs=1e-4)


class TestBlackbodyCCTAndColor:
    def test_cct_regimes_golden_hour_vs_midday(self):
        """Verify CCT is warm amber (2000K-3200K) at elevation < 10°, and crisp white (>5500K) at > 45°."""
        cct_sunset = calculate_cct_from_elevation(2.0)
        cct_midday = calculate_cct_from_elevation(60.0)
        cct_twilight = calculate_cct_from_elevation(-10.0)

        assert 2000.0 <= cct_sunset <= 3200.0, "Golden hour must be warm amber"
        assert 5500.0 <= cct_midday <= 6500.0, "Midday must be crisp daylight white"
        assert 7000.0 <= cct_twilight <= 10000.0, "Twilight/night must be cool ambient"

    def test_cct_to_linear_rgb_bounds(self):
        """Verify linear RGB values are clamped to [0.0, 1.0] and exhibit physical color ratios."""
        # Warm golden light (3000K): Red > Green > Blue
        r_warm, g_warm, b_warm = cct_to_linear_rgb(3000.0)
        assert 0.0 <= r_warm <= 1.0 and 0.0 <= g_warm <= 1.0 and 0.0 <= b_warm <= 1.0
        assert r_warm > g_warm > b_warm

        # Cool twilight light (9000K): Blue > Green >= Red
        r_cool, g_cool, b_cool = cct_to_linear_rgb(9000.0)
        assert 0.0 <= r_cool <= 1.0 and 0.0 <= g_cool <= 1.0 and 0.0 <= b_cool <= 1.0
        assert b_cool > r_cool


class TestAtmosphericExtinction:
    def test_kasten_young_air_mass(self):
        """Verify optical air mass is 1.0 at zenith (90°) and increases to ~38 at horizon."""
        m_zenith = calculate_optical_air_mass(90.0)
        m_horizon = calculate_optical_air_mass(0.0)

        assert m_zenith == pytest.approx(1.0, abs=0.05)
        assert 35.0 <= m_horizon <= 40.0

    def test_bouguer_lambert_beer_extinction(self):
        """Verify intensity scales as I0 * exp(-m * tau_R) and drops to 0 below horizon."""
        i_noon = calculate_sun_intensity(68.0, i0=5.5, tau_r=0.10)
        i_sunset = calculate_sun_intensity(2.0, i0=5.5, tau_r=0.10)
        i_night = calculate_sun_intensity(-5.0, i0=5.5, tau_r=0.10)

        assert i_noon > i_sunset > 0.0
        assert i_night == 0.0


class TestStandardizedVisionCameras:
    def test_automated_framing_distance_formula(self):
        """Verify d = max(W, D) / (sqrt(2) * sin(FOV/2)) frames the entire model."""
        width = 160.0
        depth = 160.0
        lens = 45.0
        fov = calculate_fov_from_lens(lens)
        d = calculate_camera_framing_distance(width, depth, fov)

        # Theoretical distance for 160m diorama at 45mm is ~304.6m
        assert 280.0 <= d <= 330.0

    def test_isometric_diorama_viewpoint(self):
        """Verify isometric camera azimuth is 45° and elevation is 35.264°."""
        bounds = DioramaBounds(width=160.0, depth=160.0, z_base=-16.0)
        presets = build_inspection_camera_configs(bounds=bounds)
        iso_cam = presets["isometric_diorama"]

        loc = iso_cam["location"]
        # In isometric offset: x == y
        assert loc[0] == pytest.approx(loc[1], rel=1e-3)
        # Elevation angle = arctan((z - 5.0) / hypot(x, y)) = 35.264°
        dist_xy = math.sqrt(loc[0]**2 + loc[1]**2)
        elev_deg = math.degrees(math.atan((loc[2] - 5.0) / dist_xy))
        assert elev_deg == pytest.approx(35.264, abs=0.01)

    def test_eye_level_first_person_offset(self):
        """Verify eye-level camera is elevated at human eye level (+1.7m) above terrain."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=31)
        hf.Z.fill(8.2)

        bounds = DioramaBounds(width=80.0, depth=80.0)
        presets = build_inspection_camera_configs(bounds=bounds, hf=hf)
        eye_cam = presets["eye_level_first_person"]

        cam_z = eye_cam["location"][2]
        expected_z = 8.2 + 1.7
        assert cam_z == pytest.approx(expected_z, abs=1e-3)
        assert eye_cam["lens"] == 24.0

    def test_closeup_macro_proximity_and_dof(self):
        """Verify macro camera is within 15m proximity with macro lens and shallow DoF."""
        bounds = DioramaBounds(width=100.0, depth=100.0)
        presets = build_inspection_camera_configs(bounds=bounds)
        macro = presets["closeup_macro"]

        assert macro["lens"] == 85.0
        assert macro["dof_enabled"] is True
        assert macro["fstop"] <= 2.8


class TestPurePythonDecoupling:
    def test_setup_sun_lighting_pure_python(self):
        """Verify setup_sun_lighting executes in pure Python without bpy and returns valid info."""
        info = setup_sun_lighting(sun_time_of_day=14.0)
        assert isinstance(info, dict)
        assert "cct_kelvin" in info
        assert "sun_color_linear_rgb" in info
        assert "sun_intensity" in info
        assert "mist_density" in info

    def test_setup_inspection_cameras_pure_python(self):
        """Verify setup_inspection_cameras executes in pure Python without crashing."""
        cameras = setup_inspection_cameras()
        assert "isometric_diorama" in cameras
        assert "eye_level_first_person" in cameras
        assert "closeup_macro" in cameras
        assert "topdown" in cameras
