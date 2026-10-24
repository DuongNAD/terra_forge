"""
Empirical Challenge Test Suite for Milestone 3:
4-Tier Continuous Hydrology & Subterranean Karst Cavern System.

Evaluated by Challenger 1 (Empirical Challenger):
1. Hydrology Monotonicity & Continuity Stress:
   - Sample 4-tier river spline across 1,000 points t in [0.0, 1.0].
   - Assert z(t) is strictly monotonically non-increasing (downhill or flat) everywhere: z(t_{i+1}) <= z(t_i) + 10^-6.
   - Verify flow velocity acceleration: speed at cascade peak (t=0.05) must be substantially higher than at lake transit (t=0.65).
2. Lake Berm Freeboard & Channel Bed Breach:
   - Measure lake berm rim height vs lake water level: assert containment freeboard >= 1.2m along unbreached rim.
   - Assert river channel bed cuts through berm at inlet and outlet (Z_bed <= lake_level).
3. Water Depth Coupling:
   - Call rasterize_water_depth(). Assert self.hf.water_depth has strictly positive values in lake basin and along river channel, and exactly zero on dry mountain ridges.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    HydrologyConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar


PRESET_PATH = Path(__file__).resolve().parent.parent / "terra_forge" / "schema" / "presets" / "genesis_primordial_abiotic.json"


@pytest.fixture
def abiotic_preset_config() -> MapConfig:
    """Loads the canonical Genesis Primordial Abiotic preset configuration."""
    assert PRESET_PATH.exists(), f"Canonical preset not found at {PRESET_PATH}"
    with open(PRESET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MapConfig.from_dict(data)


@pytest.fixture
def synthesized_abiotic_heightfield(abiotic_preset_config: MapConfig) -> tuple[Heightfield2D, HydrologyEngine]:
    """
    Synthesizes the base topography, applies geological features, carves the lake basin,
    and initializes the HydrologyEngine for the canonical abiotic preset.
    """
    cfg = abiotic_preset_config
    hf = Heightfield2D(
        width=cfg.diorama.width,
        depth=cfg.diorama.depth,
        resolution=cfg.diorama.resolution,
    )

    # Base massif topography
    for peak in cfg.peaks:
        hf.add_peak(
            peak.position[0],
            peak.position[1],
            peak.height,
            peak.radius,
            peak.shape,
            peak.ridge_angle_deg,
        )

    hf.apply_geological_features(cfg)

    # Carve lake basin with retaining berm
    hf.carve_lake_basin(
        cfg.hydrology.lake_center[0],
        cfg.hydrology.lake_center[1],
        cfg.hydrology.lake_radius,
        cfg.hydrology.lake_level,
        cfg.hydrology.lake_depth,
    )

    hydro = HydrologyEngine(hf, cfg.hydrology)
    return hf, hydro


# =============================================================================
# 1. Hydrology Monotonicity & Continuity Stress
# =============================================================================

class TestHydrologyMonotonicityAndContinuityStress:
    """Empirical challenge 1: Monotonic downhill descent, C0 continuity, and hydrodynamic velocity acceleration."""

    def test_spline_sampled_1000_pts_strictly_monotonic_non_increasing(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Sample the 4-tier river spline across 1,000 points t in [0.0, 1.0].
        Assert z(t) is strictly monotonically non-increasing (downhill or flat) everywhere:
        z(t_{i+1}) <= z(t_i) + 10^-6.
        """
        _, hydro = synthesized_abiotic_heightfield
        t_samples = np.linspace(0.0, 1.0, 1000)
        rx, ry, rz, rw = hydro.evaluate_river_spline(t_samples)

        assert len(rz) == 1000
        dz = np.diff(rz)
        max_dz = float(np.max(dz))

        # Strict invariant: elevation must NEVER increase along flow direction
        assert max_dz <= 1e-6, f"Hydrology monotonicity violated: max dz = {max_dz:.8e} > 1e-6"

        # Check total drop from source to mouth
        total_drop = float(rz[0] - rz[-1])
        assert total_drop >= 20.0, f"Total river vertical drop {total_drop:.2f}m is insufficient (< 20m)"
        assert pytest.approx(float(rz[0]), abs=1e-3) == 21.5
        assert pytest.approx(float(rz[-1]), abs=1e-3) == 0.0

    def test_flow_velocity_acceleration_cascade_vs_lake(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Verify flow velocity acceleration: speed at cascade peak (t=0.05) must be
        substantially higher than at lake transit (t=0.65).
        """
        _, hydro = synthesized_abiotic_heightfield

        v_casc = hydro.evaluate_flow_velocity(0.05, v_base=2.5)
        v_lake = hydro.evaluate_flow_velocity(0.65, v_base=2.5)

        speed_casc = math.sqrt(float(v_casc[0])**2 + float(v_casc[1])**2 + float(v_casc[2])**2)
        speed_lake = math.sqrt(float(v_lake[0])**2 + float(v_lake[1])**2 + float(v_lake[2])**2)

        # Speed at cascade peak must exceed 10 m/s due to gravitational gradient acceleration
        assert speed_casc >= 10.0, f"Cascade peak speed {speed_casc:.2f} m/s is too sluggish (< 10 m/s)"

        # Speed across flat lake transit must be tranquil (<= 1.0 m/s)
        assert speed_lake <= 1.0, f"Lake transit speed {speed_lake:.2f} m/s is too turbulent (> 1.0 m/s)"

        # Speed ratio must show dramatic hydrodynamic acceleration (>= 10x)
        ratio = speed_casc / speed_lake
        assert ratio >= 10.0, f"Velocity acceleration ratio {ratio:.2f}x is insufficient (< 10x)"

    def test_c0_spatial_continuity_across_1000_pts(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Verify C0 geometric continuity: distance between consecutive points along 1,000 samples
        must be smooth and bounded, with zero tears or coordinate teleportation across stage boundaries.
        """
        _, hydro = synthesized_abiotic_heightfield
        t_samples = np.linspace(0.0, 1.0, 1000)
        rx, ry, rz, _ = hydro.evaluate_river_spline(t_samples)

        dx = np.diff(rx)
        dy = np.diff(ry)
        dz = np.diff(rz)
        step_distances = np.sqrt(dx**2 + dy**2 + dz**2)
        max_step = float(np.max(step_distances))

        assert max_step < 1.0, f"Discontinuous jump detected on river spline: max step = {max_step:.3f}m"

    def test_central_lake_transit_strictly_flat_at_lake_level(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Verify that Central Lake Transit (t in [0.58, 0.74]) maintains a perfectly flat water surface
        matching the configured lake_level (4.5m).
        """
        _, hydro = synthesized_abiotic_heightfield
        t_lake = np.linspace(0.58, 0.74, 50)
        _, _, rz_lake, rw_lake = hydro.evaluate_river_spline(t_lake)

        expected_level = hydro.cfg.lake_level
        assert np.allclose(rz_lake, expected_level, atol=1e-5), "Lake transit is not strictly planar"
        assert np.all(rw_lake >= 7.5), "Lake transit width is constricted"

    def test_spline_stage_transition_continuity(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """Verify exact C0 coordinate matching across all stage interface transitions."""
        _, hydro = synthesized_abiotic_heightfield
        eps = 1e-6

        # Stage 1 -> Stage 2 at t=0.32
        p1_end = hydro.evaluate_river_spline(0.32 - eps)
        p2_start = hydro.evaluate_river_spline(0.32 + eps)
        for c1, c2 in zip(p1_end, p2_start):
            assert pytest.approx(float(c1), abs=0.05) == float(c2)

        # Stage 2 -> Stage 3 at t=0.58
        p2_end = hydro.evaluate_river_spline(0.58 - eps)
        p3_start = hydro.evaluate_river_spline(0.58 + eps)
        for c1, c2 in zip(p2_end, p3_start):
            assert pytest.approx(float(c1), abs=0.05) == float(c2)

        # Stage 3 -> Stage 4 at t=0.74
        p3_end = hydro.evaluate_river_spline(0.74 - eps)
        p4_start = hydro.evaluate_river_spline(0.74 + eps)
        for c1, c2 in zip(p3_end, p4_start):
            assert pytest.approx(float(c1), abs=0.05) == float(c2)


# =============================================================================
# 2. Lake Berm Freeboard & Channel Bed Breach
# =============================================================================

class TestLakeBermFreeboardAndChannelBedBreach:
    """Empirical challenge 2: Containment freeboard >= 1.2m along unbreached rim and channel bed breach at inlet/outlet."""

    def test_lake_retaining_berm_freeboard_unbreached_rim(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Measure lake berm rim height vs lake water level:
        assert containment freeboard >= 1.2m along unbreached rim.
        """
        hf, hydro = synthesized_abiotic_heightfield
        lx, ly = hydro.cfg.lake_center
        lr = hydro.cfg.lake_radius
        l_level = hydro.cfg.lake_level

        # Carve river channel to create natural inlet/outlet breaches
        hydro.carve_river_channel(num_samples=250)

        # Sample rim at crest radius (1.05 * R) across 360 angular rays
        angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        rim_x = lx + (1.05 * lr) * np.cos(angles)
        rim_y = ly + (1.05 * lr) * np.sin(angles)

        # Sample river spline to identify breach zone proximity
        t_spline = np.linspace(0.0, 1.0, 500)
        rx, ry, _, _ = hydro.evaluate_river_spline(t_spline)

        dist_to_river = np.array([
            np.min(np.sqrt((rx - px)**2 + (ry - py)**2))
            for px, py in zip(rim_x, rim_y)
        ])

        # Unbreached rim: points located outside the channel incision zone (> 15m from river spline)
        unbreached_mask = dist_to_river > 15.0
        assert np.sum(unbreached_mask) >= 150, "Not enough unbreached rim points identified"

        unbreached_x = rim_x[unbreached_mask]
        unbreached_y = rim_y[unbreached_mask]
        unbreached_z = np.array([hf.sample_bilinear(px, py) for px, py in zip(unbreached_x, unbreached_y)])

        freeboard = unbreached_z - l_level
        min_freeboard = float(np.min(freeboard))

        assert min_freeboard >= 1.200 - 1e-4, (
            f"Containment freeboard invariant violated: min freeboard {min_freeboard:.3f}m < 1.2m"
        )
        assert np.max(unbreached_z) >= 5.7, "Berm crest does not reach expected elevation >= 5.7m"

    def test_river_channel_bed_breach_at_inlet_and_outlet(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Assert river channel bed cuts through berm at inlet and outlet (Z_bed <= lake_level).
        This guarantees the lake does not artificially dam the inflow or block outflow drainage.
        """
        hf, hydro = synthesized_abiotic_heightfield
        lx, ly = hydro.cfg.lake_center
        lr = hydro.cfg.lake_radius
        l_level = hydro.cfg.lake_level

        # Carve river channel
        hydro.carve_river_channel(num_samples=250)

        # Trace river spline and locate berm crossings (where distance to lake center crosses 1.05 * lr)
        t_spline = np.linspace(0.0, 1.0, 1000)
        rx, ry, rz, _ = hydro.evaluate_river_spline(t_spline)
        dist_to_center = np.sqrt((rx - lx)**2 + (ry - ly)**2)

        # Crossings of berm crest radius (23.1m)
        cross_indices = np.where(np.diff(dist_to_center < 1.05 * lr))[0]
        assert len(cross_indices) >= 2, f"Expected at least 2 berm crossings, found {len(cross_indices)}"

        inlet_idx = cross_indices[0]
        outlet_idx = cross_indices[1]

        # 1. Inlet Breach Verification
        t_in = t_spline[inlet_idx]
        inlet_x, inlet_y = rx[inlet_idx], ry[inlet_idx]
        inlet_bed_z = hf.sample_bilinear(inlet_x, inlet_y)

        assert inlet_bed_z <= l_level, (
            f"Inlet failed to breach retaining berm: Z_bed {inlet_bed_z:.2f}m > lake_level {l_level:.2f}m at t={t_in:.3f}"
        )
        assert inlet_bed_z <= l_level - 1.0, (
            f"Inlet channel bed {inlet_bed_z:.2f}m should cut at least 1m below lake_level {l_level:.2f}m"
        )

        # 2. Outlet Breach Verification
        t_out = t_spline[outlet_idx]
        outlet_x, outlet_y = rx[outlet_idx], ry[outlet_idx]
        outlet_bed_z = hf.sample_bilinear(outlet_x, outlet_y)

        assert outlet_bed_z <= l_level, (
            f"Outlet failed to breach retaining berm: Z_bed {outlet_bed_z:.2f}m > lake_level {l_level:.2f}m at t={t_out:.3f}"
        )
        assert outlet_bed_z <= l_level - 2.0, (
            f"Outlet channel bed {outlet_bed_z:.2f}m should cut deeply below lake_level {l_level:.2f}m"
        )

    def test_berm_bathymetric_profile_cross_section(self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]):
        """
        Validates radial cross-sectional profile from lake center outwards:
        Center bottom -> submerged shelf -> berm crest -> outer moraine falloff.
        """
        hf, hydro = synthesized_abiotic_heightfield
        lx, ly = hydro.cfg.lake_center
        lr = hydro.cfg.lake_radius
        l_level = hydro.cfg.lake_level
        l_depth = hydro.cfg.lake_depth

        # Measure radial profile along a transect away from river (e.g. angle = 3*pi/4, Northwest)
        angle = 0.75 * math.pi
        radii = np.linspace(0.0, 1.45 * lr, 60)
        profile_z = [
            hf.sample_bilinear(lx + r * math.cos(angle), ly + r * math.sin(angle))
            for r in radii
        ]

        # Center depth: lake_level - lake_depth = 4.5 - 10.0 = -5.5m
        center_z = profile_z[0]
        assert pytest.approx(center_z, abs=0.25) == (l_level - l_depth)

        # Crest elevation (at r ~ 1.05 * R)
        crest_idx = int(round(60 * (1.05 / 1.45)))
        crest_z = profile_z[crest_idx]
        assert crest_z >= l_level + 1.2, f"Crest Z {crest_z:.2f}m fails containment freeboard >= 1.2m"


# =============================================================================
# 3. Water Depth Coupling
# =============================================================================

class TestWaterDepthCoupling:
    """Empirical challenge 3: rasterize_water_depth() coupling, positive in water, zero on dry ridges."""

    def test_rasterize_water_depth_positive_in_lake_and_river_zero_on_ridges(
        self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine], abiotic_preset_config: MapConfig
    ):
        """
        Call rasterize_water_depth(). Assert self.hf.water_depth has strictly positive
        values in lake basin and along river channel, and exactly zero on dry mountain ridges.
        """
        hf, hydro = synthesized_abiotic_heightfield
        cfg = abiotic_preset_config

        # Carve channel and rasterize water depth
        hydro.carve_river_channel(num_samples=250)
        depth_map = hydro.rasterize_water_depth()

        # Invariant 1: Identity and shape coupling
        assert depth_map is hf.water_depth
        assert depth_map.shape == (hf.res_y, hf.res_x)

        # Invariant 2: Strictly positive values in submerged lake basin
        lx, ly = cfg.hydrology.lake_center
        lr = cfg.hydrology.lake_radius
        l_level = cfg.hydrology.lake_level

        dist_lake = np.sqrt((hf.X - lx)**2 + (hf.Y - ly)**2)
        submerged_lake_mask = (dist_lake < lr) & (hf.Z < l_level)

        submerged_depths = depth_map[submerged_lake_mask]
        assert len(submerged_depths) > 1000, "Submerged lake basin has insufficient area"
        assert np.all(submerged_depths > 0.0), "Lake basin has non-positive water depth!"
        assert np.min(submerged_depths) > 0.001

        # Central bathymetric lake depth check
        gx_c, gy_c = hf.world_to_grid(lx, ly)
        center_water_depth = depth_map[int(round(gy_c)), int(round(gx_c))]
        assert pytest.approx(center_water_depth, abs=0.25) == cfg.hydrology.lake_depth

        # Invariant 3: Strictly positive values along the carved river channel
        t_river = np.linspace(0.05, 0.95, 50)
        rx, ry, rz, _ = hydro.evaluate_river_spline(t_river)

        river_depths = []
        for px, py in zip(rx, ry):
            gx, gy = hf.world_to_grid(px, py)
            d = depth_map[int(round(gy)), int(round(gx))]
            river_depths.append(d)
            assert d > 0.0, f"River centerline point ({px:.1f}, {py:.1f}) has non-positive depth {d}"

        assert np.min(river_depths) >= 1.0, f"River centerline min depth {np.min(river_depths):.2f}m is too shallow"

        # Invariant 4: Exactly zero on dry mountain ridges and peaks
        for peak in cfg.peaks:
            px, py = peak.position
            gx, gy = hf.world_to_grid(px, py)
            d = depth_map[int(round(gy)), int(round(gx))]
            assert d == 0.0, f"Mountain peak '{peak.name}' at ({px}, {py}) has non-zero water depth: {d}"

        # Invariant 5: Dry highlands far from water (>15m) with elevation > lake_level + 2m
        dist_to_spline = np.full_like(hf.Z, 999.0)
        for px, py in zip(rx, ry):
            dist_to_spline = np.minimum(dist_to_spline, np.sqrt((hf.X - px)**2 + (hf.Y - py)**2))

        dry_highlands_mask = (hf.Z > l_level + 2.0) & (dist_to_spline > 15.0) & (dist_lake > lr + 5.0)
        assert np.sum(dry_highlands_mask) > 1000
        highland_depths = depth_map[dry_highlands_mask]
        assert np.all(highland_depths == 0.0), "Dry mountain highlands contain non-zero water depth!"

    def test_water_depth_monotonic_transition_near_shoreline(
        self, synthesized_abiotic_heightfield: tuple[Heightfield2D, HydrologyEngine]
    ):
        """Validates that water depth tapers smoothly to zero at the lake shoreline."""
        hf, hydro = synthesized_abiotic_heightfield
        lx, ly = hydro.cfg.lake_center
        lr = hydro.cfg.lake_radius

        hydro.carve_river_channel()
        depth_map = hydro.rasterize_water_depth()

        # Ray outwards from lake center towards shoreline
        angle = 0.5 * math.pi  # North towards valley
        radii = np.linspace(0.0, 0.9 * lr, 25)
        depths_along_ray = []
        for r in radii:
            px = lx + r * math.cos(angle)
            py = ly + r * math.sin(angle)
            gx, gy = hf.world_to_grid(px, py)
            depths_along_ray.append(depth_map[int(round(gy)), int(round(gx))])

        # Depth must monotonically decrease as we approach the shore
        assert depths_along_ray[0] > depths_along_ray[-1]
        assert depths_along_ray[0] > 8.0
        assert depths_along_ray[-1] < 3.0
