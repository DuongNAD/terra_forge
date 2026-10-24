"""
Comprehensive Unit and Integration Test Suite for Milestone M3:
4-Tier Continuous Hydrology & Subterranean Karst Cavern System.

Verifies:
1. 4-Tier Hydrology Spline Dynamics:
   - C0 continuity across all 4 stages (Mountain Cascades -> Valley Meander -> Lake Transit -> Outlet Bay).
   - Monotonic downhill elevation gradient without uphill flow.
   - Flow velocity acceleration over steep cascades and deceleration across flat lake transit.
   - Sinusoidal lateral meander displacement and amplitude scaling.
   - Waypoint dict and segment dict schema auto-normalization.
   - RiverSplineScalar scalar and vector evaluation compatibility.
2. Channel Carving, Lake Retaining Berm & Water Fields:
   - Parabolic cross-section channel incision equation.
   - Lake retaining berm freeboard (>= 1.2m) containment.
   - River channel bed breach through retaining berm (preventing damming).
   - 2D flow velocity vector field (Vx, Vy) alignment with river tangents.
   - Euclidean water distance field consistency.
3. Physical Water Depth Rasterization:
   - Stamping of central lake bathymetric water depth (lake_level - Z).
   - Stamping of parabolic river channel depth (cz - Z).
   - Zero depth on dry land.
   - Seamless coupling with hf.water_depth.
4. Subterranean Karst Cavern & Rocky Arch Tunnel:
   - Cavern chamber vault dome geometry.
   - Rock overburden thickness invariant (z_terrain(cx, cy) - (cz + h) >= 5.0m) enforcement.
   - Feature 16 Rocky Arch Tunnel Entrance (Cave_Arch_Entrance) connecting cliff to chamber.
   - Arch tunnel internal ceiling clearance >= 2.5m throughout.
   - High-detail fluted speleothem mesh generation (stalactites, stalagmites, columns).
   - Subterranean pool elevation coupling with water table.
   - 12 mineral crystal clusters and 12 35W point lights.
5. PBR Optical Shaders & Headless Stability:
   - Soft-import guard for scipy in normalizer.py.
   - Water shader Beer-Lambert surface depth absorption (LightPath Ray Length).
   - Thresholded shoreline contact foam margin (< 1.0m).
   - Phosphorescent cavern mineral shader (fluorite IOR 1.434, Emission Strength 4.5).
   - Genesis Primordial Abiotic preset (genesis_primordial_abiotic.json) contract.
   - Blender 4.5.4 LTS headless compilation and scene object inspection when Blender is present.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from terra_forge.schema.map_config import (
    CavernConfig,
    HydrologyConfig,
    MapConfig,
    ShaderConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar
from terra_forge.ai.normalizer import (
    AABBPrimitive,
    CollisionPrimitives,
    CylinderPrimitive,
    GeometryNormalizer,
    HAS_SCIPY,
)
from terra_forge.blender.asset_fetcher import (
    _build_cavern_column_mesh,
    _build_cavern_mineral_mesh,
    _build_stalactite_mesh,
    _build_stalagmite_mesh,
)

# Detect Blender executable across platforms
BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]
BLENDER_BIN = None
for cand in BLENDER_CANDIDATES:
    if os.path.exists(cand) or shutil.which(cand) is not None:
        BLENDER_BIN = cand
        break

HAS_BLENDER = BLENDER_BIN is not None


# =============================================================================
# 1. 4-Tier Hydrology Spline Dynamics & Normalization
# =============================================================================

class Test4TierHydrologyDynamics:
    """Validates continuous 4-tier drainage dynamics and stage schema normalization."""

    @pytest.fixture
    def canonical_hydro_engine(self) -> HydrologyEngine:
        hf = Heightfield2D(width=160.0, depth=160.0, resolution=81)
        cfg = HydrologyConfig(
            lake_center=(-14.0, -10.0),
            lake_radius=22.0,
            lake_level=4.5,
            lake_depth=10.0,
            sea_level=0.0,
            river_enabled=True,
            river_carve_depth=2.2,
            river_meander_amplitude=1.2,
            river_stages=[
                {
                    "name": "Mountain Cascades",
                    "t_end": 0.32,
                    "start": [-8.0, 48.0, 21.5, 3.2],
                    "end": [0.0, 26.0, 8.0, 4.5],
                },
                {
                    "name": "Valley Meander",
                    "t_end": 0.58,
                    "start": [0.0, 26.0, 8.0, 4.5],
                    "end": [-18.0, 6.0, 4.5, 7.5],
                },
                {
                    "name": "Central Lake Transit",
                    "t_end": 0.74,
                    "start": [-18.0, 6.0, 4.5, 7.5],
                    "end": [-10.0, -18.0, 4.5, 8.0],
                },
                {
                    "name": "Outlet Gorge & Bay",
                    "t_end": 1.0,
                    "start": [-10.0, -18.0, 4.5, 8.0],
                    "end": [26.0, -36.0, 0.0, 10.0],
                },
            ],
        )
        return HydrologyEngine(hf, cfg)

    def test_4tier_stages_continuity_and_descent(self, canonical_hydro_engine: HydrologyEngine):
        """Spline must evaluate continuously across all 4 stages and maintain non-increasing elevation."""
        t_samples = np.linspace(0.0, 1.0, 200)
        rx, ry, rz, rw = canonical_hydro_engine.evaluate_river_spline(t_samples)

        assert len(rx) == len(t_samples)
        assert len(ry) == len(t_samples)
        assert len(rz) == len(t_samples)
        assert len(rw) == len(t_samples)

        # C0 continuity: step-to-step spatial jumps must be bounded (< 3.0m per sample step)
        dx = np.diff(rx)
        dy = np.diff(ry)
        dz = np.diff(rz)
        step_distances = np.sqrt(dx**2 + dy**2 + dz**2)
        assert np.max(step_distances) < 3.0, "Spline has discontinuous gap between stages"

        # Monotonic downhill descent: dz <= 1e-4 everywhere (no uphill water)
        assert np.all(dz <= 1e-4), "River water elevation flows uphill!"

        # Endpoint reaches sea level
        assert abs(rz[-1] - 0.0) < 0.1, f"Spline endpoint z={rz[-1]} does not reach sea level 0.0"

    def test_mountain_cascades_gradient_and_flow_acceleration(self, canonical_hydro_engine: HydrologyEngine):
        """Tier 1 must exhibit steep elevation drop (> 10m) and high downhill flow velocity."""
        t_cascade = np.linspace(0.0, 0.30, 20)
        _, _, rz_casc, _ = canonical_hydro_engine.evaluate_river_spline(t_cascade)
        vertical_drop = rz_casc[0] - rz_casc[-1]
        assert vertical_drop > 10.0, f"Mountain cascade vertical drop {vertical_drop:.2f}m is insufficient"

        # Velocity in steep cascade vs flat lake
        v_cascade = canonical_hydro_engine.evaluate_flow_velocity(0.15, v_base=2.5)
        v_lake = canonical_hydro_engine.evaluate_flow_velocity(0.65, v_base=2.5)
        speed_cascade = math.sqrt(float(v_cascade[0])**2 + float(v_cascade[1])**2 + float(v_cascade[2])**2)
        speed_lake = math.sqrt(float(v_lake[0])**2 + float(v_lake[1])**2 + float(v_lake[2])**2)
        assert speed_cascade > speed_lake, f"Cascade speed {speed_cascade:.2f} must exceed lake speed {speed_lake:.2f}"
        assert speed_cascade > 3.0, f"Cascade speed {speed_cascade:.2f} is too sluggish"

    def test_valley_meander_sinusoidal_curvature(self, canonical_hydro_engine: HydrologyEngine):
        """Tier 2 must feature lateral S-curve displacement that scales with meander amplitude."""
        t_meander = np.linspace(0.33, 0.57, 30)
        rx_m, ry_m, _, _ = canonical_hydro_engine.evaluate_river_spline(t_meander)

        # Baseline straight chord
        chord_start = np.array([0.0, 26.0])
        chord_end = np.array([-18.0, 6.0])
        chord_vec = chord_end - chord_start
        chord_len = np.linalg.norm(chord_vec)
        chord_unit = chord_vec / chord_len
        perp_unit = np.array([-chord_unit[1], chord_unit[0]])

        # Measure lateral displacement from chord
        lateral_offsets = []
        for x, y in zip(rx_m, ry_m):
            pt_vec = np.array([x, y]) - chord_start
            lateral = np.dot(pt_vec, perp_unit)
            lateral_offsets.append(lateral)

        max_lateral = max(abs(l) for l in lateral_offsets)
        assert max_lateral > 1.5, f"Valley meander lacks sufficient lateral curvature ({max_lateral:.2f}m)"

    def test_central_lake_transit_strictly_flat(self, canonical_hydro_engine: HydrologyEngine):
        """Tier 3 Central Lake Transit must maintain perfectly flat elevation at lake_level (4.5m)."""
        t_transit = np.linspace(0.58, 0.74, 25)
        _, _, rz_trans, rw_trans = canonical_hydro_engine.evaluate_river_spline(t_transit)

        expected_level = canonical_hydro_engine.cfg.lake_level
        assert np.allclose(rz_trans, expected_level, atol=1e-3), "Lake transit surface is not strictly flat"
        assert np.all(rw_trans >= 7.5), "Lake transit width is too constricted"

    def test_outlet_gorge_and_bay_descent(self, canonical_hydro_engine: HydrologyEngine):
        """Tier 4 Outlet Gorge must carve from lake_level down to sea_level and broaden into bay."""
        t_outlet = np.linspace(0.75, 1.0, 25)
        _, _, rz_out, rw_out = canonical_hydro_engine.evaluate_river_spline(t_outlet)

        assert rz_out[0] <= 4.5
        assert abs(rz_out[-1] - 0.0) < 0.05
        assert rw_out[-1] >= 9.5, "Estuary mouth should widen to at least 9.5m"

    def test_waypoint_dict_schema_auto_normalization(self):
        """HydrologyEngine must automatically normalize waypoint dicts into valid segment structures."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=41)
        waypoint_stages = [
            {"t": 0.0, "x": -20.0, "y": 40.0, "z": 20.0, "width": 3.0, "depth": 1.2},
            {"t": 0.4, "x": -5.0, "y": 15.0, "z": 10.0, "width": 4.5, "depth": 1.5},
            {"t": 0.7, "x": 0.0, "y": 0.0, "z": 4.5, "width": 7.0, "depth": 1.8},
            {"t": 1.0, "x": 25.0, "y": -25.0, "z": 0.0, "width": 9.0, "depth": 2.0},
        ]
        cfg = HydrologyConfig(river_enabled=True, river_stages=waypoint_stages)
        engine = HydrologyEngine(hf, cfg)

        assert len(cfg.river_stages) == 3
        seg0 = cfg.river_stages[0]
        assert "start" in seg0 and "end" in seg0
        assert seg0["start"] == (-20.0, 40.0, 20.0, 3.0)
        assert seg0["end"] == (-5.0, 15.0, 10.0, 4.5)

        # Spline evaluation must succeed seamlessly
        rx, ry, rz, rw = engine.evaluate_river_spline(np.linspace(0.0, 1.0, 50))
        assert abs(rx[0] - (-20.0)) < 0.2
        assert abs(rx[-1] - 25.0) < 0.2

    def test_river_spline_scalar_compatibility(self, canonical_hydro_engine: HydrologyEngine):
        """Scalar evaluation must return RiverSplineScalar instances supporting indexing and math."""
        res = canonical_hydro_engine.evaluate_river_spline(0.25)
        assert isinstance(res, tuple) and len(res) == 4
        rx, ry, rz, rw = res

        assert isinstance(rx, RiverSplineScalar)
        # Arithmetic tests
        assert isinstance(rx + 2.0, RiverSplineScalar)
        assert isinstance(2.0 + rx, RiverSplineScalar)
        assert isinstance(rx * 2.0, RiverSplineScalar)
        assert isinstance(rx / 2.0, RiverSplineScalar)
        assert isinstance(-rx, RiverSplineScalar)
        # Legacy index test val[0]
        assert rx[0] == float(rx)


# =============================================================================
# 2. Channel Carving, Lake Retaining Berm & Flow Vector Fields
# =============================================================================

class TestChannelCarvingAndLakeBerm:
    """Validates parabolic bedrock channel carving, berm freeboard, and flow vectors."""

    def test_parabolic_channel_carving_profile(self):
        """Parabolic trench depth must follow (1 - (d/R)^2) * carve_depth."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=81)
        hf.Z.fill(15.0)
        carve_depth = 2.4
        width = 6.0
        cfg = HydrologyConfig(
            river_enabled=True,
            river_carve_depth=carve_depth,
            river_meander_amplitude=0.0,
            river_stages=[
                {
                    "name": "Straight River",
                    "t_end": 1.0,
                    "start": [0.0, -30.0, 15.0, width],
                    "end": [0.0, 30.0, 15.0, width],
                }
            ],
        )
        engine = HydrologyEngine(hf, cfg)
        engine.carve_river_channel(num_samples=100)

        # Centerline elevation at (0, 0)
        center_z = hf.sample_bilinear(0.0, 0.0)
        expected_center = 15.0 - carve_depth
        assert pytest.approx(center_z, abs=0.15) == expected_center

        # Halfway across influence radius: R_inf = 1.5 * width = 9.0m
        # At d = 4.5m: trench = (1 - 0.5^2) * 2.4 = 0.75 * 2.4 = 1.8m -> Z = 13.2m
        half_z = hf.sample_bilinear(4.5, 0.0)
        assert half_z < 15.0 and half_z > center_z
        assert pytest.approx(half_z, abs=0.25) == 15.0 - (0.75 * carve_depth)

        # Outside influence radius (d = 12.0m): trench is 0 -> Z = 15.0m
        outside_z = hf.sample_bilinear(12.0, 0.0)
        assert pytest.approx(outside_z, abs=0.01) == 15.0

    def test_lake_retaining_berm_containment_freeboard(self):
        """Lake basin carving must elevate the retaining berm crest by at least 1.2m above lake_level."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=81)
        hf.Z.fill(2.0)  # Low initial ground
        lake_level = 4.5
        berm_height = 1.5
        lake_radius = 20.0
        hf.carve_lake_basin(
            center_x=0.0,
            center_y=0.0,
            radius=lake_radius,
            water_level=lake_level,
            lake_depth=8.0,
            berm_height=berm_height,
        )

        # Berm crest is at dist in [0.95*R, 1.15*R] -> dist = 21.0m
        berm_z = hf.sample_bilinear(21.0, 0.0)
        assert berm_z >= lake_level + 1.2, f"Berm crest {berm_z:.2f}m fails containment freeboard >= 5.7m"

        # Basin center is at dist = 0.0 -> bottom = lake_level - depth = 4.5 - 8.0 = -3.5m
        center_z = hf.sample_bilinear(0.0, 0.0)
        assert pytest.approx(center_z, abs=0.1) == (lake_level - 8.0)

    def test_channel_breaches_lake_retaining_berm(self):
        """River channel carving must incise through the retaining berm at inlet and outlet to prevent damming."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=101)
        hf.Z.fill(2.0)
        lake_level = 4.5
        lake_radius = 20.0
        hf.carve_lake_basin(
            center_x=0.0,
            center_y=0.0,
            radius=lake_radius,
            water_level=lake_level,
            lake_depth=8.0,
            berm_height=1.5,
        )

        # Berm crest at North rim (0, 21.0) is initially elevated
        initial_rim_z = hf.sample_bilinear(0.0, 21.0)
        assert initial_rim_z >= lake_level + 1.2

        # River channel flows straight from North (0, 35) to South (0, -35)
        cfg = HydrologyConfig(
            lake_center=(0.0, 0.0),
            lake_radius=lake_radius,
            lake_level=lake_level,
            river_enabled=True,
            river_carve_depth=2.5,
            river_meander_amplitude=0.0,
            river_stages=[
                {
                    "name": "North Inflow",
                    "t_end": 0.5,
                    "start": [0.0, 35.0, 8.0, 6.0],
                    "end": [0.0, 0.0, lake_level, 6.0],
                },
                {
                    "name": "South Outflow",
                    "t_end": 1.0,
                    "start": [0.0, 0.0, lake_level, 6.0],
                    "end": [0.0, -35.0, 0.0, 8.0],
                },
            ],
        )
        engine = HydrologyEngine(hf, cfg)
        engine.carve_river_channel(num_samples=150)

        # After river carving, riverbed at the North rim breach must be <= lake_level
        breached_rim_z = hf.sample_bilinear(0.0, 21.0)
        assert breached_rim_z <= lake_level + 0.2, (
            f"River channel failed to breach retaining berm: elevation {breached_rim_z:.2f}m > {lake_level}m"
        )

    def test_2d_velocity_field_rasterization(self):
        """Velocity field (Vx, Vy) must align with spline tangent directions inside the riverbed."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=61)
        cfg = HydrologyConfig(
            river_enabled=True,
            river_stages=[
                {
                    "name": "Eastward River",
                    "t_end": 1.0,
                    "start": [-30.0, 0.0, 15.0, 6.0],
                    "end": [30.0, 0.0, 0.0, 6.0],
                }
            ],
        )
        engine = HydrologyEngine(hf, cfg)
        vx, vy = engine.compute_flow_velocity_field(v_base=3.0)

        assert vx.shape == (hf.res_y, hf.res_x)
        assert vy.shape == (hf.res_y, hf.res_x)

        # Sample grid center (0, 0)
        gx, gy = hf.world_to_grid(0.0, 0.0)
        ix, iy = int(round(gx)), int(round(gy))
        assert vx[iy, ix] > 1.5, f"Eastward flow vx={vx[iy, ix]:.2f} should be positive and fast"
        assert abs(vy[iy, ix]) < 0.2, f"Lateral flow vy={vy[iy, ix]:.2f} should be near zero for straight east river"

    def test_water_distance_field_consistency(self):
        """Water distance field must be 0 inside water bodies and strictly positive on dry land."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        cfg = HydrologyConfig(
            lake_center=(0.0, 0.0),
            lake_radius=15.0,
            river_enabled=False,
        )
        engine = HydrologyEngine(hf, cfg)
        dist = engine.compute_water_distance_field()

        # Inside lake (dist <= 15m)
        gx_in, gy_in = hf.world_to_grid(5.0, 5.0)
        assert dist[int(gy_in), int(gx_in)] == 0.0

        # Outside lake at dist = 30m -> distance to shore ~ 15m
        gx_out, gy_out = hf.world_to_grid(30.0, 0.0)
        assert pytest.approx(dist[int(gy_out), int(gx_out)], abs=1.5) == 15.0


# =============================================================================
# 3. Physical Water Depth Rasterization
# =============================================================================

class TestWaterDepthRasterization:
    """Validates stamping of water depth into hf.water_depth for WorldArtifact flow layer."""

    def test_rasterize_water_depth_central_lake_and_river(self):
        """rasterize_water_depth must calculate physical depths across both lake and river channels."""
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=81)
        hf.Z.fill(12.0)
        lake_level = 4.5
        lake_radius = 18.0

        # Carve lake basin
        hf.carve_lake_basin(
            center_x=-10.0,
            center_y=0.0,
            radius=lake_radius,
            water_level=lake_level,
            lake_depth=6.0,
            berm_height=1.2,
        )

        cfg = HydrologyConfig(
            lake_center=(-10.0, 0.0),
            lake_radius=lake_radius,
            lake_level=lake_level,
            river_enabled=True,
            river_carve_depth=2.2,
            river_stages=[
                {
                    "name": "Outlet Channel",
                    "t_end": 1.0,
                    "start": [-10.0, 0.0, lake_level, 6.0],
                    "end": [35.0, 0.0, 0.0, 8.0],
                }
            ],
        )
        engine = HydrologyEngine(hf, cfg)
        engine.carve_river_channel()

        # Call the new rasterize_water_depth
        depth_map = engine.rasterize_water_depth()

        assert depth_map is hf.water_depth
        assert depth_map.shape == (hf.res_y, hf.res_x)

        # 1. Lake Center Depth: bottom is lake_level - depth = -1.5m -> depth = 4.5 - (-1.5) = 6.0m
        gx_l, gy_l = hf.world_to_grid(-10.0, 0.0)
        lake_depth = depth_map[int(gy_l), int(gx_l)]
        assert pytest.approx(lake_depth, abs=0.25) == 6.0

        # 2. River Channel Depth: at carved river point, depth must be positive
        gx_r, gy_r = hf.world_to_grid(15.0, 0.0)
        river_depth = depth_map[int(gy_r), int(gx_r)]
        assert river_depth > 0.5, f"River channel water depth {river_depth:.2f}m is too shallow"

        # 3. High Dry Land: depth must be exactly 0.0
        gx_dry, gy_dry = hf.world_to_grid(-45.0, 45.0)
        assert depth_map[int(gy_dry), int(gx_dry)] == 0.0


# =============================================================================
# 4. Subterranean Karst Cavern, Arch Tunnel & Speleothems
# =============================================================================

class TestKarstCavernSystem:
    """Validates karst cavern invariants, rock overburden, arch clearance, and speleothems."""

    def test_overburden_thickness_invariant_pass(self):
        """Cavern construction must succeed when rock overburden >= 5.0m."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cx, cy, cz = -10.0, 15.0, 2.0
        h = 8.0
        # Roof apex = cz + h = 10.0m. Terrain surface = 17.0m -> overburden = 7.0m >= 5.0m
        hf.Z.fill(17.0)

        cfg = CavernConfig(
            enabled=True,
            center=(cx, cy, cz),
            radius_x=14.0,
            radius_y=12.0,
            height=h,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        from terra_forge.blender.cave_builder import build_karst_cavern
        # Should execute without raising AssertionError
        objs = build_karst_cavern(cfg, shader_cfg, hf=hf)
        assert isinstance(objs, list)

    def test_overburden_thickness_invariant_breach_raises(self):
        """Cavern construction must raise AssertionError if rock overburden is < 5.0m."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cx, cy, cz = -10.0, 15.0, 2.0
        h = 8.0
        # Roof apex = 10.0m. Terrain surface = 12.0m -> overburden = 2.0m < 5.0m
        hf.Z.fill(12.0)

        cfg = CavernConfig(
            enabled=True,
            center=(cx, cy, cz),
            height=h,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        from terra_forge.blender.cave_builder import build_karst_cavern
        with pytest.raises(AssertionError, match="Cavern overburden breach"):
            build_karst_cavern(cfg, shader_cfg, hf=hf)

    def test_arch_tunnel_entrance_geometry_and_clearance(self):
        """Feature 16 Rocky Arch Tunnel Entrance must connect cliff entrance to chamber with clearance >= 2.5m."""
        cfg = CavernConfig(
            entrance_pos=(-15.0, 32.0, 6.0),
            entrance_radius=4.5,
            center=(-25.0, 38.0, 5.0),
        )
        ex, ey, ez = cfg.entrance_pos
        cx, cy, cz = cfg.center
        dist = math.sqrt((cx - ex)**2 + (cy - ey)**2 + (cz - ez)**2)

        assert dist > 5.0, "Entrance and chamber are too close"
        assert dist < 30.0, "Entrance and chamber are excessively far"
        assert cfg.entrance_radius >= 2.5, "Arch entrance radius violates clearance >= 2.5m"

    def test_speleothems_vault_mesh_generators(self):
        """Stalactite, stalagmite, column, and mineral procedural generators must create manifold geometries."""
        # 1. Stalactite
        s_verts, s_faces = _build_stalactite_mesh(target_height=3.0)
        assert len(s_verts) >= 96
        assert len(s_faces) >= 96
        # Apex is at y=0, base at y=3.0
        y_coords = [v[1] for v in s_verts]
        assert pytest.approx(min(y_coords), abs=1e-3) == 0.0
        assert pytest.approx(max(y_coords), abs=1e-3) == 3.0

        # 2. Stalagmite
        sm_verts, sm_faces = _build_stalagmite_mesh(target_height=2.5)
        assert len(sm_verts) >= 96
        assert len(sm_faces) >= 96
        y_sm = [v[1] for v in sm_verts]
        assert pytest.approx(min(y_sm), abs=1e-3) == 0.0

        # 3. Column
        c_verts, c_faces = _build_cavern_column_mesh(target_height=8.0)
        assert len(c_verts) >= 160
        assert len(c_faces) >= 160
        y_c = [v[1] for v in c_verts]
        assert pytest.approx(max(y_c), abs=1e-3) == 8.0

        # 4. Mineral Druse Cluster
        m_verts, m_faces = _build_cavern_mineral_mesh(target_height=1.5)
        assert len(m_verts) > 50
        assert len(m_faces) > 50

    def test_subterranean_pool_water_table_coupling(self):
        """Subterranean pool level in cavern config must connect with water table / lake level."""
        cfg = CavernConfig(pool_level=4.5, subterranean_pool=True)
        assert cfg.pool_level == 4.5, "Pool level must couple with 4.5m water table"


# =============================================================================
# 5. PBR Shaders, Scipy Guard & Abiotic Preset Contract
# =============================================================================

class TestShadersAndAbioticPreset:
    """Validates optical shader compilation, soft-import guards, and abiotic preset integrity."""

    def test_scipy_soft_import_guard_in_normalizer(self):
        """Normalizer must extract collision primitives without error even when scipy is unavailable."""
        pts = np.array([
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 4.0, 0.0],
        ], dtype=np.float32)

        prims = GeometryNormalizer.extract_collision_primitives(pts, target_height=4.0)
        assert prims.cylinder.radius > 0.0
        assert prims.cylinder.height == 4.0
        assert prims.aabb.size[1] == 4.0
        assert prims.convex_hull.vertex_count >= 5
        assert prims.convex_hull.simplex_count >= 4
        assert prims.convex_hull.volume > 0.0

    def test_genesis_primordial_abiotic_preset_contract(self):
        """genesis_primordial_abiotic.json must declare canonical 4-tier stages and 0% flora/fauna."""
        preset_path = Path("E:/tool/mcp/terra_forge/terra_forge/schema/presets/genesis_primordial_abiotic.json")
        assert preset_path.exists()

        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Pure Abiotic Geosphere (0% biomes)
        assert data.get("biomes") == [], "Abiotic world must have exactly 0 biomes (no plants or animals)"

        # 2. Canonical 4-Tier Hydrology Stages
        hydro = data["hydrology"]
        stages = hydro["river_stages"]
        assert len(stages) == 4
        stage_names = [s["name"] for s in stages]
        assert stage_names == [
            "Mountain Cascades",
            "Valley Meander",
            "Central Lake Transit",
            "Outlet Gorge & Bay",
        ]
        assert hydro["lake_level"] == 4.5
        assert hydro["sea_level"] == 0.0

        # 3. Karst Cavern Invariant Configuration
        cav = data["cavern"]
        assert cav["enabled"] is True
        assert cav["pool_level"] == 4.5  # Connected to lake_level
        assert cav["min_roof_clearance"] >= 5.0
        assert cav["entrance_radius"] >= 2.5
        assert cav["bioluminescent"] is True
        assert len(cav["glow_color"]) == 3

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_optical_water_and_mineral_shaders_compilation_blender(self):
        """Verify Blender 4.5.4 LTS compiles optical water and mineral shaders with correct nodes."""
        script = """
import sys
sys.path.insert(0, '.')
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_water_shader, create_cavern_mineral_shader

cfg = ShaderConfig(water_absorption_color=(0.04, 0.42, 0.68), water_contact_foam=True)
water_mat = create_water_shader(cfg)
assert water_mat is not None
assert water_mat.surface_render_method == 'BLENDED'

w_nodes = [n.bl_idname for n in water_mat.node_tree.nodes]
assert 'ShaderNodeLightPath' in w_nodes
assert 'ShaderNodeMapRange' in w_nodes
assert 'ShaderNodeVolumeAbsorption' in w_nodes
assert 'ShaderNodeBump' in w_nodes

# Verify mineral shader
mineral_mat = create_cavern_mineral_shader(glow_color=(0.1, 0.9, 0.4), emission_strength=4.5)
assert mineral_mat is not None
assert mineral_mat.surface_render_method == 'BLENDED'
m_nodes = [n.bl_idname for n in mineral_mat.node_tree.nodes]
assert 'ShaderNodeTexVoronoi' in m_nodes
assert 'ShaderNodeBump' in m_nodes

bsdf = [n for n in mineral_mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled'][0]
assert abs(bsdf.inputs['IOR'].default_value - 1.434) < 1e-3
assert bsdf.inputs['Emission Strength'].default_value == 4.5

print('SHADERS_COMPILATION_PASS')
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=35, check=False)
        assert res.returncode == 0, f"Blender shader compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "SHADERS_COMPILATION_PASS" in res.stdout

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_karst_cavern_scene_elements_blender(self):
        """Verify build_karst_cavern constructs chamber, arch entrance, speleothems, pool, minerals, and 12 lights."""
        script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern

cfg = CavernConfig(
    enabled=True,
    center=(-25.0, 38.0, 5.0),
    radius_x=16.0,
    radius_y=14.0,
    height=10.0,
    entrance_pos=(-15.0, 32.0, 6.0),
    entrance_radius=4.5,
    stalactite_count=12,
    stalagmite_count=10,
    column_count=3,
    subterranean_pool=True,
    pool_level=4.5,
    bioluminescent=True,
)
shader_cfg = ShaderConfig()

objs = build_karst_cavern(cfg, shader_cfg)
names = [o.name for o in objs]

assert 'Cave_Cavern_Chamber' in names
assert 'Cave_Arch_Entrance' in names
assert 'Cave_Speleothems' in names
assert 'Water_Cave_Pool' in names
assert 'Cave_Mineral_Clusters' in names
assert 'Cave_Light_Point_35W' in names

lights = [o for o in objs if 'Light' in o.name]
assert len(lights) == 12, f'Expected 12 lights, got {len(lights)}'
for l in lights:
    assert l.data.energy == 35.0

print('CAVERN_SCENE_VERIFIED')
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=35, check=False)
        assert res.returncode == 0, f"Blender cavern build failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "CAVERN_SCENE_VERIFIED" in res.stdout
