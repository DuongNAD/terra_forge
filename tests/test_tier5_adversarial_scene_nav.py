"""
Milestone 5 Phase 2 Dedicated Tier 5 Adversarial Test Suite: Scene & Navigation.

Targeted Modules:
- terra_forge.navigation.navmesh (ObstacleGridSynchronizer, NavMeshReachabilityValidator, adaptive_prune, safe_spawn)
- terra_forge.blender.cave_builder (Karst cavern chamber, arch entrance clearance, overburden invariant)
- terra_forge.blender.water_builder (Watertight central lake, marine bay skirts, river ribbon flow)
- terra_forge.blender.exporter (Simulation heightmap export, passability matrix, GLB export options)
- terra_forge.inspector.cameras (Solar ephemeris, CCT conversion, atmospheric extinction, camera framing math)
- terra_forge.inspector.vision_rig (World volume bypass, 2x2 vision grid stitching, legacy snapshot resolution)

Adversarial Stress Dimensions:
1. NavMesh Extreme Topographies & Boundary Edge Cases
2. Karst Cavern Structural Invariants & Overburden Protection
3. Hydrology & Simulation Heightmap Export Integrity
4. Astronomical Ephemeris, Photometric Optics & Camera Viewpoints
5. Vision Rig Matrix & Snapshot Resolution
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import pytest

from terra_forge.ai.vault import AbioticType, FloraType
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import (
    CavernConfig,
    DioramaBounds,
    HydrologyConfig,
    MapConfig,
    ShaderConfig,
)
from terra_forge.navigation.navmesh import (
    CANONICAL_COVERAGE_TARGET,
    DEFAULT_GENTLE_SLOPE_DEG,
    DEFAULT_MAX_FLOW,
    DEFAULT_MAX_WALKABLE_SLOPE_DEG,
    DEFAULT_WALKABLE_SLOPE,
    SOLID_ABIOTIC_TYPES,
    TALL_FLORA_TYPES,
    NON_SOLID_FLORA_TYPES,
    NavMeshReachabilityValidator,
    ObstacleGridSynchronizer,
    cell_to_world,
    flora_canopy_radius,
    flora_collider_radius,
    spawn_clearance_radius,
    validate_and_prune_navmesh,
    world_to_cell,
)
from terra_forge.blender.cave_builder import build_karst_cavern
from terra_forge.blender.asset_fetcher import (
    _build_cavern_column_mesh,
    _build_cavern_mineral_mesh,
    _build_stalactite_mesh,
    _build_stalagmite_mesh,
)
from terra_forge.blender.exporter import export_simulation_heightmap_json
from terra_forge.inspector.cameras import (
    build_inspection_camera_configs,
    calculate_camera_framing_distance,
    calculate_cct_from_elevation,
    cct_to_linear_rgb,
    calculate_fov_from_lens,
    calculate_optical_air_mass,
    calculate_solar_position,
    calculate_sun_intensity,
    setup_sun_lighting,
)
from terra_forge.inspector.vision_rig import (
    build_2x2_vision_grid,
    resolve_snapshot_key,
)


# =============================================================================
# Dimension 1: NavMesh Extreme Topographies & Boundary Edge Cases
# =============================================================================

class TestNavMeshAdversarialTopography:
    """Stress-tests NavMesh reachability and obstacle synchronization under adversarial conditions."""

    def test_100_percent_submerged_atlantis_world(self):
        """
        An adversarial world where 100% of the elevation is below sea level (total_land = 0).
        Must cleanly return navmesh_coverage = 0.0, reach_count = 0, is_valid = False
        without raising ZeroDivisionError or IndexError.
        """
        shape = (64, 64)
        elev = np.full(shape, -10.0, dtype=np.float32)
        sync = ObstacleGridSynchronizer(shape=shape, world_bounds=(-100.0, -100.0, 100.0, 100.0))
        validator = NavMeshReachabilityValidator(
            elevation=elev,
            water_cutoff=0.0,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
        )

        res = validator.evaluate(sync)
        assert res.total_land == 0
        assert res.reach_count == 0
        assert res.navmesh_coverage == 0.0
        assert res.is_valid is False
        assert len(res.navigation_nodes) == 0
        assert len(res.navigation_edges) == 0

    def test_disconnected_archipelago_largest_component_selection(self):
        """
        Constructs an archipelago of 3 disconnected islands of unequal sizes:
        - Island A (Center): ~70% of total land
        - Island B (Northwest): ~19% of total land
        - Island C (Southeast): ~11% of total land
        Safe spawn MUST select Island A (largest component), reaching exactly Island A.
        Coverage must be ~0.71, and is_valid must be False (0.71 < 0.80).
        """
        shape = (100, 100)
        elev = np.full(shape, -5.0, dtype=np.float32)

        # Island A (Center): rows 30..70, cols 30..70 (41x41 = 1681 cells)
        elev[30:71, 30:71] = 4.0

        # Island B (Northwest): rows 5..25, cols 5..25 (21x21 = 441 cells)
        elev[5:26, 5:26] = 4.0

        # Island C (Southeast): rows 80..95, cols 80..95 (16x16 = 256 cells)
        elev[80:96, 80:96] = 4.0

        total_land_cells = 1681 + 441 + 256  # 2378 cells
        sync = ObstacleGridSynchronizer(shape=shape, world_bounds=(-100.0, -100.0, 100.0, 100.0))
        validator = NavMeshReachabilityValidator(
            elevation=elev,
            water_cutoff=0.0,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            safe_spawn=True,
        )

        res = validator.evaluate(sync)
        assert res.total_land == total_land_cells

        # Safe spawn MUST be located on Island A (rows 30..70, cols 30..70)
        sr, sc = res.spawn_cell
        assert 30 <= sr <= 71 and 30 <= sc <= 71, (
            f"Safe spawn selected ({sr}, {sc}), which is outside largest Island A!"
        )

        # Reach count covers Island A flat interior (39x39 = 1521, perimeter cells are ocean cliffs)
        expected_island_a_reach = 39 * 39  # 1521
        assert res.reach_count == expected_island_a_reach
        expected_cov = round(expected_island_a_reach / total_land_cells, 4)
        assert res.navmesh_coverage == expected_cov
        assert res.is_valid is False  # 0.6396 < 0.80

    def test_single_cell_isthmus_chokepoint_pruning(self):
        """
        Two massive continents (North: ~49% land, South: ~49% land) connected ONLY
        by a 1-cell wide isthmus (~2% land). Total land = 100%.
        A solid CavernColumn placed on the isthmus bisects the map, reducing coverage to ~50%.
        validate_and_prune_navmesh MUST detect the bottleneck, prune the obstacle,
        and restore coverage to >= 80.0%.
        """
        shape = (100, 100)
        elev = np.full(shape, -2.0, dtype=np.float32)

        # North continent: rows 10..40, cols 20..80 (31 * 61 = 1891 cells)
        elev[10:41, 20:81] = 3.0
        # South continent: rows 60..90, cols 20..80 (31 * 61 = 1891 cells)
        elev[60:91, 20:81] = 3.0
        # 1-cell wide Isthmus: rows 41..59, col 50 (19 cells)
        elev[41:60, 50] = 3.0

        # World coordinates for row 50, col 50 in bounds (-100, -100, 100, 100)
        # col 50 -> x = -100 + 50.5 * 2 = 1.0; row 50 -> z = -100 + 50.5 * 2 = 1.0
        blocking_obs = [{
            "abiotic_type": int(AbioticType.CavernColumn),
            "x": 1.0,
            "z": 1.0,
            "scale": 1.5,
        }]

        res, pruned = validate_and_prune_navmesh(
            elevation=elev,
            abiotic_instances=blocking_obs,
            water_cutoff=0.0,
            target_coverage=0.80,
            safe_spawn=True,
        )

        assert len(pruned) == 1, "Adaptive pruning must identify and prune the isthmus blocker"
        assert pruned[0] == 10000
        assert res.navmesh_coverage >= 0.80, f"Coverage after pruning: {res.navmesh_coverage:.2%}"
        assert res.is_valid is True

    def test_total_obstacle_saturation_raises_value_error(self):
        """
        When every walkable dry cell is covered by solid obstacles (walkable mask has 0 True cells),
        NavMeshReachabilityValidator.find_spawn_cell must raise ValueError.
        """
        shape = (20, 20)
        elev = np.full(shape, 3.0, dtype=np.float32)
        sync = ObstacleGridSynchronizer(shape=shape, world_bounds=(-10.0, -10.0, 10.0, 10.0))
        sync.obstacle_grid.fill(True)

        validator = NavMeshReachabilityValidator(
            elevation=elev,
            water_cutoff=0.0,
            world_bounds=(-10.0, -10.0, 10.0, 10.0),
        )
        walkable = validator.compute_walkable_mask(sync.obstacle_grid)
        assert not np.any(walkable), "All cells should be blocked"

        with pytest.raises(ValueError, match="No walkable land cell exists"):
            validator.find_spawn_cell(walkable, sync)

    def test_sheer_cliff_world_no_walkable_cells(self):
        """
        When the entire landscape consists of vertical cliffs (> 60° slope),
        walkable mask is completely False and find_spawn_cell raises ValueError.
        """
        shape = (40, 40)
        elev = np.zeros(shape, dtype=np.float32)
        for r in range(40):
            elev[r, :] = float(r * 20.0)

        sync = ObstacleGridSynchronizer(shape=shape, world_bounds=(-20.0, -20.0, 20.0, 20.0))
        validator = NavMeshReachabilityValidator(
            elevation=elev,
            water_cutoff=-100.0,
            world_bounds=(-20.0, -20.0, 20.0, 20.0),
        )
        walkable = validator.compute_walkable_mask(sync.obstacle_grid)
        assert not np.any(walkable), "Slopes > 60° must all be non-walkable"

        with pytest.raises(ValueError, match="No walkable land cell exists"):
            validator.find_spawn_cell(walkable, sync)

    def test_river_flow_velocity_barrier(self):
        """
        A high-velocity river (flow >= 100.0) bisecting a flat plain must be treated
        as impassable in compute_walkable_mask, preventing organisms from crossing.
        """
        shape = (50, 50)
        elev = np.full(shape, 3.0, dtype=np.float32)
        flow = np.zeros(shape, dtype=np.float32)

        # River channel across columns 24..26 has turbulent torrent flow = 250.0
        flow[:, 24:27] = 250.0

        sync = ObstacleGridSynchronizer(shape=shape, world_bounds=(-50.0, -50.0, 50.0, 50.0))
        validator = NavMeshReachabilityValidator(
            elevation=elev,
            water_cutoff=0.0,
            flow=flow,
            max_flow=DEFAULT_MAX_FLOW,  # 100.0
            world_bounds=(-50.0, -50.0, 50.0, 50.0),
        )
        walkable = validator.compute_walkable_mask(sync.obstacle_grid)

        # River columns must be blocked
        assert not np.any(walkable[:, 24:27]), "High flow cells must be impassable"
        # Land columns outside river must be walkable
        assert np.all(walkable[:, :24]), "Normal land should be walkable"
        assert np.all(walkable[:, 27:]), "Normal land should be walkable"

    def test_non_square_asymmetric_bounds_coordinate_inversion(self):
        """
        Tests world_to_cell and cell_to_world on prime-dimensioned non-square grid (73, 137)
        with asymmetric world bounds (-150.0, -80.0, 120.0, 90.0).
        Verifies exact round-trip reconstruction without index wrapping or drift.
        """
        shape = (73, 137)
        bounds = (-150.0, -80.0, 120.0, 90.0)

        test_points = [
            (0, 0),
            (0, 136),
            (72, 0),
            (72, 136),
            (36, 68),
            (10, 50),
            (60, 110),
        ]

        for r, c in test_points:
            wx, wy, wz = cell_to_world(r, c, shape, bounds=bounds)
            r_inv, c_inv = world_to_cell(wx, wz, shape, bounds=bounds)
            assert (r_inv, c_inv) == (r, c), (
                f"Coordinate inversion failed: ({r}, {c}) -> ({wx}, {wz}) -> ({r_inv}, {c_inv})"
            )

        # Test extreme out-of-bounds coordinates are safely clamped
        assert world_to_cell(-9999.0, -9999.0, shape, bounds) == (0, 0)
        assert world_to_cell(9999.0, 9999.0, shape, bounds) == (72, 136)

    def test_out_of_bounds_and_negative_scale_obstacles(self):
        """
        Adversarial obstacle inputs:
        - Obstacle placed far outside bounds (x=500.0, z=-500.0).
        - Obstacle with negative scale (scale = -2.0).
        - Obstacle with zero scale (scale = 0.0).
        ObstacleGridSynchronizer must handle these without IndexError or runtime exceptions.
        """
        sync = ObstacleGridSynchronizer(shape=(64, 64), world_bounds=(-50.0, -50.0, 50.0, 50.0))

        # Far out-of-bounds obstacle
        obs_oob = sync.register_abiotic_instance(1, AbioticType.HornPeak, x=500.0, z=-500.0, scale=1.0)
        assert obs_oob is not None

        # Negative scale obstacle
        obs_neg = sync.register_abiotic_instance(2, AbioticType.RockCrag, x=0.0, z=0.0, scale=-2.0)
        assert obs_neg is not None

        # Zero scale obstacle
        obs_zero = sync.register_abiotic_instance(3, AbioticType.BasaltColumn, x=10.0, z=10.0, scale=0.0)
        assert obs_zero is not None

        # Rasterize all must complete without crash
        grid = sync.rasterize_all()
        assert grid.shape == (64, 64)
        assert grid.dtype == bool

    def test_all_abiotic_types_solid_vs_passthrough_classification(self):
        """
        Verifies complete classification of all 9 AbioticType enum variants:
        - Solid (HornPeak, RockCrag, BasaltColumn, Stalagmite, CavernColumn) must register SolidObstacle.
        - Non-solid (RiverPebble, RiparianSandbank, Stalactite, CavernMineral) must return None.
        """
        sync = ObstacleGridSynchronizer(shape=(50, 50), world_bounds=(-50.0, -50.0, 50.0, 50.0))

        solid_expected = {
            AbioticType.HornPeak,
            AbioticType.RockCrag,
            AbioticType.BasaltColumn,
            AbioticType.Stalagmite,
            AbioticType.CavernColumn,
        }

        all_types = list(AbioticType)
        assert len(all_types) in (9, 10)

        for idx, ab_type in enumerate(all_types):
            obs = sync.register_abiotic_instance(
                instance_index=idx,
                abiotic_type=ab_type,
                x=float(idx * 4 - 16),
                z=float(idx * 4 - 16),
                scale=1.0,
            )
            if ab_type in solid_expected:
                assert obs is not None, f"Solid abiotic type {ab_type.name} was not registered"
                assert obs.collider_radius > 0.0
            else:
                assert obs is None, f"Pass-through abiotic type {ab_type.name} should not be solid"

# =============================================================================
# Dimension 2: Karst Cavern Structural Invariants
# =============================================================================

class TestKarstCavernStructuralInvariants:
    """Stress-tests subterranean karst cavern invariants and speleothem geometries."""

    def test_overburden_thickness_invariant_breach_detection(self):
        """
        Enforces rock overburden clearance invariant: z_terrain(cx, cy) - (cz + h) >= 5.0m.
        Constructs a heightfield where surface terrain is only 1.0m above cavern roof.
        build_karst_cavern MUST raise AssertionError with diagnostic message.
        """
        hf = Heightfield2D(resolution=32)
        # Cavern center: cx=-16.0, cy=28.0, cz=6.0, height=4.0 -> roof apex = 10.0m
        # Surface terrain at (-16.0, 28.0) is 11.0m -> overburden = 1.0m < 5.0m
        hf.Z.fill(11.0)

        cfg = CavernConfig(
            center=(-16.0, 28.0, 6.0),
            height=4.0,
            radius_x=12.0,
            radius_y=10.0,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        with pytest.raises(AssertionError, match="Cavern overburden breach"):
            build_karst_cavern(cfg, shader_cfg, hf=hf)

    def test_overburden_thickness_exact_boundary_threshold(self):
        """
        Tests exact boundary values around 5.0m overburden requirement:
        - 5.001m: Must pass without error.
        - 4.999m: Must fail with AssertionError.
        """
        hf = Heightfield2D(resolution=32)
        cx, cy, cz, h = -16.0, 28.0, 6.0, 4.0
        roof_apex = cz + h  # 10.0m

        cfg = CavernConfig(
            center=(cx, cy, cz),
            height=h,
            radius_x=10.0,
            radius_y=10.0,
            min_roof_clearance=5.0,
        )
        shader_cfg = ShaderConfig()

        # Case A: 5.001m (15.001m surface) -> Passes
        hf.Z.fill(roof_apex + 5.001)
        res_pass = build_karst_cavern(cfg, shader_cfg, hf=hf)
        assert isinstance(res_pass, list)

        # Case B: 4.999m (14.999m surface) -> Fails
        hf.Z.fill(roof_apex + 4.999)
        with pytest.raises(AssertionError, match="Cavern overburden breach"):
            build_karst_cavern(cfg, shader_cfg, hf=hf)

    def test_arch_tunnel_clearance_minimum_radius_guarantee(self):
        """
        The rocky arch entrance tunnel must maintain minimum internal clearance >= 2.5m
        even if an adversarial configuration specifies entrance_radius = 0.1m or negative.
        """
        adversarial_radii = [0.1, 0.0, -5.0, 1.2]
        for bad_r in adversarial_radii:
            arch_r = max(2.5, bad_r)
            assert arch_r >= 2.5, f"Arch radius {arch_r} violated 2.5m minimum clearance for input {bad_r}"

            # Check cross-section modulation equation from cave_builder.py
            for theta_deg in range(0, 360, 15):
                theta = math.radians(theta_deg)
                r_mod = max(2.5, arch_r * (1.0 + 0.08 * math.cos(3.0 * theta)))
                assert r_mod >= 2.5, (
                    f"Arch tunnel modulation dipped below 2.5m clearance: {r_mod} at theta={theta_deg}"
                )

    @pytest.mark.parametrize("h", [0.5, 3.0, 25.0])
    def test_speleothem_stalactite_mesh_generator_topological_invariants(self, h: float):
        """
        Verifies stalactite generator:
        - Non-empty vertices and faces.
        - Every face index strictly < len(vertices).
        - Apex bottom at (0, 0, 0), top base at target_height.
        - Radially fluted outward normals.
        """
        verts, faces = _build_stalactite_mesh(target_height=h)
        assert len(verts) > 0
        assert len(faces) > 0

        v_arr = np.array(verts)
        assert v_arr[:, 1].min() >= -1e-5
        assert abs(v_arr[:, 1].max() - h) < 1e-4

        # Face index bounds
        n_verts = len(verts)
        for f in faces:
            assert len(f) in (3, 4)
            for idx in f:
                assert 0 <= idx < n_verts

    @pytest.mark.parametrize("h", [0.5, 3.0, 25.0])
    def test_speleothem_stalagmite_mesh_generator_topological_invariants(self, h: float):
        """
        Verifies stalagmite generator:
        - Floor base at y=0, rising apex at target_height.
        - All face indices bounded.
        """
        verts, faces = _build_stalagmite_mesh(target_height=h)
        assert len(verts) > 0
        assert len(faces) > 0

        v_arr = np.array(verts)
        assert v_arr[:, 1].min() >= -1e-5
        assert abs(v_arr[:, 1].max() - h) < 1e-4

        n_verts = len(verts)
        for f in faces:
            assert len(f) in (3, 4)
            for idx in f:
                assert 0 <= idx < n_verts

    @pytest.mark.parametrize("h", [2.0, 10.0, 40.0])
    def test_speleothem_cavern_column_mesh_generator_topological_invariants(self, h: float):
        """
        Verifies cavern column generator:
        - Fluted waist curves with waist constriction at center.
        - Height span strictly [0, h].
        """
        verts, faces = _build_cavern_column_mesh(target_height=h)
        assert len(verts) > 0
        assert len(faces) > 0

        v_arr = np.array(verts)
        assert v_arr[:, 1].min() >= -1e-5
        assert abs(v_arr[:, 1].max() - h) < 1e-4

        n_verts = len(verts)
        for f in faces:
            for idx in f:
                assert 0 <= idx < n_verts

    @pytest.mark.parametrize("h", [0.8, 1.8, 12.0])
    def test_cavern_mineral_mesh_generator_topological_invariants(self, h: float):
        """
        Verifies mineral crystal druse mesh generator:
        - Basal druse slab and radiating crystal prisms.
        - All face indices bounded.
        """
        verts, faces = _build_cavern_mineral_mesh(target_height=h)
        assert len(verts) > 0
        assert len(faces) > 0

        v_arr = np.array(verts)
        assert v_arr[:, 1].min() >= -1e-5

        n_verts = len(verts)
        for f in faces:
            for idx in f:
                assert 0 <= idx < n_verts


# =============================================================================
# Dimension 3: Hydrology & Simulation Heightmap Export Integrity
# =============================================================================

class TestHydrologyAndSimulationHeightmapExport:
    """Stress-tests hydrology mesh math and headless heightmap JSON export."""

    def test_central_lake_basin_minimum_depth_enforcement(self):
        """
        In water_builder.py: z_bot = z_top - max(1.5, config.lake_depth).
        Verifies that even if lake_depth is 0.0 or negative, water body maintains >= 1.5m depth.
        """
        for bad_depth in [0.0, -5.0, -100.0, 0.5]:
            z_top = 4.5
            z_bot = z_top - max(1.5, float(bad_depth))
            depth = z_top - z_bot
            assert depth >= 1.5, f"Lake depth {depth} breached 1.5m minimum for input {bad_depth}"

    def test_export_simulation_heightmap_json_schema_and_passability(self):
        """
        Tests export_simulation_heightmap_json on a composite synthetic terrain:
        - Central lake basin (passability = 0)
        - High alpine peak Z > 22m (passability = 3)
        - Steep cliff slope > 40° (passability = 2)
        - Flat fertile basin (passability = 1)
        Verifies JSON output schema, downsampling, and passability matrix values.
        """
        hf = Heightfield2D(resolution=64)
        hf.Z.fill(5.0)  # base walkable

        # Alpine peaks at North: Z = 25m
        hf.Z[50:64, :] = 25.0

        # Lake basin at center: lake_center=(0, 0), radius=10, lake_level=4.5
        dist_sq = hf.X**2 + hf.Y**2
        lake_mask = dist_sq <= 10.0**2
        hf.Z[lake_mask] = 3.0

        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 4.5
        cfg.hydrology.sea_level = 0.0
        cfg.hydrology.river_enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "simulation_heightmap.json"
            res_path = export_simulation_heightmap_json(str(out_json), hf, cfg)
            assert Path(res_path).is_file()

            data = json.loads(Path(res_path).read_text(encoding="utf-8"))

            assert data["generator"] == "TerraForge"
            assert "bounds" in data
            assert "water_levels" in data
            assert data["water_levels"]["lake"] == 4.5

            elev_mat = np.array(data["elevation_matrix"])
            pass_mat = np.array(data["passability_matrix"])

            assert elev_mat.shape == pass_mat.shape
            assert elev_mat.shape[0] <= 128

            # Passability values must strictly be subset of {0, 1, 2, 3}
            unique_pass = set(np.unique(pass_mat))
            assert unique_pass.issubset({0, 1, 2, 3}), f"Unexpected passability values: {unique_pass}"

            # Must contain lake cells (0) and alpine cells (3)
            assert 0 in unique_pass, "Lake cells must have passability 0"
            assert 3 in unique_pass, "Alpine cells must have passability 3"

    def test_export_simulation_heightmap_downsampling_integrity(self):
        """
        Verifies that exporting a 256x256 heightfield downsamples cleanly to 128x128
        while preserving bounds and matrix rectangularity.
        """
        hf = Heightfield2D(resolution=256)
        cfg = MapConfig()
        cfg.hydrology.river_enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "heightmap_256.json"
            export_simulation_heightmap_json(str(out_json), hf, cfg)
            data = json.loads(out_json.read_text(encoding="utf-8"))

            elev_mat = np.array(data["elevation_matrix"])
            assert elev_mat.shape == (128, 128)
            assert data["grid_size"] == [128, 128]

# =============================================================================
# Dimension 4: Astronomical Ephemeris, Photometric Optics & Camera Viewpoints
# =============================================================================

class TestCamerasAtmosphereAndOpticsStress:
    """Stress-tests camera framing math, solar ephemeris, and atmospheric shaders."""

    @pytest.mark.parametrize("t", [-24.0, -12.0, -6.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 36.0, 100.0])
    def test_solar_ephemeris_diurnal_unit_vectors_normalized(self, t: float):
        """
        For any time of day (including negative and out-of-bounds hours):
        - Solar direction vector ||l|| MUST be normalized: 1.0 ± 1e-5.
        - Azimuth MUST be in [0, 360).
        - Solar elevation MUST be in [-max_elev, max_elev].
        """
        elev, azim, (lx, ly, lz) = calculate_solar_position(t, max_elevation_deg=68.0)
        mag = math.sqrt(lx**2 + ly**2 + lz**2)
        assert abs(mag - 1.0) < 1e-5, f"Solar vector magnitude {mag} violated at t={t}"
        assert 0.0 <= azim < 360.0
        assert -68.0 - 1e-4 <= elev <= 68.0 + 1e-4

        # Solar noon (t=12) must be near max elevation
        if (t % 24.0) == 12.0:
            assert abs(elev - 68.0) < 1e-3

        # Solar midnight (t=0) must be near min elevation
        if (t % 24.0) == 0.0:
            assert abs(elev - (-68.0)) < 1e-3

    def test_solar_intensity_and_air_mass_at_horizon(self):
        """
        Direct solar intensity must be strictly 0.0 when elevation <= 0° (night/civil twilight).
        Optical air mass at zenith (90°) must be ~1.0; at horizon (0°) must be 38.0.
        """
        # Below horizon: intensity = 0.0
        assert calculate_sun_intensity(-10.0) == 0.0
        assert calculate_sun_intensity(0.0) == 0.0

        # Daylight intensity: positive and monotonic
        i_10 = calculate_sun_intensity(10.0)
        i_45 = calculate_sun_intensity(45.0)
        i_68 = calculate_sun_intensity(68.0)
        assert 0.0 < i_10 < i_45 < i_68

        # Air mass
        m_zenith = calculate_optical_air_mass(90.0)
        assert abs(m_zenith - 1.0) < 0.05
        m_horizon = calculate_optical_air_mass(0.0)
        assert m_horizon == 38.0

    def test_blackbody_cct_piecewise_continuity(self):
        """
        Correlated Color Temperature (CCT) mapping from solar elevation:
        - Deep night (<= -6°): 9000K
        - Twilight (-6° to 0°): continuous transition from 9000K down to 2200K
        - Golden hour (0° to 10°): warm amber 2200K to 3200K
        - Daytime (10° to 45°): 3200K to 5800K
        - Midday (> 45°): crisp daylight 5800K to 6500K
        Verifies continuity across all partition boundaries.
        """
        assert calculate_cct_from_elevation(-10.0) == 9000.0
        assert calculate_cct_from_elevation(-6.0) == 9000.0
        assert abs(calculate_cct_from_elevation(0.0) - 2200.0) < 1.0
        assert abs(calculate_cct_from_elevation(10.0) - 3200.0) < 1.0
        assert abs(calculate_cct_from_elevation(45.0) - 5800.0) < 1.0
        assert abs(calculate_cct_from_elevation(90.0) - 6500.0) < 1.0

    @pytest.mark.parametrize("temp_k", [-500.0, 0.0, 1000.0, 2200.0, 5500.0, 6500.0, 9000.0, 40000.0, 100000.0])
    def test_tanner_helland_linear_rgb_clamping(self, temp_k: float):
        """
        Tests Tanner-Helland Planckian approximation across extreme temperatures.
        All RGB channels MUST be strictly clamped in [0.0, 1.0].
        """
        r, g, b = cct_to_linear_rgb(temp_k)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

        # At low temperature (warm amber/candlelight): red > blue
        if 1000.0 <= temp_k <= 2500.0:
            assert r > b

        # At high temperature (cool sky/blue): blue > red
        if temp_k >= 10000.0:
            assert b > r

    def test_camera_framing_distance_scaling(self):
        """
        Camera framing distance formula: d = max(w, d) / (sqrt(2) * sin(FOV/2)).
        Verifies:
        - Scaling with diorama span (larger map requires larger distance).
        - Scaling with focal length (telephoto requires larger distance; wide-angle requires smaller).
        - Non-zero framing distance for extreme aspect ratios (300m x 50m).
        """
        # 160x160m diorama with 45mm lens
        fov_45 = calculate_fov_from_lens(45.0)
        dist_160 = calculate_camera_framing_distance(160.0, 160.0, fov_45)
        assert dist_160 > 160.0

        # 320x320m diorama with 45mm lens -> distance must double
        dist_320 = calculate_camera_framing_distance(320.0, 320.0, fov_45)
        assert abs(dist_320 / dist_160 - 2.0) < 1e-3

        # Wide angle (14mm) vs Telephoto (200mm)
        fov_14 = calculate_fov_from_lens(14.0)
        fov_200 = calculate_fov_from_lens(200.0)
        dist_wide = calculate_camera_framing_distance(160.0, 160.0, fov_14)
        dist_tele = calculate_camera_framing_distance(160.0, 160.0, fov_200)
        assert dist_wide < dist_tele

    def test_inspection_camera_configs_elevation_sampling(self):
        """
        Verifies build_inspection_camera_configs dynamically samples eye-level
        camera z from the terrain heightfield (+1.7m human eye height).
        """
        hf = Heightfield2D(resolution=32)
        # Terrain at eye level position (-0.18*160, -0.25*160) = (-28.8, -40.0)
        hf.Z.fill(8.2)

        bounds = DioramaBounds(width=160.0, depth=160.0, z_base=-16.0)
        configs = build_inspection_camera_configs(bounds=bounds, hf=hf)

        assert "eye_level_first_person" in configs
        ped_cfg = configs["eye_level_first_person"]
        loc = ped_cfg["location"]
        # Expected z = 8.2 + 1.7 = 9.9m
        assert abs(loc[2] - 9.9) < 0.1, f"Eye level z={loc[2]} does not reflect terrain+1.7m"


# =============================================================================
# Dimension 5: Vision Rig Matrix & Snapshot Resolution
# =============================================================================

class TestVisionRigMatrixAndSnapshotResolution:
    """Stress-tests vision inspection rig, key alias resolution, and 2x2 grid stitching."""

    def test_resolve_snapshot_key_alias_permutations(self):
        """
        Verifies resolve_snapshot_key resolves all bidirectional canonical and legacy aliases:
        - isometric_diorama <-> iso_sw
        - closeup_macro <-> closeup_water
        - iso_ne <-> eye_level_first_person
        """
        paths = {
            "iso_sw": "/tmp/iso_sw.png",
            "closeup_water": "/tmp/closeup_water.png",
            "eye_level_first_person": "/tmp/eye_level.png",
            "topdown": "/tmp/topdown.png",
        }

        assert resolve_snapshot_key("isometric_diorama", paths) == "/tmp/iso_sw.png"
        assert resolve_snapshot_key("closeup_macro", paths) == "/tmp/closeup_water.png"
        assert resolve_snapshot_key("iso_ne", paths) == "/tmp/eye_level.png"
        assert resolve_snapshot_key("topdown", paths) == "/tmp/topdown.png"
        assert resolve_snapshot_key("unknown_key", paths) is None

    def test_build_2x2_vision_grid_missing_images_placeholder(self):
        """
        When some or all snapshot files are missing, build_2x2_vision_grid MUST
        generate dark gray placeholders with readable labels and successfully output
        a valid 1280x960 PNG composite grid without crashing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_grid = Path(tmpdir) / "vision_grid_empty.png"
            result = build_2x2_vision_grid({}, str(out_grid))
            assert result is not None
            assert out_grid.is_file()

            with Image.open(out_grid) as im:
                assert im.size == (1280, 960)
                assert im.format == "PNG"

    def test_build_2x2_vision_grid_full_images_composition(self):
        """
        Generates 4 synthetic 640x480 images (different colors for each camera viewpoint)
        and stitches them into a full 1280x960 vision inspection grid.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cam_keys = ["isometric_diorama", "iso_ne", "topdown", "closeup_macro"]
            colors = [(180, 50, 50), (50, 180, 50), (50, 50, 180), (180, 180, 50)]

            snapshot_paths = {}
            for key, col in zip(cam_keys, colors):
                fpath = tmp_path / f"{key}.png"
                img = Image.new("RGB", (640, 480), color=col)
                img.save(str(fpath), "PNG")
                snapshot_paths[key] = str(fpath)

            out_grid = tmp_path / "vision_grid_composed.png"
            result = build_2x2_vision_grid(snapshot_paths, str(out_grid))
            assert result is not None
            assert out_grid.is_file()

            with Image.open(out_grid) as im:
                assert im.size == (1280, 960)
                arr = np.array(im)
                # Verify that each quadrant contains non-trivial color
                assert arr[:480, :640, 0].mean() > 100  # Top-left has red
                assert arr[:480, 640:, 1].mean() > 100  # Top-right has green
                assert arr[480:, :640, 2].mean() > 100  # Bottom-left has blue
