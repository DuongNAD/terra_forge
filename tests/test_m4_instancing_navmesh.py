"""
Unit and Integration Test Suite for Milestone M4: GPU Instancing & Navigation Sync.

Verifies:
1. 4x4 column-major matrix buffer construction, composition, and binary round-trip (.anmi).
2. Three.js `InstancedMesh` (1 draw call per species, groundLift = 0.0) and Bevy ECS matrix conformance.
3. Bottom-pivot guarantee: min_y = 0.0 ensuring flush terrain placement with zero lift.
4. Tall flora collider radius calculation (r = 0.45 + 0.25s) and non-solid pass-through for ground cover.
5. 4-connected BFS reachability flood-fill replicating Anima-Engine `gen_map_manifest.ts`.
6. Strict enforcement of invariant: navmeshCoverage >= 0.80.
7. Adaptive chokepoint pruning under high-density bottleneck conditions.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import tempfile
from typing import List

import numpy as np
import pytest

from terra_forge.ai.vault import FloraType
from terra_forge.instancing.matrices import (
    CANONICAL_FLORA_AABBS,
    HEADER_LEN,
    HEADER_STRUCT,
    INSTANCING_MAGIC,
    INSTANCING_VERSION,
    SPECIES_HEADER_LEN,
    STRIDE_BYTES,
    TALL_FLORA_TYPES,
    GPUInstancingScene,
    InstanceTransform,
    SpeciesInstancingBuffer,
    compose_transform_matrix,
    decompose_transform_matrix,
    euler_to_quaternion,
    quaternion_from_matrix3x3,
    quaternion_to_matrix3x3,
)
from terra_forge.navigation.navmesh import (
    DEFAULT_WALKABLE_SLOPE,
    FLORA_COLLIDER_BASE_RADIUS,
    FLORA_COLLIDER_SCALE_COEFF,
    NON_SOLID_FLORA_TYPES,
    NavMeshReachabilityValidator,
    NavMeshResult,
    ObstacleGridSynchronizer,
    PruningResult,
    cell_to_world,
    flora_canopy_radius,
    flora_collider_radius,
    spawn_clearance_radius,
    validate_and_prune_navmesh,
    world_to_cell,
)


# =============================================================================
# 1. GPU Instancing Matrix Transformations & Conformance Tests
# =============================================================================

class TestGPUInstancingMath:
    def test_quaternion_to_matrix3x3_identity(self):
        """Identity quaternion (0, 0, 0, 1) must produce a 3x3 identity matrix."""
        q_ident = (0.0, 0.0, 0.0, 1.0)
        R = quaternion_to_matrix3x3(q_ident)
        assert R.shape == (3, 3)
        assert np.allclose(R, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_quaternion_to_matrix3x3_cardinal_rotations(self):
        """Verify 90-degree rotations around X, Y, and Z axes."""
        # 90 deg around Y: yaw = pi/2 -> q = (0, sin(pi/4), 0, cos(pi/4))
        q_rot_y = (0.0, float(math.sin(math.pi / 4)), 0.0, float(math.cos(math.pi / 4)))
        R_y = quaternion_to_matrix3x3(q_rot_y)

        # Rotating [1, 0, 0] by 90° around Y gives [0, 0, -1]
        v_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_rotated = R_y @ v_x
        assert np.allclose(v_rotated, np.array([0.0, 0.0, -1.0]), atol=1e-5)

        # 90 deg around X: pitch = pi/2 -> q = (sin(pi/4), 0, 0, cos(pi/4))
        q_rot_x = (float(math.sin(math.pi / 4)), 0.0, 0.0, float(math.cos(math.pi / 4)))
        R_x = quaternion_to_matrix3x3(q_rot_x)
        v_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v_rot_x = R_x @ v_y
        assert np.allclose(v_rot_x, np.array([0.0, 0.0, 1.0]), atol=1e-5)

    def test_euler_to_quaternion_and_roundtrip(self):
        """Verify Euler angles (pitch, yaw, roll) convert to quaternion and back to matrix."""
        pitch, yaw, roll = 0.25, 0.75, -0.15
        q = euler_to_quaternion(pitch, yaw, roll)
        assert len(q) == 4
        # Norm of unit quaternion must be 1.0
        q_norm = math.sqrt(sum(x * x for x in q))
        assert pytest.approx(q_norm, abs=1e-6) == 1.0

        R = quaternion_to_matrix3x3(q)
        q_recon = quaternion_from_matrix3x3(R)
        R_recon = quaternion_to_matrix3x3(q_recon)
        assert np.allclose(R, R_recon, atol=1e-5)

        # Explicitly verify Y-axis yaw rotation preserves upright trunk [0, 1, 0]
        # and rotates [1, 0, 0] into [0, 0, -1]
        q_yaw90 = euler_to_quaternion(0.0, math.pi / 2.0, 0.0)
        R_yaw90 = quaternion_to_matrix3x3(q_yaw90)
        v_upright = R_yaw90 @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert np.allclose(v_upright, np.array([0.0, 1.0, 0.0]), atol=1e-5)
        v_forward = R_yaw90 @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(v_forward, np.array([0.0, 0.0, -1.0]), atol=1e-5)

    def test_column_major_matrix_layout_conformance(self):
        """
        Verify that compose_transform_matrix produces exact 4x4 column-major layout
        matching Three.js Matrix4.elements and Bevy glam::Mat4:
          Col 0 (indices 0..3):   R00*sx, R10*sx, R20*sx, 0.0
          Col 1 (indices 4..7):   R01*sy, R11*sy, R21*sy, 0.0
          Col 2 (indices 8..11):  R02*sz, R12*sz, R22*sz, 0.0
          Col 3 (indices 12..15): px, py + groundLift*sy, pz, 1.0
        """
        pos = (15.5, 4.2, -32.8)
        scale = (2.0, 3.0, 4.0)
        q_ident = (0.0, 0.0, 0.0, 1.0)  # No rotation, R = I

        mat = compose_transform_matrix(pos, q_ident, scale, ground_lift=0.0)
        assert mat.dtype == np.float32
        assert len(mat) == 16

        # Column 0: X-axis basis scaled by sx = 2.0
        assert pytest.approx(mat[0], abs=1e-6) == 2.0
        assert mat[1] == 0.0
        assert mat[2] == 0.0
        assert mat[3] == 0.0

        # Column 1: Y-axis basis scaled by sy = 3.0
        assert mat[4] == 0.0
        assert pytest.approx(mat[5], abs=1e-6) == 3.0
        assert mat[6] == 0.0
        assert mat[7] == 0.0

        # Column 2: Z-axis basis scaled by sz = 4.0
        assert mat[8] == 0.0
        assert mat[9] == 0.0
        assert pytest.approx(mat[10], abs=1e-6) == 4.0
        assert mat[11] == 0.0

        # Column 3: Translation vector [px, py, pz, 1.0]
        assert pytest.approx(mat[12], abs=1e-6) == 15.5
        assert pytest.approx(mat[13], abs=1e-6) == 4.2
        assert pytest.approx(mat[14], abs=1e-6) == -32.8
        assert mat[15] == 1.0

    def test_bottom_pivot_zero_lift_guarantee(self):
        """
        Verify bottom-pivot guarantee: because biological assets are normalized to
        min_y = 0.0 in Vault, ground_lift = 0.0 places asset base flush on terrain.
        """
        pos = (10.0, 5.0, 10.0)
        scale = (1.5, 2.0, 1.5)
        q = (0.0, 0.0, 0.0, 1.0)

        # Default ground_lift = 0.0
        mat_flush = compose_transform_matrix(pos, q, scale, ground_lift=0.0)
        assert pytest.approx(mat_flush[13], abs=1e-6) == 5.0

        # Non-zero ground_lift scales by sy
        mat_lifted = compose_transform_matrix(pos, q, scale, ground_lift=0.5)
        assert pytest.approx(mat_lifted[13], abs=1e-6) == 5.0 + 0.5 * 2.0  # 6.0

    def test_decompose_transform_roundtrip(self):
        """Verify full round-trip: (pos, quat, scale) -> matrix -> decompose -> (pos, quat, scale)."""
        original_pos = (-25.3, 8.4, 72.1)
        original_scale = (1.2, 1.8, 1.2)
        original_quat = euler_to_quaternion(0.1, 1.2, -0.3)

        mat = compose_transform_matrix(original_pos, original_quat, original_scale, ground_lift=0.0)
        pos, quat, scale = decompose_transform_matrix(mat)

        assert np.allclose(pos, original_pos, atol=1e-4)
        assert np.allclose(scale, original_scale, atol=1e-4)

        # Check orientation equivalence via rotation matrices
        R_orig = quaternion_to_matrix3x3(original_quat)
        R_recon = quaternion_to_matrix3x3(quat)
        assert np.allclose(R_orig, R_recon, atol=1e-4)


# =============================================================================
# 2. Species Instancing Buffer & Binary Round-trip (.anmi) Tests
# =============================================================================

class TestSpeciesBufferAndBinaryRoundtrip:
    @pytest.fixture
    def sample_scene(self):
        scene = GPUInstancingScene(world_bounds=(-100.0, -100.0, 100.0, 100.0))

        # Add 3 instances of Pine (species_id 0)
        pine_buf = scene.get_or_create_buffer(species_id=0, species_name="Pine")
        pine_buf.add_instance(InstanceTransform(position=(-10.0, 2.5, 5.0), rotation_quaternion=(0, 0, 0, 1), scale=1.2))
        pine_buf.add_instance(InstanceTransform(position=(15.0, 3.8, -20.0), rotation_quaternion=(0, 0.7071, 0, 0.7071), scale=1.5))
        pine_buf.add_instance(InstanceTransform(position=(0.0, 1.2, 0.0), rotation_quaternion=(0, 0, 0, 1), scale=0.9))

        # Add 2 instances of Rock (species_id 4, ground cover)
        rock_buf = scene.get_or_create_buffer(species_id=4, species_name="Rock")
        rock_buf.add_instance(InstanceTransform(position=(-5.0, 0.5, 8.0), rotation_quaternion=(0, 0, 0, 1), scale=1.0))
        rock_buf.add_instance(InstanceTransform(position=(12.0, 0.8, -15.0), rotation_quaternion=(0, 0, 0, 1), scale=1.3))

        return scene

    def test_species_buffer_matrix_dimensions(self, sample_scene):
        """Verify matrix buffer produces flat float32 array of length count * 16."""
        pine_buf = sample_scene.species_buffers[0]
        assert pine_buf.count == 3
        mat_arr = pine_buf.get_matrix_buffer()
        assert mat_arr.dtype == np.float32
        assert len(mat_arr) == 3 * 16

        rock_buf = sample_scene.species_buffers[4]
        assert rock_buf.count == 2
        assert len(rock_buf.get_matrix_buffer()) == 2 * 16

    def test_binary_header_structure(self, sample_scene):
        """Verify the 32-byte global header format of .anmi export."""
        raw_bytes = sample_scene.export_binary()
        assert len(raw_bytes) >= HEADER_LEN

        magic, ver, species_count, total_inst, b_min_x, b_min_z, b_max_x, b_max_z = HEADER_STRUCT.unpack_from(raw_bytes, 0)
        assert magic == INSTANCING_MAGIC
        assert ver == INSTANCING_VERSION
        assert species_count == 2
        assert total_inst == 5
        assert (b_min_x, b_min_z, b_max_x, b_max_z) == (-100.0, -100.0, 100.0, 100.0)

    def test_binary_serialization_and_deserialization_roundtrip(self, sample_scene):
        """Export to binary buffer and decode back, verifying exact numerical preservation."""
        raw_bytes = sample_scene.export_binary()
        decoded_scene = GPUInstancingScene.from_binary(raw_bytes)

        assert decoded_scene.species_count == sample_scene.species_count
        assert decoded_scene.total_instances == sample_scene.total_instances
        assert decoded_scene.world_bounds == sample_scene.world_bounds

        for species_id in [0, 4]:
            orig_buf = sample_scene.species_buffers[species_id]
            dec_buf = decoded_scene.species_buffers[species_id]
            assert dec_buf.count == orig_buf.count
            assert dec_buf.species_id == orig_buf.species_id

            orig_matrices = orig_buf.get_matrix_buffer()
            dec_matrices = dec_buf.get_matrix_buffer()
            assert np.allclose(orig_matrices, dec_matrices, atol=1e-5)

    def test_file_binary_roundtrip_with_tempfile(self, sample_scene):
        """Verify binary export to disk file and loading from path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "world_flora.anmi"
            sample_scene.export_binary(file_path)
            assert file_path.exists()
            assert file_path.stat().st_size > HEADER_LEN

            loaded = GPUInstancingScene.from_binary(file_path)
            assert loaded.total_instances == 5

    def test_threejs_and_bevy_export_manifests(self, sample_scene):
        """Verify companion Three.js and Bevy ECS JSON exports."""
        # Three.js manifest
        three_manifest = sample_scene.to_threejs_manifest()
        assert three_manifest["format"] == "threejs_instanced_mesh"
        assert three_manifest["drawCallCount"] == 2
        assert three_manifest["totalInstances"] == 5

        pine_three = next(s for s in three_manifest["species"] if s["speciesId"] == 0)
        assert pine_three["drawCalls"] == 1
        assert pine_three["groundLift"] == 0.0
        assert pine_three["castShadow"] is True  # Pine is tall flora
        assert pine_three["matrixFloatCount"] == 3 * 16

        rock_three = next(s for s in three_manifest["species"] if s["speciesId"] == 4)
        assert rock_three["castShadow"] is False  # Rock is ground cover -> castShadow=False

        # Bevy ECS bundles
        bevy_bundles = sample_scene.to_bevy_bundles()
        assert len(bevy_bundles) == 5
        pine_bundle = bevy_bundles[0]
        assert "transform" in pine_bundle
        assert "matrix_column_major" in pine_bundle
        assert pine_bundle["collider"]["is_solid"] is True
        assert pine_bundle["collider"]["shape"] == "cylinder"
        assert pine_bundle["collider"]["radius"] > 0.0


# =============================================================================
# 3. Obstacle Grid Synchronization Tests
# =============================================================================

class TestObstacleGridSynchronization:
    def test_tall_flora_vs_ground_cover_collider_registration(self):
        """
        Verify that tall flora types (Pine, Round, Jungle, Cactus, Acacia, Palm, DeadTree)
        register solid colliders, while ground cover types are pass-through.
        """
        sync = ObstacleGridSynchronizer(shape=(64, 64), world_bounds=(-100.0, -100.0, 100.0, 100.0))

        # Tall flora types
        for ft in [FloraType.Pine, FloraType.Round, FloraType.Jungle, FloraType.Cactus, FloraType.Acacia, FloraType.Palm, FloraType.DeadTree]:
            obs = sync.register_flora_instance(instance_index=int(ft), species_id=int(ft), x=0.0, z=0.0, scale=1.0)
            assert obs is not None, f"Expected {ft.name} to register as solid"
            assert obs.collider_radius == pytest.approx(0.70, abs=1e-5)

        initial_count = len(sync.solid_obstacles)
        assert initial_count == 7

        # Ground cover types (must NOT register solid obstacles)
        for ft in [FloraType.Rock, FloraType.Bush, FloraType.Reed, FloraType.Tuft, FloraType.Coral, FloraType.Kelp, FloraType.Seagrass]:
            obs = sync.register_flora_instance(instance_index=int(ft), species_id=int(ft), x=10.0, z=10.0, scale=1.0)
            assert obs is None, f"Expected ground cover {ft.name} to be non-solid pass-through"

        # Obstacle count unchanged
        assert len(sync.solid_obstacles) == initial_count

    def test_collider_radius_formula(self):
        """Verify r = 0.45 + 0.25 * s for various scale values."""
        assert pytest.approx(flora_collider_radius(0.0), abs=1e-5) == 0.45
        assert pytest.approx(flora_collider_radius(1.0), abs=1e-5) == 0.70
        assert pytest.approx(flora_collider_radius(1.5), abs=1e-5) == 0.825
        assert pytest.approx(flora_collider_radius(2.0), abs=1e-5) == 0.95

    def test_obstacle_rasterization_into_grid(self):
        """Verify a registered solid obstacle properly sets grid cells to True within radius."""
        sync = ObstacleGridSynchronizer(shape=(50, 50), world_bounds=(-50.0, -50.0, 50.0, 50.0))
        # Center cell is (25, 25) at (0.0, 0.0), dx = dz = 2.0
        # Place tree at (0.0, 0.0) with scale 10.0 -> r = 0.45 + 2.5 = 2.95m
        sync.register_flora_instance(instance_index=1, species_id=int(FloraType.Pine), x=0.0, z=0.0, scale=10.0)

        grid = sync.obstacle_grid
        assert grid[25, 25] is np.True_ or grid[25, 25] == 1  # Center cell must be blocked
        assert grid[0, 0] is np.False_ or grid[0, 0] == 0   # Far corner must be clear

    def test_spawn_clearance_radius_margin(self):
        """Verify spawn clearance breathing room: R_spawn = max(2*R_canopy, R_canopy + 0.5)."""
        # Small canopy: R = 0.2 -> 2*R = 0.4, R + 0.5 = 0.7 -> max is 0.7
        assert pytest.approx(spawn_clearance_radius(0.2), abs=1e-5) == 0.70
        # Large canopy: R = 1.5 -> 2*R = 3.0, R + 0.5 = 2.0 -> max is 3.0
        assert pytest.approx(spawn_clearance_radius(1.5), abs=1e-5) == 3.00

    def test_small_flora_at_boundary_junction_blocks_at_least_one_cell(self):
        """
        Verify that small tall flora (scale=0.2, collider_radius=0.50m)
        placed at cell boundary junctions produces at least 1 blocked cell.
        """
        sync = ObstacleGridSynchronizer(shape=(256, 256), world_bounds=(-100.0, -100.0, 100.0, 100.0))
        # Cell size dx = dz = 200 / 256 = 0.78125m
        # Placing at (-21.09375, -21.09375) is exactly at a cell boundary intersection
        obs = sync.register_flora_instance(
            instance_index=1,
            species_id=int(FloraType.Pine),
            x=-21.09375,
            z=-21.09375,
            scale=0.2,
        )
        assert obs is not None
        assert obs.collider_radius == pytest.approx(0.50, abs=1e-5)
        blocked_count = int(np.count_nonzero(sync.obstacle_grid))
        assert blocked_count >= 1, f"Expected >= 1 blocked cell, got {blocked_count}"


# =============================================================================
# 4. NavMesh BFS Reachability & Coverage Invariant Tests
# =============================================================================

class TestNavMeshReachability:
    @pytest.fixture
    def flat_island_terrain(self):
        """Create a 64x64 flat circular island surrounded by ocean."""
        shape = (64, 64)
        elev = np.zeros(shape, dtype=np.float32)
        cr, cc = 32, 32
        r_island = 24.0

        for r in range(64):
            for c in range(64):
                dist = math.sqrt((r - cr) ** 2 + (c - cc) ** 2)
                if dist <= r_island:
                    elev[r, c] = 5.0  # Dry land above sea level (0.20)
                else:
                    elev[r, c] = 0.1  # Submerged ocean below sea level

        return elev

    def test_open_terrain_reachability_coverage(self, flat_island_terrain):
        """An open island with zero obstacles must achieve >= 0.80 reachability coverage (satisfying invariant)."""
        sync = ObstacleGridSynchronizer(shape=(64, 64), world_bounds=(-100.0, -100.0, 100.0, 100.0))
        validator = NavMeshReachabilityValidator(
            elevation=flat_island_terrain,
            sea_level=0.20,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
        )

        res = validator.evaluate(sync)
        assert res.total_land > 0
        assert res.navmesh_coverage >= 0.80
        assert res.is_valid is True
        assert len(res.navigation_nodes) == 4
        assert len(res.navigation_edges) == 3

    def test_spawn_point_avoids_tree_canopies(self, flat_island_terrain):
        """Verify spawn point selection skips candidate cells occluded by tree canopies."""
        sync = ObstacleGridSynchronizer(shape=(64, 64), world_bounds=(-100.0, -100.0, 100.0, 100.0))
        # Place a tall tree right at the exact center (0.0, 0.0) -> cell (32, 32)
        sync.register_flora_instance(instance_index=1, species_id=int(FloraType.Pine), x=0.0, z=0.0, scale=1.5)

        validator = NavMeshReachabilityValidator(
            elevation=flat_island_terrain,
            sea_level=0.20,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
        )

        res = validator.evaluate(sync)
        # Spawn cell must NOT be the tree center (32, 32)
        assert res.spawn_cell != (32, 32)
        # Verify spawn position is outside clearance radius
        obs = sync.solid_obstacles[0]
        spawn_d = math.sqrt((res.spawn_world_pos[0] - obs.x) ** 2 + (res.spawn_world_pos[2] - obs.z) ** 2)
        assert spawn_d >= obs.spawn_clearance - 0.5  # Outside or at breathing margin

    def test_steep_cliffs_and_torrents_are_non_walkable(self):
        """Verify slope >= 0.60 and flow >= 100 cells are marked non-walkable."""
        shape = (30, 30)
        elev = np.ones(shape, dtype=np.float32) * 5.0
        slope = np.zeros(shape, dtype=np.float32)
        flow = np.zeros(shape, dtype=np.float32)

        # Set a cliff barrier across row 15
        slope[15, :] = 0.85  # Exceeds WALKABLE_SLOPE (0.60)
        # Set a torrent river at column 15
        flow[:, 15] = 250.0  # Exceeds MAX_FLOW (100.0)

        sync = ObstacleGridSynchronizer(shape=shape)
        validator = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=0.20,
            slope=slope,
            flow=flow,
        )

        walkable = validator.compute_walkable_mask(sync.obstacle_grid)
        # Cliff row and river col must be non-walkable
        assert np.all(~walkable[15, :])
        assert np.all(~walkable[:, 15])


# =============================================================================
# 5. Adaptive Chokepoint Pruning Tests
# =============================================================================

class TestAdaptiveChokepointPruning:
    @pytest.fixture
    def hourglass_bottleneck_terrain(self):
        """
        Creates a terrain with two large land regions (left & right)
        connected by a single narrow land bridge (isthmus) in the middle.
        """
        shape = (60, 60)
        elev = np.zeros(shape, dtype=np.float32)

        # Left room: rows 10..50, cols 5..25
        elev[10:50, 5:25] = 4.0
        # Right room: rows 10..50, cols 35..55
        elev[10:50, 35:55] = 4.0
        # Narrow bridge: row 28..32, cols 25..35 (4 cells wide)
        elev[28:33, 25:35] = 4.0

        return elev

    def test_bottleneck_blockage_drops_coverage_below_80(self, hourglass_bottleneck_terrain):
        """A wall of trees blocking the narrow bridge drops coverage to ~50%."""
        elev = hourglass_bottleneck_terrain
        sync = ObstacleGridSynchronizer(shape=(60, 60), world_bounds=(-60.0, -60.0, 60.0, 60.0))

        # Place trees completely blocking the bridge at col 30, rows 28..32
        # World coords for row 28..32, col 30
        dx = 120.0 / 60.0
        x_bridge = -60.0 + (30 + 0.5) * dx
        for r in range(28, 33):
            z_bridge = -60.0 + (r + 0.5) * dx
            sync.register_flora_instance(instance_index=r, species_id=int(FloraType.Pine), x=x_bridge, z=z_bridge, scale=3.0)

        validator = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=0.20,
            world_bounds=(-60.0, -60.0, 60.0, 60.0),
        )

        res = validator.evaluate(sync)
        # Because the right half is cut off from spawn (in left half), coverage is ~50%
        assert res.navmesh_coverage < 0.70
        assert res.is_valid is False

    def test_adaptive_pruning_restores_coverage_to_80_percent(self, hourglass_bottleneck_terrain):
        """
        Adaptive pruning detects trees blocking the bridge and prunes them
        until navmeshCoverage >= 0.80 is achieved.
        """
        elev = hourglass_bottleneck_terrain
        sync = ObstacleGridSynchronizer(shape=(60, 60), world_bounds=(-60.0, -60.0, 60.0, 60.0))

        dx = 120.0 / 60.0
        x_bridge = -60.0 + (30 + 0.5) * dx
        for r in range(28, 33):
            z_bridge = -60.0 + (r + 0.5) * dx
            sync.register_flora_instance(instance_index=r, species_id=int(FloraType.Pine), x=x_bridge, z=z_bridge, scale=3.0)

        validator = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=0.20,
            world_bounds=(-60.0, -60.0, 60.0, 60.0),
        )

        prune_res = validator.adaptive_prune(sync, target_coverage=0.80)

        assert prune_res.initial_coverage < 0.70
        assert prune_res.final_coverage >= 0.80
        assert prune_res.pruned_count >= 1
        assert len(prune_res.pruned_obstacle_indices) >= 1
        assert prune_res.navmesh_result.is_valid is True

    def test_convenience_validate_and_prune_navmesh(self, hourglass_bottleneck_terrain):
        """Verify the top-level validate_and_prune_navmesh function."""
        elev = hourglass_bottleneck_terrain

        # Create dummy flora instances
        flora_instances = []
        dx = 120.0 / 60.0
        x_bridge = -60.0 + (30 + 0.5) * dx
        for r in range(28, 33):
            z_bridge = -60.0 + (r + 0.5) * dx
            flora_instances.append({
                "species_id": int(FloraType.Pine),
                "x": x_bridge,
                "z": z_bridge,
                "scale": 3.0,
            })

        res, pruned = validate_and_prune_navmesh(
            elevation=elev,
            flora_instances=flora_instances,
            sea_level=0.20,
            world_bounds=(-60.0, -60.0, 60.0, 60.0),
            target_coverage=0.80,
        )

        assert res.navmesh_coverage >= 0.80
        assert res.is_valid is True
        assert len(pruned) >= 1
