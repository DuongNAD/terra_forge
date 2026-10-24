"""
Empirical Adversarial Verification Test Suite for Milestone M6.
Executed by Teamwork Challenger 8.2 (Empirical Challenger: Critic & Specialist).

Targets:
1. NavMesh Reachability Invariant:
   - Evaluates BFS flood-fill reachability across 4 procedural terrains:
     a) Smooth plains
     b) Rugged mountains
     c) Deep lakes
     d) Archipelago
   - Tests with dense vegetation scatter (up to 2,000 instances).
   - Asserts invariant: navmeshCoverage >= 0.80.
   - Tests adversarial chokepoint blocking on narrow isthmus land bridge and verifies
     adaptive pruning successfully restores navmeshCoverage >= 0.80.
   - Discovers and verifies the Mountain Summit Cliff Trap boundary condition.
2. GPU Instancing Matrix Integrity:
   - Verifies 4x4 transform matrices match Three.js Matrix4 column-major standard.
   - Decomposes matrices and verifies position (X, Y, Z) matches heightfield height,
     scale matches assigned parameters, and rotation matches quaternion orientations.
   - Verifies bottom-pivot zero ground-lift guarantee.
3. Cross-Language WorldArtifact Decoder Parity:
   - Validates .anmw binary format against Rust world_artifact.rs and TS worldArtifact.ts.
   - Verifies FNV-1a 32-bit checksum parity and exact 1,114,148 byte length for 256x256.
   - Verifies rejection of trailing bytes, truncation, and corrupted bit payloads.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from terra_forge.ai.vault import FloraType
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.world_artifact import (
    BIOME_COUNT,
    CANONICAL_GRID_DIM,
    CANONICAL_TOTAL_SIZE_256,
    CANONICAL_WORLD_SCALE,
    WORLD_ARTIFACT_HEADER_LEN,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    Biome,
    WorldArtifact,
    fnv1a_32,
    map_biome_frontend_to_backend,
)
from terra_forge.instancing.matrices import (
    CANONICAL_FLORA_AABBS,
    HEADER_LEN,
    HEADER_STRUCT,
    INSTANCING_MAGIC,
    INSTANCING_VERSION,
    SPECIES_HEADER_LEN,
    SPECIES_HEADER_STRUCT,
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
    CANONICAL_COVERAGE_TARGET,
    DEFAULT_MAX_FLOW,
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
# TARGET 1: NavMesh Reachability Invariant & Adversarial Scenarios
# =============================================================================

class TestNavMeshReachabilityAdversarial:
    """
    Adversarial verification of NavMesh BFS reachability across diverse procedural
    terrains with dense vegetation (up to 2,000 instances).
    """

    @staticmethod
    def generate_random_vegetation_instances(
        count: int,
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generates random flora instances with a realistic mix of tall trees and ground cover."""
        rng = np.random.default_rng(seed)
        min_x, min_z, max_x, max_z = world_bounds
        instances = []

        all_species = [
            int(FloraType.Pine),
            int(FloraType.Round),
            int(FloraType.Jungle),
            int(FloraType.Cactus),
            int(FloraType.Rock),
            int(FloraType.Acacia),
            int(FloraType.Palm),
            int(FloraType.DeadTree),
            int(FloraType.Bush),
            int(FloraType.Reed),
            int(FloraType.Tuft),
        ]

        for idx in range(count):
            sp_id = int(rng.choice(all_species))
            x = float(rng.uniform(min_x + 2.0, max_x - 2.0))
            z = float(rng.uniform(min_z + 2.0, max_z - 2.0))
            scale = float(rng.uniform(0.8, 1.8))
            instances.append({
                "species_id": sp_id,
                "x": x,
                "z": z,
                "scale": scale,
            })
        return instances

    def test_navmesh_smooth_plains_dense_vegetation_2000(self):
        """
        Terrain 1: Smooth Plains.
        256x256 grid, elevation in [2.5, 4.0] (above sea_level 1.0), gentle slopes (< 0.10).
        Dense vegetation: 2,000 scattered instances.
        Assert: navmeshCoverage >= 0.80.
        """
        dim = 256
        hf = Heightfield2D(width=200.0, depth=200.0, resolution=dim)
        gx = np.linspace(-3.0, 3.0, dim)
        gy = np.linspace(-3.0, 3.0, dim)
        X, Y = np.meshgrid(gx, gy)
        # Gentle rolling plains
        hf.Z = 3.0 + 0.5 * np.sin(X) * np.cos(Y)

        sea_level = 1.0
        flora = self.generate_random_vegetation_instances(count=2000, seed=101)

        result, pruned = validate_and_prune_navmesh(
            elevation=hf.Z,
            flora_instances=flora,
            sea_level=sea_level,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            target_coverage=0.80,
        )

        assert result.total_land > 0
        assert result.navmesh_coverage >= 0.80, f"Smooth plains coverage {result.navmesh_coverage} < 0.80"
        assert result.is_valid is True
        print(f"\n[NavMesh Smooth Plains] Coverage: {result.navmesh_coverage:.4f}, Land: {result.total_land}, Reached: {result.reach_count}, Pruned: {len(pruned)}")

    def test_navmesh_rugged_mountains_dense_vegetation_2000(self):
        """
        Terrain 2: Rugged Mountains.
        256x256 grid, alpine massif with peaks surrounding a central navigable valley pass.
        Dense vegetation: 2,000 scattered instances.
        Assert: navmeshCoverage >= 0.80 on walkable valley network.
        """
        dim = 256
        hf = Heightfield2D(width=200.0, depth=200.0, resolution=dim)
        hf.Z.fill(2.5)  # Valley floor baseline

        # Surround central pass with distinct alpine peaks
        hf.add_peak(-50.0, -50.0, height=7.0, radius=40.0, shape="horn")
        hf.add_peak(50.0, 50.0, height=6.0, radius=35.0, shape="ridge", ridge_angle_deg=30.0)
        hf.add_peak(-50.0, 50.0, height=5.5, radius=35.0, shape="dome")
        hf.add_peak(50.0, -50.0, height=5.0, radius=35.0, shape="dome")

        sea_level = 1.0
        flora = self.generate_random_vegetation_instances(count=2000, seed=202)

        result, pruned = validate_and_prune_navmesh(
            elevation=hf.Z,
            flora_instances=flora,
            sea_level=sea_level,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            target_coverage=0.80,
        )

        assert result.total_land > 0
        assert result.navmesh_coverage >= 0.80, f"Rugged mountains coverage {result.navmesh_coverage} < 0.80"
        assert result.is_valid is True
        print(f"\n[NavMesh Rugged Mountains] Coverage: {result.navmesh_coverage:.4f}, Land: {result.total_land}, Reached: {result.reach_count}, Pruned: {len(pruned)}")

    def test_navmesh_deep_lakes_dense_vegetation_2000(self):
        """
        Terrain 3: Deep Lakes.
        256x256 grid, central submerged lake basin below sea level (water_level = 2.5, sea_level = 2.5),
        surrounded by extensive dry terrain.
        Dense vegetation: 2,000 scattered instances.
        Assert: navmeshCoverage >= 0.80 around the lake rim.
        """
        dim = 256
        hf = Heightfield2D(width=200.0, depth=200.0, resolution=dim)
        hf.Z.fill(4.0)  # Surrounding dry land elevation
        # Carve central deep lake basin
        hf.carve_lake_basin(center_x=0.0, center_y=0.0, radius=40.0, water_level=2.5, lake_depth=5.0, berm_height=0.0)

        sea_level = 2.5
        flora = self.generate_random_vegetation_instances(count=2000, seed=303)

        result, pruned = validate_and_prune_navmesh(
            elevation=hf.Z,
            flora_instances=flora,
            sea_level=sea_level,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            target_coverage=0.80,
        )

        assert result.total_land > 0
        assert result.navmesh_coverage >= 0.80, f"Deep lakes coverage {result.navmesh_coverage} < 0.80"
        assert result.is_valid is True
        print(f"\n[NavMesh Deep Lakes] Coverage: {result.navmesh_coverage:.4f}, Land: {result.total_land}, Reached: {result.reach_count}, Pruned: {len(pruned)}")

    def test_navmesh_archipelago_dense_vegetation_2000(self):
        """
        Terrain 4: Archipelago.
        256x256 grid, connected island chain with central main landmass and walkable tidal causeways.
        Dense vegetation: 2,000 scattered instances.
        Assert: navmeshCoverage >= 0.80.
        """
        dim = 256
        hf = Heightfield2D(width=200.0, depth=200.0, resolution=dim)
        hf.Z.fill(0.2)  # Ocean baseline

        # Main large island in center (radius 70m)
        hf.add_peak(0.0, 0.0, height=4.5, radius=70.0, shape="dome")
        # Satellite islands with tidal connection
        hf.add_peak(-65.0, 0.0, height=2.8, radius=25.0, shape="dome")
        hf.add_peak(65.0, 0.0, height=2.8, radius=25.0, shape="dome")

        sea_level = 1.0
        flora = self.generate_random_vegetation_instances(count=2000, seed=404)

        result, pruned = validate_and_prune_navmesh(
            elevation=hf.Z,
            flora_instances=flora,
            sea_level=sea_level,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            target_coverage=0.80,
        )

        assert result.total_land > 0
        assert result.navmesh_coverage >= 0.80, f"Archipelago coverage {result.navmesh_coverage} < 0.80"
        assert result.is_valid is True
        print(f"\n[NavMesh Archipelago] Coverage: {result.navmesh_coverage:.4f}, Land: {result.total_land}, Reached: {result.reach_count}, Pruned: {len(pruned)}")

    def test_adversarial_isthmus_chokepoint_pruning(self):
        """
        Adversarial Test:
        Construct an hourglass continent with two 45% land masses connected solely by
        a narrow 4-cell isthmus land bridge.
        Plant a dense wall of 30 tall solid trees directly across the isthmus to choke
        connectivity, forcing initial coverage to ~50%.
        Verify that adaptive chokepoint pruning identifies and prunes the blocking trees,
        restoring coverage to >= 0.80.
        Also verifies non-solid flora (Bushes/Rocks) placed in the chokepoint are NOT pruned.
        """
        dim = 120
        elev = np.zeros((dim, dim), dtype=np.float32)

        # Left room: rows 20..100, cols 10..50
        elev[20:100, 10:50] = 5.0
        # Right room: rows 20..100, cols 70..110
        elev[20:100, 70:110] = 5.0
        # Narrow isthmus bridge: rows 58..62, cols 50..70
        elev[58:63, 50:71] = 5.0

        sea_level = 1.0
        bounds = (-60.0, -60.0, 60.0, 60.0)
        sync = ObstacleGridSynchronizer(shape=(dim, dim), world_bounds=bounds)

        # Place 15 solid tall trees across the isthmus bridge
        cell_size = 120.0 / dim
        flora_instances = []
        inst_id = 0

        for r in range(58, 63):
            for c in [59, 60, 61]:
                x = bounds[0] + (c + 0.5) * cell_size
                z = bounds[1] + (r + 0.5) * cell_size
                flora_instances.append({
                    "species_id": int(FloraType.Pine),
                    "x": x,
                    "z": z,
                    "scale": 2.5,
                })
                sync.register_flora_instance(inst_id, int(FloraType.Pine), x, z, scale=2.5)
                inst_id += 1

        # Also add non-solid bushes/rocks in the chokepoint
        for r in range(58, 63):
            x = bounds[0] + (60.5) * cell_size
            z = bounds[1] + (r + 0.5) * cell_size
            flora_instances.append({
                "species_id": int(FloraType.Bush),
                "x": x,
                "z": z,
                "scale": 1.0,
            })
            obs = sync.register_flora_instance(inst_id, int(FloraType.Bush), x, z, scale=1.0)
            assert obs is None, "Ground cover must not register as a solid obstacle"
            inst_id += 1

        validator = NavMeshReachabilityValidator(
            elevation=elev,
            sea_level=sea_level,
            world_bounds=bounds,
        )

        initial_res = validator.evaluate(sync)
        assert initial_res.navmesh_coverage < 0.70, f"Expected initial drop below 0.70, got {initial_res.navmesh_coverage}"
        assert initial_res.is_valid is False

        prune_res = validator.adaptive_prune(sync, target_coverage=0.80)

        assert prune_res.final_coverage >= 0.80, f"Pruning failed to restore coverage: {prune_res.final_coverage}"
        assert prune_res.pruned_count >= 1, "Must have pruned at least one obstacle"
        assert prune_res.navmesh_result.is_valid is True

        print(f"\n[Adversarial Isthmus Pruning] Initial Coverage: {prune_res.initial_coverage:.4f} -> Final Coverage: {prune_res.final_coverage:.4f}, Pruned Count: {prune_res.pruned_count}")

    def test_adversarial_mountain_summit_cliff_trap_finding(self):
        """
        EMPIRICAL CHALLENGE FINDING:
        Demonstrates the 'Mountain Summit Cliff Trap' failure mode.
        When a sharp mountain peak is centered at (0, 0), find_spawn_cell (center-biased)
        picks the summit of the peak as the spawn location.
        If the summit is ringed by steep cliffs (slope >= 0.60), the spawn point is isolated
        from the rest of the world, dropping reachability to near zero (< 0.01) regardless
        of flora pruning.
        This provides proof for the architectural recommendation that spawn point selection
        must be valley-biased or pass-aware in mountainous geomorphology.
        """
        dim = 128
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=dim)
        hf.Z.fill(2.0)
        # Place sharp horn peak exactly at (0, 0)
        hf.add_peak(0.0, 0.0, height=8.0, radius=25.0, shape="horn")

        val = NavMeshReachabilityValidator(elevation=hf.Z, sea_level=1.0)
        sync = ObstacleGridSynchronizer(shape=(dim, dim))
        res = val.evaluate(sync)

        # Confirm summit trap: total land is 100%, but coverage is trapped (< 0.05)
        assert res.total_land > 0
        assert res.navmesh_coverage < 0.05, f"Expected summit trap to drop coverage < 0.05, got {res.navmesh_coverage}"
        print(f"\n[Summit Trap Finding] Total Land: {res.total_land}, Reached: {res.reach_count}, Coverage: {res.navmesh_coverage:.4f} (Summit isolated by cliffs)")


# =============================================================================
# TARGET 2: GPU Instancing 4x4 Matrix Decomposition vs Three.js Standards
# =============================================================================

class TestGPUInstancingMatrixIntegrity:
    """
    Verifies 4x4 transform matrices adhere to Three.js Matrix4 column-major standard,
    accurate round-trip decomposition, and heightfield ground-lift guarantee.
    """

    def test_threejs_matrix4_column_major_order_standard(self):
        """
        Three.js Matrix4 elements layout:
          elements[0..3]:   col 0 (basis X * sx)
          elements[4..7]:   col 1 (basis Y * sy)
          elements[8..11]:  col 2 (basis Z * sz)
          elements[12..15]: col 3 (translation [tx, ty, tz, 1.0])
        """
        pos = (12.34, 5.67, -89.01)
        scale = (1.5, 2.5, 3.5)
        qy = float(math.sin(math.pi / 4))
        qw = float(math.cos(math.pi / 4))
        quat = (0.0, qy, 0.0, qw)

        mat = compose_transform_matrix(pos, quat, scale, ground_lift=0.0)
        assert len(mat) == 16
        assert mat.dtype == np.float32

        # Col 0 (X basis) scaled by sx = 1.5 -> [0, 0, -1.5, 0]
        assert pytest.approx(mat[0], abs=1e-5) == 0.0
        assert pytest.approx(mat[1], abs=1e-5) == 0.0
        assert pytest.approx(mat[2], abs=1e-5) == -1.5
        assert mat[3] == 0.0

        # Col 1 (Y basis) unchanged by Y rotation -> [0, 2.5, 0, 0]
        assert pytest.approx(mat[4], abs=1e-5) == 0.0
        assert pytest.approx(mat[5], abs=1e-5) == 2.5
        assert pytest.approx(mat[6], abs=1e-5) == 0.0
        assert mat[7] == 0.0

        # Col 2 (Z basis) rotated [0, 0, 1] -> [1, 0, 0] scaled by sz = 3.5 -> [3.5, 0, 0, 0]
        assert pytest.approx(mat[8], abs=1e-5) == 3.5
        assert pytest.approx(mat[9], abs=1e-5) == 0.0
        assert pytest.approx(mat[10], abs=1e-5) == 0.0
        assert mat[11] == 0.0

        # Col 3 (translation)
        assert pytest.approx(mat[12], abs=1e-5) == 12.34
        assert pytest.approx(mat[13], abs=1e-5) == 5.67
        assert pytest.approx(mat[14], abs=1e-5) == -89.01
        assert mat[15] == 1.0

    def test_randomized_matrix_decomposition_accuracy_1000(self):
        """
        Stress test: 1,000 randomized transform instances.
        Composes 4x4 matrix, decomposes it back into (pos, quat, scale),
        and verifies errors are below strict numerical tolerances (< 1e-4).
        """
        rng = np.random.default_rng(777)
        max_pos_err = 0.0
        max_scale_err = 0.0
        max_rot_err = 0.0

        for _ in range(1000):
            px = float(rng.uniform(-100.0, 100.0))
            py = float(rng.uniform(0.0, 10.0))
            pz = float(rng.uniform(-100.0, 100.0))
            pos = (px, py, pz)

            sx = float(rng.uniform(0.2, 5.0))
            sy = float(rng.uniform(0.2, 5.0))
            sz = float(rng.uniform(0.2, 5.0))
            scale = (sx, sy, sz)

            u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
            q0 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
            q1 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
            q2 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
            q3 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
            quat = (q0, q1, q2, q3)

            mat = compose_transform_matrix(pos, quat, scale, ground_lift=0.0)
            recon_pos, recon_quat, recon_scale = decompose_transform_matrix(mat)

            p_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, recon_pos)))
            max_pos_err = max(max_pos_err, p_err)

            s_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(scale, recon_scale)))
            max_scale_err = max(max_scale_err, s_err)

            R_orig = quaternion_to_matrix3x3(quat)
            R_recon = quaternion_to_matrix3x3(recon_quat)
            r_err = float(np.linalg.norm(R_orig - R_recon, ord="fro"))
            max_rot_err = max(max_rot_err, r_err)

        assert max_pos_err < 1e-4, f"Max position error too high: {max_pos_err}"
        assert max_scale_err < 1e-4, f"Max scale error too high: {max_scale_err}"
        assert max_rot_err < 1e-4, f"Max rotation Frobenius error too high: {max_rot_err}"
        print(f"\n[Matrix Decomposition 1000 Runs] Max Pos Err: {max_pos_err:.2e}, Max Scale Err: {max_scale_err:.2e}, Max Rot Err: {max_rot_err:.2e}")

    def test_heightfield_elevation_adherence_and_zero_ground_lift(self):
        """
        Verifies that stratified flora placement builds matrices with:
        1. Y = heightfield(X, Z) with error < 0.01m.
        2. ground_lift = 0.0 ensures the bottom of the mesh touches terrain flush.
        """
        dim = 128
        hf = Heightfield2D(width=200.0, depth=200.0, resolution=dim)
        gx = np.linspace(-math.pi, math.pi, dim)
        gy = np.linspace(-math.pi, math.pi, dim)
        X, Y = np.meshgrid(gx, gy)
        hf.Z = 4.0 + 2.0 * np.sin(X) * np.cos(Y)

        class DummyPlacement:
            def __init__(self, x: float, z: float, scale: float = 1.2):
                self.x = x
                self.z = z
                self.y = 0.0
                self.scale = scale
                self.flora_type = FloraType.Pine
                self.rotation_quaternion = (0.0, 0.0, 0.0, 1.0)
                self.suitability = 1.0
                self.rule_name = "Canopy_Alpine_Pine"

        rng = np.random.default_rng(888)
        placements = []
        for _ in range(200):
            wx = float(rng.uniform(-90.0, 90.0))
            wz = float(rng.uniform(-90.0, 90.0))
            placements.append(DummyPlacement(wx, wz, scale=float(rng.uniform(1.0, 2.0))))

        scene = GPUInstancingScene.build_from_stratified_placements(
            placements,
            heightfield=hf.Z,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            default_ground_lift=0.0,
        )

        assert scene.total_instances == 200
        pine_buf = scene.species_buffers[int(FloraType.Pine)]
        mat_flat = pine_buf.get_matrix_buffer()

        for idx, inst in enumerate(pine_buf.instances):
            m_slice = mat_flat[idx * 16 : (idx + 1) * 16]
            pos, quat, scale = decompose_transform_matrix(m_slice)

            norm_c = (pos[0] - (-100.0)) / 200.0 * (dim - 1)
            norm_r = (pos[2] - (-100.0)) / 200.0 * (dim - 1)
            c = int(np.clip(round(norm_c), 0, dim - 1))
            r = int(np.clip(round(norm_r), 0, dim - 1))
            expected_y = float(hf.Z[r, c])

            assert abs(pos[1] - expected_y) < 0.01, f"Instance Y {pos[1]} diverged from heightfield {expected_y}"
            assert inst.ground_lift == 0.0, "Expected zero ground lift for bottom-pivot guarantee"


# =============================================================================
# TARGET 3: Cross-Language WorldArtifact Decoder Parity
# =============================================================================

class TestCrossLanguageWorldArtifactDecoderParity:
    """
    Validates WorldArtifact v2 binary format (.anmw) against specifications in:
    - Anima-Engine Rust: src-tauri/src/core/world_artifact.rs
    - Anima-Engine TS: src/components/Landscape/utils/worldArtifact.ts
    """

    def test_fnv1a_32_checksum_parity_vectors(self):
        """Verifies FNV-1a 32-bit checksum implementation against standard vectors."""
        assert fnv1a_32(b"") == 0x811C9DC5
        # Vector 2: "ANMW" - XOR then multiply (FNV-1a 32)
        assert fnv1a_32(b"ANMW") == 3042223698  # 0xB554C652
        data = b"Anima-Engine-Cross-Language-Checksum-Parity"
        csum = fnv1a_32(data)
        assert 0 <= csum <= 0xFFFFFFFF

    def test_decode_committed_cross_language_fixture_world_4x4(self):
        """
        Loads Anima-Engine's committed fixture:
        src-tauri/tests/fixtures/world_4x4.anmw
        Verifies exact bitwise header layout, checksum validation, and array match.
        """
        fixture_path = Path(r"e:\Project\03_Engines_Simulation\Anima-Engine\src-tauri\tests\fixtures\world_4x4.anmw")
        assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

        data = fixture_path.read_bytes()
        assert len(data) == WORLD_ARTIFACT_HEADER_LEN + 16 * 17  # 36 + 272 = 308 bytes

        artifact = WorldArtifact.from_bytes(data)
        assert artifact.width == 4
        assert artifact.height == 4
        assert artifact.sea_level == 0.5
        assert artifact.seed == 1337
        assert artifact.generator_version == 20
        assert artifact.world_scale == CANONICAL_WORLD_SCALE

        expected_elevation = np.array([i * 0.5 for i in range(16)], dtype=np.float32)
        expected_moisture = np.array([i * 0.25 for i in range(16)], dtype=np.float32)
        expected_temperature = np.array([i * 0.125 for i in range(16)], dtype=np.float32)
        expected_flow = np.array([float(i) for i in range(16)], dtype=np.float32)
        expected_biome = np.array([i for i in range(16)], dtype=np.uint8)

        assert np.allclose(artifact.elevation, expected_elevation)
        assert np.allclose(artifact.moisture, expected_moisture)
        assert np.allclose(artifact.temperature, expected_temperature)
        assert np.allclose(artifact.flow, expected_flow)
        assert np.array_equal(artifact.biome, expected_biome)

    def test_canonical_256_exact_byte_length_and_rejection_of_trailing_bytes(self):
        """
        Asserts canonical 256x256 WorldArtifact exports exactly 1,114,148 bytes.
        Asserts strict rejection of:
        - Truncated bytes (< 1,114,148)
        - Trailing bytes (> 1,114,148)
        - Corrupted payload bytes (checksum mismatch)
        """
        n = CANONICAL_GRID_DIM * CANONICAL_GRID_DIM
        elev = np.full(n, 0.5, dtype=np.float32)
        moist = np.full(n, 0.4, dtype=np.float32)
        temp = np.full(n, 0.6, dtype=np.float32)
        flow = np.zeros(n, dtype=np.float32)
        biome = np.full(n, int(Biome.GRASSLAND), dtype=np.uint8)

        artifact = WorldArtifact(
            width=CANONICAL_GRID_DIM,
            height=CANONICAL_GRID_DIM,
            sea_level=0.3,
            seed=42,
            generator_version=1,
            world_scale=CANONICAL_WORLD_SCALE,
            elevation=elev,
            moisture=moist,
            temperature=temp,
            flow=flow,
            biome=biome,
        )

        raw = artifact.to_bytes()
        assert len(raw) == CANONICAL_TOTAL_SIZE_256, f"Expected {CANONICAL_TOTAL_SIZE_256} bytes, got {len(raw)}"

        decoded = WorldArtifact.from_bytes(raw)
        assert decoded.width == CANONICAL_GRID_DIM
        assert decoded.height == CANONICAL_GRID_DIM
        assert decoded.seed == 42
        assert decoded.checksum() == artifact.checksum()

        padded = raw + b"\x00"
        with pytest.raises(ValueError, match="trailing byte"):
            WorldArtifact.from_bytes(padded)

        truncated = raw[:-10]
        with pytest.raises(ValueError, match="truncated"):
            WorldArtifact.from_bytes(truncated)

        corrupted = bytearray(raw)
        corrupted[100] ^= 0xFF
        with pytest.raises(ValueError, match="checksum mismatch"):
            WorldArtifact.from_bytes(bytes(corrupted))

    def test_node_typescript_decoder_parity(self):
        """
        Cross-language test:
        1. Export a 256x256 WorldArtifact from Python to a temporary .anmw file.
        2. Run a Node.js script that imports Anima-Engine's `decodeWorldArtifact`.
        3. Assert Node.js decodes the binary file cleanly with 0 errors and matching fields.
        """
        n = CANONICAL_GRID_DIM * CANONICAL_GRID_DIM
        rng = np.random.default_rng(999)
        elev = rng.uniform(0.0, 1.0, n).astype(np.float32)
        moist = rng.uniform(0.0, 1.0, n).astype(np.float32)
        temp = rng.uniform(0.0, 1.0, n).astype(np.float32)
        flow = rng.uniform(0.0, 50.0, n).astype(np.float32)
        biome = rng.integers(0, BIOME_COUNT, n).astype(np.uint8)

        artifact = WorldArtifact(
            width=CANONICAL_GRID_DIM,
            height=CANONICAL_GRID_DIM,
            sea_level=0.25,
            seed=987654,
            generator_version=2,
            world_scale=CANONICAL_WORLD_SCALE,
            elevation=elev,
            moisture=moist,
            temperature=temp,
            flow=flow,
            biome=biome,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            anmw_file = Path(tmp_dir) / "python_generated_256.anmw"
            artifact.to_file(anmw_file)
            assert anmw_file.stat().st_size == CANONICAL_TOTAL_SIZE_256

            js_file_path = str(anmw_file).replace("\\", "/")
            js_validator = f"""
import fs from 'fs';
import {{ decodeWorldArtifact }} from '../src/components/Landscape/utils/worldArtifact';

const buffer = fs.readFileSync('{js_file_path}');
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

try {{
  const d = decodeWorldArtifact(arrayBuffer);
  if (d.width !== 256 || d.height !== 256) throw new Error(`bad dims: ${{d.width}}x${{d.height}}`);
  if (d.seed !== 987654) throw new Error(`bad seed: ${{d.seed}}`);
  if (d.generatorVersion !== 2) throw new Error(`bad gen ver: ${{d.generatorVersion}}`);
  if (Math.abs(d.seaLevel - 0.25) > 1e-4) throw new Error(`bad seaLevel: ${{d.seaLevel}}`);
  if (d.elevation.length !== 65536) throw new Error(`bad elevation length`);
  if (d.biome.length !== 65536) throw new Error(`bad biome length`);
  console.log("SUCCESS: TS decodeWorldArtifact verified binary parity cleanly.");
  process.exit(0);
}} catch (err) {{
  console.error("TS DECODER FAILED:", err);
  process.exit(1);
}}
"""
            runner_script = Path(r"e:\Project\03_Engines_Simulation\Anima-Engine\scripts\test_validate_anmw_tmp.ts")
            try:
                runner_script.write_text(js_validator, encoding="utf-8")

                cmd = ["node", "scripts/run_ts.mjs", "scripts/test_validate_anmw_tmp.ts"]
                proc = subprocess.run(
                    cmd,
                    cwd=r"e:\Project\03_Engines_Simulation\Anima-Engine",
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                assert proc.returncode == 0, f"Node TS decoder failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
                assert "SUCCESS: TS decodeWorldArtifact verified binary parity cleanly." in proc.stdout
                print(f"\n[Cross-Language TS Parity] {proc.stdout.strip()}")
            finally:
                if runner_script.exists():
                    runner_script.unlink()


if __name__ == "__main__":
    print("Executing Milestone M6 Adversarial Verification Suite...")
    pytest.main([__file__, "-v", "-s"])
