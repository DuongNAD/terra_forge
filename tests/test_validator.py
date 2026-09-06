"""
Unit test suite for Decoupled Pure-Python 2-Manifold Validator.
Verifies:
1. Pure Python importability without bpy/bmesh.
2. Standard geometric solids (tetrahedron, cube, octahedron, diorama slab).
3. Degenerate and boundary topologies (open hole, non-manifold edge, bowtie vertex, flipped normal).
4. Hydrological invariants (berm freeboard >= 0.50m, lake exterior wall breach).
5. Cavern rock overburden clearance (>= 5.0m).
6. validate_all end-to-end execution.
"""

import numpy as np

from terra_forge.inspector.validator import QualityValidator
from terra_forge.schema.map_config import MapConfig, DioramaBounds
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder


class TestPurePythonImport:
    def test_validator_importable_without_blender(self):
        """Verifies QualityValidator can be imported and initialized in pure Python."""
        cfg = MapConfig()
        validator = QualityValidator(cfg)
        assert validator is not None
        assert callable(QualityValidator.validate_mesh_topology)


class TestMeshTopologyPrimitives:
    def test_tetrahedron_closed_manifold(self):
        """Tetrahedron: V=4, E=6, F=4 -> chi = 4 - 6 + 4 = 2."""
        vertices = [(1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)]
        faces = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0

    def test_cube_closed_manifold(self):
        """Cube: V=8, E=12, F=6 quads -> chi = 8 - 12 + 6 = 2."""
        vertices = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            [0, 3, 2, 1],  # Bottom (-Z)
            [4, 5, 6, 7],  # Top (+Z)
            [0, 1, 5, 4],  # South (-Y)
            [2, 3, 7, 6],  # North (+Y)
            [3, 0, 4, 7],  # West (-X)
            [1, 2, 6, 5],  # East (+X)
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0

    def test_diorama_slab_builder_mesh(self):
        """Validates real diorama slab output from DioramaSlabBuilder."""
        res_grid = 12
        bounds = DioramaBounds(width=30.0, depth=30.0, z_base=-10.0, resolution=res_grid)
        hf = Heightfield2D(width=30.0, depth=30.0, resolution=res_grid)
        hf.Z.fill(5.0)

        slab = DioramaSlabBuilder(hf, bounds)
        data = slab.build_mesh_data()

        res = QualityValidator.validate_mesh_topology(data["vertices"], data["faces"])
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0


class TestMeshTopologyDefects:
    def test_open_boundary_hole_detected(self):
        """Removing one face from a cube creates an open boundary hole (4 boundary edges)."""
        vertices = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            # [0, 3, 2, 1] missing bottom
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["boundary_edges"] == 4
        assert res["metrics"]["euler_characteristic"] == 1  # 8 - 12 + 5 = 1 (disk topology)
        assert any("boundary edges" in e for e in res["errors"])

    def test_non_manifold_t_junction_edge(self):
        """Adding a fin/divider face inside a cube creates non-manifold edges with 3 incident faces."""
        vertices = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
            [0, 1, 6, 7],  # Diagonal internal fin sharing edges (0, 1) and (6, 7)
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["non_manifold_edges"] > 0
        assert any("non-manifold edges" in e for e in res["errors"])

    def test_non_manifold_bowtie_pinch_point_vertex(self):
        """Two tetrahedra meeting at a single shared apex vertex (pinch point)."""
        # Apex is vertex 0
        vertices = [
            (0, 0, 0),       # Shared apex 0
            # Tet 1
            (-1, -1, 1), (1, -1, 1), (0, 1, 1),
            # Tet 2
            (-1, -1, -1), (1, -1, -1), (0, 1, -1),
        ]
        faces = [
            # Tet 1
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2],
            # Tet 2
            [0, 5, 4], [0, 6, 5], [0, 4, 6], [4, 5, 6],
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["non_manifold_vertices"] >= 1
        assert res["metrics"]["pinch_point_vertices"] >= 1
        assert any("non-manifold vertices" in e for e in res["errors"])

    def test_inconsistent_normal_winding(self):
        """Cube with one face wound clockwise instead of counter-clockwise."""
        vertices = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 7, 6, 5],  # INVERTED winding
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
        ]
        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is False
        assert res["metrics"]["orientation_conflicts"] > 0
        assert any("winding" in e for e in res["errors"])


class TestHydrologicalQualityValidation:
    def test_lake_berm_freeboard_pass(self):
        """Lake with high moraine berm crest (clearance >= 0.50m) passes."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(0.0)
        # Carve lake basin with 1.2m berm height, water level 4.5m
        hf.carve_lake_basin(center_x=0.0, center_y=0.0, radius=15.0, water_level=4.5, lake_depth=10.0, berm_height=1.2)

        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 15.0
        cfg.hydrology.lake_level = 4.5
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)
        res = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res["passed"] is True
        assert res["metrics"]["lake_min_freeboard_berm"] >= 0.50

    def test_lake_berm_freeboard_fail_on_low_rim(self):
        """Lake with low rim (< 0.50m) triggers containment violation error."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(4.60)  # Rim is at 4.60m, water level at 4.50m -> freeboard = 0.10m < 0.50m

        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 15.0
        cfg.hydrology.lake_level = 4.50
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)
        res = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res["passed"] is False
        assert any("freeboard clearance" in e for e in res["errors"])

    def test_water_basin_wall_breach_detection(self):
        """Lake radius intersecting diorama exterior wall is flagged."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(10.0)

        cfg = MapConfig()
        cfg.diorama.width = 40.0  # half_w = 20.0
        cfg.diorama.depth = 40.0  # half_d = 20.0
        # Lake center at x = 15.0, radius = 10.0 -> extends to x = 25.0 > 20.0 (breach!)
        cfg.hydrology.lake_center = (15.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 5.0

        validator = QualityValidator(cfg, hf)
        res = validator.validate_water_basin_containment()
        assert res["passed"] is False
        assert any("intersects exterior diorama wall" in e for e in res["errors"])


class TestCavernClearanceValidation:
    def test_cavern_rock_overburden_pass_and_fail(self):
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(20.0)

        cfg = MapConfig()
        cfg.cavern.enabled = True
        cfg.cavern.center = (0.0, 0.0, -5.0)
        cfg.cavern.height = 10.0  # Roof apex = -5 + 5 = 0.0. Surface = 20.0 -> Clearance = 20.0m >= 5.0m
        cfg.cavern.min_roof_clearance = 5.0

        validator = QualityValidator(cfg, hf)
        assert validator.validate_cavern_clearance()["passed"] is True

        # Lower surface to 3.0m -> clearance = 3.0m < 5.0m
        hf.Z.fill(3.0)
        fail_res = validator.validate_cavern_clearance()
        assert fail_res["passed"] is False
        assert any("Cavern clearance violation" in e for e in fail_res["errors"])


class TestValidateAllIntegration:
    def test_validate_all_pure_python_pipeline(self):
        """End-to-end validate_all on generated heightfield without Blender."""
        bounds = DioramaBounds(width=40.0, depth=40.0, z_base=-10.0, resolution=21)
        hf = Heightfield2D(width=bounds.width, depth=bounds.depth, resolution=bounds.resolution)
        hf.Z.fill(0.0)
        hf.carve_lake_basin(0.0, 0.0, radius=8.0, water_level=2.0, lake_depth=6.0, berm_height=1.5)

        cfg = MapConfig()
        cfg.diorama = bounds
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 8.0
        cfg.hydrology.lake_level = 2.0
        cfg.hydrology.river_enabled = False
        cfg.cavern.enabled = False

        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        validator = QualityValidator(cfg, hf)
        results = validator.validate_all(mesh_data=mesh_data)

        assert results["passed"] is True
        assert results["metrics"]["is_watertight"] is True
        assert results["metrics"]["euler_characteristic"] == 2
        assert results["metrics"]["boundary_edges"] == 0
        assert results["metrics"]["non_manifold_edges"] == 0
        assert results["metrics"]["non_manifold_vertices"] == 0
        assert results["metrics"]["lake_min_freeboard_berm"] >= 0.50


class TestValidatorRemediationM4:
    """
    Remediation tests for Milestone 4:
    1. NumPy 2D array input for vertices and faces.
    2. Top-level convenience keys adhering to PROJECT.md § Interface Contracts.
    3. Lake berm freeboard epsilon tolerance.
    """

    def test_validate_mesh_topology_numpy_2d_arrays(self):
        """QualityValidator.validate_mesh_topology accepts faces and vertices as 2D np.ndarray."""
        vertices = np.array([
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ], dtype=np.int64)

        res = QualityValidator.validate_mesh_topology(vertices, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["euler_characteristic"] == 2
        assert res["boundary_edges"] == 0
        assert res["non_manifold_edges"] == 0
        assert res["non_manifold_vertices"] == 0

    def test_validate_mesh_topology_top_level_contract_keys(self):
        """Returned dict exposes top-level keys per PROJECT.md § Interface Contracts."""
        vertices = [(1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)]
        faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]

        res = QualityValidator.validate_mesh_topology(vertices, faces)

        # Assert mandatory top-level interface contract keys exist directly on res
        assert "euler_characteristic" in res, "Missing euler_characteristic at top level"
        assert "is_watertight" in res, "Missing is_watertight at top level"
        assert "boundary_edges" in res, "Missing boundary_edges at top level"
        assert "non_manifold_edges" in res, "Missing non_manifold_edges at top level"
        assert "non_manifold_vertices" in res, "Missing non_manifold_vertices at top level"
        assert "passed" in res, "Missing passed at top level"
        assert "errors" in res, "Missing errors at top level"
        assert "warnings" in res, "Missing warnings at top level"
        assert "metrics" in res, "Missing metrics dictionary"

        # Check values
        assert res["euler_characteristic"] == 2
        assert res["is_watertight"] is True
        assert res["boundary_edges"] == 0
        assert res["non_manifold_edges"] == 0
        assert res["non_manifold_vertices"] == 0
        assert res["passed"] is True

    def test_validate_mesh_topology_degenerate_mesh_top_level_keys(self):
        """Early return on degenerate mesh also contains top-level keys."""
        res = QualityValidator.validate_mesh_topology([], [])
        assert "euler_characteristic" in res
        assert "is_watertight" in res
        assert "boundary_edges" in res
        assert "non_manifold_edges" in res
        assert "non_manifold_vertices" in res
        assert res["passed"] is False

    def test_lake_berm_epsilon_tolerance_nominal_freeboard(self):
        """Nominal 0.50m freeboard with epsilon=1e-6 passes without false positive error."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 10.0
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)
        hf.Z.fill(10.50)  # Nominal 10.50 - 10.0 = 0.50m
        res = validator.validate_lake_berm(min_freeboard_threshold=0.50, epsilon=1e-6)
        assert res["passed"] is True
        assert len(res["errors"]) == 0
        assert res["metrics"]["lake_min_freeboard_berm"] == 0.5
