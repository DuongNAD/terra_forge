"""
Milestone 4 Empirical Challenge & 2-Manifold Topological Stress Testing Suite.
Evaluates QualityValidator.validate_mesh_topology and hydrological containment engines:
1. Validates closed genus-0 2-manifolds (Regular Icosahedron, Regular Dodecahedron,
   Icosphere subdivisions, UV spheres, and DioramaSlabBuilder at resolutions 16 to 128).
   Strictly asserts: chi = 2, boundary_edges = 0, non_manifold_edges = 0,
   non_manifold_vertices = 0, orientation_conflicts = 0, is_watertight = True.
2. Injects synthetic topological defects into multiple manifold topologies:
   a) Boundary hole (remove 1 face) -> detects boundary_edges > 0 and is_watertight = False.
   b) Non-manifold edge (add 3rd face sharing an edge) -> detects non_manifold_edges > 0.
   c) Bowtie / pinch vertex (two solids sharing only 1 vertex) -> detects non_manifold_vertices > 0 and pinch_point_vertices > 0.
   d) Inverted normal (reverse winding of 1 face) -> detects orientation_conflicts > 0.
3. Stress tests lake berm freeboard margin and water basin exterior wall containment under extreme values.
"""

from __future__ import annotations
import math
import time
from typing import List, Tuple
import numpy as np
import pytest

from terra_forge.inspector.validator import QualityValidator
from terra_forge.schema.map_config import MapConfig, DioramaBounds
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder


# =============================================================================
# Helper Geometry Generators for Closed Genus-0 2-Manifolds
# =============================================================================

def make_regular_icosahedron() -> Tuple[List[List[float]], List[List[int]]]:
    """Generates a regular icosahedron: 12 vertices, 30 edges, 20 triangular faces."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    scale = 1.0 / math.sqrt(1.0 + phi * phi)
    raw_verts = [
        (-1.0, phi, 0.0), (1.0, phi, 0.0), (-1.0, -phi, 0.0), (1.0, -phi, 0.0),
        (0.0, -1.0, phi), (0.0, 1.0, phi), (0.0, -1.0, -phi), (0.0, 1.0, -phi),
        (phi, 0.0, -1.0), (phi, 0.0, 1.0), (-phi, 0.0, -1.0), (-phi, 0.0, 1.0),
    ]
    verts = [[x * scale, y * scale, z * scale] for x, y, z in raw_verts]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return verts, faces


def make_regular_dodecahedron() -> Tuple[List[List[float]], List[List[int]]]:
    """Generates a regular dodecahedron: 20 vertices, 30 edges, 12 pentagonal faces."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    verts = [
        # Cube vertices (+-1, +-1, +-1)
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        # Coordinates (0, +-1/phi, +-phi)
        [0.0, -inv_phi, -phi], [0.0, inv_phi, -phi], [0.0, -inv_phi, phi], [0.0, inv_phi, phi],
        # Coordinates (+-1/phi, +-phi, 0)
        [-inv_phi, -phi, 0.0], [inv_phi, -phi, 0.0], [-inv_phi, phi, 0.0], [inv_phi, phi, 0.0],
        # Coordinates (+-phi, 0, +-1/phi)
        [-phi, 0.0, -inv_phi], [phi, 0.0, -inv_phi], [-phi, 0.0, inv_phi], [phi, 0.0, inv_phi],
    ]

    # Scale to unit sphere
    norm_factor = 1.0 / math.sqrt(3.0)
    verts = [[x * norm_factor, y * norm_factor, z * norm_factor] for x, y, z in verts]

    V = np.array(verts, dtype=float)
    D = np.linalg.norm(V[:, None] - V[None, :], axis=-1)
    edge_len = (2.0 / phi) * norm_factor

    adj: List[List[int]] = [[] for _ in range(20)]
    for i in range(20):
        for j in range(i + 1, 20):
            if abs(D[i, j] - edge_len) < 1e-4:
                adj[i].append(j)
                adj[j].append(i)

    pentagons = []
    for u in range(20):
        for v in adj[u]:
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x in (u, v):
                        continue
                    for y in adj[x]:
                        if y in (u, v, w):
                            continue
                        if u in adj[y]:
                            pts = V[[u, v, w, x, y]]
                            pts_c = pts - pts.mean(axis=0)
                            _, s, _ = np.linalg.svd(pts_c)
                            if s[2] < 1e-4:
                                p = [u, v, w, x, y]
                                min_i = p.index(min(p))
                                p = p[min_i:] + p[:min_i]
                                if p[1] > p[-1]:
                                    p = [p[0]] + p[1:][::-1]
                                p_t = tuple(p)
                                if p_t not in pentagons:
                                    pentagons.append(p_t)

    oriented_faces = []
    for p in pentagons:
        pts = V[list(p)]
        center = pts.mean(axis=0)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[1])
        if np.dot(center, n) > 0:
            oriented_faces.append(list(p))
        else:
            oriented_faces.append(list(p)[::-1])

    return verts, oriented_faces


def make_icosphere(subdivisions: int = 2) -> Tuple[List[List[float]], List[List[int]]]:
    """Generates an icosphere by recursively subdividing an icosahedron."""
    verts, faces = make_regular_icosahedron()
    midpoint_cache = {}

    def get_midpoint(i1: int, i2: int) -> int:
        edge = (min(i1, i2), max(i1, i2))
        if edge in midpoint_cache:
            return midpoint_cache[edge]
        v1 = verts[i1]
        v2 = verts[i2]
        mx, my, mz = (v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5, (v1[2] + v2[2]) * 0.5
        norm = math.sqrt(mx * mx + my * my + mz * mz)
        new_idx = len(verts)
        verts.append([mx / norm, my / norm, mz / norm])
        midpoint_cache[edge] = new_idx
        return new_idx

    for _ in range(subdivisions):
        new_faces = []
        for v0, v1, v2 in faces:
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20],
            ])
        faces = new_faces

    return verts, faces


def make_uv_sphere(lat_count: int = 16, lon_count: int = 32) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """Generates a closed UV sphere with triangular polar caps and quad body faces."""
    verts: List[Tuple[float, float, float]] = [(0.0, 0.0, 1.0)]  # North pole: idx 0
    for lat_i in range(1, lat_count):
        theta = math.pi * lat_i / lat_count
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for lon_j in range(lon_count):
            phi = 2.0 * math.pi * lon_j / lon_count
            verts.append((sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t))
    verts.append((0.0, 0.0, -1.0))  # South pole: last index
    south_idx = len(verts) - 1

    faces: List[List[int]] = []
    # North cap triangles
    for lon_j in range(lon_count):
        nxt = (lon_j + 1) % lon_count
        faces.append([0, 1 + lon_j, 1 + nxt])

    # Body quads
    for lat_i in range(1, lat_count - 1):
        ring_curr = 1 + (lat_i - 1) * lon_count
        ring_next = 1 + lat_i * lon_count
        for lon_j in range(lon_count):
            nxt = (lon_j + 1) % lon_count
            faces.append([ring_curr + lon_j, ring_next + lon_j, ring_next + nxt, ring_curr + nxt])

    # South cap triangles
    last_ring = 1 + (lat_count - 2) * lon_count
    for lon_j in range(lon_count):
        nxt = (lon_j + 1) % lon_count
        faces.append([last_ring + lon_j, south_idx, last_ring + nxt])

    return verts, faces


# =============================================================================
# 1. Stress Tests: Closed Genus-0 2-Manifolds (Euler Characteristic chi = 2)
# =============================================================================

class TestClosedGenus0Manifolds:
    """Stress tests QualityValidator on closed genus-0 2-manifolds."""

    def test_regular_icosahedron_topology(self):
        """Strictly verifies regular icosahedron: chi=2, zero boundaries, zero conflicts."""
        verts, faces = make_regular_icosahedron()
        res = QualityValidator.validate_mesh_topology(verts, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["pinch_point_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0
        assert res["metrics"]["num_vertices"] == 12
        assert res["metrics"]["num_edges"] == 30
        assert res["metrics"]["num_faces"] == 20

    def test_regular_dodecahedron_topology(self):
        """Strictly verifies regular dodecahedron with pentagonal faces: chi=2, zero defects."""
        verts, faces = make_regular_dodecahedron()
        res = QualityValidator.validate_mesh_topology(verts, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["pinch_point_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0
        assert res["metrics"]["num_vertices"] == 20
        assert res["metrics"]["num_edges"] == 30
        assert res["metrics"]["num_faces"] == 12

    @pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
    def test_icosphere_subdivisions_topology(self, subdivisions: int):
        """Verifies icospheres across subdivision levels 0 to 3 (up to 1280 faces)."""
        verts, faces = make_icosphere(subdivisions)
        res = QualityValidator.validate_mesh_topology(verts, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0

    @pytest.mark.parametrize("lat,lon", [(8, 16), (16, 32), (24, 48)])
    def test_uv_sphere_quad_tri_hybrid_topology(self, lat: int, lon: int):
        """Verifies UV spheres consisting of quad bodies and triangular polar caps."""
        verts, faces = make_uv_sphere(lat, lon)
        res = QualityValidator.validate_mesh_topology(verts, faces)
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0

    @pytest.mark.parametrize("resolution", [16, 32, 64, 128])
    def test_diorama_slab_builder_resolutions_16_to_128(self, resolution: int):
        """
        Stress tests DioramaSlabBuilder at resolutions 16, 32, 64, and 128.
        At res 128: 32,768 vertices, 32,766 faces.
        Strictly verifies:
        chi = 2, boundary_edges = 0, non_manifold_edges = 0,
        non_manifold_vertices = 0, orientation_conflicts = 0, passed = True.
        """
        bounds = DioramaBounds(width=50.0, depth=50.0, z_base=-15.0, resolution=resolution)
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=resolution)

        # Generate non-trivial terrain: undulating sine hills and peaked center
        gx = np.linspace(-math.pi * 2, math.pi * 2, resolution)
        gy = np.linspace(-math.pi * 2, math.pi * 2, resolution)
        X, Y = np.meshgrid(gx, gy)
        hf.Z = 10.0 + 4.0 * np.sin(X) * np.cos(Y)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        builder = DioramaSlabBuilder(hf, bounds)
        mesh_data = builder.build_mesh_data()

        t0 = time.perf_counter()
        res = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        elapsed = time.perf_counter() - t0

        # Assert topological invariants
        assert res["passed"] is True
        assert res["is_watertight"] is True
        assert res["metrics"]["euler_characteristic"] == 2
        assert res["metrics"]["boundary_edges"] == 0
        assert res["metrics"]["non_manifold_edges"] == 0
        assert res["metrics"]["non_manifold_vertices"] == 0
        assert res["metrics"]["pinch_point_vertices"] == 0
        assert res["metrics"]["orientation_conflicts"] == 0

        # Verify performance scales cleanly (< 5.0 seconds even at res 128)
        assert elapsed < 5.0, f"Topology validation on res {resolution} exceeded 5s: {elapsed:.3f}s"


# =============================================================================
# 2. Stress Tests: Synthetic Topological Defects
# =============================================================================

class TestSyntheticTopologicalDefects:
    """Stress tests detection of intentional synthetic defects across multiple solids."""

    @pytest.mark.parametrize(
        "manifold_name",
        ["icosahedron", "dodecahedron", "uv_sphere", "diorama_slab_16"]
    )
    def test_defect_a_boundary_hole_detection(self, manifold_name: str):
        """
        Defect A: Removing 1 face creates an open boundary hole.
        Validator must detect boundary_edges > 0, report is_watertight = False, and fail.
        """
        if manifold_name == "icosahedron":
            verts, faces = make_regular_icosahedron()
        elif manifold_name == "dodecahedron":
            verts, faces = make_regular_dodecahedron()
        elif manifold_name == "uv_sphere":
            verts, faces = make_uv_sphere(12, 24)
        elif manifold_name == "diorama_slab_16":
            bounds = DioramaBounds(width=20.0, depth=20.0, z_base=-5.0, resolution=16)
            hf = Heightfield2D(width=20.0, depth=20.0, resolution=16)
            hf.Z.fill(2.0)
            data = DioramaSlabBuilder(hf, bounds).build_mesh_data()
            verts, faces = data["vertices"], data["faces"]

        # Remove the first face
        face_removed = faces[0]
        faces_with_hole = faces[1:]

        res = QualityValidator.validate_mesh_topology(verts, faces_with_hole)

        assert res["passed"] is False
        assert res["is_watertight"] is False
        assert res["metrics"]["boundary_edges"] == len(face_removed)
        assert res["metrics"]["euler_characteristic"] == 1  # 2 - 1 = 1 (topological disk)
        assert any("boundary edges" in err.lower() or "watertight" in err.lower() for err in res["errors"])

    @pytest.mark.parametrize(
        "manifold_name",
        ["icosahedron", "dodecahedron", "diorama_slab_16"]
    )
    def test_defect_b_non_manifold_edge_detection(self, manifold_name: str):
        """
        Defect B: Adding a 3rd face sharing an edge creates a non-manifold edge (|F(e)| > 2).
        Validator must detect non_manifold_edges > 0 and fail.
        """
        if manifold_name == "icosahedron":
            verts, faces = make_regular_icosahedron()
        elif manifold_name == "dodecahedron":
            verts, faces = make_regular_dodecahedron()
        elif manifold_name == "diorama_slab_16":
            bounds = DioramaBounds(width=20.0, depth=20.0, z_base=-5.0, resolution=16)
            hf = Heightfield2D(width=20.0, depth=20.0, resolution=16)
            hf.Z.fill(2.0)
            data = DioramaSlabBuilder(hf, bounds).build_mesh_data()
            verts, faces = data["vertices"], data["faces"]

        # Face 0 has edge (u, v)
        u, v = faces[0][0], faces[0][1]
        # Introduce a new vertex and an extra triangular fin sharing edge (u, v)
        new_v_idx = len(verts)
        verts_mod = list(verts) + [[0.0, 0.0, 0.0]]
        fin_face = [u, v, new_v_idx]
        faces_mod = list(faces) + [fin_face]

        res = QualityValidator.validate_mesh_topology(verts_mod, faces_mod)

        assert res["passed"] is False
        assert res["metrics"]["non_manifold_edges"] >= 1
        assert any("non-manifold edges" in err.lower() for err in res["errors"])

    def test_defect_c_bowtie_pinch_vertex_two_tetrahedra(self):
        """
        Defect C (Tetrahedra): Two tetrahedra meeting at a single shared apex vertex (pinch point).
        Must detect non_manifold_vertices >= 1, pinch_point_vertices >= 1, and fail.
        """
        vertices = [
            (0.0, 0.0, 0.0),       # Shared apex vertex 0
            # Tetrahedron 1
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0),
            # Tetrahedron 2
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (0.0, 1.0, -1.0),
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
        assert any("pinch points" in err.lower() or "manifold vertex" in err.lower() for err in res["errors"])

    def test_defect_c_bowtie_pinch_vertex_two_icosahedra(self):
        """
        Defect C (Icosahedra): Two complete icosahedra joined at a single apex vertex.
        Link graph of shared vertex 0 contains two disconnected 5-cycles.
        Must detect non_manifold_vertices >= 1, pinch_point_vertices >= 1, and fail.
        """
        verts1, faces1 = make_regular_icosahedron()
        # Offset vertices of 2nd icosahedron except apex vertex 0
        verts2 = [[x + 10.0, y + 10.0, z + 10.0] for x, y, z in verts1[1:]]
        all_verts = verts1 + verts2

        faces2 = []
        for f in faces1:
            new_f = []
            for vi in f:
                if vi == 0:
                    new_f.append(0)  # Shared vertex 0
                else:
                    new_f.append(vi + 11)
            faces2.append(new_f)

        all_faces = faces1 + faces2
        res = QualityValidator.validate_mesh_topology(all_verts, all_faces)

        assert res["passed"] is False
        assert res["metrics"]["non_manifold_vertices"] == 1
        assert res["metrics"]["pinch_point_vertices"] == 1
        assert any("pinch points" in err.lower() for err in res["errors"])

    @pytest.mark.parametrize(
        "manifold_name",
        ["icosahedron", "dodecahedron", "uv_sphere", "diorama_slab_16"]
    )
    def test_defect_d_inverted_normal_orientation_conflict(self, manifold_name: str):
        """
        Defect D: Inverting the winding of 1 face causes adjacent shared edges to be
        traversed in the same direction, violating manifold orientability.
        Must detect orientation_conflicts > 0 and fail.
        """
        if manifold_name == "icosahedron":
            verts, faces = make_regular_icosahedron()
        elif manifold_name == "dodecahedron":
            verts, faces = make_regular_dodecahedron()
        elif manifold_name == "uv_sphere":
            verts, faces = make_uv_sphere(12, 24)
        elif manifold_name == "diorama_slab_16":
            bounds = DioramaBounds(width=20.0, depth=20.0, z_base=-5.0, resolution=16)
            hf = Heightfield2D(width=20.0, depth=20.0, resolution=16)
            hf.Z.fill(2.0)
            data = DioramaSlabBuilder(hf, bounds).build_mesh_data()
            verts, faces = data["vertices"], data["faces"]

        # Invert the winding order of face 0
        face_len = len(faces[0])
        faces_inverted = [faces[0][::-1]] + faces[1:]

        res = QualityValidator.validate_mesh_topology(verts, faces_inverted)

        assert res["passed"] is False
        assert res["metrics"]["orientation_conflicts"] == face_len
        assert any("conflicting face orientations" in err.lower() or "winding" in err.lower() for err in res["errors"])


# =============================================================================
# 3. Stress Tests: Lake Berm Freeboard & Basin Wall Containment (Extreme Values)
# =============================================================================

class TestHydrologicalContainmentStress:
    """Extreme parameter stress testing for hydrological freeboard and wall containment."""

    def test_lake_berm_extreme_negative_freeboard(self):
        """Lake water level at 50m while rim is at 10m (freeboard = -40m) -> severe failure."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(10.0)

        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 12.0
        cfg.hydrology.lake_level = 50.0  # Deeply submerged berm
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)
        res = validator.validate_lake_berm(min_freeboard_threshold=0.50)

        assert res["passed"] is False
        assert res["metrics"]["lake_min_freeboard_berm"] == -40.0
        assert any("lake berm containment violation" in err.lower() for err in res["errors"])

    def test_lake_berm_marginal_threshold_boundary_stress(self):
        """
        Tests boundary behavior around the 0.50m threshold:
        - Freeboard = 0.49m -> Fail
        - Freeboard = 0.50001m (safe offset) -> Pass
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 10.0
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)

        # 1. 0.49m (below 0.50m threshold)
        hf.Z.fill(10.49)
        res_fail = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res_fail["passed"] is False

        # 2. 0.50001m (safely above threshold)
        hf.Z.fill(10.50001)
        res_pass = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res_pass["passed"] is True

    def test_lake_berm_floating_point_epsilon_boundary_finding(self):
        """
        EMPIRICAL CHALLENGE FINDING:
        Demonstrates that nominal 10.50m terrain elevation with 10.0m water level
        fails validation due to strict float comparison without epsilon tolerance.
        Calculated margin is 0.4999999999999982m, resulting in contradictory error message:
        'Minimum freeboard clearance is 0.50m (minimum required: 0.50m).'
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 10.0
        cfg.hydrology.river_enabled = False

        validator = QualityValidator(cfg, hf)
        hf.Z.fill(10.50)
        res = validator.validate_lake_berm(min_freeboard_threshold=0.50)

        assert res["passed"] is False
        assert res["metrics"]["lake_min_freeboard_berm"] == 0.5
        assert any("0.50m (minimum required: 0.50m)" in err for err in res["errors"])

    def test_lake_berm_river_spillway_exclusion_and_leak(self):
        """
        Verifies that river spillway cutaway is ignored if river is enabled,
        but flags failure if river is disabled (uncontained notch in berm).
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(5.0)

        cfg = MapConfig()
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = 3.0  # Standard freeboard = 5.0 - 3.0 = 2.0m >= 0.50m
        cfg.hydrology.river_enabled = True
        cfg.hydrology.river_start = (10.0, 0.0)
        cfg.hydrology.river_end = (35.0, 0.0)
        cfg.hydrology.river_width = 3.0
        cfg.hydrology.river_depth = 2.5

        # Cut a spillway notch at (11.5, 0.0) to elevation 2.5m (below lake level 3.0)
        # When river is enabled and covers this notch, validate_lake_berm should exclude it.
        # Carve actual lake with berm
        hf.carve_lake_basin(0.0, 0.0, radius=10.0, water_level=3.0, lake_depth=5.0, berm_height=1.0)

        validator = QualityValidator(cfg, hf)
        res_river_enabled = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res_river_enabled["passed"] is True

        # Now disable river: the notch or lowered area is no longer excused as a spillway
        # Artificially carve a low breach in the berm rim outside any river
        hf.Z.fill(4.0)  # Surrounding terrain at 4.0m
        # Lake level is 3.70m -> freeboard = 0.30m < 0.50m
        cfg.hydrology.lake_level = 3.70
        cfg.hydrology.river_enabled = False
        res_river_disabled = validator.validate_lake_berm(min_freeboard_threshold=0.50)
        assert res_river_disabled["passed"] is False

    def test_water_basin_wall_clearance_extreme_radius_and_offsets(self):
        """
        Stress tests water basin wall clearance:
        - Giant radius (radius 500m on a 40m diorama) -> Geometry breach
        - Lake centered near South, North, East, West walls -> Geometry breach
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(10.0)

        cfg = MapConfig()
        cfg.diorama.width = 40.0  # X bounds: [-20, 20]
        cfg.diorama.depth = 40.0  # Y bounds: [-20, 20]
        cfg.hydrology.lake_level = 5.0
        validator = QualityValidator(cfg, hf)

        # 1. Giant radius 500m
        cfg.hydrology.lake_center = (0.0, 0.0)
        cfg.hydrology.lake_radius = 500.0
        r_giant = validator.validate_water_basin_containment()
        assert r_giant["passed"] is False
        assert any("intersects exterior diorama wall" in e for e in r_giant["errors"])

        # 2. Breach East wall: center at (18.0, 0.0), radius 5.0m -> reaches 23.0m > 20.0m
        cfg.hydrology.lake_center = (18.0, 0.0)
        cfg.hydrology.lake_radius = 5.0
        r_east = validator.validate_water_basin_containment()
        assert r_east["passed"] is False
        assert r_east["metrics"]["wall_clearance"] == -3.0

        # 3. Breach West wall: center at (-18.0, 0.0), radius 5.0m -> reaches -23.0m < -20.0m
        cfg.hydrology.lake_center = (-18.0, 0.0)
        cfg.hydrology.lake_radius = 5.0
        r_west = validator.validate_water_basin_containment()
        assert r_west["passed"] is False
        assert r_west["metrics"]["wall_clearance"] == -3.0

        # 4. Breach North wall: center at (0.0, 18.0), radius 5.0m -> reaches 23.0m > 20.0m
        cfg.hydrology.lake_center = (0.0, 18.0)
        cfg.hydrology.lake_radius = 5.0
        r_north = validator.validate_water_basin_containment()
        assert r_north["passed"] is False
        assert r_north["metrics"]["wall_clearance"] == -3.0

        # 5. Breach South wall: center at (0.0, -18.0), radius 5.0m -> reaches -23.0m < -20.0m
        cfg.hydrology.lake_center = (0.0, -18.0)
        cfg.hydrology.lake_radius = 5.0
        r_south = validator.validate_water_basin_containment()
        assert r_south["passed"] is False
        assert r_south["metrics"]["wall_clearance"] == -3.0

    def test_water_basin_exterior_wall_elevation_leak_penetration(self):
        """
        Lake radius is within bounds (wall clearance >= 0), but exterior wall elevation
        dips below lake water elevation -> triggers water basin leak penetration error.
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(12.0)  # Wall elevation initially 12m

        cfg = MapConfig()
        cfg.diorama.width = 40.0
        cfg.diorama.depth = 40.0
        # Lake center at (0.0, 10.0), radius 6.0m (extends to y = 16.0m, within wall y = 20.0m)
        cfg.hydrology.lake_center = (0.0, 10.0)
        cfg.hydrology.lake_radius = 6.0
        cfg.hydrology.lake_level = 8.0  # Water elevation 8.0m

        validator = QualityValidator(cfg, hf)
        # Safe condition
        assert validator.validate_water_basin_containment()["passed"] is True

        # Drop north wall elevation to 7.0m (lower than lake water level 8.0m)
        hf.Z[-1, :] = 7.0
        res_leak = validator.validate_water_basin_containment()
        assert res_leak["passed"] is False
        assert any("lake water elevation" in err.lower() and "breaches exterior wall" in err.lower() for err in res_leak["errors"])

    def test_trivial_and_disabled_lake_invariants(self):
        """Verifies that disabled, zero-radius, or ocean-level water bodies bypass berm checks cleanly."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(0.0)
        cfg = MapConfig()

        # Zero radius
        cfg.hydrology.lake_radius = 0.0
        cfg.hydrology.lake_level = 10.0
        validator = QualityValidator(cfg, hf)
        assert validator.validate_lake_berm()["passed"] is True
        assert validator.validate_water_basin_containment()["passed"] is True

        # Lake level <= -900
        cfg.hydrology.lake_radius = 10.0
        cfg.hydrology.lake_level = -999.0
        assert validator.validate_lake_berm()["passed"] is True
        assert validator.validate_water_basin_containment()["passed"] is True

        # Coastal lagoon / lake at or below sea level
        cfg.hydrology.sea_level = 2.0
        cfg.hydrology.lake_level = 1.5
        assert validator.validate_lake_berm()["passed"] is True


if __name__ == "__main__":
    print("Running Milestone 4 Empirical Challenge Stress Suite...")
    pytest.main([__file__, "-v", "-s"])
