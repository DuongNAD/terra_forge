"""
Algorithmic Invariant & Geometry Quality Validator for TerraForge.
Decoupled pure-Python implementation with optional Blender scene hooks.

Validates:
1. Watertight 2-Manifold mesh topology without bpy/bmesh:
   - Euler characteristic chi = V - E + F == 2 (sphere-equivalent closed solid).
   - Exactly 0 boundary edges (watertight).
   - Exactly 0 non-manifold edges (<= 2 faces per edge).
   - Exactly 0 non-manifold vertices (link graph is a single connected circle S^1).
   - Consistent normal winding orientation across all shared edges.
2. Hydrological containment:
   - Lake retaining berm freeboard margin >= 0.50m (excluding river spillways).
   - Leak-proof water basin: lake volume does not penetrate diorama exterior walls.
3. Subsurface geology:
   - Karst cavern rock overburden thickness >= 5.0m.
4. Optional Blender scene complexity budget (if running under Blender).
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Sequence, Optional, Set
import numpy as np

from terra_forge.schema.map_config import MapConfig
from terra_forge.core.heightfield import Heightfield2D

# Lazy / conditional Blender imports
try:
    import bpy
    import bmesh
    HAS_BLENDER = True
except ImportError:
    bpy = None
    bmesh = None
    HAS_BLENDER = False


class QualityValidator:
    """
    Quality and invariant validation engine for TerraForge terrain models.
    Supports headless pure-Python validation and optional Blender scene inspection.
    """

    def __init__(self, config: MapConfig, hf: Optional[Heightfield2D] = None):
        self.config = config
        self.hf = hf

    # -------------------------------------------------------------------------
    # 1. Pure-Python Topological 2-Manifold Validator
    # -------------------------------------------------------------------------
    @staticmethod
    def validate_mesh_topology(
        vertices: Sequence[Sequence[float]] | np.ndarray,
        faces: Sequence[Sequence[int]],
    ) -> Dict[str, Any]:
        """
        Pure-Python 2-manifold topological validator.
        Evaluates discrete differential geometry invariants without Blender:
        1. V - E + F = 2 (Euler-Poincaré characteristic for genus-0 closed diorama block).
        2. Boundary edges == 0 (watertight closed surface, no open holes).
        3. Non-manifold edges == 0 (no edge incident to > 2 faces).
        4. Non-manifold vertices == 0 (each vertex link graph is a single connected circle S^1).
        5. Orientation consistency (adjacent faces traverse shared edges in opposing directions).

        Args:
            vertices: (V, 3) vertex coordinates.
            faces: list of indexed polygonal faces (triangles, quads, or n-gons).

        Returns:
            Dict containing pass/fail flag, errors list, and detailed topological metrics.
        """
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        faces = [list(int(idx) for idx in f) for f in faces]

        V = len(vertices)
        F = len(faces)

        errors: List[str] = []
        warnings: List[str] = []

        if V < 4 or F < 4:
            return {
                "passed": False,
                "is_watertight": False,
                "euler_characteristic": 0,
                "boundary_edges": 0,
                "non_manifold_edges": 0,
                "non_manifold_vertices": 0,
                "errors": [f"Degenerate mesh: vertex count ({V}) or face count ({F}) too small."],
                "warnings": [],
                "metrics": {
                    "num_vertices": V,
                    "num_faces": F,
                    "num_edges": 0,
                    "euler_characteristic": 0,
                    "boundary_edges": 0,
                    "non_manifold_edges": 0,
                    "non_manifold_vertices": 0,
                    "pinch_point_vertices": 0,
                    "orientation_conflicts": 0,
                    "is_watertight": False,
                },
            }

        # Validate vertex index bounds & face degeneracy
        for f_idx, face in enumerate(faces):
            if len(face) < 3:
                errors.append(f"Face {f_idx} is degenerate with < 3 vertices ({len(face)}).")
                continue
            if len(set(face)) != len(face):
                errors.append(f"Face {f_idx} contains duplicate vertex indices: {face}.")
            for v_idx in face:
                if v_idx < 0 or v_idx >= V:
                    errors.append(f"Face {f_idx} references out-of-bounds vertex index {v_idx} (total V={V}).")

        if errors:
            return {
                "passed": False,
                "is_watertight": False,
                "euler_characteristic": 0,
                "boundary_edges": 0,
                "non_manifold_edges": 0,
                "non_manifold_vertices": 0,
                "errors": errors,
                "warnings": warnings,
                "metrics": {
                    "num_vertices": V,
                    "num_faces": F,
                    "num_edges": 0,
                    "euler_characteristic": 0,
                    "boundary_edges": 0,
                    "non_manifold_edges": 0,
                    "non_manifold_vertices": 0,
                    "pinch_point_vertices": 0,
                    "orientation_conflicts": 0,
                    "is_watertight": False,
                },
            }

        # Construct edge adjacency & traversal maps
        # Undirected edge key: (min(u, v), max(u, v))
        edge_incident_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        edge_directed_counts: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
        vertex_faces: Dict[int, List[int]] = defaultdict(list)

        for f_idx, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                u = face[i]
                v = face[(i + 1) % n]
                if u == v:
                    continue
                undir_edge = (min(u, v), max(u, v))
                edge_incident_faces[undir_edge].append(f_idx)
                edge_directed_counts[undir_edge][(u, v)] += 1
                vertex_faces[u].append(f_idx)

        E = len(edge_incident_faces)

        # 1. Boundary & Non-Manifold Edges
        boundary_edges = 0
        non_manifold_edges = 0
        orientation_conflicts = 0

        for edge, inc_faces in edge_incident_faces.items():
            count = len(inc_faces)
            if count == 1:
                boundary_edges += 1
            elif count > 2:
                non_manifold_edges += 1
            elif count == 2:
                # Check normal winding consistency:
                # One face must traverse (u, v) and the other (v, u)
                dirs = edge_directed_counts[edge]
                u, v = edge
                fwd = dirs.get((u, v), 0)
                rev = dirs.get((v, u), 0)
                if fwd != 1 or rev != 1:
                    orientation_conflicts += 1

        # 2. Euler Characteristic
        euler_characteristic = V - E + F

        # 3. Non-Manifold Vertices (Link Condition Analysis)
        # For a closed 2-manifold without boundary:
        # The link graph of every vertex must be a single simple cycle graph (connected 2-regular).
        non_manifold_vertices = 0
        pinch_point_vertices = 0

        for v in range(V):
            inc_f_indices = vertex_faces.get(v, [])
            if not inc_f_indices:
                non_manifold_vertices += 1
                continue

            # Build link graph G_v = (V_v, E_v)
            # V_v: neighbors of v in incident faces
            # E_v: edges between neighbors flanking v in each face
            link_adj: Dict[int, Set[int]] = defaultdict(set)
            link_nodes: Set[int] = set()

            for f_idx in inc_f_indices:
                face = faces[f_idx]
                n = len(face)
                v_pos = face.index(v)
                prev_v = face[(v_pos - 1) % n]
                next_v = face[(v_pos + 1) % n]

                link_nodes.add(prev_v)
                link_nodes.add(next_v)
                link_adj[prev_v].add(next_v)
                link_adj[next_v].add(prev_v)

            # Degree check: in a closed manifold, each neighbor must connect to exactly 2 other neighbors
            is_2_regular = all(len(link_adj[node]) == 2 for node in link_nodes)

            # Connected component count in link graph
            visited: Set[int] = set()
            components = 0
            for start_node in link_nodes:
                if start_node not in visited:
                    components += 1
                    queue = [start_node]
                    visited.add(start_node)
                    while queue:
                        curr = queue.pop(0)
                        for nbr in link_adj[curr]:
                            if nbr not in visited:
                                visited.add(nbr)
                                queue.append(nbr)

            if components != 1 or not is_2_regular:
                non_manifold_vertices += 1
                if components > 1:
                    pinch_point_vertices += 1

        # Global Invariant Assertions
        if boundary_edges > 0:
            errors.append(f"Watertight violation: Mesh has {boundary_edges} open boundary edges.")
        if non_manifold_edges > 0:
            errors.append(f"Manifold edge violation: Mesh has {non_manifold_edges} non-manifold edges (> 2 incident faces).")
        if non_manifold_vertices > 0:
            errors.append(f"Manifold vertex violation: Mesh has {non_manifold_vertices} non-manifold vertices ({pinch_point_vertices} pinch points).")
        if orientation_conflicts > 0:
            errors.append(f"Normal winding violation: Mesh has {orientation_conflicts} edges with conflicting face orientations.")
        if euler_characteristic != 2:
            errors.append(f"Euler characteristic violation: chi = {euler_characteristic} (expected 2 for sphere-equivalent closed solid).")

        passed = len(errors) == 0
        is_watertight = (boundary_edges == 0) and (non_manifold_edges == 0) and (euler_characteristic == 2)

        return {
            "passed": passed,
            "is_watertight": is_watertight,
            "euler_characteristic": euler_characteristic,
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "non_manifold_vertices": non_manifold_vertices,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "num_vertices": V,
                "num_edges": E,
                "num_faces": F,
                "euler_characteristic": euler_characteristic,
                "boundary_edges": boundary_edges,
                "non_manifold_edges": non_manifold_edges,
                "non_manifold_vertices": non_manifold_vertices,
                "pinch_point_vertices": pinch_point_vertices,
                "orientation_conflicts": orientation_conflicts,
                "is_watertight": is_watertight,
            },
        }

    # -------------------------------------------------------------------------
    # 2. Hydrological Berm & Water Basin Validation
    # -------------------------------------------------------------------------
    def validate_lake_berm(
        self,
        min_freeboard_threshold: float = 0.50,
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Verifies minimum vertical freeboard clearance between lake surface elevation
        and the surrounding moraine/berm rim (>= 0.50m).
        Excludes intentional river spillway channels.

        Args:
            min_freeboard_threshold: Minimum required berm clearance in meters (default: 0.50m).
            epsilon: Optional floating-point tolerance (e.g. 1e-6) to guard against
                sub-millimeter precision loss in bilinear interpolation at exact nominal clearance.
        """
        if not self.hf:
            return {"passed": True, "warnings": ["Heightfield not supplied; skipping lake berm check."]}

        hydro = self.config.hydrology
        lx, ly = hydro.lake_center
        lr = hydro.lake_radius
        ll = hydro.lake_level

        # If lake is disabled, trivial radius, or at/below sea level (coastal lagoon/bay), return clean pass
        if lr <= 0.0 or ll <= -900.0 or ll <= hydro.sea_level:
            return {"passed": True, "metrics": {"lake_min_freeboard_berm": 999.0}}

        from terra_forge.core.hydrology import HydrologyEngine
        he = HydrologyEngine(self.hf, hydro)

        # River spillway exclusion
        rx, ry, rw = [], [], []
        if hydro.river_enabled:
            rx, ry, _, rw = he.evaluate_river_spline(np.linspace(0.0, 1.0, 200))

        min_berm_margin = 999.0
        num_samples = 32
        rim_offset = 1.5  # Crest of retaining berm

        for i in range(num_samples):
            ang = (i / float(num_samples)) * 2.0 * math.pi
            px = lx + math.cos(ang) * (lr + rim_offset)
            py = ly + math.sin(ang) * (lr + rim_offset)

            # Check if point is inside river spillway channel
            near_river = False
            if hydro.river_enabled and len(rx) > 0:
                for k in range(len(rx)):
                    if (px - rx[k]) ** 2 + (py - ry[k]) ** 2 < (rw[k] * 2.0) ** 2:
                        near_river = True
                        break
            if near_river:
                continue

            pz = self.hf.sample_bilinear(px, py)
            margin = pz - ll
            if margin < min_berm_margin:
                min_berm_margin = margin

        freeboard = min_berm_margin
        # When checking if freeboard < (min_freeboard_threshold - 1e-6), add epsilon tolerance (e.g. 1e-6)
        # so floating-point inaccuracy at exact nominal 0.50m freeboard does not trigger false positive warnings
        effective_threshold = min_freeboard_threshold - epsilon
        passed = not (freeboard < effective_threshold)
        errors: List[str] = []
        warnings: List[str] = []

        if not passed:
            errors.append(
                f"Lake berm containment violation: Minimum freeboard clearance is {freeboard:.2f}m "
                f"(minimum required: {min_freeboard_threshold:.2f}m). Lake is at risk of overflowing!"
            )

        return {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "lake_min_freeboard_berm": round(freeboard, 3),
                "threshold": min_freeboard_threshold,
            },
        }

    def validate_water_basin_containment(self) -> Dict[str, Any]:
        """
        Validates that water volume boundaries do not penetrate or escape through
        the diorama exterior walls (x = +-W/2, y = +-D/2).
        """
        if not self.hf:
            return {"passed": True, "warnings": ["Heightfield not supplied; skipping water containment check."]}

        bounds = self.config.diorama
        hydro = self.config.hydrology
        lx, ly = hydro.lake_center
        lr = hydro.lake_radius
        ll = hydro.lake_level

        if lr <= 0.0 or ll <= -900.0 or abs(lx) >= 900.0:
            return {"passed": True}

        half_w = bounds.width * 0.5
        half_d = bounds.depth * 0.5

        errors: List[str] = []

        # 1. Spatial boundary clearance
        dist_west = lx - (-half_w)
        dist_east = half_w - lx
        dist_south = ly - (-half_d)
        dist_north = half_d - ly

        wall_clearance = min(dist_west, dist_east, dist_south, dist_north) - lr

        if wall_clearance < 0.0:
            errors.append(
                f"Water basin geometry breach: Lake radius ({lr:.1f}m) intersects exterior diorama wall! "
                f"Negative wall clearance: {wall_clearance:.2f}m."
            )

        # 2. Exterior wall elevation penetration check
        # Sample terrain elevation along outer border in vicinity of lake
        res_x = self.hf.res_x
        res_y = self.hf.res_y

        wall_coords = []
        # South wall (y = 0)
        for x_idx in range(res_x):
            wall_coords.append((x_idx, 0))
        # North wall (y = res_y - 1)
        for x_idx in range(res_x):
            wall_coords.append((x_idx, res_y - 1))
        # West wall (x = 0)
        for y_idx in range(res_y):
            wall_coords.append((0, y_idx))
        # East wall (x = res_x - 1)
        for y_idx in range(res_y):
            wall_coords.append((res_x - 1, y_idx))

        min_wall_elevation_near_lake = 999.0

        for gx, gy in wall_coords:
            wx, wy = self.hf.grid_to_world(gx, gy)
            dist_to_center = math.hypot(wx - lx, wy - ly)
            if dist_to_center <= lr + 5.0:
                wz = float(self.hf.Z[gy, gx])
                if wz < min_wall_elevation_near_lake:
                    min_wall_elevation_near_lake = wz
                if wz <= ll:
                    errors.append(
                        f"Water basin leak violation: Lake water elevation ({ll:.2f}m) breaches "
                        f"exterior wall at ({wx:.1f}, {wy:.1f}, {wz:.2f}m)!"
                    )
                    break

        passed = len(errors) == 0
        return {
            "passed": passed,
            "errors": errors,
            "metrics": {
                "wall_clearance": round(wall_clearance, 2),
                "min_wall_elevation_near_lake": round(min_wall_elevation_near_lake, 2) if min_wall_elevation_near_lake < 900 else None,
            },
        }

    # -------------------------------------------------------------------------
    # 3. Subsurface Cavern Clearance Validation
    # -------------------------------------------------------------------------
    def validate_cavern_clearance(self) -> Dict[str, Any]:
        """Verifies karst cavern roof apex has minimum rock overburden (>= 5.0m)."""
        if not self.hf or not self.config.cavern.enabled:
            return {"passed": True}

        cx, cy, cz = self.config.cavern.center
        ch = self.config.cavern.height
        roof_apex = cz + ch * 0.5
        surface_z = self.hf.sample_bilinear(cx, cy)
        clearance = surface_z - roof_apex
        min_req = self.config.cavern.min_roof_clearance

        passed = clearance >= min_req
        errors = []
        if not passed:
            errors.append(
                f"Cavern clearance violation: Rock thickness is {clearance:.2f}m "
                f"(minimum required: {min_req:.2f}m). Cavern roof risks punching through surface!"
            )

        return {
            "passed": passed,
            "errors": errors,
            "metrics": {
                "cavern_roof_apex": round(roof_apex, 2),
                "surface_above_cavern": round(surface_z, 2),
                "rock_overburden_clearance": round(clearance, 2),
                "min_required": min_req,
            },
        }

    # -------------------------------------------------------------------------
    # 4. Blender BMesh Validation (Optional)
    # -------------------------------------------------------------------------
    def validate_bmesh_object(self, obj_name: str = "Diorama_Island_Block") -> Dict[str, Any]:
        """Validates a Blender scene object using BMesh if running in Blender runtime."""
        if not HAS_BLENDER or not bpy or not bmesh:
            return {
                "passed": False,
                "errors": ["Blender bpy/bmesh runtime is not available in the current Python environment."],
                "metrics": {},
            }

        obj = bpy.data.objects.get(obj_name)
        if not obj:
            return {
                "passed": False,
                "errors": [f"Object '{obj_name}' not found in Blender scene."],
                "metrics": {},
            }

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        boundary_edges = sum(1 for e in bm.edges if e.is_boundary)
        non_manifold_edges = sum(1 for e in bm.edges if not e.is_manifold)
        verts_count = len(bm.verts)
        faces_count = len(bm.faces)
        bm.free()

        passed = (boundary_edges == 0) and (non_manifold_edges == 0)
        errors = []
        if boundary_edges > 0:
            errors.append(f"BMesh watertight violation: {boundary_edges} boundary edges.")
        if non_manifold_edges > 0:
            errors.append(f"BMesh manifold violation: {non_manifold_edges} non-manifold edges.")

        return {
            "passed": passed,
            "errors": errors,
            "metrics": {
                "total_vertices": verts_count,
                "total_faces": faces_count,
                "boundary_edges": boundary_edges,
                "non_manifold_edges": non_manifold_edges,
            },
        }

    # -------------------------------------------------------------------------
    # 5. Unified Validation Entry Point
    # -------------------------------------------------------------------------
    def validate_all(
        self,
        mesh_data: Optional[Dict[str, Any]] = None,
        obj_name: str = "Diorama_Island_Block",
    ) -> Dict[str, Any]:
        """
        Runs complete automated quality assurance and invariant validation suite.
        Can run on raw in-memory mesh_data dict {"vertices": ..., "faces": ...}
        or directly on Blender scene objects.
        """
        results: Dict[str, Any] = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        # 1. Mesh Topology
        if mesh_data and "vertices" in mesh_data and "faces" in mesh_data:
            topo_res = self.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
            results["metrics"].update(topo_res["metrics"])
            results["metrics"]["is_watertight"] = topo_res.get("is_watertight", False)
            if not topo_res["passed"]:
                results["passed"] = False
                results["errors"].extend(topo_res["errors"])
            results["warnings"].extend(topo_res["warnings"])
        elif HAS_BLENDER and bpy and bpy.data.objects.get(obj_name):
            bm_res = self.validate_bmesh_object(obj_name)
            results["metrics"].update(bm_res["metrics"])
            if not bm_res["passed"]:
                results["passed"] = False
                results["errors"].extend(bm_res["errors"])
        elif self.hf is not None:
            # Synthesize slab mesh data directly from heightfield for pure-Python validation
            from terra_forge.core.diorama_slab import DioramaSlabBuilder
            slab = DioramaSlabBuilder(self.hf, self.config.diorama)
            synth_data = slab.build_mesh_data()
            topo_res = self.validate_mesh_topology(synth_data["vertices"], synth_data["faces"])
            results["metrics"].update(topo_res["metrics"])
            results["metrics"]["is_watertight"] = topo_res.get("is_watertight", False)
            if not topo_res["passed"]:
                results["passed"] = False
                results["errors"].extend(topo_res["errors"])
            results["warnings"].extend(topo_res["warnings"])
        else:
            results["warnings"].append("No mesh_data, Blender scene object, or Heightfield provided for topology validation.")

        # 2. Cavern Clearance
        cavern_res = self.validate_cavern_clearance()
        if "metrics" in cavern_res:
            results["metrics"].update(cavern_res["metrics"])
        if not cavern_res["passed"]:
            results["passed"] = False
            results["errors"].extend(cavern_res["errors"])

        # 3. Lake Berm Freeboard (>= 0.50m)
        berm_res = self.validate_lake_berm(min_freeboard_threshold=0.50, epsilon=1e-6)
        if "metrics" in berm_res:
            results["metrics"].update(berm_res["metrics"])
        if not berm_res["passed"]:
            results["passed"] = False
            results["errors"].extend(berm_res["errors"])
        results["warnings"].extend(berm_res.get("warnings", []))

        # 4. Water Basin Containment & Wall Breach
        contain_res = self.validate_water_basin_containment()
        if "metrics" in contain_res:
            results["metrics"].update(contain_res["metrics"])
        if not contain_res["passed"]:
            results["passed"] = False
            results["errors"].extend(contain_res["errors"])

        # 5. Scene Complexity Budget (Blender only)
        if HAS_BLENDER and bpy:
            results["metrics"]["scene_objects"] = len(bpy.data.objects)
            results["metrics"]["scene_materials"] = len(bpy.data.materials)

        return results
