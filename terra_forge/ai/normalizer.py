"""
Geometry Normalizer & Collision Primitives Generator for TerraForge.
Normalizes generative 3D assets (Meshy GLB / procedural) to meet game engine standards:
- Pivot alignment: bottom-aligned with ground plane (min_y == 0.0).
- Horizontal centering: centered around origin (x_center == 0.0, z_center == 0.0).
- Metric scaling: uniformly scaled to real-world target height (H_target in meters).
- Physics collision primitives extraction:
  - Cylinder: radius, height, center [0, H/2, 0].
  - AABB: min_vec, max_vec, size, center.
  - Convex Hull: vertices, triangular simplices, volume, surface area via scipy.spatial.ConvexHull.
- Pure-Python glTF 2.0 binary (.glb) reading, updating, and repacking.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
    HAS_SCIPY = True
except ImportError:
    ConvexHull = None
    QhullError = Exception
    HAS_SCIPY = False

from terra_forge.ai.vault import FloraType

logger = logging.getLogger("terra_forge.ai.normalizer")


@dataclass
class CylinderPrimitive:
    radius: float
    height: float
    center: List[float]  # [0.0, height / 2, 0.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AABBPrimitive:
    min_vec: List[float]  # [min_x, 0.0, min_z]
    max_vec: List[float]  # [max_x, height, max_z]
    size: List[float]     # [width, height, depth]
    center: List[float]   # [0.0, height / 2, 0.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConvexHullPrimitive:
    vertex_count: int
    simplex_count: int
    vertices: List[List[float]]
    simplices: List[List[int]]
    volume: float
    area: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollisionPrimitives:
    cylinder: CylinderPrimitive
    aabb: AABBPrimitive
    convex_hull: ConvexHullPrimitive

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cylinder": self.cylinder.to_dict(),
            "aabb": self.aabb.to_dict(),
            "convex_hull": self.convex_hull.to_dict(),
        }


@dataclass
class NormalizedMesh:
    """Result of geometry normalization."""
    vertices: np.ndarray
    triangles: np.ndarray
    original_bounds: Dict[str, List[float]]
    normalized_bounds: Dict[str, List[float]]
    scale_factor: float
    offset: List[float]
    target_height: float
    collision_primitives: CollisionPrimitives

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


class GeometryNormalizer:
    """
    Core geometry normalizer.
    Performs pivot normalization (min_y=0), horizontal centering, metric scaling,
    and collision primitives extraction.
    """

    def __init__(self, default_target_height: float = 5.0):
        self.default_target_height = default_target_height

    @staticmethod
    def normalize_vertices(
        vertices: np.ndarray,
        target_height: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transform 3D vertex positions [N, 3]:
        1. Shift horizontal center (X, Z) to (0.0, 0.0).
        2. Shift vertical base (Y) so min(Y) == 0.0.
        3. Uniformly scale so max(Y) - min(Y) == target_height (if target_height > 0).

        Returns:
            (normalized_vertices, transform_stats_dict)
        """
        pts = np.asarray(vertices, dtype=np.float32).copy()
        if len(pts) == 0:
            raise ValueError("Cannot normalize empty vertex array")

        x_min, y_min, z_min = np.min(pts, axis=0)
        x_max, y_max, z_max = np.max(pts, axis=0)

        raw_height = float(y_max - y_min)
        c_x = float((x_min + x_max) / 2.0)
        c_z = float((z_min + z_max) / 2.0)

        # Step 1 & 2: Shift horizontal center to 0, vertical base to 0
        pts[:, 0] -= c_x
        pts[:, 1] -= y_min
        pts[:, 2] -= c_z

        # Step 3: Uniform metric scaling
        scale = 1.0
        if target_height is not None and target_height > 0.0 and raw_height > 1e-6:
            scale = float(target_height / raw_height)
            pts *= scale

        new_x_min, new_y_min, new_z_min = np.min(pts, axis=0)
        new_x_max, new_y_max, new_z_max = np.max(pts, axis=0)

        stats = {
            "original_bounds": {
                "min": [float(x_min), float(y_min), float(z_min)],
                "max": [float(x_max), float(y_max), float(z_max)],
                "height": raw_height,
            },
            "normalized_bounds": {
                "min": [float(new_x_min), float(new_y_min), float(new_z_min)],
                "max": [float(new_x_max), float(new_y_max), float(new_z_max)],
                "height": float(new_y_max - new_y_min),
            },
            "scale_factor": scale,
            "offset": [c_x, float(y_min), c_z],
        }

        return pts, stats

    @staticmethod
    def extract_collision_primitives(
        vertices: np.ndarray,
        target_height: float,
        flora_type: Optional[FloraType] = None,
    ) -> CollisionPrimitives:
        """
        Extract analytical and polyhedral collision shapes from normalized vertices:
        1. Cylinder: radius sampled from lower trunk band (Y in [0, 0.25*H]), height H.
        2. AABB: Axis-aligned bounding box bounding all vertices.
        3. Convex Hull: Watertight convex polyhedron via scipy.spatial.ConvexHull.
        """
        pts = np.asarray(vertices, dtype=np.float32)
        x_min, y_min, z_min = np.min(pts, axis=0)
        x_max, y_max, z_max = np.max(pts, axis=0)
        h = float(y_max - y_min)
        if h <= 0.0:
            h = target_height

        # Cylinder radius: sample lower 20% of geometry (trunk / base)
        lower_mask = pts[:, 1] <= (0.20 * h)
        if np.any(lower_mask):
            lower_pts = pts[lower_mask]
            radii = np.sqrt(lower_pts[:, 0] ** 2 + lower_pts[:, 2] ** 2)
            cyl_radius = float(np.percentile(radii, 90))
        else:
            radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
            cyl_radius = float(np.percentile(radii, 75))

        # Enforce reasonable minimum and maximum bounds for stability
        max_extent = max(abs(float(x_max)), abs(float(x_min)), abs(float(z_max)), abs(float(z_min)))
        safe_max = max(float(max_extent), 1e-4)
        # Upper bound: cylinder radius must never exceed horizontal bounding extent
        cyl_radius = min(float(cyl_radius), safe_max)
        # Lower bound: enforce minimum positive radius for physics engine stability (at least 1e-4)
        min_radius = max(1e-4, min(0.01, safe_max * 0.1))
        cyl_radius = float(max(cyl_radius, min_radius))

        cylinder = CylinderPrimitive(
            radius=round(float(cyl_radius), 4),
            height=round(float(h), 4),
            center=[0.0, round(float(h) / 2.0, 4), 0.0],
        )

        aabb = AABBPrimitive(
            min_vec=[round(float(x_min), 4), 0.0, round(float(z_min), 4)],
            max_vec=[round(float(x_max), 4), round(float(h), 4), round(float(z_max), 4)],
            size=[round(float(x_max - x_min), 4), round(float(h), 4), round(float(z_max - z_min), 4)],
            center=[0.0, round(float(h) / 2.0, 4), 0.0],
        )

        # Convex Hull
        # If vertex count is large (> 48), subsample representative vertices
        if len(pts) > 48:
            # Pick bounding extrema + quantile slices + uniform stride
            extrema_idx = [
                int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0])),
                int(np.argmin(pts[:, 1])), int(np.argmax(pts[:, 1])),
                int(np.argmin(pts[:, 2])), int(np.argmax(pts[:, 2])),
            ]
            stride = max(1, len(pts) // 40)
            sample_idx = list(range(0, len(pts), stride))
            all_idx = list(set(extrema_idx + sample_idx))
            hull_input = pts[all_idx]
        else:
            hull_input = pts

        hull_computed = False
        if HAS_SCIPY and ConvexHull is not None and len(hull_input) >= 4:
            try:
                hull = ConvexHull(hull_input)
                hull_verts = hull.points[hull.vertices].tolist()
                hull_faces = hull.simplices.tolist()
                hull_volume = float(hull.volume)
                hull_area = float(hull.area)
                vertex_count = len(hull.vertices)
                simplex_count = len(hull.simplices)
                hull_computed = True
            except (QhullError, ValueError) as e:
                logger.warning("ConvexHull calculation failed (%s); falling back to bounding box hull", e)

        if not hull_computed:
            x0, x1 = float(x_min), float(x_max)
            y0, y1 = 0.0, max(float(h), 0.01)
            z0, z1 = float(z_min), float(z_max)
            if abs(x1 - x0) < 0.01:
                x0 -= 0.005
                x1 += 0.005
            if abs(z1 - z0) < 0.01:
                z0 -= 0.005
                z1 += 0.005
            hull_verts = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ]
            hull_faces = [
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1],
                [3, 2, 6], [3, 6, 7],
                [0, 3, 7], [0, 7, 4],
                [1, 5, 6], [1, 6, 2],
            ]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            dz = abs(z1 - z0)
            hull_volume = float(dx * dy * dz)
            hull_area = float(2.0 * (dx * dy + dx * dz + dy * dz))
            vertex_count = 8
            simplex_count = 12

        convex_hull = ConvexHullPrimitive(
            vertex_count=vertex_count,
            simplex_count=simplex_count,
            vertices=[[round(float(c), 4) for c in pt] for pt in hull_verts],
            simplices=[[int(idx) for idx in tri] for tri in hull_faces],
            volume=round(float(hull_volume), 4),
            area=round(float(hull_area), 4),
        )

        return CollisionPrimitives(
            cylinder=cylinder,
            aabb=aabb,
            convex_hull=convex_hull,
        )


# ---------------------------------------------------------------------------
# Pure-Python glTF 2.0 Binary (.glb) Parser & Repacker
# ---------------------------------------------------------------------------

GLB_MAGIC = 0x46546C67  # 'glTF'
GLB_JSON_CHUNK = 0x4E4F534A  # 'JSON'
GLB_BIN_CHUNK = 0x004E4942  # 'BIN\x00'


def parse_glb(glb_bytes: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Parse glTF 2.0 binary (.glb) into its JSON manifest dict and binary buffer."""
    if len(glb_bytes) < 12:
        raise ValueError("Invalid GLB: file smaller than 12 bytes header")

    magic, version, length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header: magic={magic!r}, version={version}")

    offset = 12
    json_dict: Optional[Dict[str, Any]] = None
    bin_data: bytes = b""

    while offset < len(glb_bytes):
        if offset + 8 > len(glb_bytes):
            break
        chunk_len, chunk_type = struct.unpack("<II", glb_bytes[offset:offset + 8])
        chunk_data = glb_bytes[offset + 8:offset + 8 + chunk_len]
        offset += 8 + chunk_len

        if chunk_type == GLB_JSON_CHUNK:
            json_text = chunk_data.decode("utf-8").rstrip(" ")
            json_dict = json.loads(json_text)
        elif chunk_type == GLB_BIN_CHUNK:
            bin_data = chunk_data

    if json_dict is None:
        raise ValueError("Invalid GLB: missing JSON chunk")

    return json_dict, bin_data


def repack_glb(json_dict: Dict[str, Any], bin_data: bytes) -> bytes:
    """Pack glTF manifest dict and binary buffer into valid glTF 2.0 binary (.glb)."""
    json_bytes = json.dumps(json_dict, separators=(",", ":")).encode("utf-8")
    # Pad JSON with spaces to 4-byte multiple
    pad_json = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad_json

    # Pad BIN with null bytes to 4-byte multiple
    pad_bin = (4 - (len(bin_data) % 4)) % 4
    padded_bin = bin_data + (b"\x00" * pad_bin)

    json_chunk_len = len(json_bytes)
    bin_chunk_len = len(padded_bin)
    total_length = 12 + 8 + json_chunk_len + (8 + bin_chunk_len if bin_chunk_len > 0 else 0)

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    chunk0 = struct.pack("<II", json_chunk_len, GLB_JSON_CHUNK) + json_bytes

    if bin_chunk_len > 0:
        chunk1 = struct.pack("<II", bin_chunk_len, GLB_BIN_CHUNK) + padded_bin
        return header + chunk0 + chunk1
    return header + chunk0


def extract_mesh_from_glb(glb_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertex positions (float32 [N, 3]) and triangle indices ([M, 3]) from GLB."""
    json_dict, bin_data = parse_glb(glb_bytes)

    # Locate POSITION accessor
    pos_accessor_idx: Optional[int] = None
    indices_accessor_idx: Optional[int] = None

    for mesh in json_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" in attrs:
                pos_accessor_idx = attrs["POSITION"]
            if "indices" in prim:
                indices_accessor_idx = prim["indices"]
            if pos_accessor_idx is not None:
                break
        if pos_accessor_idx is not None:
            break

    if pos_accessor_idx is None:
        raise ValueError("No POSITION attribute found in GLB primitives")

    accessors = json_dict["accessors"]
    buffer_views = json_dict["bufferViews"]

    # Read POSITION
    pos_acc = accessors[pos_accessor_idx]
    pos_bv = buffer_views[pos_acc["bufferView"]]
    pos_offset = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
    pos_count = pos_acc["count"]
    pos_bytes = bin_data[pos_offset:pos_offset + (pos_count * 12)]
    vertices = np.frombuffer(pos_bytes, dtype=np.float32).reshape((pos_count, 3))

    # Read Indices
    if indices_accessor_idx is not None:
        idx_acc = accessors[indices_accessor_idx]
        idx_bv = buffer_views[idx_acc["bufferView"]]
        idx_offset = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)
        idx_count = idx_acc["count"]
        comp_type = idx_acc["componentType"]

        if comp_type == 5121:  # UNSIGNED_BYTE
            dtype = np.uint8
            item_size = 1
        elif comp_type == 5123:  # UNSIGNED_SHORT
            dtype = np.uint16
            item_size = 2
        elif comp_type == 5125:  # UNSIGNED_INT
            dtype = np.uint32
            item_size = 4
        else:
            dtype = np.uint16
            item_size = 2

        idx_bytes = bin_data[idx_offset:idx_offset + (idx_count * item_size)]
        indices = np.frombuffer(idx_bytes, dtype=dtype)
        triangles = indices[: (len(indices) // 3) * 3].reshape((-1, 3))
    else:
        # Non-indexed triangles: sequential vertices
        triangles = np.arange(len(vertices) // 3 * 3, dtype=np.uint32).reshape((-1, 3))

    return vertices, triangles


def build_glb_from_mesh(vertices: np.ndarray, triangles: np.ndarray) -> bytes:
    """Create a minimal valid glTF 2.0 binary (.glb) from vertices and triangle indices."""
    v_arr = np.asarray(vertices, dtype=np.float32)
    i_arr = np.asarray(triangles.flatten(), dtype=np.uint32 if len(vertices) > 65535 else np.uint16)

    v_bytes = v_arr.tobytes()
    i_bytes = i_arr.tobytes()
    pad_i = (4 - (len(i_bytes) % 4)) % 4
    i_bytes += b"\x00" * pad_i

    bin_data = v_bytes + i_bytes
    idx_comp_type = 5125 if i_arr.dtype == np.uint32 else 5123

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "TerraForge Normalizer"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4,  # TRIANGLES
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(v_arr),
                "type": "VEC3",
                "min": [float(x) for x in np.min(v_arr, axis=0)],
                "max": [float(x) for x in np.max(v_arr, axis=0)],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": idx_comp_type,
                "count": len(i_arr),
                "type": "SCALAR",
                "min": [int(np.min(i_arr))],
                "max": [int(np.max(i_arr))],
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(v_bytes),
                "target": 34962,  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(v_bytes),
                "byteLength": len(i_bytes),
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }

    return repack_glb(gltf_dict, bin_data)


def normalize_glb_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_height: float = 5.0,
    flora_type: Optional[FloraType] = None,
) -> NormalizedMesh:
    """
    Read an input GLB, normalize its geometry (pivot at bottom min_y=0, centered X/Z, scaled to target_height),
    update accessor min/max bounds, extract collision primitives, and write the output GLB.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(in_p, "rb") as f:
        raw_glb = f.read()

    json_dict, bin_data = parse_glb(raw_glb)
    vertices, triangles = extract_mesh_from_glb(raw_glb)

    normalizer = GeometryNormalizer(default_target_height=target_height)
    norm_verts, stats = normalizer.normalize_vertices(vertices, target_height=target_height)
    colliders = normalizer.extract_collision_primitives(norm_verts, target_height=target_height, flora_type=flora_type)

    # Update bin_data with normalized vertices
    pos_accessor_idx = None
    for mesh in json_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" in attrs:
                pos_accessor_idx = attrs["POSITION"]
                break
        if pos_accessor_idx is not None:
            break

    if pos_accessor_idx is not None:
        pos_acc = json_dict["accessors"][pos_accessor_idx]
        pos_bv = json_dict["bufferViews"][pos_acc["bufferView"]]
        pos_offset = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)

        # In-place overwrite or buffer update
        bin_bytearray = bytearray(bin_data)
        norm_v_bytes = norm_verts.astype(np.float32).tobytes()
        bin_bytearray[pos_offset:pos_offset + len(norm_v_bytes)] = norm_v_bytes

        # Update accessor min/max bounds
        pos_acc["min"] = [float(x) for x in np.min(norm_verts, axis=0)]
        pos_acc["max"] = [float(x) for x in np.max(norm_verts, axis=0)]

        repacked = repack_glb(json_dict, bytes(bin_bytearray))
    else:
        # Rebuild fresh glb
        repacked = build_glb_from_mesh(norm_verts, triangles)

    with open(out_p, "wb") as f:
        f.write(repacked)

    logger.info(f"Normalized GLB saved to {out_p} (vertices: {len(norm_verts)}, target_h: {target_height})")

    return NormalizedMesh(
        vertices=norm_verts,
        triangles=triangles,
        original_bounds=stats["original_bounds"],
        normalized_bounds=stats["normalized_bounds"],
        scale_factor=stats["scale_factor"],
        offset=stats["offset"],
        target_height=target_height,
        collision_primitives=colliders,
    )


def create_procedural_primitive(
    primitive_type: str = "pine",
    target_height: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clean procedural 3D primitives (pure numpy, no Blender required)
    for testing, offline fallback, and pipeline validation.
    """
    p_type = primitive_type.lower()

    if "pine" in p_type or "conifer" in p_type:
        # Tiered conical boughs over central trunk
        verts = [[0.0, -0.5, 0.0], [0.0, 3.0, 0.0]]  # trunk
        faces = []
        n_tiers = 4
        segs = 8
        for t in range(n_tiers):
            tier_y = 0.5 + t * 0.7
            tier_r = 1.4 - t * 0.3
            top_y = tier_y + 0.9
            top_idx = len(verts)
            verts.append([0.0, top_y, 0.0])

            base_start = len(verts)
            for i in range(segs):
                angle = 2.0 * np.pi * i / segs
                verts.append([tier_r * np.cos(angle), tier_y, tier_r * np.sin(angle)])

            for i in range(segs):
                nxt = (i + 1) % segs
                faces.append([top_idx, base_start + i, base_start + nxt])
                # Base closure
                faces.append([top_idx, base_start + nxt, base_start + i])

        v_arr = np.array(verts, dtype=np.float32)
        i_arr = np.array(faces, dtype=np.uint32)
        return v_arr, i_arr

    elif "rock" in p_type or "boulder" in p_type:
        # Deformed icosphere
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        v = [
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ]
        f = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        v_arr = np.array(v, dtype=np.float32)
        # Asymmetric noise
        v_arr[:, 0] *= 1.4
        v_arr[:, 1] *= 0.8
        v_arr[:, 2] *= 1.1
        i_arr = np.array(f, dtype=np.uint32)
        return v_arr, i_arr

    else:
        # Generic pyramidal tree
        v = [
            [-0.8, -0.5, -0.8], [0.8, -0.5, -0.8], [0.8, -0.5, 0.8], [-0.8, -0.5, 0.8],
            [0.0, 2.5, 0.0]
        ]
        f = [
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 2, 1], [0, 3, 2]
        ]
        return np.array(v, dtype=np.float32), np.array(f, dtype=np.uint32)
