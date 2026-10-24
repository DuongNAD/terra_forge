"""
Continuous Open-World Heightfield & Chunked Mesh Generator for Terra Forge.

Generates seamless top-surface only terrain meshes without diorama slab walls
or bottom plate constraints. Supports:
- Full continuous heightfield mesh synthesis
- Seamless NxM chunking with shared boundary vertices (no seams/T-junctions)
- Pure-Python/NumPy direct headless export to:
    1. Raw binary heightfield (.bin / .f32)
    2. Wavefront OBJ (.obj)
    3. Binary glTF 2.0 (.glb) with POSITION, NORMAL, TEXCOORD_0 accessors
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# -----------------------------------------------------------------------------
# GLB Binary Encoding Utilities (Pure Python / Zero external dependencies)
# -----------------------------------------------------------------------------
def _pack_glb(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    generator_name: str = "terra_forge.core.open_mesh",
    colors: Optional[np.ndarray] = None,
) -> bytes:
    """
    Serializes mesh geometry into a self-contained glTF 2.0 Binary (.glb) blob.
    Conforms to Khronos glTF 2.0 Specification.
    Optionally embeds per-vertex RGBA or RGB color attributes (COLOR_0).
    """
    v_arr = np.ascontiguousarray(vertices, dtype="<f4")
    n_arr = np.ascontiguousarray(normals, dtype="<f4")
    uv_arr = np.ascontiguousarray(uvs, dtype="<f4")
    i_arr = np.ascontiguousarray(indices, dtype="<u4")

    v_count = len(v_arr)
    i_count = len(i_arr)

    v_bytes = v_arr.tobytes()
    n_bytes = n_arr.tobytes()
    uv_bytes = uv_arr.tobytes()
    i_bytes = i_arr.tobytes()

    v_len = len(v_bytes)
    n_len = len(n_bytes)
    uv_len = len(uv_bytes)
    i_len = len(i_bytes)

    # Process optional vertex colors
    c_arr: Optional[np.ndarray] = None
    c_bytes = b""
    c_len = 0
    c_type = "VEC4"
    min_c = [0.0, 0.0, 0.0, 0.0]
    max_c = [1.0, 1.0, 1.0, 1.0]

    if colors is not None:
        c_raw = np.asarray(colors, dtype="<f4")
        if c_raw.ndim == 3:
            c_raw = c_raw.reshape(-1, c_raw.shape[-1])
        elif c_raw.ndim == 1:
            comp = len(c_raw) // v_count if v_count > 0 else 4
            c_raw = c_raw.reshape(v_count, comp)
        if len(c_raw) != v_count:
            raise ValueError(f"Color count {len(c_raw)} does not match vertex count {v_count}")
        num_comp = c_raw.shape[-1]
        if num_comp not in (3, 4):
            raise ValueError(f"Colors must have 3 (RGB) or 4 (RGBA) channels, got {num_comp}")
        c_type = "VEC4" if num_comp == 4 else "VEC3"
        c_arr = np.ascontiguousarray(np.clip(c_raw, 0.0, 1.0), dtype="<f4")
        c_bytes = c_arr.tobytes()
        c_len = len(c_bytes)
        min_c = [float(x) for x in np.min(c_arr, axis=0)] if v_count > 0 else [0.0] * num_comp
        max_c = [float(x) for x in np.max(c_arr, axis=0)] if v_count > 0 else [1.0] * num_comp

    # Calculate 4-byte aligned offsets
    def align4(length: int) -> int:
        return (length + 3) & ~3

    v_off = 0
    n_off = align4(v_off + v_len)
    uv_off = align4(n_off + n_len)

    if c_arr is not None:
        c_off = align4(uv_off + uv_len)
        i_off = align4(c_off + c_len)
        total_bin_len = align4(i_off + i_len)

        bin_data = bytearray(total_bin_len)
        bin_data[v_off : v_off + v_len] = v_bytes
        bin_data[n_off : n_off + n_len] = n_bytes
        bin_data[uv_off : uv_off + uv_len] = uv_bytes
        bin_data[c_off : c_off + c_len] = c_bytes
        bin_data[i_off : i_off + i_len] = i_bytes
    else:
        c_off = 0
        i_off = align4(uv_off + uv_len)
        total_bin_len = align4(i_off + i_len)

        bin_data = bytearray(total_bin_len)
        bin_data[v_off : v_off + v_len] = v_bytes
        bin_data[n_off : n_off + n_len] = n_bytes
        bin_data[uv_off : uv_off + uv_len] = uv_bytes
        bin_data[i_off : i_off + i_len] = i_bytes

    # Bounding box for POSITION accessor
    min_v = [float(x) for x in np.min(v_arr, axis=0)] if v_count > 0 else [0.0, 0.0, 0.0]
    max_v = [float(x) for x in np.max(v_arr, axis=0)] if v_count > 0 else [0.0, 0.0, 0.0]

    primitives_attrs: Dict[str, int] = {
        "POSITION": 0,
        "NORMAL": 1,
        "TEXCOORD_0": 2,
    }

    accessors: List[Dict[str, Any]] = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": v_count,
            "type": "VEC3",
            "min": min_v,
            "max": max_v,
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": v_count,
            "type": "VEC3",
        },
        {
            "bufferView": 2,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": v_count,
            "type": "VEC2",
        },
    ]

    buffer_views: List[Dict[str, Any]] = [
        {
            "buffer": 0,
            "byteOffset": v_off,
            "byteLength": v_len,
            "target": 34962,  # ARRAY_BUFFER
        },
        {
            "buffer": 0,
            "byteOffset": n_off,
            "byteLength": n_len,
            "target": 34962,  # ARRAY_BUFFER
        },
        {
            "buffer": 0,
            "byteOffset": uv_off,
            "byteLength": uv_len,
            "target": 34962,  # ARRAY_BUFFER
        },
    ]

    if c_arr is not None:
        c_view_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": c_off,
            "byteLength": c_len,
            "target": 34962,  # ARRAY_BUFFER
        })
        c_acc_idx = len(accessors)
        accessors.append({
            "bufferView": c_view_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": v_count,
            "type": c_type,
            "min": min_c,
            "max": max_c,
        })
        primitives_attrs["COLOR_0"] = c_acc_idx

    indices_view_idx = len(buffer_views)
    buffer_views.append({
        "buffer": 0,
        "byteOffset": i_off,
        "byteLength": i_len,
        "target": 34963,  # ELEMENT_ARRAY_BUFFER
    })

    indices_acc_idx = len(accessors)
    accessors.append({
        "bufferView": indices_view_idx,
        "byteOffset": 0,
        "componentType": 5125,  # UNSIGNED_INT
        "count": i_count,
        "type": "SCALAR",
    })

    gltf_dict: Dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": generator_name,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": primitives_attrs,
                        "indices": indices_acc_idx,
                        "mode": 4,  # TRIANGLES
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": total_bin_len}],
    }

    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    # Align JSON chunk to 4 bytes using trailing spaces
    json_pad_len = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad_len

    # GLB Header: magic=0x46546C67 ("glTF"), version=2, length
    total_glb_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_data))

    header = struct.pack("<4sII", b"glTF", 2, total_glb_len)
    chunk0_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    chunk1_header = struct.pack("<I4s", len(bin_data), b"BIN\x00")

    return header + chunk0_header + json_bytes + chunk1_header + bytes(bin_data)


# -----------------------------------------------------------------------------
# MeshChunk Representation
# -----------------------------------------------------------------------------
class MeshChunk:
    """Individual tile/chunk of an open-world heightfield mesh."""

    def __init__(
        self,
        chunk_x: int,
        chunk_y: int,
        vertices: np.ndarray,
        normals: np.ndarray,
        uvs: np.ndarray,
        indices: np.ndarray,
        aabb_min: Tuple[float, float, float],
        aabb_max: Tuple[float, float, float],
        colors: Optional[np.ndarray] = None,
    ):
        self.chunk_x = int(chunk_x)
        self.chunk_y = int(chunk_y)
        self.vertices = vertices
        self.normals = normals
        self.uvs = uvs
        self.indices = indices
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.colors = colors

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def export_obj(self, path: Union[str, Path]) -> Path:
        """Exports chunk to Wavefront OBJ format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# terra_forge mesh chunk ({self.chunk_x}, {self.chunk_y})"]
        for vx, vy, vz in self.vertices:
            lines.append(f"v {vx:.5f} {vy:.5f} {vz:.5f}")
        for nx, ny, nz in self.normals:
            lines.append(f"vn {nx:.5f} {ny:.5f} {nz:.5f}")
        for u, v in self.uvs:
            lines.append(f"vt {u:.5f} {v:.5f}")
        for i in range(0, len(self.indices), 3):
            i0 = self.indices[i] + 1
            i1 = self.indices[i + 1] + 1
            i2 = self.indices[i + 2] + 1
            lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def export_glb(self, path: Union[str, Path], colors: Optional[np.ndarray] = None) -> Path:
        """Exports chunk to binary glTF 2.0 (.glb) format with optional vertex colors."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        active_colors = colors if colors is not None else self.colors
        data = _pack_glb(
            self.vertices,
            self.normals,
            self.uvs,
            self.indices,
            generator_name=f"terra_forge.core.open_mesh.chunk_{self.chunk_x}_{self.chunk_y}",
            colors=active_colors,
        )
        p.write_bytes(data)
        return p


# -----------------------------------------------------------------------------
# OpenWorldMesh Generator
# -----------------------------------------------------------------------------
class OpenWorldMesh:
    """
    Continuous open-world heightfield mesh generator.
    Generates top-surface only geometry (no diorama cutaway walls or baseplate).
    """

    def __init__(
        self,
        elevation: Union[np.ndarray, Any],
        world_min_xz: float = -100.0,
        world_max_xz: float = 100.0,
        world_min_y: float = 0.0,
        world_max_y: float = 10.0,
        y_up: bool = True,
        colors: Optional[np.ndarray] = None,
    ):
        """
        Initializes the open-world mesh.

        Args:
            elevation: 2D array [H, W] or 1D array of elevation values.
                       If values are normalized in [0, 1], they scale to [world_min_y, world_max_y].
            world_min_xz: World minimum coordinate for X and Z axes (canonical -100.0).
            world_max_xz: World maximum coordinate for X and Z axes (canonical 100.0).
            world_min_y: Minimum elevation in world units (canonical 0.0).
            world_max_y: Maximum elevation in world units (canonical 10.0).
            y_up: True for Y-up standard (glTF / Bevy / Three.js). False for Z-up.
            colors: Optional per-vertex RGBA or RGB color array or VertexColorMasks.
        """
        if hasattr(elevation, "elevation"):
            # WorldArtifact
            elev_raw = getattr(elevation, "elevation")
            w = int(getattr(elevation, "width", 256))
            h = int(getattr(elevation, "height", 256))
            elev_grid = np.asarray(elev_raw, dtype=np.float32).reshape((h, w))
        elif hasattr(elevation, "Z"):
            # Heightfield2D
            elev_grid = np.asarray(getattr(elevation, "Z"), dtype=np.float32)
        else:
            elev_grid = np.asarray(elevation, dtype=np.float32)
            if elev_grid.ndim == 1:
                side = int(np.sqrt(len(elev_grid)))
                elev_grid = elev_grid.reshape((side, side))

        self.height, self.width = elev_grid.shape
        self.world_min_xz = float(world_min_xz)
        self.world_max_xz = float(world_max_xz)
        self.world_min_y = float(world_min_y)
        self.world_max_y = float(world_max_y)
        self.y_up = bool(y_up)

        # Rescale elevation if normalized [0, 1]
        z_min = float(np.min(elev_grid))
        z_max = float(np.max(elev_grid))
        if 0.0 <= z_min and z_max <= 1.0001:
            self.elevation = elev_grid * (self.world_max_y - self.world_min_y) + self.world_min_y
        else:
            self.elevation = elev_grid

        self._build_geometry()

        self.colors: Optional[np.ndarray] = None
        if colors is not None:
            self.set_vertex_colors(colors)

    def set_vertex_colors(self, colors: Union[np.ndarray, Any]) -> None:
        """
        Embeds per-vertex color/mask attributes (RGBA or RGB) into the mesh.
        Supports:
          - Flat array [N, 4] or [N, 3] matching vertex count.
          - 2D grid array [H, W, 4] or [H, W, 3].
          - VertexColorMasks object from terra_forge.ecology.dynamics.
        """
        if hasattr(colors, "rgba"):
            c_raw = getattr(colors, "rgba")
        else:
            c_raw = colors
        c_arr = np.asarray(c_raw, dtype=np.float32)
        if c_arr.ndim == 3 and c_arr.shape[0] == self.height and c_arr.shape[1] == self.width:
            c_arr = c_arr.reshape(-1, c_arr.shape[2])
        elif c_arr.ndim == 1:
            comp = len(c_raw) // len(self.vertices) if len(self.vertices) > 0 else 4
            c_arr = c_arr.reshape(len(self.vertices), comp)
        if len(c_arr) != len(self.vertices):
            raise ValueError(
                f"Vertex color count {len(c_arr)} does not match mesh vertex count {len(self.vertices)}"
            )
        self.colors = np.ascontiguousarray(np.clip(c_arr, 0.0, 1.0), dtype=np.float32)

    def _build_geometry(self) -> None:
        """Synthesizes vertex coordinates, normals, UVs, and triangle face indices."""
        W = self.width
        H = self.height

        x_coords = np.linspace(self.world_min_xz, self.world_max_xz, W, dtype=np.float32)
        z_coords = np.linspace(self.world_min_xz, self.world_max_xz, H, dtype=np.float32)
        grid_x, grid_z = np.meshgrid(x_coords, z_coords)
        grid_y = self.elevation.astype(np.float32)

        # Central difference slope calculation for per-vertex normals
        dx = (self.world_max_xz - self.world_min_xz) / max(1, W - 1)
        dz = (self.world_max_xz - self.world_min_xz) / max(1, H - 1)

        grad_y_z, grad_y_x = np.gradient(grid_y, dz, dx)
        # Normal vector: (-dy/dx, 1, -dy/dz) normalized
        norm_x = -grad_y_x
        norm_y = np.ones_like(grid_y)
        norm_z = -grad_y_z
        norm_mag = np.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z)
        norm_x /= norm_mag
        norm_y /= norm_mag
        norm_z /= norm_mag

        # UV coordinates: [0, 1] along both axes
        u_coords = np.linspace(0.0, 1.0, W, dtype=np.float32)
        v_coords = np.linspace(0.0, 1.0, H, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(u_coords, v_coords)

        if self.y_up:
            # Standard glTF / Three.js / Bevy: X right, Y up, Z forward/depth
            vertices = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
            normals = np.stack([norm_x, norm_y, norm_z], axis=-1).reshape(-1, 3)
        else:
            # Z-up format (Blender internal / standard CAD)
            vertices = np.stack([grid_x, grid_z, grid_y], axis=-1).reshape(-1, 3)
            normals = np.stack([norm_x, norm_z, norm_y], axis=-1).reshape(-1, 3)

        uvs = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 2)

        # Triangle face indices (CCW winding looking from top)
        # Vectorized quad generation via NumPy meshgrid (identical CCW winding)
        iy, ix = np.meshgrid(
            np.arange(H - 1, dtype=np.uint32),
            np.arange(W - 1, dtype=np.uint32),
            indexing="ij",
        )
        v00 = iy * np.uint32(W) + ix
        v10 = v00 + np.uint32(1)
        v01 = v00 + np.uint32(W)
        v11 = v01 + np.uint32(1)

        quads = np.empty((H - 1, W - 1, 2, 3), dtype=np.uint32)
        # Triangle 1: (v00, v01, v10)
        quads[:, :, 0, 0] = v00
        quads[:, :, 0, 1] = v01
        quads[:, :, 0, 2] = v10
        # Triangle 2: (v10, v01, v11)
        quads[:, :, 1, 0] = v10
        quads[:, :, 1, 1] = v01
        quads[:, :, 1, 2] = v11

        self.vertices = vertices
        self.normals = normals
        self.uvs = uvs
        self.indices = quads.ravel()

        self.aabb_min = (
            float(np.min(vertices[:, 0])),
            float(np.min(vertices[:, 1])),
            float(np.min(vertices[:, 2])),
        )
        self.aabb_max = (
            float(np.max(vertices[:, 0])),
            float(np.max(vertices[:, 1])),
            float(np.max(vertices[:, 2])),
        )

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def to_mesh_data(self) -> Dict[str, Any]:
        """Returns mesh dictionary matching terra_forge builder conventions."""
        data: Dict[str, Any] = {
            "vertices": [tuple(v) for v in self.vertices],
            "normals": [tuple(n) for n in self.normals],
            "uvs": [tuple(uv) for uv in self.uvs],
            "faces": [
                [int(self.indices[i]), int(self.indices[i + 1]), int(self.indices[i + 2])]
                for i in range(0, len(self.indices), 3)
            ],
            "aabb_min": self.aabb_min,
            "aabb_max": self.aabb_max,
        }
        if self.colors is not None:
            data["colors"] = [tuple(c) for c in self.colors]
        return data

    def create_chunks(self, chunks_x: int = 4, chunks_y: int = 4) -> List[MeshChunk]:
        """
        Partitions the continuous heightfield into a grid of seamless chunks.
        Adjacent chunks share boundary vertices for continuous stitching.
        """
        chunks: List[MeshChunk] = []
        W = self.width
        H = self.height

        # Grid cells per chunk
        cells_x = (W - 1) // chunks_x
        cells_y = (H - 1) // chunks_y

        for cy in range(chunks_y):
            iy_start = cy * cells_y
            iy_end = H if cy == chunks_y - 1 else (cy + 1) * cells_y + 1

            for cx in range(chunks_x):
                ix_start = cx * cells_x
                ix_end = W if cx == chunks_x - 1 else (cx + 1) * cells_x + 1

                chunk_h = iy_end - iy_start
                chunk_w = ix_end - ix_start

                # Extract sub-arrays
                sub_elev = self.elevation[iy_start:iy_end, ix_start:ix_end]

                # Map vertices
                v_list: List[Tuple[float, float, float]] = []
                n_list: List[Tuple[float, float, float]] = []
                uv_list: List[Tuple[float, float]] = []
                c_list: Optional[List[Tuple[float, ...]]] = [] if self.colors is not None else None
                i_list: List[int] = []

                for y_idx in range(chunk_h):
                    global_y = iy_start + y_idx
                    for x_idx in range(chunk_w):
                        global_x = ix_start + x_idx
                        global_idx = global_y * W + global_x
                        v_list.append(tuple(self.vertices[global_idx]))
                        n_list.append(tuple(self.normals[global_idx]))
                        uv_list.append(tuple(self.uvs[global_idx]))
                        if self.colors is not None and c_list is not None:
                            c_list.append(tuple(self.colors[global_idx]))

                for y_idx in range(chunk_h - 1):
                    row0 = y_idx * chunk_w
                    row1 = (y_idx + 1) * chunk_w
                    for x_idx in range(chunk_w - 1):
                        v00 = row0 + x_idx
                        v10 = row0 + x_idx + 1
                        v01 = row1 + x_idx
                        v11 = row1 + x_idx + 1
                        i_list.extend([v00, v01, v10, v10, v01, v11])

                v_arr = np.array(v_list, dtype=np.float32)
                chunk_aabb_min = (
                    float(np.min(v_arr[:, 0])),
                    float(np.min(v_arr[:, 1])),
                    float(np.min(v_arr[:, 2])),
                )
                chunk_aabb_max = (
                    float(np.max(v_arr[:, 0])),
                    float(np.max(v_arr[:, 1])),
                    float(np.max(v_arr[:, 2])),
                )

                chunk = MeshChunk(
                    chunk_x=cx,
                    chunk_y=cy,
                    vertices=v_arr,
                    normals=np.array(n_list, dtype=np.float32),
                    uvs=np.array(uv_list, dtype=np.float32),
                    indices=np.array(i_list, dtype=np.uint32),
                    aabb_min=chunk_aabb_min,
                    aabb_max=chunk_aabb_max,
                    colors=np.array(c_list, dtype=np.float32) if c_list is not None else None,
                )
                chunks.append(chunk)

        return chunks

    def export_binary_heightfield(self, path: Union[str, Path]) -> Path:
        """Exports raw 32-bit float elevation values in row-major order."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(np.ascontiguousarray(self.elevation, dtype="<f4").tobytes())
        return p

    def export_obj(self, path: Union[str, Path]) -> Path:
        """Exports full continuous mesh to Wavefront OBJ format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# terra_forge continuous open-world mesh ({self.width}x{self.height})"]
        for vx, vy, vz in self.vertices:
            lines.append(f"v {vx:.5f} {vy:.5f} {vz:.5f}")
        for nx, ny, nz in self.normals:
            lines.append(f"vn {nx:.5f} {ny:.5f} {nz:.5f}")
        for u, v in self.uvs:
            lines.append(f"vt {u:.5f} {v:.5f}")
        for i in range(0, len(self.indices), 3):
            i0 = self.indices[i] + 1
            i1 = self.indices[i + 1] + 1
            i2 = self.indices[i + 2] + 1
            lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def export_glb(self, path: Union[str, Path], colors: Optional[np.ndarray] = None) -> Path:
        """Exports full continuous mesh to binary glTF 2.0 (.glb) format with optional vertex colors."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        active_colors = colors if colors is not None else self.colors
        glb_data = _pack_glb(
            self.vertices,
            self.normals,
            self.uvs,
            self.indices,
            generator_name="terra_forge.core.open_mesh.OpenWorldMesh",
            colors=active_colors,
        )
        p.write_bytes(glb_data)
        return p

    @classmethod
    def from_world_artifact(
        cls,
        artifact: Any,
        world_min_xz: float = -100.0,
        world_max_xz: float = 100.0,
        world_min_y: float = 0.0,
        world_max_y: float = 10.0,
        y_up: bool = True,
        colors: Optional[np.ndarray] = None,
    ) -> OpenWorldMesh:
        """Factory creating OpenWorldMesh from a WorldArtifact."""
        return cls(
            elevation=artifact,
            world_min_xz=world_min_xz,
            world_max_xz=world_max_xz,
            world_min_y=world_min_y,
            world_max_y=world_max_y,
            y_up=y_up,
            colors=colors,
        )

    @classmethod
    def from_heightfield(
        cls,
        hf: Any,
        world_min_xz: float = -100.0,
        world_max_xz: float = 100.0,
        world_min_y: float = 0.0,
        world_max_y: float = 10.0,
        y_up: bool = True,
        colors: Optional[np.ndarray] = None,
    ) -> OpenWorldMesh:
        """Factory creating OpenWorldMesh from a Heightfield2D."""
        return cls(
            elevation=hf,
            world_min_xz=world_min_xz,
            world_max_xz=world_max_xz,
            world_min_y=world_min_y,
            world_max_y=world_max_y,
            y_up=y_up,
            colors=colors,
        )
