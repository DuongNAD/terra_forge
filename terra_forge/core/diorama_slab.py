"""
Watertight Diorama Slab Mesh Synthesizer.
Generates complete closed manifold geometry:
Top surface + 4 vertical cutaway sides + bottom baseplate at z_base.
Calculates geological strata vertex colors.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any
import numpy as np
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import DioramaBounds


class DioramaSlabBuilder:
    def __init__(self, hf: Heightfield2D, bounds: DioramaBounds):
        self.hf = hf
        self.bounds = bounds

    def build_mesh_data(self) -> Dict[str, Any]:
        """
        Builds raw vertex, face, and vertex color arrays for the watertight diorama slab.
        Returns:
            {
                "vertices": List[Tuple[float, float, float]],
                "faces": List[List[int]],
                "vertex_colors": List[Tuple[float, float, float, float]],
            }
        """
        Z = self.hf.Z
        res_x = self.hf.res_x
        res_y = self.hf.res_y
        z_base = self.bounds.z_base

        vertices: List[Tuple[float, float, float]] = []
        faces: List[List[int]] = []

        # 1. Top Surface Grid Vertices
        # Index formula: idx = y * res_x + x
        for y_idx in range(res_y):
            for x_idx in range(res_x):
                wx, wy = self.hf.grid_to_world(x_idx, y_idx)
                wz = float(Z[y_idx, x_idx])
                vertices.append((wx, wy, wz))

        # Top Surface Quads
        for y_idx in range(res_y - 1):
            for x_idx in range(res_x - 1):
                v0 = y_idx * res_x + x_idx
                v1 = y_idx * res_x + (x_idx + 1)
                v2 = (y_idx + 1) * res_x + (x_idx + 1)
                v3 = (y_idx + 1) * res_x + x_idx
                # Counter-clockwise for upwards normal
                faces.append([v0, v1, v2, v3])

        # 2. Bottom Baseplate Grid Vertices
        # Index formula: base_offset + y * res_x + x
        base_offset = len(vertices)
        for y_idx in range(res_y):
            for x_idx in range(res_x):
                wx, wy = self.hf.grid_to_world(x_idx, y_idx)
                vertices.append((wx, wy, z_base))

        # Bottom Surface Quads (facing downwards: clockwise from top)
        for y_idx in range(res_y - 1):
            for x_idx in range(res_x - 1):
                b0 = base_offset + y_idx * res_x + x_idx
                b1 = base_offset + y_idx * res_x + (x_idx + 1)
                b2 = base_offset + (y_idx + 1) * res_x + (x_idx + 1)
                b3 = base_offset + (y_idx + 1) * res_x + x_idx
                # Face pointing downwards
                faces.append([b0, b3, b2, b1])

        # 3. Four Sheared Vertical Cutaway Sides
        # South Side (y = 0): connect top row (y=0) to base row (y=0)
        for x_idx in range(res_x - 1):
            top_curr = 0 * res_x + x_idx
            top_next = 0 * res_x + (x_idx + 1)
            bot_curr = base_offset + 0 * res_x + x_idx
            bot_next = base_offset + 0 * res_x + (x_idx + 1)
            # Facing -Y (South)
            faces.append([top_curr, bot_curr, bot_next, top_next])

        # North Side (y = res_y - 1): connect top row to base row
        ny = res_y - 1
        for x_idx in range(res_x - 1):
            top_curr = ny * res_x + x_idx
            top_next = ny * res_x + (x_idx + 1)
            bot_curr = base_offset + ny * res_x + x_idx
            bot_next = base_offset + ny * res_x + (x_idx + 1)
            # Facing +Y (North)
            faces.append([top_curr, top_next, bot_next, bot_curr])

        # West Side (x = 0): connect top col (x=0) to base col (x=0)
        for y_idx in range(res_y - 1):
            top_curr = y_idx * res_x + 0
            top_next = (y_idx + 1) * res_x + 0
            bot_curr = base_offset + y_idx * res_x + 0
            bot_next = base_offset + (y_idx + 1) * res_x + 0
            # Facing -X (West)
            faces.append([top_curr, top_next, bot_next, bot_curr])

        # East Side (x = res_x - 1): connect top col to base col
        ex = res_x - 1
        for y_idx in range(res_y - 1):
            top_curr = y_idx * res_x + ex
            top_next = (y_idx + 1) * res_x + ex
            bot_curr = base_offset + y_idx * res_x + ex
            bot_next = base_offset + (y_idx + 1) * res_x + ex
            # Facing +X (East)
            faces.append([top_curr, bot_curr, bot_next, top_next])

        # 4. Geological Strata Vertex Colors (COLOR_0)
        # Calculates rock stratification bands along the vertical cutaways
        colors: List[Tuple[float, float, float, float]] = []
        for vx, vy, vz in vertices:
            rel_z = (vz - z_base) / max(1.0, 35.0 - z_base)
            # Strata banding using sine waves
            band = 0.5 + 0.5 * math.sin(vz * 1.5)

            if vz <= z_base + 1.0:
                # Dense dark basaltic bedrock at baseplate
                col = (0.12, 0.11, 0.13, 1.0)
            elif vz < 0.0:
                # Deep subterranean strata
                col = (0.18 + 0.05 * band, 0.16 + 0.04 * band, 0.19, 1.0)
            elif vz < 6.0:
                # Alluvial soil & gravel layer
                col = (0.35 + 0.06 * band, 0.28 + 0.05 * band, 0.20, 1.0)
            elif vz < 20.0:
                # Exposed limestone/granite mountain rock
                col = (0.42 + 0.08 * band, 0.40 + 0.07 * band, 0.38, 1.0)
            else:
                # Alpine summit rock & snow dusting
                col = (0.65, 0.68, 0.72, 1.0)

            colors.append(col)

        return {
            "vertices": vertices,
            "faces": faces,
            "vertex_colors": colors,
        }
