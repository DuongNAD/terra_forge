"""
Shared fixtures and pure-Python topological validator for TerraForge E2E test suite.
"""

import math
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    PeakConfig,
    ErosionConfig,
    HydrologyConfig,
    CavernConfig,
    BiomeScatterConfig,
    ShaderConfig,
    ExportConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder


def compute_mesh_topology_metrics(vertices: List[Tuple[float, float, float]], faces: List[List[int]]) -> Dict[str, Any]:
    """
    Pure Python topological validator.
    Evaluates 2-manifold invariants without bpy/bmesh:
    1. V - E + F = 2 (Euler characteristic for genus-0 sphere-equivalent diorama slab)
    2. Boundary edges == 0 (every edge has exactly 2 incident faces)
    3. Non-manifold edges == 0 (no edge has > 2 incident faces)
    """
    V = len(vertices)
    F = len(faces)

    # Map each undirected edge (min(u, v), max(u, v)) to its incident faces count
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for face in faces:
        n = len(face)
        for i in range(n):
            u = face[i]
            v = face[(i + 1) % n]
            edge = (min(u, v), max(u, v))
            edge_counts[edge] += 1

    E = len(edge_counts)
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    euler_characteristic = V - E + F

    is_watertight = (boundary_edges == 0) and (non_manifold_edges == 0) and (euler_characteristic == 2)

    return {
        "num_vertices": V,
        "num_edges": E,
        "num_faces": F,
        "euler_characteristic": euler_characteristic,
        "is_watertight": is_watertight,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
    }


@pytest.fixture
def topology_validator():
    """Provides pure-Python topological validator callable."""
    return compute_mesh_topology_metrics


@pytest.fixture
def sample_heightfield():
    """Returns a baseline 31x31 heightfield with a peak and basin."""
    hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
    hf.add_peak(center_x=0.0, center_y=10.0, height=25.0, radius=18.0, shape="horn")
    hf.carve_lake_basin(center_x=-10.0, center_y=-10.0, radius=12.0, water_level=4.0, lake_depth=8.0)
    return hf


@pytest.fixture
def flat_heightfield():
    """Returns a zero-elevation flat heightfield."""
    hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
    hf.Z.fill(0.0)
    return hf
