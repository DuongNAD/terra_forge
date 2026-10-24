"""
Navigation Mesh & Obstacle Synchronization for Terra Forge.

Provides synchronized obstacle grids (tall flora circular colliders r = 0.45 + 0.25s,
ground cover pass-through), 4-connected BFS reachability flood-fill enforcing
navmeshCoverage >= 0.80, and adaptive chokepoint pruning.
"""

from terra_forge.navigation.navmesh import (
    FLORA_COLLIDER_BASE_RADIUS,
    FLORA_COLLIDER_SCALE_COEFF,
    TALL_FLORA_TYPES,
    NON_SOLID_FLORA_TYPES,
    flora_collider_radius,
    flora_canopy_radius,
    spawn_clearance_radius,
    ObstacleGridSynchronizer,
    NavMeshReachabilityValidator,
    NavMeshResult,
    PruningResult,
    validate_and_prune_navmesh,
)

__all__ = [
    "FLORA_COLLIDER_BASE_RADIUS",
    "FLORA_COLLIDER_SCALE_COEFF",
    "TALL_FLORA_TYPES",
    "NON_SOLID_FLORA_TYPES",
    "flora_collider_radius",
    "flora_canopy_radius",
    "spawn_clearance_radius",
    "ObstacleGridSynchronizer",
    "NavMeshReachabilityValidator",
    "NavMeshResult",
    "PruningResult",
    "validate_and_prune_navmesh",
]
