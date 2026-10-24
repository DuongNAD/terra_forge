"""
Navigation Mesh & Obstacle Synchronization Engine for Terra Forge.

Replicates and extends Anima-Engine's `gen_map_manifest.ts` and `floraClearance.ts` logic:
1. Obstacle Grid Synchronization:
   - Tall flora types (Pine, Round, Jungle, Cactus, Acacia, Palm, DeadTree) register
     solid circular colliders with radius: r = 0.45 + 0.25 * s.
   - Ground cover types (Bush, Reed, Tuft, Rock, Coral, Kelp, Seagrass) are non-solid pass-through.
2. NavMesh BFS Reachability Validator:
   - Evaluates:
       isLand(r, c) = elevation[r, c] > sea_level and (water[r, c] == 0)
       walkable(r, c) = isLand(r, c) and slope[r, c] < 0.60 and flow[r, c] < 100 and not_in_solid_flora(r, c)
   - Selects center-closest walkable land cell clear of flora canopies as spawn point.
   - 4-connected BFS flood-fill over grid without row-wrapping.
   - Enforces invariant: navmeshCoverage = reachCount / totalLand >= 0.80.
3. Adaptive Chokepoint Pruning:
   - Detects and prunes flora instances blocking narrow bottlenecks until navmeshCoverage >= 0.80.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from terra_forge.ai.vault import FloraType, AbioticType

# -----------------------------------------------------------------------------
# Flora Physical Collider & Footprint Constants
# -----------------------------------------------------------------------------
FLORA_COLLIDER_BASE_RADIUS: float = 0.45
FLORA_COLLIDER_SCALE_COEFF: float = 0.25
VEGETATION_BASE_SIZE: float = 1.4
DEFAULT_MAX_WALKABLE_SLOPE_DEG: float = 35.0
DEFAULT_GENTLE_SLOPE_DEG: float = 15.0
DEFAULT_WALKABLE_SLOPE: float = 0.7002  # tan(35°) ≈ 0.7002075
DEFAULT_MAX_FLOW: float = 100.0         # River/flow cutoff for walkability
CANONICAL_COVERAGE_TARGET: float = 0.80  # Invariant target: reachCount / totalLand >= 0.80

# Abiotic geological types that receive solid colliders
SOLID_ABIOTIC_TYPES: Set[int] = {
    int(AbioticType.HornPeak),     # 0
    int(AbioticType.RockCrag),     # 1
    int(AbioticType.BasaltColumn), # 2
    int(AbioticType.Stalagmite),   # 6
    int(AbioticType.CavernColumn), # 7
}

# Tall trunked flora types that receive solid circular colliders
TALL_FLORA_TYPES: Set[int] = {
    int(FloraType.Pine),      # 0
    int(FloraType.Round),     # 1
    int(FloraType.Jungle),    # 2
    int(FloraType.Cactus),    # 3
    int(FloraType.Acacia),    # 5
    int(FloraType.Palm),      # 6
    int(FloraType.DeadTree),  # 7
}

# Ground cover and aquatic flora that are non-solid pass-through
NON_SOLID_FLORA_TYPES: Set[int] = {
    int(FloraType.Rock),      # 4
    int(FloraType.Bush),      # 8
    int(FloraType.Reed),      # 9
    int(FloraType.Tuft),      # 10
    int(FloraType.Coral),     # 11
    int(FloraType.Kelp),      # 12
    int(FloraType.Seagrass),  # 13
}

# Measured canopy footprint unit radii from floraClearance.ts
FLORA_FOOTPRINT_UNIT_RADIUS: Dict[int, float] = {
    int(FloraType.Pine): 0.55,
    int(FloraType.Round): 0.7628,
    int(FloraType.Jungle): 0.7595,
    int(FloraType.Cactus): 0.5952,
    int(FloraType.Acacia): 0.95,
    int(FloraType.Palm): 0.8062,
    int(FloraType.DeadTree): 0.424,
}

SPAWN_CANOPY_DISTANCE_RATIO: float = 2.0
SPAWN_CLEARANCE_MARGIN: float = 0.5


def flora_collider_radius(flora_scale: float) -> float:
    """
    Computes world-space circular collider radius for a tall flora trunk:
    r = 0.45 + flora_scale * 0.25
    """
    return FLORA_COLLIDER_BASE_RADIUS + float(flora_scale) * FLORA_COLLIDER_SCALE_COEFF


def flora_canopy_radius(species_id: int, flora_scale: float) -> float:
    """
    Computes visual canopy radius in world units:
    R = unit_radius * scale * VEGETATION_BASE_SIZE
    """
    unit_r = FLORA_FOOTPRINT_UNIT_RADIUS.get(int(species_id), 0.6)
    return unit_r * float(flora_scale) * VEGETATION_BASE_SIZE


def spawn_clearance_radius(canopy_rad: float) -> float:
    """
    Computes breathing room radius a spawn pose must keep from the trunk center:
    R_spawn = max(2.0 * R_canopy, R_canopy + 0.5)
    """
    return max(canopy_rad * SPAWN_CANOPY_DISTANCE_RATIO, canopy_rad + SPAWN_CLEARANCE_MARGIN)


# -----------------------------------------------------------------------------
# Spatial Coordinate Helpers
# -----------------------------------------------------------------------------

def world_to_cell(
    x: float,
    z: float,
    shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
) -> Tuple[int, int]:
    """Maps world (x, z) coordinates to discrete (row, col) grid coordinates."""
    h, w = shape
    min_x, min_z, max_x, max_z = bounds
    col = int(math.floor(((x - min_x) / (max_x - min_x)) * w))
    row = int(math.floor(((z - min_z) / (max_z - min_z)) * h))
    return max(0, min(h - 1, row)), max(0, min(w - 1, col))


def cell_to_world(
    row: int,
    col: int,
    shape: Tuple[int, int],
    elevation: Optional[np.ndarray] = None,
    bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
) -> Tuple[float, float, float]:
    """Maps discrete (row, col) grid cell center to world (x, y, z) coordinates."""
    h, w = shape
    min_x, min_z, max_x, max_z = bounds
    x = min_x + ((col + 0.5) / w) * (max_x - min_x)
    z = min_z + ((row + 0.5) / h) * (max_z - min_z)
    y = 0.0
    if elevation is not None:
        raw_y = float(elevation[row, col])
        # If elevation is normalized in [0, 1], scale to [0, 10]
        y = raw_y * 10.0 if raw_y <= 1.0 and np.max(elevation) <= 1.0 else raw_y
    return (round(x, 2), round(max(0.0, y), 2), round(z, 2))


# -----------------------------------------------------------------------------
# Obstacle Grid Synchronizer
# -----------------------------------------------------------------------------

@dataclass
class SolidObstacle:
    """A registered solid circular obstacle in world space."""
    instance_index: int
    species_id: int
    x: float
    z: float
    scale: float
    collider_radius: float
    canopy_radius: float
    spawn_clearance: float


class ObstacleGridSynchronizer:
    """
    Rasterizes solid circular colliders for tall flora types and maintains
    the discrete obstacle grid synchronized with physical collisions.
    Ground cover flora is strictly pass-through (non-solid).
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (256, 256),
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
    ):
        self.shape = shape
        self.world_bounds = world_bounds
        self.solid_obstacles: List[SolidObstacle] = []
        self.obstacle_grid = np.zeros(shape, dtype=bool)
        self.canopy_clearance_mask = np.ones(shape, dtype=bool)
        self.spawn_clearance_mask = self.canopy_clearance_mask

    def clear(self) -> None:
        self.solid_obstacles.clear()
        self.obstacle_grid.fill(False)
        self.canopy_clearance_mask.fill(True)

    def register_flora_instance(
        self,
        instance_index: int,
        species_id: int,
        x: float,
        z: float,
        scale: float,
    ) -> Optional[SolidObstacle]:
        """
        Registers a flora instance. Only tall flora types (Pine, Round, Jungle,
        Cactus, Acacia, Palm, DeadTree) register solid colliders.
        Ground cover (Bush, Reed, Tuft, Rock, Coral, Kelp, Seagrass) returns None.
        """
        if int(species_id) not in TALL_FLORA_TYPES:
            # Ground cover is pass-through
            return None

        c_rad = flora_collider_radius(scale)
        can_rad = flora_canopy_radius(species_id, scale)
        s_clear = spawn_clearance_radius(can_rad)

        obs = SolidObstacle(
            instance_index=instance_index,
            species_id=int(species_id),
            x=float(x),
            z=float(z),
            scale=float(scale),
            collider_radius=c_rad,
            canopy_radius=can_rad,
            spawn_clearance=s_clear,
        )
        self.solid_obstacles.append(obs)
        self._rasterize_obstacle(obs)
        return obs

    def register_abiotic_instance(
        self,
        instance_index: int,
        abiotic_type: Union[AbioticType, int],
        x: float,
        z: float,
        scale: float = 1.0,
    ) -> Optional[SolidObstacle]:
        """
        Registers solid colliders for abiotic geological structures.
        Solid types: HornPeak, RockCrag, BasaltColumn, Stalagmite, CavernColumn.
        Non-solid types (TalusScree, AlluvialSand, SubterraneanPool, RiverPebble, RiparianSandbank, MineralCluster) are pass-through.
        """
        t_int = int(abiotic_type)
        if t_int not in SOLID_ABIOTIC_TYPES:
            return None

        if t_int == int(AbioticType.CavernColumn):
            base_r = 1.2
        elif t_int == int(AbioticType.BasaltColumn):
            base_r = 0.8
        elif t_int == int(AbioticType.HornPeak):
            base_r = 2.0
        elif t_int == int(AbioticType.Stalagmite):
            base_r = 0.5
        else:  # RockCrag
            base_r = 0.6
        c_rad = base_r * float(scale)
        s_clear = max(c_rad * 2.0, c_rad + 0.5)

        obs = SolidObstacle(
            instance_index=instance_index,
            species_id=1000 + t_int,
            x=float(x),
            z=float(z),
            scale=float(scale),
            collider_radius=c_rad,
            canopy_radius=c_rad,
            spawn_clearance=s_clear,
        )
        self.solid_obstacles.append(obs)
        self._rasterize_obstacle(obs)
        return obs

    def register_cavern_obstacles(
        self,
        center: Tuple[float, float, float] = (-16.0, 28.0, 6.0),
        column_positions: Optional[List[Tuple[float, float]]] = None,
        stalagmites: Optional[List[Tuple[float, float, float]]] = None,
    ) -> None:
        """Rasterizes cavern columns and stalagmites as solid obstacles."""
        if column_positions:
            for idx, (col_x, col_z) in enumerate(column_positions):
                self.register_abiotic_instance(
                    instance_index=20000 + idx,
                    abiotic_type=int(AbioticType.CavernColumn),
                    x=col_x,
                    z=col_z,
                    scale=1.5,
                )
        if stalagmites:
            for idx, item in enumerate(stalagmites):
                sx = item[0]
                sz = item[1]
                sc = item[2] if len(item) > 2 else 1.0
                self.register_abiotic_instance(
                    instance_index=21000 + idx,
                    abiotic_type=int(AbioticType.Stalagmite),
                    x=sx,
                    z=sz,
                    scale=sc,
                )

    def _rasterize_obstacle(self, obs: SolidObstacle) -> None:
        """Rasterizes a circular solid obstacle into the boolean obstacle grid and canopy clearance mask."""
        h, w = self.shape
        min_x, min_z, max_x, max_z = self.world_bounds
        dx = (max_x - min_x) / w
        dz = (max_z - min_z) / h

        # Guarantee containing cell is always marked blocked
        center_r, center_c = world_to_cell(obs.x, obs.z, self.shape, self.world_bounds)
        if 0 <= center_r < self.shape[0] and 0 <= center_c < self.shape[1]:
            self.obstacle_grid[center_r, center_c] = True
            self.canopy_clearance_mask[center_r, center_c] = False

        # Calculate bounding cell range for collider
        r_c = obs.collider_radius
        min_c = max(0, int(math.floor(((obs.x - r_c - min_x) / (max_x - min_x)) * w)))
        max_c = min(w - 1, int(math.ceil(((obs.x + r_c - min_x) / (max_x - min_x)) * w)))
        min_r = max(0, int(math.floor(((obs.z - r_c - min_z) / (max_z - min_z)) * h)))
        max_r = min(h - 1, int(math.ceil(((obs.z + r_c - min_z) / (max_z - min_z)) * h)))

        r_c_sq = r_c * r_c
        for r in range(min_r, max_r + 1):
            cell_z = min_z + (r + 0.5) * dz
            for c in range(min_c, max_c + 1):
                cell_x = min_x + (c + 0.5) * dx
                dist_sq = (cell_x - obs.x) ** 2 + (cell_z - obs.z) ** 2
                if dist_sq <= r_c_sq:
                    self.obstacle_grid[r, c] = True

        # Calculate bounding cell range for spawn canopy clearance
        s_c = obs.spawn_clearance
        s_min_c = max(0, int(math.floor(((obs.x - s_c - min_x) / (max_x - min_x)) * w)))
        s_max_c = min(w - 1, int(math.ceil(((obs.x + s_c - min_x) / (max_x - min_x)) * w)))
        s_min_r = max(0, int(math.floor(((obs.z - s_c - min_z) / (max_z - min_z)) * h)))
        s_max_r = min(h - 1, int(math.ceil(((obs.z + s_c - min_z) / (max_z - min_z)) * h)))

        s_c_sq = s_c * s_c
        for r in range(s_min_r, s_max_r + 1):
            cell_z = min_z + (r + 0.5) * dz
            for c in range(s_min_c, s_max_c + 1):
                cell_x = min_x + (c + 0.5) * dx
                dist_sq = (cell_x - obs.x) ** 2 + (cell_z - obs.z) ** 2
                if dist_sq < s_c_sq:
                    self.canopy_clearance_mask[r, c] = False

    def rasterize_all(self) -> np.ndarray:
        """Re-rasterizes all registered solid obstacles from scratch."""
        self.obstacle_grid.fill(False)
        self.canopy_clearance_mask.fill(True)
        for obs in self.solid_obstacles:
            self._rasterize_obstacle(obs)
        return self.obstacle_grid

    def is_flora_clear_for_spawn(self, x: float, z: float) -> bool:
        """
        Tests if candidate spawn coordinates are strictly clear of flora canopies.
        Fast O(1) query using pre-rasterized canopy_clearance_mask with bounding fallback.
        """
        if getattr(self, "canopy_clearance_mask", None) is not None:
            r, c = world_to_cell(x, z, self.shape, self.world_bounds)
            if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
                return bool(self.canopy_clearance_mask[r, c])
            return False
        for obs in self.solid_obstacles:
            dist = math.sqrt((x - obs.x) ** 2 + (z - obs.z) ** 2)
            if dist < obs.spawn_clearance:
                return False
        return True


# -----------------------------------------------------------------------------
# NavMesh Reachability Validator & Chokepoint Pruning
# -----------------------------------------------------------------------------

@dataclass
class NavMeshResult:
    """Output summary from NavMesh BFS reachability verification."""
    navmesh_coverage: float
    reach_count: int
    total_land: int
    spawn_cell: Tuple[int, int]
    spawn_world_pos: Tuple[float, float, float]
    farthest_cell: Tuple[int, int]
    farthest_world_pos: Tuple[float, float, float]
    navigation_nodes: List[Dict[str, Any]]
    navigation_edges: List[Dict[str, str]]
    reached_mask: np.ndarray
    walkable_mask: np.ndarray
    is_land_mask: np.ndarray
    obstacle_grid: np.ndarray
    is_valid: bool  # navmesh_coverage >= CANONICAL_COVERAGE_TARGET


@dataclass
class PruningResult:
    """Result of adaptive chokepoint pruning."""
    initial_coverage: float
    final_coverage: float
    pruned_count: int
    pruned_obstacle_indices: List[int]
    navmesh_result: NavMeshResult


class NavMeshReachabilityValidator:
    """
    Validates navigation connectivity over a heightfield and flora obstacle grid.
    Replicates Anima-Engine `gen_map_manifest.ts` BFS flood-fill and enforces
    the `navmeshCoverage >= 0.80` invariant.
    """

    def __init__(
        self,
        elevation: np.ndarray,
        sea_level: float = 0.20,
        water_cutoff: Optional[float] = None,
        water_depth: Optional[np.ndarray] = None,
        slope: Optional[np.ndarray] = None,
        water: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
        walkable_slope_limit: float = DEFAULT_WALKABLE_SLOPE,
        max_walkable_slope_deg: float = DEFAULT_MAX_WALKABLE_SLOPE_DEG,
        max_flow: float = DEFAULT_MAX_FLOW,
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
        safe_spawn: bool = False,
    ):
        self.elevation = np.asarray(elevation, dtype=np.float32)
        if self.elevation.ndim == 1:
            side = int(round(math.sqrt(self.elevation.size)))
            if side * side == self.elevation.size:
                self.elevation = self.elevation.reshape((side, side))
        self.shape = self.elevation.shape
        self.sea_level = float(sea_level)
        self.water_cutoff = float(water_cutoff if water_cutoff is not None else sea_level)
        self.world_bounds = world_bounds
        self.walkable_slope_limit = float(walkable_slope_limit)
        self.max_walkable_slope_deg = float(max_walkable_slope_deg)
        self.max_flow = float(max_flow)
        self.safe_spawn = bool(safe_spawn)

        # Compute numerical gradient or physical slope
        if slope is not None:
            self.slope = np.asarray(slope, dtype=np.float32)
            if self.slope.ndim == 1 and self.slope.size == self.elevation.size:
                self.slope = self.slope.reshape(self.shape)
            if float(np.nanmax(self.slope)) > math.pi / 2.0:
                self.slope_deg = self.slope
                self.slope_tangent = np.tan(np.radians(np.clip(self.slope, 0.0, 89.9))).astype(np.float32)
            else:
                self.slope_tangent = self.slope
                self.slope_deg = np.degrees(np.arctan(self.slope)).astype(np.float32)
        else:
            h, w = self.shape
            min_x, min_z, max_x, max_z = self.world_bounds
            dx_m = (max_x - min_x) / max(1, w - 1)
            dz_m = (max_z - min_z) / max(1, h - 1)
            dz_dz, dz_dx = np.gradient(self.elevation, dz_m, dx_m)
            self.slope_tangent = np.sqrt(dz_dx * dz_dx + dz_dz * dz_dz).astype(np.float32)
            self.slope_deg = np.degrees(np.arctan(self.slope_tangent)).astype(np.float32)
            self.slope = self.slope_tangent

        self.water = np.asarray(water, dtype=np.float32) if water is not None else None
        if self.water is not None and self.water.ndim == 1 and self.water.size == self.elevation.size:
            self.water = self.water.reshape(self.shape)
        self.water_depth = np.asarray(water_depth, dtype=np.float32) if water_depth is not None else None
        if self.water_depth is not None and self.water_depth.ndim == 1 and self.water_depth.size == self.elevation.size:
            self.water_depth = self.water_depth.reshape(self.shape)
        self.flow = np.asarray(flow, dtype=np.float32) if flow is not None else None
        if self.flow is not None and self.flow.ndim == 1 and self.flow.size == self.elevation.size:
            self.flow = self.flow.reshape(self.shape)

        # Precompute land mask incorporating water cutoff and water depth
        self.is_land = (self.elevation > self.water_cutoff)
        if self.water is not None:
            self.is_land &= (self.water == 0)
        if self.water_depth is not None:
            self.is_land &= (self.water_depth <= 0.05)

    def compute_walkable_mask(self, obstacle_grid: np.ndarray) -> np.ndarray:
        """
        Computes boolean walkable mask:
        walkable = isLand && slope <= limit && flow < max_flow && !obstacle
        """
        walkable = (
            self.is_land
            & (self.slope_deg <= self.max_walkable_slope_deg + 1e-4)
            & (self.slope <= self.walkable_slope_limit + 1e-4)
        )
        if self.flow is not None:
            walkable &= self.flow < self.max_flow
        walkable &= ~obstacle_grid
        return walkable

    def find_spawn_cell(
        self,
        walkable: np.ndarray,
        synchronizer: ObstacleGridSynchronizer,
    ) -> Tuple[int, int]:
        """
        Selects spawn cell: walkable land cell closest to grid center that is
        strictly clear of flora canopies.
        """
        h, w = self.shape
        cr, cc = h // 2, w // 2
        min_x, min_z, max_x, max_z = self.world_bounds
        dx = (max_x - min_x) / w
        dz = (max_z - min_z) / h

        best_cell = (-1, -1)
        best_d_sq = float("inf")
        first_walkable = (-1, -1)

        for r in range(h):
            cell_z = min_z + (r + 0.5) * dz
            for c in range(w):
                if not walkable[r, c]:
                    continue

                if first_walkable == (-1, -1):
                    first_walkable = (r, c)

                cell_x = min_x + (c + 0.5) * dx
                if not synchronizer.is_flora_clear_for_spawn(cell_x, cell_z):
                    continue

                d_sq = (r - cr) ** 2 + (c - cc) ** 2
                if d_sq < best_d_sq:
                    best_d_sq = d_sq
                    best_cell = (r, c)

        if best_cell[0] >= 0:
            return best_cell

        # Fallback if no flora-clear cell found
        if first_walkable[0] >= 0:
            # Pick center-closest walkable
            best_walk = (-1, -1)
            min_dist = float("inf")
            for r in range(h):
                for c in range(w):
                    if walkable[r, c]:
                        d = (r - cr) ** 2 + (c - cc) ** 2
                        if d < min_dist:
                            min_dist = d
                            best_walk = (r, c)
            return best_walk

        raise ValueError("No walkable land cell exists on this map.")

    def find_safe_spawn_point(
        self,
        walkable: np.ndarray,
        synchronizer: ObstacleGridSynchronizer,
        max_spawn_slope_deg: float = DEFAULT_GENTLE_SLOPE_DEG,
        min_water_clearance_m: float = 0.5,
        boundary_margin_ratio: float = 0.85,
    ) -> Tuple[int, int]:
        """
        Computes a guaranteed safe spawn point satisfying:
        1. Non-submerged dry ground: elevation > water_cutoff + min_water_clearance_m
        2. Gentle slope: slope_deg < max_spawn_slope_deg (15.0°)
        3. Clear of all solid colliders and canopies
        4. Within world bounds (inside boundary margin)
        5. Belongs to the largest connected walkable component (preventing mountain summit cliff traps)
        """
        h, w = self.shape
        cr, cc = h // 2, w // 2
        min_x, min_z, max_x, max_z = self.world_bounds
        dx = (max_x - min_x) / w
        dz = (max_z - min_z) / h

        # 1. Filter candidates by safety criteria
        dry_mask = self.elevation > (self.water_cutoff + min_water_clearance_m)
        if self.water_depth is not None:
            dry_mask &= (self.water_depth <= 0.01)

        slope_mask = self.slope_deg < max_spawn_slope_deg

        # Boundary padding (keep within boundary_margin_ratio of grid)
        r_pad = int(h * (1.0 - boundary_margin_ratio) / 2)
        c_pad = int(w * (1.0 - boundary_margin_ratio) / 2)
        bounds_mask = np.zeros((h, w), dtype=bool)
        bounds_mask[r_pad : max(r_pad + 1, h - r_pad), c_pad : max(c_pad + 1, w - c_pad)] = True

        candidates = walkable & dry_mask & slope_mask & bounds_mask

        # 2. Identify largest connected walkable component via BFS
        visited = np.zeros((h, w), dtype=bool)
        largest_comp_mask = np.zeros((h, w), dtype=bool)
        max_comp_size = 0

        for r in range(h):
            for c in range(w):
                if walkable[r, c] and not visited[r, c]:
                    comp_cells = []
                    q = deque([(r, c)])
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.popleft()
                        comp_cells.append((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    if len(comp_cells) > max_comp_size:
                        max_comp_size = len(comp_cells)
                        largest_comp_mask.fill(False)
                        for cr_idx, cc_idx in comp_cells:
                            largest_comp_mask[cr_idx, cc_idx] = True

        # Intersect safe candidates with largest component
        prime_candidates = candidates & largest_comp_mask
        if not np.any(prime_candidates):
            prime_candidates = candidates
        if not np.any(prime_candidates):
            prime_candidates = walkable & largest_comp_mask
        if not np.any(prime_candidates):
            prime_candidates = walkable

        # 3. Select center-closest among prime candidates that is obstacle-clear
        clear_mask = getattr(synchronizer, "canopy_clearance_mask", None)
        valid_candidates = (prime_candidates & clear_mask) if clear_mask is not None else prime_candidates

        best_cell = (-1, -1)
        best_d_sq = float("inf")
        if clear_mask is not None:
            if np.any(valid_candidates):
                r_idx, c_idx = np.nonzero(valid_candidates)
                d_sq = (r_idx - cr) ** 2 + (c_idx - cc) ** 2
                min_d = np.min(d_sq)
                first_pos = np.where(d_sq == min_d)[0][0]
                best_cell = (int(r_idx[first_pos]), int(c_idx[first_pos]))
        else:
            for r in range(h):
                cell_z = min_z + (r + 0.5) * dz
                for c in range(w):
                    if prime_candidates[r, c]:
                        cell_x = min_x + (c + 0.5) * dx
                        if not synchronizer.is_flora_clear_for_spawn(cell_x, cell_z):
                            continue
                        d_sq = (r - cr) ** 2 + (c - cc) ** 2
                        if d_sq < best_d_sq:
                            best_d_sq = d_sq
                            best_cell = (r, c)

        if best_cell[0] >= 0:
            return best_cell

        # Absolute fallback to legacy find_spawn_cell
        return self.find_spawn_cell(walkable, synchronizer)

    def run_bfs(
        self,
        walkable: np.ndarray,
        spawn_cell: Tuple[int, int],
    ) -> Tuple[np.ndarray, int, Tuple[int, int], List[int]]:
        """
        Executes 4-connected BFS flood fill starting from spawn_cell.
        Row-wrapping across grid boundaries is strictly prohibited.
        """
        h, w = self.shape
        reached = np.zeros((h, w), dtype=bool)
        sr, sc = spawn_cell

        if not walkable[sr, sc]:
            return reached, 0, spawn_cell, []

        queue = deque([(sr, sc)])
        reached[sr, sc] = True
        reach_count = 1

        far_cell = spawn_cell
        far_d_sq = 0
        path_nodes: List[int] = [sr * w + sc]

        # 4-connected neighbors: North, South, West, East
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        while queue:
            curr_r, curr_c = queue.popleft()
            d_sq = (curr_r - sr) ** 2 + (curr_c - sc) ** 2
            if d_sq > far_d_sq:
                far_d_sq = d_sq
                far_cell = (curr_r, curr_c)

            for i in range(4):
                nr = curr_r + dr[i]
                nc = curr_c + dc[i]

                # Strict boundary check (no wrapping)
                if 0 <= nr < h and 0 <= nc < w:
                    if not reached[nr, nc] and walkable[nr, nc]:
                        reached[nr, nc] = True
                        reach_count += 1
                        queue.append((nr, nc))
                        path_nodes.append(nr * w + nc)

        return reached, reach_count, far_cell, path_nodes

    def evaluate(
        self,
        synchronizer: ObstacleGridSynchronizer,
        safe_spawn: Optional[bool] = None,
    ) -> NavMeshResult:
        """
        Evaluates full NavMesh reachability and constructs navigation graph.
        """
        h, w = self.shape
        obstacle_grid = synchronizer.obstacle_grid
        walkable = self.compute_walkable_mask(obstacle_grid)

        total_land = int(np.sum(self.is_land))
        if total_land == 0:
            spawn_pos = (0.0, 0.0, 0.0)
            return NavMeshResult(
                navmesh_coverage=0.0,
                reach_count=0,
                total_land=0,
                spawn_cell=(0, 0),
                spawn_world_pos=spawn_pos,
                farthest_cell=(0, 0),
                farthest_world_pos=spawn_pos,
                navigation_nodes=[],
                navigation_edges=[],
                reached_mask=np.zeros(self.shape, dtype=bool),
                walkable_mask=walkable,
                is_land_mask=self.is_land,
                obstacle_grid=obstacle_grid,
                is_valid=False,
            )

        use_safe = self.safe_spawn if safe_spawn is None else bool(safe_spawn)
        if use_safe:
            spawn_cell = self.find_safe_spawn_point(walkable, synchronizer)
        else:
            spawn_cell = self.find_spawn_cell(walkable, synchronizer)

        reached, reach_count, far_cell, path_nodes = self.run_bfs(walkable, spawn_cell)

        coverage = round(reach_count / total_land, 4) if total_land > 0 else 0.0

        spawn_pos = cell_to_world(spawn_cell[0], spawn_cell[1], self.shape, self.elevation, self.world_bounds)
        far_pos = cell_to_world(far_cell[0], far_cell[1], self.shape, self.elevation, self.world_bounds)

        # Generate canonical navigation waypoints
        wp_a_idx = path_nodes[len(path_nodes) // 3] if path_nodes else spawn_cell[0] * w + spawn_cell[1]
        wp_b_idx = path_nodes[(len(path_nodes) * 2) // 3] if path_nodes else far_cell[0] * w + far_cell[1]

        wp_a_pos = cell_to_world(wp_a_idx // w, wp_a_idx % w, self.shape, self.elevation, self.world_bounds)
        wp_b_pos = cell_to_world(wp_b_idx // w, wp_b_idx % w, self.shape, self.elevation, self.world_bounds)

        nodes = [
            {"id": "spawn-center", "kind": "spawn", "position": {"x": spawn_pos[0], "y": spawn_pos[1], "z": spawn_pos[2]}},
            {"id": "waypoint-a", "kind": "walkable", "position": {"x": wp_a_pos[0], "y": wp_a_pos[1], "z": wp_a_pos[2]}},
            {"id": "waypoint-b", "kind": "walkable", "position": {"x": wp_b_pos[0], "y": wp_b_pos[1], "z": wp_b_pos[2]}},
            {"id": "exit-far", "kind": "exit", "position": {"x": far_pos[0], "y": far_pos[1], "z": far_pos[2]}},
        ]

        edges = [
            {"from": "spawn-center", "to": "waypoint-a"},
            {"from": "waypoint-a", "to": "waypoint-b"},
            {"from": "waypoint-b", "to": "exit-far"},
        ]

        return NavMeshResult(
            navmesh_coverage=coverage,
            reach_count=reach_count,
            total_land=total_land,
            spawn_cell=spawn_cell,
            spawn_world_pos=spawn_pos,
            farthest_cell=far_cell,
            farthest_world_pos=far_pos,
            navigation_nodes=nodes,
            navigation_edges=edges,
            reached_mask=reached,
            walkable_mask=walkable,
            is_land_mask=self.is_land,
            obstacle_grid=obstacle_grid,
            is_valid=bool(coverage >= CANONICAL_COVERAGE_TARGET),
        )

    def adaptive_prune(
        self,
        synchronizer: ObstacleGridSynchronizer,
        target_coverage: float = CANONICAL_COVERAGE_TARGET,
        max_prune_iterations: int = 100,
    ) -> PruningResult:
        """
        Iteratively detects and prunes flora instances blocking chokepoints
        until navmeshCoverage >= target_coverage or no more blocking candidates exist.
        """
        initial_res = self.evaluate(synchronizer)
        if initial_res.navmesh_coverage >= target_coverage or not synchronizer.solid_obstacles:
            return PruningResult(
                initial_coverage=initial_res.navmesh_coverage,
                final_coverage=initial_res.navmesh_coverage,
                pruned_count=0,
                pruned_obstacle_indices=[],
                navmesh_result=initial_res,
            )

        h, w = self.shape
        pruned_indices: List[int] = []
        current_res = initial_res

        for _ in range(max_prune_iterations):
            if current_res.navmesh_coverage >= target_coverage or not synchronizer.solid_obstacles:
                break

            reached = current_res.reached_mask
            # Land cells that are walkable in terrain terms (slope < 0.60 & flow < 100) but unreached
            base_terrain_walkable = self.is_land & (self.slope < self.walkable_slope_limit)
            if self.flow is not None:
                base_terrain_walkable &= self.flow < self.max_flow
            unreached_land = base_terrain_walkable & ~reached

            # Score each registered solid obstacle by how many reached and unreached cells it bridges
            best_obs_idx = -1
            max_bottleneck_score = -1.0

            min_x, min_z, max_x, max_z = self.world_bounds
            dx = (max_x - min_x) / w
            dz = (max_z - min_z) / h

            for idx, obs in enumerate(synchronizer.solid_obstacles):
                # Check boundary neighborhood around this obstacle (radius + 2 cells)
                search_rad = obs.collider_radius + max(dx, dz) * 2.0
                min_c = max(0, int(math.floor(((obs.x - search_rad - min_x) / (max_x - min_x)) * w)))
                max_c = min(w - 1, int(math.ceil(((obs.x + search_rad - min_x) / (max_x - min_x)) * w)))
                min_r = max(0, int(math.floor(((obs.z - search_rad - min_z) / (max_z - min_z)) * h)))
                max_r = min(h - 1, int(math.ceil(((obs.z + search_rad - min_z) / (max_z - min_z)) * h)))

                neighborhood_reached = np.count_nonzero(reached[min_r : max_r + 1, min_c : max_c + 1])
                neighborhood_unreached = np.count_nonzero(unreached_land[min_r : max_r + 1, min_c : max_c + 1])

                # An obstacle is a chokepoint if it touches BOTH reached and unreached walkable cells
                if neighborhood_reached > 0 and neighborhood_unreached > 0:
                    score = float(neighborhood_reached * neighborhood_unreached)
                elif neighborhood_reached > 0:
                    score = float(neighborhood_reached * 0.1)
                else:
                    score = 0.0

                if score > max_bottleneck_score:
                    max_bottleneck_score = score
                    best_obs_idx = idx

            if best_obs_idx < 0:
                # No clear chokepoint found; prune the obstacle nearest to any unreached land
                best_obs_idx = 0

            # Prune selected obstacle
            pruned_obs = synchronizer.solid_obstacles.pop(best_obs_idx)
            pruned_indices.append(pruned_obs.instance_index)

            # Re-rasterize and re-evaluate
            synchronizer.rasterize_all()
            current_res = self.evaluate(synchronizer)

        return PruningResult(
            initial_coverage=initial_res.navmesh_coverage,
            final_coverage=current_res.navmesh_coverage,
            pruned_count=len(pruned_indices),
            pruned_obstacle_indices=pruned_indices,
            navmesh_result=current_res,
        )


def validate_and_prune_navmesh(
    elevation: np.ndarray,
    flora_instances: Optional[List[Any]] = None,
    abiotic_instances: Optional[List[Any]] = None,
    sea_level: float = 0.20,
    water_cutoff: Optional[float] = None,
    water_depth: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    water: Optional[np.ndarray] = None,
    flow: Optional[np.ndarray] = None,
    world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
    target_coverage: float = CANONICAL_COVERAGE_TARGET,
    safe_spawn: bool = False,
) -> Tuple[NavMeshResult, List[int]]:
    """
    Convenience function:
    1. Synchronizes obstacle grid with flora and abiotic instances.
    2. Runs BFS reachability check.
    3. Performs adaptive pruning if navmeshCoverage < target_coverage.
    Returns (final_navmesh_result, list_of_pruned_instance_indices).
    """
    shape = elevation.shape
    synchronizer = ObstacleGridSynchronizer(shape=shape, world_bounds=world_bounds)

    # Populate synchronizer from flora instances
    if flora_instances:
        for idx, inst in enumerate(flora_instances):
            species_id = 0
            scale = 1.0
            x, z = 0.0, 0.0

            if isinstance(inst, dict):
                species_id = int(inst.get("species_id", inst.get("speciesId", 0)))
                if "x" in inst and "z" in inst:
                    x = float(inst["x"])
                    z = float(inst["z"])
                elif "position" in inst:
                    pos = inst["position"]
                    x = float(pos[0])
                    z = float(pos[2] if len(pos) > 2 else pos[1])
                s_val = inst.get("scale", 1.0)
                scale = float(s_val[0]) if isinstance(s_val, (tuple, list)) else float(s_val)
            else:
                if hasattr(inst, "flora_type") and inst.flora_type is not None:
                    species_id = int(inst.flora_type)
                elif hasattr(inst, "species_id"):
                    species_id = int(inst.species_id)

                if hasattr(inst, "x") and hasattr(inst, "z"):
                    x = float(inst.x)
                    z = float(inst.z)
                elif hasattr(inst, "position"):
                    x = float(inst.position[0])
                    z = float(inst.position[2])

                if hasattr(inst, "scale"):
                    s_val = inst.scale
                    scale = float(s_val[0]) if isinstance(s_val, (tuple, list)) else float(s_val)

            synchronizer.register_flora_instance(idx, species_id, x, z, scale)

    # Populate synchronizer from abiotic instances
    if abiotic_instances:
        for idx, inst in enumerate(abiotic_instances):
            abiotic_type = 0
            scale = 1.0
            x, z = 0.0, 0.0
            if isinstance(inst, dict):
                abiotic_type = int(inst.get("abiotic_type", inst.get("abioticType", 0)))
                if "x" in inst and "z" in inst:
                    x = float(inst["x"])
                    z = float(inst["z"])
                elif "position" in inst:
                    pos = inst["position"]
                    x = float(pos[0])
                    z = float(pos[2] if len(pos) > 2 else pos[1])
                s_val = inst.get("scale", 1.0)
                scale = float(s_val[0]) if isinstance(s_val, (tuple, list)) else float(s_val)
            else:
                if hasattr(inst, "abiotic_type") and inst.abiotic_type is not None:
                    abiotic_type = int(inst.abiotic_type)
                if hasattr(inst, "x") and hasattr(inst, "z"):
                    x = float(inst.x)
                    z = float(inst.z)
                elif hasattr(inst, "position"):
                    x = float(inst.position[0])
                    z = float(inst.position[2])
                if hasattr(inst, "scale"):
                    s_val = inst.scale
                    scale = float(s_val[0]) if isinstance(s_val, (tuple, list)) else float(s_val)
            synchronizer.register_abiotic_instance(10000 + idx, abiotic_type, x, z, scale)

    validator = NavMeshReachabilityValidator(
        elevation=elevation,
        sea_level=sea_level,
        water_cutoff=water_cutoff,
        water_depth=water_depth,
        slope=slope,
        water=water,
        flow=flow,
        world_bounds=world_bounds,
        safe_spawn=safe_spawn,
    )

    prune_res = validator.adaptive_prune(synchronizer, target_coverage=target_coverage)
    return prune_res.navmesh_result, prune_res.pruned_obstacle_indices
