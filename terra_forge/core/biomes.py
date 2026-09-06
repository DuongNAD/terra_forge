r"""
Biomes & Multi-Tier Ecological Niche Simulation:
Calculates environmental suitability based on microclimate drivers:
- Elevation temperature lapse rate: T(z) = T_0 - \Gamma (z - z_{sea})
- Continuous soil moisture index field M(x, y) from proximity to rivers, lakes, and oceans
- Solar radiation insolation aspect: -\vec{n} \cdot \vec{L}_{sun} from surface normals

Implements multi-tier ecological stratification:
1. Canopy trees (alpine pines, temperate broadleaf oaks, tropical palms)
2. Understory shrubs and bushes
3. Ground flora (wildflowers, dense grass clumps, turf grass, ferns, mushrooms, rock moss)
4. Natural tree decay and windthrow (fallen logs, snapped trunks, rotting wood, weathered stumps)
5. Biological & creature impacts (animal game trails, trampled grass, disturbed soil patches)
6. Underwater ecosystems (aquatic seaweed, waterweeds, reeds, riverbed pebbles, coral reefs)

Provides fast O(N) blue-noise Poisson-Disk clustering with biological density maps,
strictly clamping all scatter coordinates within [-W/2, W/2] to prevent perimeter jitter overshoot.
"""

from __future__ import annotations
import enum
import math
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import BiomeScatterConfig


class EcologicalTier(str, enum.Enum):
    """Vertical and functional ecological stratification tiers."""
    CANOPY = "canopy"
    UNDERSTORY = "understory"
    GROUND_FLORA = "ground_flora"
    DECAY_WINDTHROW = "decay_windthrow"
    BIOLOGICAL_IMPACT = "biological_impact"
    UNDERWATER = "underwater"


# Stratified prototype taxonomy mapping
PROTOTYPE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # Canopy Trees
    "pine_tall": {"tier": EcologicalTier.CANOPY, "min_alt": 8.0, "max_alt": 35.0, "max_slope": 35.0, "density": 0.08, "poisson_dist": 3.0},
    "pine_medium": {"tier": EcologicalTier.CANOPY, "min_alt": 6.0, "max_alt": 30.0, "max_slope": 32.0, "density": 0.07, "poisson_dist": 2.8},
    "broadleaf": {"tier": EcologicalTier.CANOPY, "min_alt": 2.0, "max_alt": 22.0, "max_slope": 30.0, "density": 0.09, "poisson_dist": 3.5},
    "broadleaf_mature": {"tier": EcologicalTier.CANOPY, "min_alt": 2.0, "max_alt": 20.0, "max_slope": 28.0, "density": 0.08, "poisson_dist": 4.0},
    "palm_tropical": {"tier": EcologicalTier.CANOPY, "min_alt": 0.5, "max_alt": 18.0, "max_slope": 35.0, "density": 0.07, "poisson_dist": 3.2},
    # Understory Shrubs & Bushes
    "shrub": {"tier": EcologicalTier.UNDERSTORY, "min_alt": 2.0, "max_alt": 26.0, "max_slope": 35.0, "density": 0.12, "poisson_dist": 2.0},
    "bush": {"tier": EcologicalTier.UNDERSTORY, "min_alt": 1.5, "max_alt": 24.0, "max_slope": 32.0, "density": 0.14, "poisson_dist": 1.8},
    "krummholz": {"tier": EcologicalTier.UNDERSTORY, "min_alt": 12.0, "max_alt": 30.0, "max_slope": 35.0, "density": 0.08, "poisson_dist": 2.2},
    # Rich Ground Flora
    "wildflower": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 1.0, "max_alt": 22.0, "max_slope": 25.0, "density": 0.20, "poisson_dist": 1.2},
    "dense_grass": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 0.5, "max_alt": 25.0, "max_slope": 30.0, "density": 0.25, "poisson_dist": 1.0},
    "turf_grass": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 0.5, "max_alt": 22.0, "max_slope": 28.0, "density": 0.22, "poisson_dist": 1.1},
    "fern": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 1.5, "max_alt": 25.0, "max_slope": 35.0, "density": 0.15, "poisson_dist": 1.5},
    "mushroom": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 1.0, "max_alt": 20.0, "max_slope": 25.0, "density": 0.16, "poisson_dist": 1.3},
    "forest_mushroom": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 1.0, "max_alt": 18.0, "max_slope": 24.0, "density": 0.14, "poisson_dist": 1.4},
    "rock_moss": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 2.0, "max_alt": 38.0, "max_slope": 65.0, "density": 0.18, "poisson_dist": 1.2},
    # Natural Tree Decay & Windthrow
    "fallen_log": {"tier": EcologicalTier.DECAY_WINDTHROW, "min_alt": 1.5, "max_alt": 22.0, "max_slope": 22.0, "density": 0.04, "poisson_dist": 4.5},
    "snapped_trunk": {"tier": EcologicalTier.DECAY_WINDTHROW, "min_alt": 2.0, "max_alt": 24.0, "max_slope": 25.0, "density": 0.03, "poisson_dist": 5.0},
    "rotting_wood": {"tier": EcologicalTier.DECAY_WINDTHROW, "min_alt": 1.0, "max_alt": 20.0, "max_slope": 20.0, "density": 0.04, "poisson_dist": 4.0},
    "weathered_stump": {"tier": EcologicalTier.DECAY_WINDTHROW, "min_alt": 1.5, "max_alt": 25.0, "max_slope": 26.0, "density": 0.04, "poisson_dist": 4.2},
    # Biological & Creature Impacts
    "game_trail": {"tier": EcologicalTier.BIOLOGICAL_IMPACT, "min_alt": 0.5, "max_alt": 20.0, "max_slope": 18.0, "density": 0.05, "poisson_dist": 2.0},
    "trampled_grass": {"tier": EcologicalTier.BIOLOGICAL_IMPACT, "min_alt": 0.5, "max_alt": 18.0, "max_slope": 20.0, "density": 0.08, "poisson_dist": 1.5},
    "disturbed_soil": {"tier": EcologicalTier.BIOLOGICAL_IMPACT, "min_alt": 0.5, "max_alt": 22.0, "max_slope": 22.0, "density": 0.06, "poisson_dist": 2.5},
    # Underwater Ecosystems
    "seaweed": {"tier": EcologicalTier.UNDERWATER, "min_alt": -18.0, "max_alt": 0.0, "max_slope": 35.0, "density": 0.18, "poisson_dist": 1.6},
    "waterweed": {"tier": EcologicalTier.UNDERWATER, "min_alt": -12.0, "max_alt": 0.2, "max_slope": 30.0, "density": 0.20, "poisson_dist": 1.4},
    "reeds": {"tier": EcologicalTier.UNDERWATER, "min_alt": 0.1, "max_alt": 3.5, "max_slope": 15.0, "density": 0.22, "poisson_dist": 1.3},
    "riverbed_pebble": {"tier": EcologicalTier.UNDERWATER, "min_alt": -10.0, "max_alt": 2.0, "max_slope": 25.0, "density": 0.25, "poisson_dist": 0.9},
    "coral_reef": {"tier": EcologicalTier.UNDERWATER, "min_alt": -15.0, "max_alt": -0.5, "max_slope": 40.0, "density": 0.08, "poisson_dist": 3.0},
    "submerged_reef": {"tier": EcologicalTier.UNDERWATER, "min_alt": -20.0, "max_alt": 0.0, "max_slope": 45.0, "density": 0.07, "poisson_dist": 3.5},
    "rock_boulder": {"tier": EcologicalTier.GROUND_FLORA, "min_alt": 1.0, "max_alt": 35.0, "max_slope": 60.0, "density": 0.06, "poisson_dist": 3.0},
}


def bridson_poisson_disk_sampling(
    width: float,
    depth: float,
    r_min: float,
    density_evaluator: Optional[Callable[[float, float], float]] = None,
    k: int = 30,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """
    Bridson's O(N) blue-noise Poisson-disk distribution with spatial hash grid
    and continuous biological probability sampling.
    Returns points centered in [-width/2, width/2] x [-depth/2, depth/2].
    """
    rng = np.random.default_rng(seed)
    cell_size = r_min / math.sqrt(2.0)
    gw = int(math.ceil(width / cell_size)) + 1
    gh = int(math.ceil(depth / cell_size)) + 1
    grid = np.full((gh, gw), -1, dtype=int)

    half_w = width / 2.0
    half_d = depth / 2.0
    r_min_sq = r_min * r_min

    # Seed point within domain
    p0 = (float(rng.uniform(-half_w * 0.8, half_w * 0.8)), float(rng.uniform(-half_d * 0.8, half_d * 0.8)))
    points: List[Tuple[float, float]] = [p0]
    active: List[int] = [0]

    gx0 = int((p0[0] + half_w) / cell_size)
    gy0 = int((p0[1] + half_d) / cell_size)
    if 0 <= gx0 < gw and 0 <= gy0 < gh:
        grid[gy0, gx0] = 0

    while active:
        idx = rng.integers(0, len(active))
        p_idx = active[idx]
        px, py = points[p_idx]
        found = False

        for _ in range(k):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(r_min, 2.0 * r_min)
            cx = px + rad * math.cos(theta)
            cy = py + rad * math.sin(theta)

            # Strict boundary clamping within [-W/2, W/2] and [-D/2, D/2]
            if cx < -half_w or cx > half_w or cy < -half_d or cy > half_d:
                continue

            cgx = int((cx + half_w) / cell_size)
            cgy = int((cy + half_d) / cell_size)
            if cgx < 0 or cgx >= gw or cgy < 0 or cgy >= gh:
                continue

            # Density acceptance test
            if density_evaluator is not None:
                prob = density_evaluator(cx, cy)
                if prob <= 0.0 or rng.uniform(0.0, 1.0) > prob:
                    continue

            # Check 5x5 grid neighborhood for collision
            conflict = False
            for ny in range(max(0, cgy - 2), min(gh, cgy + 3)):
                for nx in range(max(0, cgx - 2), min(gw, cgx + 3)):
                    nb_idx = grid[ny, nx]
                    if nb_idx != -1:
                        ox, oy = points[nb_idx]
                        if (cx - ox) ** 2 + (cy - oy) ** 2 < r_min_sq:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                new_idx = len(points)
                points.append((float(cx), float(cy)))
                active.append(new_idx)
                grid[cgy, cgx] = new_idx
                found = True
                break

        if not found:
            active.pop(idx)

    return points


class BiomeEngine:
    """
    Comprehensive procedural biome and ecological simulation engine:
    - Microclimate physics: lapse rate temperature, solar radiation insolation, soil moisture index
    - Multi-tier ecological stratification: canopy, understory, ground flora, decay/windthrow, creature impacts, underwater
    - Blue-noise Poisson-Disk clustering with strict boundary clamping [-W/2, W/2]
    """

    def __init__(
        self,
        hf: Heightfield2D,
        dist_to_water: np.ndarray,
        sun_vector: Optional[np.ndarray] = None,
        sea_level: float = 0.0,
    ):
        self.hf = hf
        self.dist_to_water = dist_to_water
        self.sea_level = sea_level
        self.slopes = hf.compute_slope_degrees()
        self.normals = hf.compute_normals()

        # Default sun vector (approx 135 deg SE, 45 deg elevation)
        if sun_vector is None:
            az = math.radians(135.0)
            el = math.radians(45.0)
            self.sun_vector = np.array([
                math.cos(el) * math.sin(az),
                math.cos(el) * math.cos(az),
                math.sin(el)
            ], dtype=np.float64)
        else:
            v = np.asarray(sun_vector, dtype=np.float64)
            norm = np.linalg.norm(v)
            self.sun_vector = v / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    # -------------------------------------------------------------------------
    # Microclimate Physics Formulations
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_temperature(
        elevation: np.ndarray,
        sea_level: float = 0.0,
        lapse_rate: float = 0.0065,
        base_temp: float = 20.0,
    ) -> np.ndarray:
        """
        Elevation temperature lapse rate: T(z) = T_0 - Gamma * (z - z_sea).
        Standard atmospheric lapse rate is 0.0065 C/m (0.65 C per 100m).
        """
        elev_above_sea = np.maximum(0.0, elevation - sea_level)
        return base_temp - lapse_rate * elev_above_sea

    @staticmethod
    def compute_insolation(
        normals: np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray],
        sun_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solar radiation insolation aspect: dot(n, L_sun) or -dot(n, L_ray), clamped to [0.0, 1.0].
        Calculates direct solar irradiation based on terrain surface normals.
        """
        if sun_vector is None:
            az = math.radians(135.0)
            el = math.radians(45.0)
            sun_vec = np.array([
                math.cos(el) * math.sin(az),
                math.cos(el) * math.cos(az),
                math.sin(el)
            ], dtype=np.float64)
        else:
            sun_vec = np.asarray(sun_vector, dtype=np.float64)
            norm = np.linalg.norm(sun_vec)
            sun_vec = sun_vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        if isinstance(normals, (tuple, list)):
            nx, ny, nz = normals
            norm_arr = np.dstack([nx, ny, nz])
        else:
            norm_arr = normals

        # Dot product with sun vector
        dot = np.sum(norm_arr * sun_vec, axis=-1)
        # If sun_vector points down towards ground (ray direction), invert dot
        if sun_vec[2] < 0:
            insolation = -dot
        else:
            insolation = dot
        return np.clip(insolation, 0.0, 1.0)

    @staticmethod
    def compute_moisture(
        water_dist_field: np.ndarray,
        slopes: Optional[np.ndarray] = None,
        insolation: Optional[np.ndarray] = None,
        sigma_moisture: float = 15.0,
        base_rain: float = 0.2,
    ) -> np.ndarray:
        """
        Continuous soil moisture index field M(x, y) in [0.0, 1.0].
        Synthesizes proximity to water bodies (rivers, lakes, ocean),
        topographic slope drainage (cos theta), and solar insolation drying.
        """
        water_factor = np.exp(-water_dist_field / max(1.0, sigma_moisture))
        if slopes is not None:
            drainage = np.cos(np.radians(slopes))
        else:
            drainage = 1.0

        moisture = 0.5 * water_factor * drainage + base_rain
        if insolation is not None:
            moisture -= 0.15 * insolation

        return np.clip(moisture, 0.0, 1.0)

    def get_temperature_field(
        self,
        lapse_rate: float = 0.0065,
        base_temp: float = 20.0,
    ) -> np.ndarray:
        """Calculates temperature field across heightfield."""
        return self.compute_temperature(self.hf.Z, self.sea_level, lapse_rate, base_temp)

    def get_insolation_field(self) -> np.ndarray:
        """Calculates solar insolation aspect field across heightfield."""
        return self.compute_insolation(self.normals, self.sun_vector)

    def get_moisture_field(
        self,
        sigma_moisture: float = 15.0,
        base_rain: float = 0.2,
    ) -> np.ndarray:
        """Calculates soil moisture index field across heightfield."""
        insolation = self.get_insolation_field()
        return self.compute_moisture(self.dist_to_water, self.slopes, insolation, sigma_moisture, base_rain)

    def compute_microclimate(
        self,
        lapse_rate: float = 0.0065,
        base_temp: float = 20.0,
        sigma_moisture: float = 15.0,
    ) -> Dict[str, np.ndarray]:
        """Synthesizes all microclimate fields into a dictionary."""
        temp = self.get_temperature_field(lapse_rate, base_temp)
        insolation = self.get_insolation_field()
        moisture = self.compute_moisture(self.dist_to_water, self.slopes, insolation, sigma_moisture)
        return {
            "temperature": temp,
            "insolation": insolation,
            "moisture": moisture,
            "slopes": self.slopes,
        }

    # -------------------------------------------------------------------------
    # Ecological Suitability Evaluation
    # -------------------------------------------------------------------------

    def evaluate_suitability(self, config: BiomeScatterConfig) -> np.ndarray:
        """
        Calculates a [0.0, 1.0] suitability mask across the heightfield for a specific biome.
        Takes into account altitude, slope, and water distance.
        Maintains backward compatibility with all baseline tests.
        """
        Z = self.hf.Z
        slopes = self.slopes
        d_water = self.dist_to_water

        # 1. Altitude Mask
        alt_mask = (Z >= config.min_altitude) & (Z <= config.max_altitude)

        # 2. Slope Mask
        slope_mask = (slopes >= config.min_slope_deg) & (slopes <= config.max_slope_deg)

        # 3. Water Distance Mask
        water_mask = (d_water >= config.min_water_dist) & (d_water <= config.max_water_dist)

        suitability = (alt_mask & slope_mask & water_mask).astype(np.float64)
        return suitability

    def evaluate_microclimate_suitability(
        self,
        config: BiomeScatterConfig,
        min_temp: float = -10.0,
        max_temp: float = 35.0,
        min_moisture: float = 0.0,
        max_moisture: float = 1.0,
    ) -> np.ndarray:
        """
        Continuous ecological niche suitability modulated by temperature lapse rate,
        soil moisture index, and solar insolation.
        """
        base_suit = self.evaluate_suitability(config)
        if not np.any(base_suit > 0.0):
            return base_suit

        climate = self.compute_microclimate()
        temp = climate["temperature"]
        moist = climate["moisture"]

        temp_mask = (temp >= min_temp) & (temp <= max_temp)
        moist_mask = (moist >= min_moisture) & (moist <= max_moisture)

        temp_center = 0.5 * (min_temp + max_temp)
        temp_sigma = max(1.0, 0.4 * (max_temp - min_temp))
        temp_factor = np.exp(-0.5 * ((temp - temp_center) / temp_sigma) ** 2)

        return np.clip(base_suit * temp_mask * moist_mask * temp_factor * (0.5 + 0.5 * moist), 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Biological Game Trails, Disturbed Soil & Underwater Zones
    # -------------------------------------------------------------------------

    def generate_game_trails(
        self,
        count: int = 3,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """
        Generates continuous animal game trails / creature paths traversing from
        inland biomes towards water sources along gentle slopes (least cost paths).
        Returns list of (N, 2) world coordinate trails.
        """
        rng = np.random.default_rng(seed)
        trails: List[np.ndarray] = []
        half_w = self.hf.width / 2.0
        half_d = self.hf.depth / 2.0

        for _ in range(count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist_start = rng.uniform(half_w * 0.4, half_w * 0.8)
            curr_x = math.cos(angle) * dist_start
            curr_y = math.sin(angle) * dist_start

            trail_pts = [(curr_x, curr_y)]
            for _ in range(35):
                gx, gy = self.hf.world_to_grid(curr_x, curr_y)
                gx_i = int(np.clip(round(gx), 0, self.hf.res_x - 1))
                gy_i = int(np.clip(round(gy), 0, self.hf.res_y - 1))
                if self.dist_to_water[gy_i, gx_i] < 2.0:
                    break

                best_step = None
                best_score = float("inf")
                for step_angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                    step_x = curr_x + math.cos(step_angle) * 2.5
                    step_y = curr_y + math.sin(step_angle) * 2.5
                    if -half_w <= step_x <= half_w and -half_d <= step_y <= half_d:
                        sgx, sgy = self.hf.world_to_grid(step_x, step_y)
                        sgx_i = int(np.clip(round(sgx), 0, self.hf.res_x - 1))
                        sgy_i = int(np.clip(round(sgy), 0, self.hf.res_y - 1))
                        water_d = self.dist_to_water[sgy_i, sgx_i]
                        slope_val = self.slopes[sgy_i, sgx_i]
                        score = water_d + 1.5 * slope_val
                        if score < best_score:
                            best_score = score
                            best_step = (step_x, step_y)

                if best_step is None or best_step == (curr_x, curr_y):
                    break
                curr_x, curr_y = best_step
                trail_pts.append((curr_x, curr_y))

            if len(trail_pts) >= 4:
                trails.append(np.array(trail_pts, dtype=np.float64))

        return trails

    def compute_game_trail_mask(
        self,
        trails: Optional[List[np.ndarray]] = None,
        trail_width: float = 2.0,
    ) -> np.ndarray:
        """
        Rasterizes animal game trails into a [0.0, 1.0] continuous proximity mask
        for processing trampled grass zones and creature trampling impacts.
        """
        if trails is None:
            trails = self.generate_game_trails(count=3)

        mask = np.zeros_like(self.hf.Z)
        X, Y = self.hf.X, self.hf.Y

        for trail in trails:
            for i in range(len(trail) - 1):
                p0, p1 = trail[i], trail[i + 1]
                v = p1 - p0
                v_len_sq = float(np.dot(v, v))
                if v_len_sq < 1e-6:
                    continue
                u = np.clip(((X - p0[0]) * v[0] + (Y - p0[1]) * v[1]) / v_len_sq, 0.0, 1.0)
                proj_x = p0[0] + u * v[0]
                proj_y = p0[1] + u * v[1]
                dist_sq = (X - proj_x) ** 2 + (Y - proj_y) ** 2
                seg_mask = np.exp(-dist_sq / (2.0 * (trail_width ** 2)))
                mask = np.maximum(mask, seg_mask)

        return np.clip(mask, 0.0, 1.0)

    def compute_disturbed_soil_mask(
        self,
        trail_mask: Optional[np.ndarray] = None,
        churn_patches: int = 5,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Computes naturally churned / disturbed soil patches around animal trails
        and creature foraging areas.
        """
        if trail_mask is None:
            trail_mask = self.compute_game_trail_mask()

        rng = np.random.default_rng(seed)
        soil_mask = trail_mask.copy() * 0.75

        valid_y, valid_x = np.where((self.dist_to_water < 25.0) & (self.slopes < 20.0))
        if len(valid_y) > 0:
            for _ in range(churn_patches):
                idx = rng.integers(0, len(valid_y))
                cy, cx = valid_y[idx], valid_x[idx]
                wx, wy = self.hf.grid_to_world(cx, cy)
                radius = rng.uniform(2.5, 6.0)
                dist_sq = (self.hf.X - wx) ** 2 + (self.hf.Y - wy) ** 2
                patch = np.exp(-dist_sq / (2.0 * (radius ** 2)))
                soil_mask = np.maximum(soil_mask, patch)

        return np.clip(soil_mask, 0.0, 1.0)

    def compute_underwater_ecosystem_mask(
        self,
        max_depth: float = 15.0,
        max_slope_deg: float = 45.0,
    ) -> np.ndarray:
        """
        Computes suitability mask for underwater ecosystems:
        aquatic seaweed, waterweeds, reeds, riverbed pebbles, and submerged reefs.
        Area must be at or below sea level (or water level) and within slope limits.
        """
        depth = self.sea_level - self.hf.Z
        submerged = (depth >= -0.2) & (depth <= max_depth) & (self.slopes <= max_slope_deg)
        return submerged.astype(np.float64)

    # -------------------------------------------------------------------------
    # Fast O(N) Blue-Noise Poisson-Disk Scatter
    # -------------------------------------------------------------------------

    def generate_poisson_scatter(
        self,
        config: BiomeScatterConfig,
        max_samples: int = 2000,
        density_map: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Generates Poisson Disk distributed instance coordinates (x, y, z, scale, rot_z)
        meeting the biome's ecological criteria.
        Guarantees:
        - Blue-noise spatial distribution with minimum separation distance r_min
        - Strict clamping of all (x, y) coordinates within [-W/2, W/2] and [-D/2, D/2]
        - Seed-based deterministic reproducibility
        """
        suitability = self.evaluate_suitability(config)
        if density_map is not None:
            suitability = suitability * density_map

        valid_indices = np.argwhere(suitability > 0.05)
        if len(valid_indices) == 0:
            return []

        rng = np.random.default_rng(config.seed)
        area_per_cell = self.hf.dx * self.hf.dy
        total_valid_area = len(valid_indices) * area_per_cell
        target_count = min(max_samples, int(total_valid_area * config.density))

        if target_count <= 0:
            return []

        r_min = config.poisson_dist_min
        r_min_sq = r_min * r_min

        cell_size = r_min / math.sqrt(2.0)
        grid_w = int(self.hf.width / cell_size) + 1
        grid_h = int(self.hf.depth / cell_size) + 1
        spatial_grid = np.full((grid_h, grid_w), -1, dtype=int)

        half_w = self.hf.width / 2.0
        half_d = self.hf.depth / 2.0
        points: List[Tuple[float, float, float, float, float]] = []

        shuffled = rng.permutation(valid_indices)

        for gy, gx in shuffled:
            if len(points) >= target_count:
                break

            jx = rng.uniform(-0.35, 0.35)
            jy = rng.uniform(-0.35, 0.35)
            wx, wy = self.hf.grid_to_world(gx + jx, gy + jy)

            # Strict coordinate clamping within [-W/2, W/2] and [-D/2, D/2]
            wx = float(np.clip(wx, -half_w, half_w))
            wy = float(np.clip(wy, -half_d, half_d))

            prob = float(suitability[gy, gx])
            if prob < 1.0 and rng.uniform(0.0, 1.0) > prob:
                continue

            sgx = int((wx + half_w) / cell_size)
            sgy = int((wy + half_d) / cell_size)
            if sgx < 0 or sgx >= grid_w or sgy < 0 or sgy >= grid_h:
                continue

            conflict = False
            for dy in range(max(0, sgy - 2), min(grid_h, sgy + 3)):
                for dx in range(max(0, sgx - 2), min(grid_w, sgx + 3)):
                    pidx = spatial_grid[dy, dx]
                    if pidx != -1:
                        px, py = points[pidx][0], points[pidx][1]
                        if (wx - px) ** 2 + (wy - py) ** 2 < r_min_sq:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                wz = float(self.hf.sample_bilinear(wx, wy))
                scale = float(rng.uniform(config.scale_min, config.scale_max))
                rot_z = float(rng.uniform(0.0, 2.0 * math.pi))

                spatial_grid[sgy, sgx] = len(points)
                points.append((wx, wy, wz, scale, rot_z))

        return points

    def poisson_scatter(
        self,
        suitability_map: np.ndarray,
        min_radius: float,
        max_count: int = 1000,
        seed: int = 42,
    ) -> List[Tuple[float, float, float]]:
        """
        Interface contract with M1/M2/M4:
        Given a 2D suitability map, generates blue-noise scatter points (x, y, z)
        strictly clamped to [-W/2, W/2] and [-D/2, D/2].
        """
        valid_indices = np.argwhere(suitability_map > 0.05)
        if len(valid_indices) == 0:
            return []

        rng = np.random.default_rng(seed)
        r_min = max(0.1, min_radius)
        r_min_sq = r_min * r_min
        cell_size = r_min / math.sqrt(2.0)
        grid_w = int(self.hf.width / cell_size) + 1
        grid_h = int(self.hf.depth / cell_size) + 1
        spatial_grid = np.full((grid_h, grid_w), -1, dtype=int)

        half_w = self.hf.width / 2.0
        half_d = self.hf.depth / 2.0
        points: List[Tuple[float, float, float]] = []

        shuffled = rng.permutation(valid_indices)

        for gy, gx in shuffled:
            if len(points) >= max_count:
                break

            jx = rng.uniform(-0.35, 0.35)
            jy = rng.uniform(-0.35, 0.35)
            wx, wy = self.hf.grid_to_world(gx + jx, gy + jy)

            # Clamp coordinates
            wx = float(np.clip(wx, -half_w, half_w))
            wy = float(np.clip(wy, -half_d, half_d))

            prob = float(suitability_map[gy, gx])
            if prob < 1.0 and rng.uniform(0.0, 1.0) > prob:
                continue

            sgx = int((wx + half_w) / cell_size)
            sgy = int((wy + half_d) / cell_size)
            if sgx < 0 or sgx >= grid_w or sgy < 0 or sgy >= grid_h:
                continue

            conflict = False
            for dy in range(max(0, sgy - 2), min(grid_h, sgy + 3)):
                for dx in range(max(0, sgx - 2), min(grid_w, sgx + 3)):
                    pidx = spatial_grid[dy, dx]
                    if pidx != -1:
                        px, py = points[pidx][0], points[pidx][1]
                        if (wx - px) ** 2 + (wy - py) ** 2 < r_min_sq:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                wz = float(self.hf.sample_bilinear(wx, wy))
                spatial_grid[sgy, sgx] = len(points)
                points.append((wx, wy, wz))

        return points

    # -------------------------------------------------------------------------
    # Stratified Biome Config Generator Helper
    # -------------------------------------------------------------------------

    @classmethod
    def create_stratified_biome_configs(
        cls,
        include_underwater: bool = True,
        include_decay: bool = True,
        include_creature_trails: bool = True,
    ) -> List[BiomeScatterConfig]:
        """
        Creates a comprehensive suite of stratified BiomeScatterConfig objects
        covering all ecological tiers:
        - Canopy trees (alpine pines, deciduous oaks, tropical palms)
        - Understory shrubs & bushes
        - Rich ground flora (wildflowers, dense grass, ferns, forest mushrooms, rock moss)
        - Natural tree decay and windthrow (fallen logs, snapped trunks, weathered stumps)
        - Biological trails & disturbed soil zones
        - Underwater ecosystems (seaweed, waterweeds, reeds, riverbed pebbles, coral reefs)
        """
        configs: List[BiomeScatterConfig] = []
        seed_counter = 100

        for proto_name, props in PROTOTYPE_TAXONOMY.items():
            tier = props["tier"]
            if tier == EcologicalTier.UNDERWATER and not include_underwater:
                continue
            if tier == EcologicalTier.DECAY_WINDTHROW and not include_decay:
                continue
            if tier == EcologicalTier.BIOLOGICAL_IMPACT and not include_creature_trails:
                continue

            cfg = BiomeScatterConfig(
                name=f"Tier_{tier.value}_{proto_name}",
                prototype=proto_name,
                min_altitude=props["min_alt"],
                max_altitude=props["max_alt"],
                min_slope_deg=0.0,
                max_slope_deg=props["max_slope"],
                density=props["density"],
                poisson_dist_min=props["poisson_dist"],
                seed=seed_counter,
            )
            configs.append(cfg)
            seed_counter += 1

        return configs
