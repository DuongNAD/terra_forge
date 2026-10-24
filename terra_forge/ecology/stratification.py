"""
Organic Ecological Stratification Engine for TerraForge.
Simulates microclimate stratification based on geology, slope gradients, hydrological flow,
solar exposure aspect, and soil moisture indices.

Physics-Informed Placement Rules:
1. Fallen timber oriented along slope gradients (-grad Z) or flood vectors (flow).
2. Fluvial pebbles deposited in stream beds based on flow velocity and low slope basins.
3. Moisture-loving moss, ferns, and fungi concentrated in shadowed, high-humidity microclimates.
4. Deep-root canopies strictly avoiding sheer vertical cliff faces (slope > 35 deg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from terra_forge.ai.vault import FloraType


class OrientationRule(str, Enum):
    """Orientation behavior for stratified biological and geological assets."""
    RANDOM_YAW = "random_yaw"              # Random rotation around vertical axis (standard trees)
    SLOPE_GRADIENT = "slope_gradient"      # Long axis oriented along downslope descent vector
    FLOW_VECTOR = "flow_vector"            # Aligned with hydrological stream / flood flow velocity
    SURFACE_NORMAL = "surface_normal"      # Tilted perpendicular to terrain surface (cliff moss)
    UPRIGHT_SURFACE_LEAN = "upright_lean"  # Upright with subtle lean towards surface normal


@dataclass
class StratificationRule:
    """Physics-informed rule defining ecological fitness and spatial distribution."""
    name: str
    flora_type: Optional[FloraType] = None
    tier: str = "canopy"

    # Slope constraints in degrees [0, 90]
    min_slope_deg: float = 0.0
    max_slope_deg: float = 35.0
    optimal_slope_deg: Optional[float] = None

    # Moisture constraints [0.0, 1.0]
    min_moisture: float = 0.20
    max_moisture: float = 0.85
    optimal_moisture: Optional[float] = None

    # Solar insolation aspect [0.0, 1.0] (0.0 = deep shadow/north, 1.0 = direct sunlight/south)
    min_insolation: float = 0.0
    max_insolation: float = 1.0
    optimal_insolation: Optional[float] = None

    # Hydrological flow velocity / accumulation
    min_flow: float = 0.0
    max_flow: float = 80.0
    optimal_flow: Optional[float] = None

    # Elevation range in meters
    min_elevation: float = 0.5
    max_elevation: float = 80.0

    # Orientation rule
    orientation_rule: OrientationRule = OrientationRule.RANDOM_YAW

    # Spatial Poisson parameters
    base_density: float = 0.10
    min_distance: float = 2.5
    scale_range: Tuple[float, float] = (0.85, 1.25)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "flora_type": self.flora_type.name if self.flora_type else None,
            "tier": self.tier,
            "slope_deg": [self.min_slope_deg, self.max_slope_deg],
            "moisture": [self.min_moisture, self.max_moisture],
            "insolation": [self.min_insolation, self.max_insolation],
            "flow": [self.min_flow, self.max_flow],
            "elevation": [self.min_elevation, self.max_elevation],
            "orientation_rule": self.orientation_rule.value,
            "base_density": self.base_density,
            "min_distance": self.min_distance,
        }


@dataclass
class StratifiedPlacement:
    """A verified, physics-aligned instance placement."""
    x: float
    y: float
    z: float
    rotation_euler: Tuple[float, float, float]       # (pitch, yaw, roll) in radians
    rotation_quaternion: Tuple[float, float, float, float]  # (x, y, z, w)
    scale: float
    suitability: float
    rule_name: str
    flora_type: Optional[FloraType] = None


# -----------------------------------------------------------------------------
# Canonical Stratification Rules
# -----------------------------------------------------------------------------

CANOPY_ALPINE_PINE_RULE = StratificationRule(
    name="Canopy_Alpine_Pine",
    flora_type=FloraType.Pine,
    tier="canopy",
    min_slope_deg=0.0,
    max_slope_deg=35.0,  # Deep-root canopy strictly avoids steep cliff faces > 35°
    optimal_slope_deg=12.0,
    min_moisture=0.20,
    max_moisture=0.75,
    optimal_moisture=0.45,
    min_insolation=0.20,
    max_insolation=1.00,
    min_flow=0.0,
    max_flow=40.0,       # Avoids riverbed scour channels
    min_elevation=3.0,
    max_elevation=50.0,
    orientation_rule=OrientationRule.UPRIGHT_SURFACE_LEAN,
    base_density=0.08,
    min_distance=3.2,
)

CANOPY_BROADLEAF_OAK_RULE = StratificationRule(
    name="Canopy_Broadleaf_Oak",
    flora_type=FloraType.Round,
    tier="canopy",
    min_slope_deg=0.0,
    max_slope_deg=28.0,  # Broadleaf trees prefer gentle lowlands and rolling hills
    optimal_slope_deg=8.0,
    min_moisture=0.35,
    max_moisture=0.80,
    optimal_moisture=0.60,
    min_insolation=0.30,
    max_insolation=1.00,
    min_flow=0.0,
    max_flow=35.0,
    min_elevation=1.0,
    max_elevation=25.0,
    orientation_rule=OrientationRule.UPRIGHT_SURFACE_LEAN,
    base_density=0.09,
    min_distance=3.5,
)

FALLEN_TIMBER_RULE = StratificationRule(
    name="Fallen_Timber_Windthrow",
    flora_type=FloraType.DeadTree,
    tier="decay",
    min_slope_deg=2.0,
    max_slope_deg=30.0,
    min_moisture=0.30,
    max_moisture=0.95,   # Rotting wood thrives in damp microclimates
    optimal_moisture=0.70,
    min_insolation=0.0,
    max_insolation=0.85,
    min_flow=0.0,
    max_flow=150.0,      # Can settle in stream drift margins
    min_elevation=0.8,
    max_elevation=35.0,
    orientation_rule=OrientationRule.SLOPE_GRADIENT,  # Downslope or stream flow alignment
    base_density=0.04,
    min_distance=4.0,
)

FLUVIAL_PEBBLE_RULE = StratificationRule(
    name="Fluvial_River_Pebbles",
    flora_type=FloraType.Rock,
    tier="aquatic",
    min_slope_deg=0.0,
    max_slope_deg=18.0,  # Deposited in low-slope alluvial beds and stream depressions
    optimal_slope_deg=5.0,
    min_moisture=0.50,
    max_moisture=1.00,
    optimal_moisture=None,
    min_insolation=0.0,
    max_insolation=1.0,
    min_flow=25.0,       # High stream velocity deposits rounded fluvial pebbles
    max_flow=500.0,
    optimal_flow=60.0,
    min_elevation=0.1,
    max_elevation=20.0,
    orientation_rule=OrientationRule.FLOW_VECTOR,
    base_density=0.25,
    min_distance=1.0,
)

SHADOW_MOISTURE_MOSS_FERN_RULE = StratificationRule(
    name="Shadow_Moisture_Moss_Fern",
    flora_type=FloraType.Bush,
    tier="ground",
    min_slope_deg=0.0,
    max_slope_deg=75.0,  # Thrives on steep rock walls and cliff faces
    optimal_slope_deg=None,
    min_moisture=0.55,   # High-humidity zones
    max_moisture=1.00,
    optimal_moisture=None,
    min_insolation=0.0,
    max_insolation=0.40, # Concentrated in shadowed north-facing aspects and grottoes
    optimal_insolation=None,
    min_flow=0.0,
    max_flow=60.0,
    min_elevation=1.0,
    max_elevation=45.0,
    orientation_rule=OrientationRule.SURFACE_NORMAL,  # Clings to cliff surface
    base_density=0.18,
    min_distance=1.5,
)

CLIFF_SCREE_BASALT_RULE = StratificationRule(
    name="Cliff_Scree_Basalt",
    flora_type=FloraType.Rock,
    tier="mineral",
    min_slope_deg=35.0,  # Sheer rock faces and scree talus slopes
    max_slope_deg=85.0,
    optimal_slope_deg=55.0,
    min_moisture=0.0,
    max_moisture=0.60,
    min_insolation=0.0,
    max_insolation=1.0,
    min_flow=0.0,
    max_flow=30.0,
    min_elevation=5.0,
    max_elevation=75.0,
    orientation_rule=OrientationRule.SURFACE_NORMAL,
    base_density=0.10,
    min_distance=2.8,
)

WETLAND_REEDS_RULE = StratificationRule(
    name="Wetland_Riparian_Reeds",
    flora_type=FloraType.Reed,
    tier="aquatic",
    min_slope_deg=0.0,
    max_slope_deg=15.0,  # Flat shoreline margins
    min_moisture=0.75,
    max_moisture=1.00,
    optimal_moisture=0.90,
    min_insolation=0.20,
    max_insolation=1.0,
    min_flow=0.0,
    max_flow=60.0,
    min_elevation=0.1,
    max_elevation=6.0,
    orientation_rule=OrientationRule.RANDOM_YAW,
    base_density=0.20,
    min_distance=1.2,
)

STANDARD_ECOLOGY_RULES: List[StratificationRule] = [
    CANOPY_ALPINE_PINE_RULE,
    CANOPY_BROADLEAF_OAK_RULE,
    FALLEN_TIMBER_RULE,
    FLUVIAL_PEBBLE_RULE,
    SHADOW_MOISTURE_MOSS_FERN_RULE,
    CLIFF_SCREE_BASALT_RULE,
    WETLAND_REEDS_RULE,
]


# -----------------------------------------------------------------------------
# Ecological Stratifier Core
# -----------------------------------------------------------------------------

class EcologicalStratifier:
    """
    Evaluates microclimate fitness and applies physics-informed orientation,
    enforcing slope exclusions, stream bed fluvial accumulation, and moisture/shadow niches.
    """

    def __init__(self, rules: Optional[List[StratificationRule]] = None):
        self.rules = rules or list(STANDARD_ECOLOGY_RULES)

    @staticmethod
    def evaluate_factor_curve(
        value: float,
        val_min: float,
        val_max: float,
        val_opt: Optional[float] = None,
    ) -> float:
        """
        Smooth trapezoidal/Gaussian fitness curve returning [0.0, 1.0].
        Strict 0.0 outside [val_min, val_max].
        """
        if value < val_min or value > val_max:
            return 0.0

        if val_opt is None:
            # Flat plateau with smooth edge taper (10% margins)
            span = val_max - val_min
            if span <= 1e-6:
                return 1.0
            margin = span * 0.12
            if value < val_min + margin:
                return float((value - val_min) / margin)
            if value > val_max - margin:
                return float((val_max - value) / margin)
            return 1.0

        # Asymmetric triangular/Gaussian around optimal
        if value <= val_opt:
            left_span = val_opt - val_min
            return 1.0 if left_span <= 1e-6 else float((value - val_min) / left_span)
        else:
            right_span = val_max - val_opt
            return 1.0 if right_span <= 1e-6 else float((val_max - value) / right_span)

    def evaluate_point(
        self,
        rule: StratificationRule,
        elevation: float,
        slope_deg: float,
        flow: float,
        insolation: float,
        moisture: float,
    ) -> float:
        """
        Evaluate ecological suitability score in [0.0, 1.0] for a single point.
        Strict zero tolerance for hard constraints (e.g. slope ceiling on canopy trees).
        """
        # 1. Hard Cliff Exclusion for Canopies
        if rule.tier == "canopy" and slope_deg > rule.max_slope_deg:
            return 0.0

        # 2. Hard Elevation Range
        if elevation < rule.min_elevation or elevation > rule.max_elevation:
            return 0.0

        # 3. Factor Evaluations
        s_slope = self.evaluate_factor_curve(slope_deg, rule.min_slope_deg, rule.max_slope_deg, rule.optimal_slope_deg)
        s_moist = self.evaluate_factor_curve(moisture, rule.min_moisture, rule.max_moisture, rule.optimal_moisture)
        s_insol = self.evaluate_factor_curve(insolation, rule.min_insolation, rule.max_insolation, rule.optimal_insolation)
        s_flow = self.evaluate_factor_curve(flow, rule.min_flow, rule.max_flow, rule.optimal_flow)

        # Multiplicative composite with geometric mean scaling
        total = s_slope * s_moist * s_insol * s_flow
        return float(np.clip(total, 0.0, 1.0))

    def compute_suitability_map(
        self,
        rule: StratificationRule,
        elevation: np.ndarray,
        slope_deg: np.ndarray,
        moisture: np.ndarray,
        insolation: np.ndarray,
        flow: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized 2D array evaluation across continuous environmental grids.
        Returns float32 array in [0.0, 1.0] with exact dimensions of input grids.
        """
        elev_arr = np.asarray(elevation, dtype=np.float32)
        slope_arr = np.asarray(slope_deg, dtype=np.float32)
        moist_arr = np.asarray(moisture, dtype=np.float32)
        insol_arr = np.asarray(insolation, dtype=np.float32)
        flow_arr = np.asarray(flow, dtype=np.float32)

        # Elevation mask
        mask = (elev_arr >= rule.min_elevation) & (elev_arr <= rule.max_elevation)

        # Hard cliff canopy constraint
        if rule.tier == "canopy":
            mask &= (slope_arr <= rule.max_slope_deg)

        # Slope suitability
        s_slope = np.zeros_like(slope_arr)
        slope_valid = (slope_arr >= rule.min_slope_deg) & (slope_arr <= rule.max_slope_deg)
        if rule.optimal_slope_deg is not None and slope_valid.any():
            opt = rule.optimal_slope_deg
            left = (slope_arr <= opt) & slope_valid
            right = (slope_arr > opt) & slope_valid
            if opt > rule.min_slope_deg:
                s_slope[left] = (slope_arr[left] - rule.min_slope_deg) / (opt - rule.min_slope_deg)
            else:
                s_slope[left] = 1.0
            if rule.max_slope_deg > opt:
                s_slope[right] = (rule.max_slope_deg - slope_arr[right]) / (rule.max_slope_deg - opt)
            else:
                s_slope[right] = 1.0
        else:
            s_slope[slope_valid] = 1.0

        # Moisture suitability
        s_moist = np.zeros_like(moist_arr)
        moist_valid = (moist_arr >= rule.min_moisture) & (moist_arr <= rule.max_moisture)
        if rule.optimal_moisture is not None and moist_valid.any():
            opt_m = rule.optimal_moisture
            left_m = (moist_arr <= opt_m) & moist_valid
            right_m = (moist_arr > opt_m) & moist_valid
            if opt_m > rule.min_moisture:
                s_moist[left_m] = (moist_arr[left_m] - rule.min_moisture) / (opt_m - rule.min_moisture)
            else:
                s_moist[left_m] = 1.0
            if rule.max_moisture > opt_m:
                s_moist[right_m] = (rule.max_moisture - moist_arr[right_m]) / (rule.max_moisture - opt_m)
            else:
                s_moist[right_m] = 1.0
        else:
            s_moist[moist_valid] = 1.0

        # Insolation suitability
        s_insol = np.zeros_like(insol_arr)
        insol_valid = (insol_arr >= rule.min_insolation) & (insol_arr <= rule.max_insolation)
        if rule.optimal_insolation is not None and insol_valid.any():
            opt_i = rule.optimal_insolation
            left_i = (insol_arr <= opt_i) & insol_valid
            right_i = (insol_arr > opt_i) & insol_valid
            if opt_i > rule.min_insolation:
                s_insol[left_i] = (insol_arr[left_i] - rule.min_insolation) / (opt_i - rule.min_insolation)
            else:
                s_insol[left_i] = 1.0
            if rule.max_insolation > opt_i:
                s_insol[right_i] = (rule.max_insolation - insol_arr[right_i]) / (rule.max_insolation - opt_i)
            else:
                s_insol[right_i] = 1.0
        else:
            s_insol[insol_valid] = 1.0

        # Flow suitability
        s_flow = np.zeros_like(flow_arr)
        flow_valid = (flow_arr >= rule.min_flow) & (flow_arr <= rule.max_flow)
        if rule.optimal_flow is not None and flow_valid.any():
            opt_f = rule.optimal_flow
            left_f = (flow_arr <= opt_f) & flow_valid
            right_f = (flow_arr > opt_f) & flow_valid
            if opt_f > rule.min_flow:
                s_flow[left_f] = (flow_arr[left_f] - rule.min_flow) / (opt_f - rule.min_flow)
            else:
                s_flow[left_f] = 1.0
            if rule.max_flow > opt_f:
                s_flow[right_f] = (rule.max_flow - flow_arr[right_f]) / (rule.max_flow - opt_f)
            else:
                s_flow[right_f] = 1.0
        else:
            s_flow[flow_valid] = 1.0

        suitability = s_slope * s_moist * s_insol * s_flow * mask
        return np.clip(suitability, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def calculate_orientation(
        rule: StratificationRule,
        slope_gradient: Tuple[float, float],  # (dz/dx, dz/dy)
        flow_vector: Optional[Tuple[float, float]] = None,  # (vx, vy)
        surface_normal: Optional[Tuple[float, float, float]] = None,  # (nx, ny, nz)
        random_seed: Optional[int] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Compute physically motivated 3D Euler angles (pitch, yaw, roll in radians)
        and unit quaternion (x, y, z, w).

        Rules:
        - SLOPE_GRADIENT: Rotates yaw to point downhill along -grad(Z).
        - FLOW_VECTOR: Rotates yaw to point along stream flow velocity (vx, vy).
        - SURFACE_NORMAL: Aligns up vector with terrain normal (cliff-clinging moss/scree).
        - UPRIGHT_SURFACE_LEAN: Upright vertical with gentle lean towards normal (< 12°).
        - RANDOM_YAW: Standard stochastic yaw.
        """
        rng = np.random.default_rng(random_seed)
        gx, gy = slope_gradient
        grad_mag = math.sqrt(gx * gx + gy * gy)

        pitch = 0.0
        yaw = 0.0
        roll = 0.0

        if rule.orientation_rule == OrientationRule.SLOPE_GRADIENT:
            # If in high stream flow, flow vector takes precedence over gravity
            if flow_vector is not None and (abs(flow_vector[0]) > 0.1 or abs(flow_vector[1]) > 0.1):
                vx, vy = flow_vector
                yaw = math.atan2(vy, vx)
            elif grad_mag > 1e-4:
                # Downslope direction is -gradient
                yaw = math.atan2(-gy, -gx)
                # Pitch tilts slightly along the slope incline
                pitch = math.atan(grad_mag)
            else:
                yaw = float(rng.uniform(0, 2.0 * math.pi))

        elif rule.orientation_rule == OrientationRule.FLOW_VECTOR:
            if flow_vector is not None and (abs(flow_vector[0]) > 1e-4 or abs(flow_vector[1]) > 1e-4):
                yaw = math.atan2(flow_vector[1], flow_vector[0])
            else:
                yaw = float(rng.uniform(0, 2.0 * math.pi))

        elif rule.orientation_rule == OrientationRule.SURFACE_NORMAL and surface_normal is not None:
            nx, ny, nz = surface_normal
            # Align Y-axis to normal
            # pitch around X, roll around Z
            norm_xz = math.sqrt(nx * nx + nz * nz)
            if norm_xz > 1e-4:
                yaw = math.atan2(nx, nz)
                pitch = math.atan2(norm_xz, ny)
            else:
                yaw = float(rng.uniform(0, 2.0 * math.pi))

        elif rule.orientation_rule == OrientationRule.UPRIGHT_SURFACE_LEAN and surface_normal is not None:
            # Mostly upright with subtle normal bias
            nx, ny, nz = surface_normal
            yaw = float(rng.uniform(0, 2.0 * math.pi))
            pitch = float(np.clip(-math.atan2(nz, ny) * 0.25, -0.2, 0.2))
            roll = float(np.clip(math.atan2(nx, ny) * 0.25, -0.2, 0.2))

        else:
            yaw = float(rng.uniform(0, 2.0 * math.pi))

        # Convert Euler (pitch, yaw, roll) to Quaternion (x, y, z, w)
        # Order: Yaw (Y) -> Pitch (X) -> Roll (Z) (q = q_Y * q_X * q_Z)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sp * cy * cr + sy * cp * sr
        qy = sy * cp * cr - sp * cy * sr
        qz = cy * cp * sr - sy * sp * cr
        qw = cy * cp * cr + sy * sp * sr

        # Normalize quaternion
        q_len = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if q_len > 1e-6:
            qx /= q_len
            qy /= q_len
            qz /= q_len
            qw /= q_len
        else:
            qw = 1.0

        return (pitch, yaw, roll), (qx, qy, qz, qw)

    def generate_stratified_points(
        self,
        rule: StratificationRule,
        elevation: np.ndarray,
        slope_deg: np.ndarray,
        moisture: np.ndarray,
        insolation: np.ndarray,
        flow: np.ndarray,
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
        suitability_threshold: float = 0.25,
        max_instances: int = 500,
        seed: int = 42,
    ) -> List[StratifiedPlacement]:
        """
        Generate placed instances adhering to biological suitability masks and physical orientations.
        Coordinates are centered in world space [min_x, max_x] x [min_z, max_z].
        """
        rng = np.random.default_rng(seed)
        suitability = self.compute_suitability_map(rule, elevation, slope_deg, moisture, insolation, flow)

        h, w = elevation.shape
        min_x, min_z, max_x, max_z = world_bounds
        span_x = max_x - min_x
        span_z = max_z - min_z

        # Compute gradient fields for slope descent
        dy, dx = np.gradient(elevation)

        placements: List[StratifiedPlacement] = []
        r_min = rule.min_distance
        r_min_sq = r_min * r_min

        # Sample candidate points proportional to suitability
        cand_count = min(max_instances * 8, h * w)
        flat_suit = suitability.flatten()
        prob_sum = float(flat_suit.sum())
        if prob_sum <= 1e-6:
            return []

        prob = flat_suit / prob_sum
        indices = rng.choice(len(flat_suit), size=cand_count, p=prob, replace=True)

        occupied: List[Tuple[float, float]] = []

        for idx in indices:
            if len(placements) >= max_instances:
                break

            r = idx // w
            c = idx % w

            score = float(suitability[r, c])
            if score < suitability_threshold:
                continue

            # Convert grid cell (r, c) with sub-pixel jitter to world coordinates
            jitter_x = float(rng.uniform(-0.45, 0.45))
            jitter_z = float(rng.uniform(-0.45, 0.45))
            wx = min_x + ((c + 0.5 + jitter_x) / w) * span_x
            wz = min_z + ((r + 0.5 + jitter_z) / h) * span_z
            wy = float(elevation[r, c])

            # Spatial distance pruning (blue noise approximation)
            clash = False
            for ox, oz in occupied[-150:]:  # Check local recent window
                dist_sq = (wx - ox) ** 2 + (wz - oz) ** 2
                if dist_sq < r_min_sq:
                    clash = True
                    break
            if clash:
                continue

            # Physical orientation calculation
            gx = float(dx[r, c])
            gz = float(dy[r, c])
            fl_val = float(flow[r, c])

            flow_vec = (fl_val, 0.0) if fl_val > 20.0 else None

            # Surface normal approximation from gradients: (-gx, 1.0, -gz)
            norm_len = math.sqrt(gx * gx + 1.0 + gz * gz)
            normal = (-gx / norm_len, 1.0 / norm_len, -gz / norm_len)

            euler, quat = self.calculate_orientation(
                rule=rule,
                slope_gradient=(gx, gz),
                flow_vector=flow_vec,
                surface_normal=normal,
                random_seed=int(rng.integers(0, 1000000)),
            )

            scale_val = float(rng.uniform(rule.scale_range[0], rule.scale_range[1]))

            placements.append(
                StratifiedPlacement(
                    x=round(wx, 3),
                    y=round(wy, 3),
                    z=round(wz, 3),
                    rotation_euler=(round(euler[0], 4), round(euler[1], 4), round(euler[2], 4)),
                    rotation_quaternion=(round(quat[0], 6), round(quat[1], 6), round(quat[2], 6), round(quat[3], 6)),
                    scale=round(scale_val, 3),
                    suitability=round(score, 3),
                    rule_name=rule.name,
                    flora_type=rule.flora_type,
                )
            )
            occupied.append((wx, wz))

        return placements
