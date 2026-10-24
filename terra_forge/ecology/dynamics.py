"""
Ecological Dynamics & Seasonal State Engine for Terra Forge.

Implements Milestone M5:
1. Dynamic water level modulator (water_level / sea_level), simulating seasonal
   water rise/fall and shoreline expansion/contraction without topological tearing.
2. Seasonal moisture cycles: surface soil moisture modulation across seasons
   (monsoon alluvial flooding of floodplains vs dry-season exposed gravel).
3. Whittaker 1975 Net Primary Productivity (NPP) biomass carrying capacity and
   discrete logistic regrowth dynamics for vegetation biomass (aligned with Anima-Engine
   src-tauri/src/core/ecology.rs and seasonal fertility (1.0 + 0.5 * sin(phase))).
4. Shader Uniforms Generator for WebGL / Three.js terrain and vegetation shaders:
   uTime, uWindSpeed, uWindDirection, uSeasonPhase, uWaterLevel, uDryingFactor.
5. Per-vertex colors & mask planes generator (RGBA: aridity, moisture, snowline, NPP biomass).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from terra_forge.core.world_artifact import (
    Biome,
    LegacyBiome,
    map_biome_frontend_to_backend,
)

# -----------------------------------------------------------------------------
# Ecological & Physical Constants
# -----------------------------------------------------------------------------

# Canonical dry-matter Net Primary Productivity (Whittaker 1975 means, g/m^2/yr)
# Matches Anima-Engine src-tauri/src/core/ecology.rs byte-for-byte
WHITTAKER_NPP_LEGACY: Dict[int, float] = {
    int(LegacyBiome.DEEP_OCEAN): 125.0,
    int(LegacyBiome.OCEAN): 250.0,
    int(LegacyBiome.BEACH): 60.0,
    int(LegacyBiome.RIVER): 500.0,  # productive riparian margins
    int(LegacyBiome.GRASSLAND): 700.0,
    int(LegacyBiome.TEMPERATE_FOREST): 1200.0,
    int(LegacyBiome.BOREAL_FOREST): 800.0,
    int(LegacyBiome.RAINFOREST): 2200.0,
    int(LegacyBiome.DESERT): 90.0,
    int(LegacyBiome.MOUNTAIN_ROCK): 50.0,
    int(LegacyBiome.SNOW): 140.0,  # tundra-equivalent
}

# Whittaker 1975 canonical NPP values for all 22 Anima-Engine canonical biomes
WHITTAKER_NPP_CANONICAL: Dict[int, float] = {
    int(Biome.OCEAN): 250.0,
    int(Biome.BEACH): 60.0,
    int(Biome.DESERT): 90.0,
    int(Biome.SAVANNA): 900.0,
    int(Biome.GRASSLAND): 700.0,
    int(Biome.SHRUBLAND): 600.0,
    int(Biome.FOREST): 1200.0,
    int(Biome.JUNGLE): 2200.0,
    int(Biome.TAIGA): 800.0,
    int(Biome.TUNDRA): 140.0,
    int(Biome.SWAMP): 2000.0,
    int(Biome.ROCK): 50.0,
    int(Biome.SNOW): 140.0,
    int(Biome.RIVER): 500.0,
    int(Biome.LAKE): 400.0,
    int(Biome.MANGROVE): 2000.0,
    int(Biome.CHAPARRAL): 700.0,
    int(Biome.STEPPE): 600.0,
    int(Biome.ALPINE): 300.0,
    int(Biome.BADLANDS): 90.0,
    int(Biome.GLACIER): 10.0,
    int(Biome.BOG): 800.0,
}

# Scale from NPP (g/m^2/yr) to simulation carrying capacity energy units
NPP_TO_CAPACITY: float = 0.01

# Amplitude of the seasonal fertility swing around mean 1.0 (Anima-Engine ecology.rs)
SEASON_AMPLITUDE: float = 0.5

# Canonical seasonal clock parameters
DEFAULT_SEASON_PERIOD_SEC: float = 100.0  # 1 full year per 100 seconds
DEFAULT_WATER_LEVEL: float = 2.0         # 2.0 meters canonical sea level
DEFAULT_FLOOD_AMPLITUDE: float = 0.35    # seasonal water rise in meters
DEFAULT_SNOWLINE_BASE: float = 7.0       # elevation (meters) where snow starts in equinox


# -----------------------------------------------------------------------------
# 1. Dynamic Ecological Variables & Seasonal Cycles
# -----------------------------------------------------------------------------

def seasonal_fertility(phase: float) -> float:
    """
    Computes global vegetation fertility multiplier over seasonal phase (radians).
    Formula: max(0.0, 1.0 + 0.5 * sin(phase))
    - Summer (phase = pi/2): peak fertility ~1.5
    - Winter (phase = 3*pi/2): lowest fertility ~0.5 (or 0.0 if amplitude >= 1.0)
    """
    return float(max(0.0, 1.0 + SEASON_AMPLITUDE * math.sin(phase)))


def seasonal_drying_factor(phase: float) -> float:
    """
    Computes surface soil/vegetation drying factor in [0.0, 1.0].
    - Wet season / monsoon (phase = pi/2): drying factor = 0.0 (high humidity)
    - Dry season (phase = 3*pi/2): drying factor = 1.0 (exposed gravel, desiccated scrub)
    """
    val = (1.0 - math.sin(phase)) * 0.5
    return float(np.clip(val, 0.0, 1.0))


def compute_dynamic_water_level(
    base_water_level: float = DEFAULT_WATER_LEVEL,
    season_phase: float = 0.0,
    amplitude: float = DEFAULT_FLOOD_AMPLITUDE,
) -> float:
    """
    Computes dynamic water level (meters) modulated by seasonal monsoon cycles.
    Formula: base_water_level + amplitude * sin(season_phase)
    During monsoon (phase ~ pi/2), water level rises, expanding shorelines over low floodplains.
    During dry season (phase ~ 3*pi/2), water level drops, exposing river gravel and mudflats.
    """
    return float(base_water_level + amplitude * math.sin(season_phase))


def modulate_seasonal_moisture(
    base_moisture: np.ndarray,
    season_phase: float,
    elevation: Optional[np.ndarray] = None,
    water_level: Optional[float] = None,
    flood_margin: float = 0.5,
) -> np.ndarray:
    """
    Modulates surface soil moisture across seasons.
    During monsoon (wet season):
      - Floodplain cells (within flood_margin above water_level) become saturated with alluvial water.
      - General terrain moisture increases with fertility/rainfall.
    During dry season:
      - Exposed gravel and elevated slopes lose moisture.
    
    Returns:
        np.ndarray: Modulated moisture grid strictly in [0.0, 1.0].
    """
    m = np.asarray(base_moisture, dtype=np.float32)
    dryness = seasonal_drying_factor(season_phase)
    fertility = seasonal_fertility(season_phase)

    # General seasonal moisture swing (+/- 25%)
    seasonal_swing = 0.75 + 0.25 * (fertility / 1.5)
    modulated = m * seasonal_swing * (1.0 - 0.3 * dryness)

    # Alluvial flood saturation if elevation is provided
    if elevation is not None and water_level is not None:
        elev = np.asarray(elevation, dtype=np.float32)
        # Cells below or at water level are 100% saturated
        submerged = elev <= water_level
        # Cells in alluvial floodplain transition
        flood_zone = (elev > water_level) & (elev <= water_level + flood_margin)
        flood_frac = (water_level + flood_margin - elev) / max(1e-4, flood_margin)
        # Wet season flood boost
        wet_intensity = max(0.0, math.sin(season_phase))
        alluvial_boost = flood_frac * wet_intensity * 0.8

        modulated = np.where(submerged, 1.0, modulated)
        modulated = np.where(flood_zone, np.maximum(modulated, alluvial_boost), modulated)

    return np.clip(modulated, 0.0, 1.0).astype(np.float32)


# -----------------------------------------------------------------------------
# 2. Whittaker 1975 NPP & Logistic Regrowth Dynamics
# -----------------------------------------------------------------------------

def biome_npp(biome: Union[int, Biome, LegacyBiome], use_canonical: bool = True) -> float:
    """
    Returns Net Primary Productivity (g/m^2/yr) for a biome according to Whittaker 1975.
    If use_canonical is True, checks 22 canonical biomes first, falling back to legacy projection.
    """
    b_idx = int(biome)
    if use_canonical and b_idx in WHITTAKER_NPP_CANONICAL:
        return WHITTAKER_NPP_CANONICAL[b_idx]
    if b_idx in WHITTAKER_NPP_LEGACY:
        return WHITTAKER_NPP_LEGACY[b_idx]
    # Map from canonical to legacy
    leg_idx = map_biome_frontend_to_backend(b_idx)
    return WHITTAKER_NPP_LEGACY.get(leg_idx, 700.0)


def biome_carrying_capacity(
    biome: Union[int, Biome, LegacyBiome],
    use_canonical: bool = True,
) -> float:
    """
    Returns per-cell resource carrying capacity R_max (energy units) for a biome.
    Formula: biome_npp(biome) * NPP_TO_CAPACITY.
    """
    return float(biome_npp(biome, use_canonical=use_canonical) * NPP_TO_CAPACITY)


def compute_carrying_capacity_grid(
    biomes: np.ndarray,
    use_canonical: bool = True,
) -> np.ndarray:
    """
    Vectorized computation of resource carrying capacity R_max for a grid of biomes.
    """
    b_arr = np.asarray(biomes, dtype=np.int32)
    # Build lookup array
    max_idx = int(np.max(b_arr)) if b_arr.size > 0 else 0
    lut_size = max(22, max_idx + 1)
    lut = np.zeros(lut_size, dtype=np.float32)
    for i in range(lut_size):
        lut[i] = biome_carrying_capacity(i, use_canonical=use_canonical)
    return lut[b_arr].astype(np.float32)


def logistic_regrowth(r: float, g: float, r_max: float) -> float:
    """
    Discrete logistic regrowth of standing biomass toward ceiling: R + seed + g * R * (1 - R / R_max).
    A tiny seed keeps a fully grazed/depleted cell (R = 0) from remaining permanently barren.
    Matches Anima-Engine src-tauri/src/core/ecology.rs line 92.
    """
    if r_max <= 0.0:
        return 0.0
    seed = 1e-3 * r_max if r < 1e-4 else 0.0
    next_r = r + seed + g * r * (1.0 - r / r_max)
    return float(np.clip(next_r, 0.0, r_max))


def step_biomass_regrowth_grid(
    biomass: np.ndarray,
    r_max: np.ndarray,
    growth_rate: float = 0.08,
    season_phase: float = 0.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Vectorized discrete logistic regrowth step across a 2D biomass field.
    Incorporates seasonal fertility: effective_growth = growth_rate * seasonal_fertility(phase) * dt.
    """
    b = np.asarray(biomass, dtype=np.float32)
    cap = np.asarray(r_max, dtype=np.float32)

    fertility = seasonal_fertility(season_phase)
    eff_growth = growth_rate * fertility * dt

    safe_cap = np.maximum(cap, 1e-6)
    seed = np.where((b < 1e-4) & (cap > 0.0), 1e-3 * cap, 0.0).astype(np.float32)
    delta = seed + eff_growth * b * (1.0 - b / safe_cap)
    new_b = np.clip(b + delta, 0.0, cap)
    return np.where(cap <= 0.0, 0.0, new_b).astype(np.float32)


# -----------------------------------------------------------------------------
# 3. Ecological Simulation State Trackers
# -----------------------------------------------------------------------------

class SeasonClock:
    """
    Seasonal cycle clock advancing phase theta at rate omega = 2*pi / period_sec.
    Matches Anima-Engine SeasonClock default of 1 full cycle per 100 seconds.
    """

    def __init__(self, initial_phase: float = 0.0, period_sec: float = DEFAULT_SEASON_PERIOD_SEC):
        self.phase: float = float(initial_phase) % (2.0 * math.pi)
        self.period_sec: float = float(period_sec)
        self.rate: float = (2.0 * math.pi) / self.period_sec

    def tick(self, dt: float) -> float:
        """Advances clock by dt seconds and returns current seasonal fertility."""
        self.phase = (self.phase + self.rate * dt) % (2.0 * math.pi)
        return self.fertility

    @property
    def fertility(self) -> float:
        return seasonal_fertility(self.phase)

    @property
    def drying_factor(self) -> float:
        return seasonal_drying_factor(self.phase)

    def water_level(
        self,
        base_level: float = DEFAULT_WATER_LEVEL,
        amplitude: float = DEFAULT_FLOOD_AMPLITUDE,
    ) -> float:
        return compute_dynamic_water_level(base_level, self.phase, amplitude)


class BiomassField:
    """
    Standing vegetation NPP biomass field over a terrain grid regrowing logistically.
    Maintains conserved energy capacity derived from Whittaker 1975 canonical biomes.
    """

    def __init__(
        self,
        biomes: np.ndarray,
        initial_ratio: float = 1.0,
        growth_rate: float = 0.08,
        use_canonical: bool = True,
    ):
        self.r_max: np.ndarray = compute_carrying_capacity_grid(biomes, use_canonical=use_canonical)
        self.biomass: np.ndarray = (self.r_max * np.clip(initial_ratio, 0.0, 1.0)).astype(np.float32)
        self.growth_rate: float = float(growth_rate)
        self.height, self.width = self.r_max.shape

    def step(self, season_phase: float, dt: float = 1.0) -> np.ndarray:
        """Advances discrete logistic regrowth across the grid."""
        self.biomass = step_biomass_regrowth_grid(
            self.biomass,
            self.r_max,
            growth_rate=self.growth_rate,
            season_phase=season_phase,
            dt=dt,
        )
        return self.biomass

    def graze(self, y: int, x: int, amount: float) -> float:
        """Herbivore grazing at specific cell (y, x). Returns actual biomass consumed."""
        curr = float(self.biomass[y, x])
        consumed = min(curr, max(0.0, amount))
        self.biomass[y, x] = max(0.0, curr - consumed)
        return consumed

    def graze_mask(self, mask: np.ndarray, fraction: float = 0.5) -> float:
        """Grazes biomass across a boolean or fractional mask. Returns total consumed biomass."""
        m = np.asarray(mask, dtype=np.float32)
        target_consumed = self.biomass * m * np.clip(fraction, 0.0, 1.0)
        total = float(np.sum(target_consumed))
        self.biomass = np.maximum(0.0, self.biomass - target_consumed).astype(np.float32)
        return total

    @property
    def normalized_biomass(self) -> np.ndarray:
        """Returns cell biomass as fraction of carrying capacity [0.0, 1.0]."""
        safe_cap = np.maximum(self.r_max, 1e-6)
        ratio = self.biomass / safe_cap
        return np.where(self.r_max > 0.0, np.clip(ratio, 0.0, 1.0), 0.0).astype(np.float32)

    @property
    def total_biomass(self) -> float:
        return float(np.sum(self.biomass))


# -----------------------------------------------------------------------------
# 4. Shader Uniforms Generator (Three.js / WebGL)
# -----------------------------------------------------------------------------

def generate_shader_uniforms(
    season_phase: float = 0.0,
    time_sec: float = 0.0,
    wind_speed: float = 1.0,
    wind_direction: float = 0.0,
    water_level: float = DEFAULT_WATER_LEVEL,
    drying_factor: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generates standard uniform dictionary formatted for WebGL / Three.js shaders.
    Required uniforms:
        uTime: elapsed simulation time in seconds.
        uWindSpeed: wind velocity multiplier driving vegetation vertex sway.
        uWindDirection: wind bearing angle in radians.
        uSeasonPhase: seasonal phase [0, 2*pi).
        uWaterLevel: dynamic water surface level in meters.
        uDryingFactor: seasonal surface desiccation index [0, 1].

    Also injects twin aliases (time, windSpeed, windAngle) matching Anima-Engine
    Landscape/Vegetation.tsx and Map2/Map2Vegetation.tsx shader specifications.
    """
    if drying_factor is None:
        drying_factor = seasonal_drying_factor(season_phase)

    return {
        # Canonical uniforms for Three.js terrain & seasonal shaders
        "uTime": {"value": float(time_sec)},
        "uWindSpeed": {"value": float(wind_speed)},
        "uWindDirection": {"value": float(wind_direction)},
        "uSeasonPhase": {"value": float(season_phase)},
        "uWaterLevel": {"value": float(water_level)},
        "uDryingFactor": {"value": float(drying_factor)},
        # Anima-Engine Vegetation.tsx compatibility aliases
        "time": {"value": float(time_sec)},
        "windSpeed": {"value": float(wind_speed)},
        "windAngle": {"value": float(wind_direction)},
    }


# -----------------------------------------------------------------------------
# 5. Per-Vertex Colors & Mask Planes Generator
# -----------------------------------------------------------------------------

@dataclass
class VertexColorMasks:
    """
    Container holding per-vertex and per-grid seasonal attribute masks (RGBA):
    - R: seasonal aridity / dryness
    - G: seasonal moisture / alluvial flood saturation
    - B: snowline / alpine frost mask
    - A: normalized NPP biomass density
    """
    rgba: np.ndarray       # Flat [N, 4] float32 array matching OpenWorldMesh vertex count
    grid_rgba: np.ndarray  # 2D [H, W, 4] float32 array
    aridity: np.ndarray    # [H, W] Channel R
    moisture: np.ndarray   # [H, W] Channel G
    snowline: np.ndarray   # [H, W] Channel B
    biomass: np.ndarray    # [H, W] Channel A
    season_phase: float
    water_level: float

    @property
    def vertex_count(self) -> int:
        return len(self.rgba)

    def to_bytes(self) -> bytes:
        """Returns little-endian Float32 byte buffer."""
        return np.ascontiguousarray(self.rgba, dtype="<f4").tobytes()


def generate_vertex_colors_and_masks(
    heightfield: Any,
    season_phase: float = 0.0,
    water_level: Optional[float] = None,
    base_snowline: float = DEFAULT_SNOWLINE_BASE,
    flood_margin: float = 0.5,
    biomass_field: Optional[Union[BiomassField, np.ndarray]] = None,
) -> VertexColorMasks:
    """
    Generates per-vertex RGBA color attributes and 2D mask planes for seasonal terrain transitions:
    - Channel R (Aridity): desiccation index, high in dry seasons and exposed scree slopes.
    - Channel G (Moisture): alluvial flood saturation, peaking in floodplains during wet season.
    - Channel B (Snowline): alpine snow and seasonal frost mask descending into valleys in winter.
    - Channel A (Biomass): normalized Whittaker NPP biomass density.

    Args:
        heightfield: WorldArtifact, Heightfield2D, OpenWorldMesh, or 2D NumPy array.
        season_phase: Seasonal cycle angle in [0, 2*pi).
        water_level: Water surface level (meters). Defaults to dynamically modulated level.
        base_snowline: Equinox snowline elevation in meters (canonical 7.0m in [0, 10] world).
        flood_margin: Elevation margin above water level subjected to alluvial floods.
        biomass_field: Optional precomputed BiomassField or biomass array.
    """
    # 1. Extract elevation grid in world metric units [0, 10]
    if hasattr(heightfield, "elevation"):
        raw_elev = getattr(heightfield, "elevation")
        w = int(getattr(heightfield, "width", 256))
        h = int(getattr(heightfield, "height", 256))
        elev_grid = np.asarray(raw_elev, dtype=np.float32).reshape((h, w))
    elif hasattr(heightfield, "Z"):
        elev_grid = np.asarray(getattr(heightfield, "Z"), dtype=np.float32)
    else:
        elev_grid = np.asarray(heightfield, dtype=np.float32)
        if elev_grid.ndim == 1:
            side = int(math.isqrt(len(elev_grid)))
            elev_grid = elev_grid.reshape((side, side))

    H, W = elev_grid.shape

    # If elevation is normalized [0, 1], scale to world range [0, 10]
    e_min, e_max = float(np.min(elev_grid)), float(np.max(elev_grid))
    if 0.0 <= e_min and e_max <= 1.0001:
        world_elev = elev_grid * 10.0
    else:
        world_elev = elev_grid

    # 2. Determine dynamic water level
    if water_level is None:
        base_wl = 2.0
        if hasattr(heightfield, "sea_level"):
            raw_sl = float(getattr(heightfield, "sea_level"))
            base_wl = raw_sl * 10.0 if raw_sl <= 1.0001 else raw_sl
        wl = compute_dynamic_water_level(base_wl, season_phase)
    else:
        wl = float(water_level)

    # 3. Base moisture
    if hasattr(heightfield, "moisture"):
        raw_m = getattr(heightfield, "moisture")
        base_m = np.asarray(raw_m, dtype=np.float32).reshape((H, W))
        if np.max(base_m) > 1.001:
            base_m = base_m / 255.0
    else:
        dist_above_water = np.maximum(0.0, world_elev - wl)
        base_m = np.clip(1.0 / (1.0 + 0.35 * dist_above_water), 0.0, 1.0).astype(np.float32)

    # 4. Base temperature
    if hasattr(heightfield, "temperature"):
        raw_t = getattr(heightfield, "temperature")
        base_t = np.asarray(raw_t, dtype=np.float32).reshape((H, W))
    else:
        # Standard lapse rate: 22C at sea level down to -8C at 10m peak
        base_t = (22.0 - (world_elev / 10.0) * 30.0).astype(np.float32)

    # 5. Compute Channel R: Seasonal Aridity / Dryness
    # Dryness peaks when sin(phase) < 0 (winter / dry season)
    s_dry = seasonal_drying_factor(season_phase)
    # Distance above water increases desiccation potential
    height_drying_coeff = np.clip((world_elev - wl) / 6.0, 0.0, 1.0)
    # Seasonal aridity formula: low moisture combined with drying factor
    aridity = (1.0 - base_m) * (0.35 + 0.65 * s_dry) + 0.25 * height_drying_coeff * s_dry
    # Submerged cells are 0% arid
    aridity = np.where(world_elev <= wl, 0.0, aridity)
    channel_r = np.clip(aridity, 0.0, 1.0).astype(np.float32)

    # 6. Compute Channel G: Seasonal Moisture / Alluvial Flood Saturation
    channel_g = modulate_seasonal_moisture(
        base_moisture=base_m,
        season_phase=season_phase,
        elevation=world_elev,
        water_level=wl,
        flood_margin=flood_margin,
    )

    # 7. Compute Channel B: Snowline / Alpine Frost Mask
    # In summer (phase = pi/2), snowline retreats to highest peaks (base + 1.5m).
    # In winter (phase = 3*pi/2), snowline advances down to lower slopes (base - 1.5m).
    effective_snowline = base_snowline + 1.5 * math.sin(season_phase)
    # Smooth transition over 0.8m band
    t_snow = np.clip((world_elev - (effective_snowline - 0.4)) / 0.8, 0.0, 1.0)
    smooth_snow = t_snow * t_snow * (3.0 - 2.0 * t_snow)

    # Additional winter frost on sub-zero cells during deep winter
    winter_intensity = max(0.0, -math.sin(season_phase))
    frost_mask = np.clip((0.0 - base_t) / 10.0, 0.0, 1.0) * winter_intensity * 0.45
    channel_b = np.clip(np.maximum(smooth_snow, frost_mask), 0.0, 1.0).astype(np.float32)

    # 8. Compute Channel A: Normalized NPP Biomass Density
    if biomass_field is not None:
        if isinstance(biomass_field, BiomassField):
            norm_b = biomass_field.normalized_biomass
        else:
            norm_b = np.clip(np.asarray(biomass_field, dtype=np.float32), 0.0, 1.0)
    elif hasattr(heightfield, "biome"):
        raw_biomes = np.asarray(getattr(heightfield, "biome"), dtype=np.int32).reshape((H, W))
        r_max = compute_carrying_capacity_grid(raw_biomes)
        # Normalize relative to lush Rainforest ceiling (2200 * 0.01 = 22.0)
        norm_b = np.clip(r_max / 22.0, 0.0, 1.0).astype(np.float32)
    else:
        # Procedural biomass estimate from moisture, elevation, and frost
        lushness = base_m * (1.0 - channel_r * 0.7)
        norm_b = np.clip(lushness, 0.0, 1.0).astype(np.float32)

    # Modulate standing active chlorophyll by seasonal fertility
    fert = seasonal_fertility(season_phase)
    # Active biomass density swings between ~60% in winter and 100% in summer
    seasonal_biomass = norm_b * (0.6 + 0.4 * (fert / 1.5))
    # Snow cover suppresses visible green biomass
    seasonal_biomass = seasonal_biomass * (1.0 - channel_b * 0.9)
    # Submerged cells have minimal phytoplankton / benthic algae
    seasonal_biomass = np.where(world_elev <= wl, np.minimum(seasonal_biomass, 0.2), seasonal_biomass)
    channel_a = np.clip(seasonal_biomass, 0.0, 1.0).astype(np.float32)

    # 9. Pack RGBA arrays
    grid_rgba = np.stack([channel_r, channel_g, channel_b, channel_a], axis=-1).astype(np.float32)
    flat_rgba = grid_rgba.reshape(-1, 4)

    return VertexColorMasks(
        rgba=flat_rgba,
        grid_rgba=grid_rgba,
        aridity=channel_r,
        moisture=channel_g,
        snowline=channel_b,
        biomass=channel_a,
        season_phase=float(season_phase),
        water_level=float(wl),
    )
