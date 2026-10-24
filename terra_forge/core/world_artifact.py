"""
WorldArtifact v2 Binary Codec & Canonical Biome Taxonomy for Terra Forge.

Provides a pure-Python/NumPy (zero-bpy) implementation of the Anima-Engine
WorldArtifact v2 (.anmw) binary specification with FNV-1a 32-bit checksum,
strict payload bounds, 22-biome canonical taxonomy, and bidirectional
projection with the simulation's 11 legacy biomes.

Binary Layout (Little-Endian):
    Offset  Type       Field
    0       u8[4]      magic = b"ANMW"
    4       u32        version = 2
    8       u32        width
    12      u32        height
    16      f32        sea_level
    20      u32        seed
    24      u32        generator_version
    28      f32        world_scale (canonical 200.0)
    32      u32        checksum (FNV-1a 32 of bytes 36..end)
    36      f32[n]     elevation (n = width * height)
    ...     f32[n]     moisture
    ...     f32[n]     temperature
    ...     f32[n]     flow
    ...     u8[n]      biome (canonical 22 biomes 0..21)
"""

from __future__ import annotations

import enum
import hashlib
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# -----------------------------------------------------------------------------
# Binary Format Constants
# -----------------------------------------------------------------------------
WORLD_ARTIFACT_MAGIC: bytes = b"ANMW"
WORLD_ARTIFACT_VERSION: int = 2
WORLD_ARTIFACT_HEADER_LEN: int = 36
CANONICAL_WORLD_SCALE: float = 200.0
CANONICAL_GRID_DIM: int = 256
CANONICAL_TOTAL_SIZE_256: int = 1_114_148  # 36 + 256*256*17

# Header struct packing format:
# magic[4s], version[I], width[I], height[I], sea_level[f], seed[I],
# generator_version[I], world_scale[f], checksum[I]
HEADER_STRUCT: struct.Struct = struct.Struct("<4sIIIfIIfI")

# FNV-1a 32-bit parameters
FNV1A_32_BASIS: int = 0x811C9DC5
FNV1A_32_PRIME: int = 0x01000193


try:
    import numba

    @numba.njit(fastmath=True)
    def _fnv1a_32_numba(data: np.ndarray) -> np.uint32:
        h = np.uint32(0x811C9DC5)
        prime = np.uint32(0x01000193)
        for i in range(len(data)):
            h = np.uint32((h ^ np.uint32(data[i])) * prime)
        return h

    # Warm up JIT compilation on import
    _fnv1a_32_numba(np.zeros(1, dtype=np.uint8))
    _HAS_NUMBA_FNV = True
except Exception:
    _HAS_NUMBA_FNV = False


def fnv1a_32(data: Union[bytes, bytearray, memoryview, np.ndarray]) -> int:
    """
    Computes a 32-bit FNV-1a hash over data bytes.
    Matches Anima-Engine Rust and TypeScript decoders byte-for-byte.
    Uses accelerated Numba JIT kernel when available with pure-Python fallback.
    """
    if _HAS_NUMBA_FNV:
        try:
            if isinstance(data, np.ndarray):
                arr = data if data.dtype == np.uint8 and data.ndim == 1 else np.frombuffer(data, dtype=np.uint8)
            else:
                arr = np.frombuffer(data, dtype=np.uint8)
            return int(_fnv1a_32_numba(arr))
        except Exception:
            pass

    h = FNV1A_32_BASIS
    for b in data:
        h ^= b
        h = (h * FNV1A_32_PRIME) & 0xFFFFFFFF
    return h


# -----------------------------------------------------------------------------
# 22 Canonical Biomes Taxonomy & 11 Legacy Biomes
# -----------------------------------------------------------------------------
class Biome(enum.IntEnum):
    """Canonical 22 Biomes of Anima-Engine."""
    OCEAN = 0
    BEACH = 1
    DESERT = 2
    SAVANNA = 3
    GRASSLAND = 4
    SHRUBLAND = 5
    FOREST = 6
    JUNGLE = 7
    TAIGA = 8
    TUNDRA = 9
    SWAMP = 10
    ROCK = 11
    SNOW = 12
    RIVER = 13
    LAKE = 14
    MANGROVE = 15
    CHAPARRAL = 16
    STEPPE = 17
    ALPINE = 18
    BADLANDS = 19
    GLACIER = 20
    BOG = 21


class LegacyBiome(enum.IntEnum):
    """Legacy 11 Biomes of Anima-Engine simulation backend."""
    DEEP_OCEAN = 0
    OCEAN = 1
    BEACH = 2
    RIVER = 3
    GRASSLAND = 4
    TEMPERATE_FOREST = 5
    BOREAL_FOREST = 6
    RAINFOREST = 7
    DESERT = 8
    MOUNTAIN_ROCK = 9
    SNOW = 10


BIOME_COUNT: int = 22
LEGACY_BIOME_COUNT: int = 11

BIOME_NAMES_EN: List[str] = [
    "Ocean", "Beach", "Desert", "Savanna", "Grassland", "Shrubland",
    "Forest", "Jungle", "Taiga", "Tundra", "Swamp", "Rock",
    "Snow", "River", "Lake", "Mangrove", "Chaparral", "Steppe",
    "Alpine", "Badlands", "Glacier", "Bog",
]

BIOME_NAMES_VI: List[str] = [
    "Đại dương", "Bãi biển", "Sa mạc", "Xavan", "Đồng cỏ", "Vùng cây bụi",
    "Rừng ôn đới", "Rừng nhiệt đới", "Rừng taiga", "Đài nguyên", "Đầm lầy", "Núi đá",
    "Đỉnh tuyết", "Dòng sông", "Hồ nước", "Rừng ngập mặn", "Rừng bụi khô", "Thảo nguyên",
    "Đồng cỏ núi cao", "Đất cằn", "Sông băng", "Đầm than bùn",
]

BIOME_RGB: List[Tuple[int, int, int]] = [
    (26, 60, 120),    # Ocean
    (234, 216, 162),  # Beach
    (230, 196, 104),  # Desert
    (200, 190, 96),   # Savanna
    (132, 196, 92),   # Grassland
    (150, 170, 92),   # Shrubland
    (40, 124, 50),    # Forest
    (20, 98, 40),     # Jungle
    (50, 104, 80),    # Taiga
    (166, 176, 154),  # Tundra
    (66, 90, 54),     # Swamp
    (134, 128, 120),  # Rock
    (248, 251, 255),  # Snow
    (58, 132, 188),   # River
    (42, 118, 176),   # Lake
    (60, 106, 68),    # Mangrove
    (178, 158, 92),   # Chaparral
    (190, 186, 120),  # Steppe
    (122, 162, 116),  # Alpine
    (176, 104, 66),   # Badlands
    (220, 238, 248),  # Glacier
    (72, 82, 58),     # Bog
]

BIOME_EMOJI: List[str] = [
    "🌊", "🏖", "🏜", "🦁", "🌾", "🌿",
    "🌲", "🌴", "🌲", "❄", "🐊", "⛰",
    "🏔", "🏞", "💧", "🌴", "🍂", "🌾",
    "🏔", "🪨", "🧊", "🍄",
]

# Lookup table for 22 frontend biomes -> 11 backend legacy biomes
_FRONTEND_TO_BACKEND: np.ndarray = np.array([
    1,   # 0: Ocean           -> 1 (Ocean)
    2,   # 1: Beach           -> 2 (Beach)
    8,   # 2: Desert          -> 8 (Desert)
    4,   # 3: Savanna         -> 4 (Grassland)
    4,   # 4: Grassland       -> 4 (Grassland)
    4,   # 5: Shrubland       -> 4 (Grassland)
    5,   # 6: Forest          -> 5 (TemperateForest)
    7,   # 7: Jungle          -> 7 (Rainforest)
    6,   # 8: Taiga           -> 6 (BorealForest)
    6,   # 9: Tundra          -> 6 (BorealForest)
    7,   # 10: Swamp          -> 7 (Rainforest)
    9,   # 11: Rock           -> 9 (MountainRock)
    10,  # 12: Snow           -> 10 (Snow)
    3,   # 13: River          -> 3 (River)
    1,   # 14: Lake           -> 1 (Ocean)
    7,   # 15: Mangrove       -> 7 (Rainforest)
    4,   # 16: Chaparral      -> 4 (Grassland)
    4,   # 17: Steppe         -> 4 (Grassland)
    4,   # 18: Alpine         -> 4 (Grassland)
    8,   # 19: Badlands       -> 8 (Desert)
    10,  # 20: Glacier        -> 10 (Snow)
    6,   # 21: Bog            -> 6 (BorealForest)
], dtype=np.uint8)

# Lookup table for 11 backend legacy biomes -> 22 frontend canonical biomes
_BACKEND_TO_FRONTEND: np.ndarray = np.array([
    0,   # 0: DeepOcean       -> 0 (Ocean)
    0,   # 1: Ocean           -> 0 (Ocean)
    1,   # 2: Beach           -> 1 (Beach)
    13,  # 3: River           -> 13 (River)
    4,   # 4: Grassland       -> 4 (Grassland)
    6,   # 5: TemperateForest -> 6 (Forest)
    8,   # 6: BorealForest    -> 8 (Taiga)
    7,   # 7: Rainforest      -> 7 (Jungle)
    2,   # 8: Desert          -> 2 (Desert)
    11,  # 9: MountainRock    -> 11 (Rock)
    12,  # 10: Snow           -> 12 (Snow)
], dtype=np.uint8)


def map_biome_frontend_to_backend(front: int) -> int:
    """Projects a 22-biome canonical index to an 11-biome legacy index."""
    if 0 <= front < len(_FRONTEND_TO_BACKEND):
        return int(_FRONTEND_TO_BACKEND[front])
    return int(LegacyBiome.GRASSLAND)


def map_biome_backend_to_frontend(back: int) -> int:
    """Lifts an 11-biome legacy index to a 22-biome canonical index."""
    if 0 <= back < len(_BACKEND_TO_FRONTEND):
        return int(_BACKEND_TO_FRONTEND[back])
    return int(Biome.GRASSLAND)


def map_biome_array_to_backend(biomes: np.ndarray) -> np.ndarray:
    """Vectorized projection of canonical 22 biomes to legacy 11 biomes."""
    safe = np.clip(biomes, 0, len(_FRONTEND_TO_BACKEND) - 1)
    return _FRONTEND_TO_BACKEND[safe]


def map_biome_array_to_frontend(biomes: np.ndarray) -> np.ndarray:
    """Vectorized lifting of legacy 11 biomes to canonical 22 biomes."""
    safe = np.clip(biomes, 0, len(_BACKEND_TO_FRONTEND) - 1)
    return _BACKEND_TO_FRONTEND[safe]


# -----------------------------------------------------------------------------
# Biome Classification (Whittaker Matrix & Physical Zonation)
# -----------------------------------------------------------------------------
BEACH_TOP: float = 0.02
COAST_CLIFF_SLOPE: float = 0.45
INLAND_CLIFF_SLOPE: float = 0.80

T_GLACIER: float = 0.045
T_SNOW: float = 0.125
T_ROCK: float = 0.190
T_ALPINE: float = 0.260


def classify_biome(
    elev: float,
    temp: float,
    moist: float,
    flow: float,
    slope: float,
    coast: float = 0.0,
    sea_level: float = 0.3,
) -> Biome:
    """
    Classifies continuous physical parameters into one of the 22 canonical biomes.
    Matches Anima-Engine `worldGen.ts` classify logic exactly.
    """
    if elev < sea_level:
        return Biome.OCEAN

    # Thin coastal ribbon
    if coast > 0.65 and elev < sea_level + BEACH_TOP:
        if slope > COAST_CLIFF_SLOPE:
            return Biome.ROCK
        if temp > 0.62 and moist > 0.6 and slope < 0.18:
            return Biome.MANGROVE
        return Biome.BEACH

    # Mountain caps by lapsed temperature
    cap_jit = (moist - 0.45) * 0.06
    if temp < T_GLACIER + cap_jit * 0.5:
        return Biome.GLACIER
    if temp < T_SNOW + cap_jit - slope * 0.05 or (elev > 0.94 and temp < T_ROCK):
        return Biome.SNOW
    if temp < T_ROCK + cap_jit * 0.7 + slope * 0.04 or slope > INLAND_CLIFF_SLOPE:
        return Biome.ROCK

    # Rivers
    if flow > 0.615:
        return Biome.RIVER

    # Alpine meadow
    if temp < T_ALPINE + cap_jit * 0.5 and elev > sea_level + 0.18 and moist > 0.25:
        return Biome.ALPINE

    # Wetlands
    if elev < sea_level + 0.075 and moist > 0.72 and slope < 0.22 and flow > 0.18:
        return Biome.BOG if temp < 0.32 else Biome.SWAMP
    if temp < 0.3 and moist > 0.68 and slope < 0.15:
        return Biome.BOG

    # Whittaker matrix: temperature x moisture
    if temp < 0.24:
        return Biome.TAIGA if moist >= 0.6 else Biome.TUNDRA

    if temp < 0.42:
        if moist < 0.16:
            return Biome.STEPPE
        if moist < 0.30:
            return Biome.SHRUBLAND
        return Biome.TAIGA

    if temp < 0.60:
        if moist < 0.14:
            return Biome.DESERT
        if moist < 0.28:
            return Biome.STEPPE
        if moist < 0.42:
            return Biome.GRASSLAND
        if moist < 0.56:
            return Biome.SHRUBLAND
        return Biome.FOREST

    if temp < 0.72:
        if moist < 0.15:
            return Biome.BADLANDS if slope > 0.24 else Biome.DESERT
        if moist < 0.24:
            return Biome.BADLANDS if slope > 0.30 else Biome.STEPPE
        if moist < 0.38:
            return Biome.CHAPARRAL
        if moist < 0.52:
            return Biome.GRASSLAND
        if moist < 0.74:
            return Biome.FOREST
        return Biome.JUNGLE

    # Tropical
    if moist < 0.18:
        return Biome.BADLANDS if slope > 0.24 else Biome.DESERT
    if moist < 0.38:
        return Biome.BADLANDS if slope > 0.30 else Biome.SAVANNA
    if moist < 0.52:
        return Biome.SHRUBLAND
    if moist < 0.68:
        return Biome.FOREST
    return Biome.JUNGLE


def classify_biomes_vectorized(
    elev: np.ndarray,
    temp: np.ndarray,
    moist: np.ndarray,
    flow: np.ndarray,
    slope: np.ndarray,
    coast: Optional[np.ndarray] = None,
    sea_level: float = 0.3,
) -> np.ndarray:
    """
    Vectorized classification across full 2D or 1D arrays into canonical biomes.
    """
    shape = elev.shape
    elev_f = elev.ravel()
    temp_f = temp.ravel()
    moist_f = moist.ravel()
    flow_f = flow.ravel()
    slope_f = slope.ravel()
    coast_f = coast.ravel() if coast is not None else np.zeros_like(elev_f)

    out = np.full(elev_f.shape, int(Biome.GRASSLAND), dtype=np.uint8)

    # Per-cell evaluation loop (fast for 65536 cells)
    for i in range(len(elev_f)):
        out[i] = int(classify_biome(
            elev=float(elev_f[i]),
            temp=float(temp_f[i]),
            moist=float(moist_f[i]),
            flow=float(flow_f[i]),
            slope=float(slope_f[i]),
            coast=float(coast_f[i]),
            sea_level=sea_level,
        ))

    return out.reshape(shape)


# -----------------------------------------------------------------------------
# WorldArtifact Container Class
# -----------------------------------------------------------------------------
class WorldArtifact:
    """
    Neutral binary WorldArtifact v2 reader/writer.
    Represents an immutable or mutable 5-layer landscape snapshot.
    """

    def __init__(
        self,
        width: int,
        height: int,
        sea_level: float,
        seed: int,
        generator_version: int,
        world_scale: float,
        elevation: np.ndarray,
        moisture: np.ndarray,
        temperature: np.ndarray,
        flow: np.ndarray,
        biome: np.ndarray,
    ):
        self.width = int(width)
        self.height = int(height)
        self.sea_level = float(sea_level)
        self.seed = int(seed) & 0xFFFFFFFF
        self.generator_version = int(generator_version) & 0xFFFFFFFF
        self.world_scale = float(world_scale)

        n = self.width * self.height
        if n == 0:
            raise ValueError("artifact has zero cells")

        self.elevation = np.ascontiguousarray(elevation.ravel()[:n], dtype="<f4")
        self.moisture = np.ascontiguousarray(moisture.ravel()[:n], dtype="<f4")
        self.temperature = np.ascontiguousarray(temperature.ravel()[:n], dtype="<f4")
        self.flow = np.ascontiguousarray(flow.ravel()[:n], dtype="<f4")
        self.biome = np.ascontiguousarray(biome.ravel()[:n], dtype=np.uint8)

        if len(self.elevation) != n:
            raise ValueError(f"elevation array length {len(self.elevation)} != width*height {n}")
        if len(self.moisture) != n:
            raise ValueError(f"moisture array length {len(self.moisture)} != width*height {n}")
        if len(self.temperature) != n:
            raise ValueError(f"temperature array length {len(self.temperature)} != width*height {n}")
        if len(self.flow) != n:
            raise ValueError(f"flow array length {len(self.flow)} != width*height {n}")
        if len(self.biome) != n:
            raise ValueError(f"biome array length {len(self.biome)} != width*height {n}")

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    def payload_bytes(self) -> bytes:
        """
        Returns the contiguous 5-layer payload bytes (elevation, moisture,
        temperature, flow, biome) in the exact layout the checksum covers.
        """
        return (
            self.elevation.tobytes()
            + self.moisture.tobytes()
            + self.temperature.tobytes()
            + self.flow.tobytes()
            + self.biome.tobytes()
        )

    def checksum(self) -> int:
        """Computes the 32-bit FNV-1a checksum over the payload bytes."""
        return fnv1a_32(self.payload_bytes())

    def to_bytes(self) -> bytes:
        """
        Serializes the artifact to its exact little-endian binary representation.
        Always recomputes the payload checksum into the 36-byte header.
        """
        payload = self.payload_bytes()
        csum = fnv1a_32(payload)

        header = HEADER_STRUCT.pack(
            WORLD_ARTIFACT_MAGIC,
            WORLD_ARTIFACT_VERSION,
            self.width,
            self.height,
            self.sea_level,
            self.seed,
            self.generator_version,
            self.world_scale,
            csum,
        )
        return header + payload

    def to_file(self, path: Union[str, Path]) -> Path:
        """Writes the serialized artifact to disk, creating parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return p

    def sha256(self) -> str:
        """Returns the SHA-256 hash string 'sha256:<64 hex chars>' of full binary."""
        return f"sha256:{hashlib.sha256(self.to_bytes()).hexdigest()}"

    @classmethod
    def peek_seed(cls, data: bytes) -> int:
        """Quickly extracts the world seed at offset 20 without full payload parse."""
        if len(data) < WORLD_ARTIFACT_HEADER_LEN:
            raise ValueError("artifact too short for header")
        magic, ver = struct.unpack_from("<4sI", data, 0)
        if magic != WORLD_ARTIFACT_MAGIC:
            raise ValueError("bad artifact magic (expected ANMW)")
        if ver != WORLD_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported artifact version {ver} (expected {WORLD_ARTIFACT_VERSION}); regenerate the world"
            )
        seed: int = struct.unpack_from("<I", data, 20)[0]
        return seed

    @classmethod
    def from_bytes(cls, data: bytes) -> WorldArtifact:
        """
        Decodes a WorldArtifact from its binary bytes.
        Strictly validates magic, version, payload checksum, and exact length.
        """
        if len(data) < WORLD_ARTIFACT_HEADER_LEN:
            raise ValueError("artifact too short for header")

        magic, ver, width, height, sea_level, seed, gen_ver, world_scale, header_csum = (
            HEADER_STRUCT.unpack_from(data, 0)
        )

        if magic != WORLD_ARTIFACT_MAGIC:
            raise ValueError("bad artifact magic (expected ANMW)")

        if ver != WORLD_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported artifact version {ver} (expected {WORLD_ARTIFACT_VERSION}); regenerate the world"
            )

        n = width * height
        if n == 0:
            raise ValueError("artifact has zero cells")

        payload_len = n * 4 * 4 + n  # 4 layers of f32 + 1 layer of u8
        expected_len = WORLD_ARTIFACT_HEADER_LEN + payload_len

        if len(data) < expected_len:
            raise ValueError(
                f"artifact truncated (expected {expected_len} bytes, got {len(data)})"
            )

        if len(data) > expected_len:
            raise ValueError(
                f"artifact has {len(data) - expected_len} trailing byte(s) after the payload"
            )

        payload = data[WORLD_ARTIFACT_HEADER_LEN:expected_len]
        computed_csum = fnv1a_32(payload)

        if computed_csum != header_csum:
            raise ValueError(
                f"artifact checksum mismatch (header {header_csum:#010x}, computed {computed_csum:#010x}); world data is corrupt"
            )

        off = WORLD_ARTIFACT_HEADER_LEN
        elevation = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
        off += n * 4
        moisture = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
        off += n * 4
        temperature = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
        off += n * 4
        flow = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
        off += n * 4
        biome = np.frombuffer(data, dtype=np.uint8, count=n, offset=off).copy()

        return cls(
            width=width,
            height=height,
            sea_level=sea_level,
            seed=seed,
            generator_version=gen_ver,
            world_scale=world_scale,
            elevation=elevation,
            moisture=moisture,
            temperature=temperature,
            flow=flow,
            biome=biome,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> WorldArtifact:
        """Loads and parses a WorldArtifact from a binary file."""
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_heightfield(
        cls,
        hf: Any,
        seed: int = 1337,
        sea_level: float = 0.3,
        generator_version: int = 1,
        world_scale: float = CANONICAL_WORLD_SCALE,
        moisture: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
        biome: Optional[np.ndarray] = None,
    ) -> WorldArtifact:
        """
        Synthesizes a WorldArtifact directly from a Heightfield2D instance.
        Automatically computes missing physical fields and biomes if not provided.
        """
        res_x = int(getattr(hf, "res_x", 256))
        res_y = int(getattr(hf, "res_y", 256))
        Z = np.asarray(getattr(hf, "Z", np.zeros((res_y, res_x), dtype=np.float32)))

        # Normalize elevation to [0.0, 1.0] if needed
        z_min = float(np.min(Z))
        z_max = float(np.max(Z))
        if z_max > z_min:
            elevation = ((Z - z_min) / (z_max - z_min)).astype("<f4")
        else:
            elevation = np.full_like(Z, 0.5, dtype="<f4")

        # Slope map
        if hasattr(hf, "compute_slopes"):
            slopes_deg = hf.compute_slopes()
            slopes_norm = (slopes_deg / 90.0).astype("<f4")
        else:
            slopes_norm = np.zeros_like(elevation)

        # Flow map
        if flow is None:
            if hasattr(hf, "water_depth"):
                w_max = float(np.max(hf.water_depth))
                if w_max > 0.0:
                    flow_arr = (hf.water_depth / w_max).astype("<f4")
                else:
                    flow_arr = np.zeros_like(elevation)
            else:
                flow_arr = np.zeros_like(elevation)
        else:
            flow_arr = np.asarray(flow, dtype="<f4")

        # Temperature map
        if temperature is None:
            # Standard lapse rate from altitude: colder at high elevation
            temp_arr = np.clip(0.85 - 0.70 * elevation, 0.0, 1.0).astype("<f4")
        else:
            temp_arr = np.asarray(temperature, dtype="<f4")

        # Moisture map
        if moisture is None:
            # Baseline moisture: higher near lowlands and water
            moist_arr = np.clip(0.70 - 0.35 * elevation + 0.30 * flow_arr, 0.0, 1.0).astype("<f4")
        else:
            moist_arr = np.asarray(moisture, dtype="<f4")

        # Coast distance approximation
        is_water = elevation < sea_level
        coast_arr = np.zeros_like(elevation)
        if np.any(is_water):
            # Simple boundary detection for coast
            from scipy.ndimage import binary_dilation  # type: ignore
            # Fallback if scipy not present
            dilated = np.pad(is_water[1:-1, 1:-1], 1, mode="edge")
            coast_arr = np.where((~is_water) & dilated, 0.8, 0.0).astype("<f4")

        # Biome classification
        if biome is None:
            biome_arr = classify_biomes_vectorized(
                elev=elevation,
                temp=temp_arr,
                moist=moist_arr,
                flow=flow_arr,
                slope=slopes_norm,
                coast=coast_arr,
                sea_level=sea_level,
            )
        else:
            biome_arr = np.asarray(biome, dtype=np.uint8)

        return cls(
            width=res_x,
            height=res_y,
            sea_level=sea_level,
            seed=seed,
            generator_version=generator_version,
            world_scale=world_scale,
            elevation=elevation,
            moisture=moist_arr,
            temperature=temp_arr,
            flow=flow_arr,
            biome=biome_arr,
        )

    def to_terrain_map(self, target_w: int, target_h: int) -> Dict[str, Any]:
        """
        Downsamples the artifact into a backend simulation terrain map dictionary
        of size (target_w, target_h), projecting 22 canonical biomes to 11 legacy biomes.
        Matches Anima-Engine `WorldArtifact::to_terrain_map` nearest-neighbour logic.
        """
        elev_2d = self.elevation.reshape((self.height, self.width))
        flow_2d = self.flow.reshape((self.height, self.width))
        biome_2d = self.biome.reshape((self.height, self.width))

        step_x = self.width / target_w
        step_y = self.height / target_h

        out_elev = np.zeros((target_h, target_w), dtype=np.float32)
        out_flow = np.zeros((target_h, target_w), dtype=np.float32)
        out_biome = np.zeros((target_h, target_w), dtype=np.uint8)

        for ty in range(target_h):
            sy = min(int(ty * step_y), self.height - 1)
            for tx in range(target_w):
                sx = min(int(tx * step_x), self.width - 1)
                out_elev[ty, tx] = elev_2d[sy, sx]
                out_flow[ty, tx] = flow_2d[sy, sx]
                out_biome[ty, tx] = map_biome_frontend_to_backend(int(biome_2d[sy, sx]))

        return {
            "width": target_w,
            "height": target_h,
            "elevations": out_elev.ravel().tolist(),
            "flows": out_flow.ravel().tolist(),
            "biomes": out_biome.ravel().tolist(),
        }
