"""
GPU Instancing Transformation Matrix & Buffer Management for Terra Forge.

Generates GPU instancing transformation matrices: 4x4 column-major `Float32Array` buffers
(length `count * 16` floats per flora species) and companion JSON manifests directly
consumable by Three.js `WorldVegetation.tsx` (1 draw call per species) and Bevy ECS.

Guarantees:
- Bottom-pivot guarantee: because Vault/biological assets are normalized to min(y) = 0.0,
  groundLift = 0.0.
- Accurate world bounds and elevation adherence: y = heightfield(x, z) with y >= 0.
- Little-endian binary format (.anmi / .bin) with 32-byte header and per-species table.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from terra_forge.ai.vault import FloraType

# -----------------------------------------------------------------------------
# Binary Protocol Constants (.anmi - Anima Instancing)
# -----------------------------------------------------------------------------
INSTANCING_MAGIC: bytes = b"ANMI"
INSTANCING_VERSION: int = 1
HEADER_LEN: int = 32
SPECIES_HEADER_LEN: int = 32
STRIDE_BYTES: int = 64  # 16 floats * 4 bytes per 4x4 matrix

# Header format: magic[4s], version[I], species_count[I], total_instances[I],
# bounds_min_x[f], bounds_min_z[f], bounds_max_x[f], bounds_max_z[f]
HEADER_STRUCT: struct.Struct = struct.Struct("<4sIII4f")

# Species header format:
# species_id[H], stride[H], instance_count[I],
# local_aabb[6f]: min_x, min_y, min_z, max_x, max_y, max_z
SPECIES_HEADER_STRUCT: struct.Struct = struct.Struct("<HHI6f")

# Set of tall flora species types with solid trunk colliders
TALL_FLORA_TYPES: set[int] = {
    int(FloraType.Pine),
    int(FloraType.Round),
    int(FloraType.Jungle),
    int(FloraType.Cactus),
    int(FloraType.Acacia),
    int(FloraType.Palm),
    int(FloraType.DeadTree),
}

# Canonical local bounding boxes for FloraTypes with bottom pivot min_y = 0.0
CANONICAL_FLORA_AABBS: Dict[int, Tuple[float, float, float, float, float, float]] = {
    int(FloraType.Pine): (-0.55, 0.0, -0.55, 0.55, 2.5, 0.55),
    int(FloraType.Round): (-0.76, 0.0, -0.76, 0.76, 2.2, 0.76),
    int(FloraType.Jungle): (-0.76, 0.0, -0.76, 0.76, 3.0, 0.76),
    int(FloraType.Cactus): (-0.60, 0.0, -0.60, 0.60, 1.8, 0.60),
    int(FloraType.Rock): (-0.50, 0.0, -0.50, 0.50, 0.7, 0.50),
    int(FloraType.Acacia): (-0.95, 0.0, -0.95, 0.95, 2.0, 0.95),
    int(FloraType.Palm): (-0.81, 0.0, -0.81, 0.81, 2.8, 0.81),
    int(FloraType.DeadTree): (-0.42, 0.0, -0.42, 0.42, 2.0, 0.42),
    int(FloraType.Bush): (-0.40, 0.0, -0.40, 0.40, 0.8, 0.40),
    int(FloraType.Reed): (-0.30, 0.0, -0.30, 0.30, 1.2, 0.30),
    int(FloraType.Tuft): (-0.25, 0.0, -0.25, 0.25, 0.5, 0.25),
    int(FloraType.Coral): (-0.60, 0.0, -0.60, 0.60, 0.8, 0.60),
    int(FloraType.Kelp): (-0.30, 0.0, -0.30, 0.30, 2.5, 0.30),
    int(FloraType.Seagrass): (-0.20, 0.0, -0.20, 0.20, 0.4, 0.20),
}


# -----------------------------------------------------------------------------
# Math Helpers: Quaternion & Matrix Transformations
# -----------------------------------------------------------------------------

def quaternion_to_matrix3x3(q: Union[Tuple[float, float, float, float], List[float], np.ndarray]) -> np.ndarray:
    """
    Converts a unit quaternion (x, y, z, w) to a 3x3 rotation matrix.
    Formula matches Three.js Matrix4.makeRotationFromQuaternion and glam::Mat3.
    """
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    q_len = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_len > 1e-8:
        qx /= q_len
        qy /= q_len
        qz /= q_len
        qw /= q_len
    else:
        return np.eye(3, dtype=np.float32)

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz

    xx = qx * x2
    xy = qx * y2
    xz = qx * z2
    yy = qy * y2
    yz = qy * z2
    zz = qz * z2
    wx = qw * x2
    wy = qw * y2
    wz = qw * z2

    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1.0 - (yy + zz)
    R[0, 1] = xy - wz
    R[0, 2] = xz + wy

    R[1, 0] = xy + wz
    R[1, 1] = 1.0 - (xx + zz)
    R[1, 2] = yz - wx

    R[2, 0] = xz - wy
    R[2, 1] = yz + wx
    R[2, 2] = 1.0 - (xx + yy)

    return R


def quaternion_from_matrix3x3(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extracts a unit quaternion (x, y, z, w) from a 3x3 rotation matrix
    using Shepperd's robust diagonal trace algorithm.
    """
    r = np.asarray(R, dtype=np.float32)
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # s = 4 * qw
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0  # s = 4 * qx
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0  # s = 4 * qy
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0  # s = 4 * qz
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    # Normalize
    q_len = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_len > 1e-8:
        return (qx / q_len, qy / q_len, qz / q_len, qw / q_len)
    return (0.0, 0.0, 0.0, 1.0)


def euler_to_quaternion(pitch: float, yaw: float, roll: float) -> Tuple[float, float, float, float]:
    """
    Converts Euler angles (pitch, yaw, roll) in radians to a unit quaternion (x, y, z, w).
    Intrinsic order: Yaw (Y) -> Pitch (X) -> Roll (Z) (q = q_Y * q_X * q_Z).
    Standard Y-up 3D convention (matching Three.js and Bevy ECS).
    """
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

    q_len = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_len > 1e-8:
        return (qx / q_len, qy / q_len, qz / q_len, qw / q_len)
    return (0.0, 0.0, 0.0, 1.0)


def compose_transform_matrix(
    position: Union[Tuple[float, float, float], List[float], np.ndarray],
    rotation: Union[Tuple[float, float, float, float], Tuple[float, float, float], List[float], np.ndarray],
    scale: Union[Tuple[float, float, float], List[float], float, int],
    ground_lift: float = 0.0,
) -> np.ndarray:
    """
    Composes a 4x4 affine transformation matrix in standard column-major order
    (matching Three.js InstancedMesh and OpenGL / WebGL / Bevy ECS).

    Returns:
        np.ndarray of shape (16,), dtype=np.float32.
        Elements:
          [0..2]: Col 0 (X basis scaled by sx), [3]: 0
          [4..6]: Col 1 (Y basis scaled by sy), [7]: 0
          [8..10]: Col 2 (Z basis scaled by sz), [11]: 0
          [12..14]: Translation (x, y + ground_lift * sy, z), [15]: 1.0
    """
    # Scale components
    if isinstance(scale, (int, float)):
        sx = sy = sz = float(scale)
    else:
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])

    # Rotation matrix
    if len(rotation) == 4:
        # Quaternion (x, y, z, w)
        R = quaternion_to_matrix3x3(rotation)
    elif len(rotation) == 3:
        # Euler (pitch, yaw, roll)
        q = euler_to_quaternion(rotation[0], rotation[1], rotation[2])
        R = quaternion_to_matrix3x3(q)
    else:
        raise ValueError(f"Rotation must have 3 (Euler) or 4 (Quat) components, got len={len(rotation)}")

    # Position
    px = float(position[0])
    py = float(position[1]) + ground_lift * sy
    pz = float(position[2])

    # Column-major flat array of 16 float32 elements
    # Col 0: (R00*sx, R10*sx, R20*sx, 0)
    # Col 1: (R01*sy, R11*sy, R21*sy, 0)
    # Col 2: (R02*sz, R12*sz, R22*sz, 0)
    # Col 3: (px, py, pz, 1)
    m = np.zeros(16, dtype=np.float32)

    m[0] = R[0, 0] * sx
    m[1] = R[1, 0] * sx
    m[2] = R[2, 0] * sx
    m[3] = 0.0

    m[4] = R[0, 1] * sy
    m[5] = R[1, 1] * sy
    m[6] = R[2, 1] * sy
    m[7] = 0.0

    m[8] = R[0, 2] * sz
    m[9] = R[1, 2] * sz
    m[10] = R[2, 2] * sz
    m[11] = 0.0

    m[12] = px
    m[13] = py
    m[14] = pz
    m[15] = 1.0

    return m


def decompose_transform_matrix(
    m: Union[np.ndarray, List[float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], Tuple[float, float, float]]:
    """
    Decomposes a 16-element column-major 4x4 matrix into (position, quaternion, scale).
    """
    arr = np.asarray(m, dtype=np.float32).flatten()
    if len(arr) != 16:
        raise ValueError(f"Expected 16 elements for 4x4 matrix, got {len(arr)}")

    # Col 0: arr[0..2]
    # Col 1: arr[4..6]
    # Col 2: arr[8..10]
    # Col 3: arr[12..14]
    c0 = arr[0:3]
    c1 = arr[4:7]
    c2 = arr[8:11]
    c3 = arr[12:15]

    sx = float(np.linalg.norm(c0))
    sy = float(np.linalg.norm(c1))
    sz = float(np.linalg.norm(c2))

    pos = (float(c3[0]), float(c3[1]), float(c3[2]))
    scale = (sx, sy, sz)

    # Normalized rotation matrix columns
    R = np.zeros((3, 3), dtype=np.float32)
    R[:, 0] = c0 / sx if sx > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R[:, 1] = c1 / sy if sy > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R[:, 2] = c2 / sz if sz > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    quat = quaternion_from_matrix3x3(R)
    return pos, quat, scale


# -----------------------------------------------------------------------------
# Transform & Instance Data Classes
# -----------------------------------------------------------------------------

class InstanceTransform:
    """
    Represents an individual instanced entity placement with position,
    rotation (quaternion), scale, and zero-lift guarantee.
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        rotation_quaternion: Tuple[float, float, float, float],
        scale: Union[Tuple[float, float, float], float] = (1.0, 1.0, 1.0),
        ground_lift: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.position = (float(position[0]), float(position[1]), float(position[2]))
        self.rotation_quaternion = (
            float(rotation_quaternion[0]),
            float(rotation_quaternion[1]),
            float(rotation_quaternion[2]),
            float(rotation_quaternion[3]),
        )
        if isinstance(scale, (int, float)):
            self.scale = (float(scale), float(scale), float(scale))
        else:
            self.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
        self.ground_lift = float(ground_lift)
        self.metadata = metadata or {}

    def to_matrix(self) -> np.ndarray:
        """Returns 16-element column-major float32 matrix."""
        return compose_transform_matrix(
            self.position,
            self.rotation_quaternion,
            self.scale,
            ground_lift=self.ground_lift,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation suitable for JSON serialization."""
        mat = self.to_matrix().tolist()
        return {
            "position": [round(p, 4) for p in self.position],
            "rotation": [round(q, 6) for q in self.rotation_quaternion],
            "scale": [round(s, 4) for s in self.scale],
            "groundLift": round(self.ground_lift, 4),
            "matrix": [round(m, 6) for m in mat],
            "metadata": self.metadata,
        }

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, List[float]], ground_lift: float = 0.0) -> InstanceTransform:
        """Reconstructs an InstanceTransform from a 4x4 column-major matrix."""
        pos, quat, scale = decompose_transform_matrix(matrix)
        # Adjust position for ground lift if specified
        adjusted_pos = (pos[0], pos[1] - ground_lift * scale[1], pos[2])
        return cls(
            position=adjusted_pos,
            rotation_quaternion=quat,
            scale=scale,
            ground_lift=ground_lift,
        )


class SpeciesInstancingBuffer:
    """
    Manages GPU instancing matrix buffers and metadata for all instances of a single flora species.
    Directly maps to 1 draw call in Three.js InstancedMesh.
    """

    def __init__(
        self,
        species_id: int,
        species_name: str,
        mesh_asset: str = "",
        instances: Optional[List[InstanceTransform]] = None,
        local_aabb: Optional[Tuple[float, float, float, float, float, float]] = None,
        collider_radius_base: float = 0.45,
        collider_radius_scale: float = 0.25,
        is_solid: Optional[bool] = None,
    ):
        self.species_id = int(species_id)
        self.species_name = species_name
        self.mesh_asset = mesh_asset
        self.instances: List[InstanceTransform] = instances or []

        # Local AABB default from canonical flora database (with bottom pivot min_y = 0.0)
        if local_aabb is not None:
            self.local_aabb = local_aabb
        else:
            self.local_aabb = CANONICAL_FLORA_AABBS.get(
                self.species_id, (-0.5, 0.0, -0.5, 0.5, 2.0, 0.5)
            )

        self.collider_radius_base = float(collider_radius_base)
        self.collider_radius_scale = float(collider_radius_scale)

        if is_solid is not None:
            self.is_solid = is_solid
        else:
            self.is_solid = self.species_id in TALL_FLORA_TYPES

    @property
    def count(self) -> int:
        return len(self.instances)

    def add_instance(self, instance: InstanceTransform) -> None:
        self.instances.append(instance)

    def get_matrix_buffer(self) -> np.ndarray:
        """
        Returns a flat contiguous float32 numpy array of length `count * 16`.
        Each 16-element slice is a column-major 4x4 transformation matrix.
        """
        if not self.instances:
            return np.empty(0, dtype=np.float32)

        matrices = [inst.to_matrix() for inst in self.instances]
        return np.concatenate(matrices).astype(np.float32)

    def calculate_collider_radius(self, scale: float) -> float:
        """Computes obstacle collision radius: r = base + scale * coeff (0.45 + 0.25*s)."""
        return self.collider_radius_base + scale * self.collider_radius_scale

    def to_binary(self) -> bytes:
        """
        Serializes species header (32 bytes) + matrix data (N * 64 bytes).
        """
        mat_buf = self.get_matrix_buffer()
        n = len(self.instances)

        header = SPECIES_HEADER_STRUCT.pack(
            self.species_id,
            STRIDE_BYTES,
            n,
            self.local_aabb[0],
            self.local_aabb[1],
            self.local_aabb[2],
            self.local_aabb[3],
            self.local_aabb[4],
            self.local_aabb[5],
        )
        return header + mat_buf.tobytes()

    def to_dict(self, binary_offset: int = 0) -> Dict[str, Any]:
        """Generates species JSON manifest entry."""
        n = len(self.instances)
        byte_len = n * STRIDE_BYTES
        return {
            "id": self.species_name,
            "speciesId": self.species_id,
            "meshAsset": self.mesh_asset,
            "count": n,
            "stride": STRIDE_BYTES,
            "isSolid": self.is_solid,
            "colliderRadiusBase": self.collider_radius_base,
            "colliderRadiusScale": self.collider_radius_scale,
            "localAABB": list(self.local_aabb),
            "binaryBufferOffset": binary_offset,
            "binaryBufferLength": byte_len,
            "instances": [inst.to_dict() for inst in self.instances],
        }


# -----------------------------------------------------------------------------
# GPU Instancing Scene (World Container)
# -----------------------------------------------------------------------------

class GPUInstancingScene:
    """
    Authoritative container for all flora species instancing buffers across the world.
    Provides binary (.anmi) serialization, round-trip deserialization, and JSON exports
    tailored for Three.js and Bevy ECS.
    """

    def __init__(
        self,
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
        species_buffers: Optional[Dict[int, SpeciesInstancingBuffer]] = None,
    ):
        self.world_bounds = (
            float(world_bounds[0]),
            float(world_bounds[1]),
            float(world_bounds[2]),
            float(world_bounds[3]),
        )
        self.species_buffers: Dict[int, SpeciesInstancingBuffer] = species_buffers or {}

    @property
    def total_instances(self) -> int:
        return sum(buf.count for buf in self.species_buffers.values())

    @property
    def species_count(self) -> int:
        return len(self.species_buffers)

    def get_or_create_buffer(
        self,
        species_id: int,
        species_name: Optional[str] = None,
        mesh_asset: str = "",
    ) -> SpeciesInstancingBuffer:
        if species_id not in self.species_buffers:
            name = species_name
            if not name:
                ft = FloraType.from_any(species_id)
                name = ft.name if ft else f"Species_{species_id}"
            self.species_buffers[species_id] = SpeciesInstancingBuffer(
                species_id=species_id,
                species_name=name,
                mesh_asset=mesh_asset or f"vault://assets/flora/{name.lower()}_lod1.glb",
            )
        return self.species_buffers[species_id]

    def add_instance(
        self,
        species_id: int,
        position: Tuple[float, float, float],
        rotation_quaternion: Tuple[float, float, float, float],
        scale: Union[Tuple[float, float, float], float] = 1.0,
        ground_lift: float = 0.0,
        species_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InstanceTransform:
        buf = self.get_or_create_buffer(species_id, species_name)
        inst = InstanceTransform(
            position=position,
            rotation_quaternion=rotation_quaternion,
            scale=scale,
            ground_lift=ground_lift,
            metadata=metadata,
        )
        buf.add_instance(inst)
        return inst

    @classmethod
    def build_from_stratified_placements(
        cls,
        placements: List[Any],
        heightfield: Optional[np.ndarray] = None,
        world_bounds: Tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0),
        default_ground_lift: float = 0.0,
    ) -> GPUInstancingScene:
        """
        Constructs a GPUInstancingScene directly from stratified placements.
        Ensures bottom-pivot alignment (ground_lift = 0.0) and heightfield adherence.
        """
        scene = cls(world_bounds=world_bounds)

        for p in placements:
            # Determine species_id
            species_id = 0
            species_name = "Pine"
            if hasattr(p, "flora_type") and p.flora_type is not None:
                species_id = int(p.flora_type)
                species_name = p.flora_type.name
            elif hasattr(p, "rule_name"):
                species_name = p.rule_name.split("_")[0]
                ft = FloraType.from_any(species_name)
                if ft is not None:
                    species_id = int(ft)
                    species_name = ft.name

            wx = float(p.x)
            wz = float(p.z)
            wy = float(p.y)

            # Sample heightfield if provided and needed
            if heightfield is not None and (wy == 0.0 or not hasattr(p, "y")):
                min_x, min_z, max_x, max_z = world_bounds
                h, w = heightfield.shape
                norm_c = (wx - min_x) / (max_x - min_x) * (w - 1)
                norm_r = (wz - min_z) / (max_z - min_z) * (h - 1)
                c = int(np.clip(round(norm_c), 0, w - 1))
                r = int(np.clip(round(norm_r), 0, h - 1))
                wy = float(heightfield[r, c])

            wy = max(0.0, wy)

            # Rotation quaternion
            if hasattr(p, "rotation_quaternion"):
                quat = p.rotation_quaternion
            elif hasattr(p, "rotation_euler"):
                quat = euler_to_quaternion(p.rotation_euler[0], p.rotation_euler[1], p.rotation_euler[2])
            else:
                quat = (0.0, 0.0, 0.0, 1.0)

            scale_val = float(getattr(p, "scale", 1.0))

            scene.add_instance(
                species_id=species_id,
                position=(wx, wy, wz),
                rotation_quaternion=quat,
                scale=scale_val,
                ground_lift=default_ground_lift,
                species_name=species_name,
                metadata={"suitability": getattr(p, "suitability", 1.0), "rule": getattr(p, "rule_name", "")},
            )

        return scene

    def export_binary(self, output_path: Optional[Union[str, Path]] = None) -> bytes:
        """
        Encodes the scene into .anmi binary format.
        Layout:
        - Global Header (32 bytes)
        - For each species:
          - Species Header (32 bytes)
          - Matrix Data (N * 64 bytes)
        """
        stream = io.BytesIO()

        # Write global header
        global_header = HEADER_STRUCT.pack(
            INSTANCING_MAGIC,
            INSTANCING_VERSION,
            self.species_count,
            self.total_instances,
            self.world_bounds[0],
            self.world_bounds[1],
            self.world_bounds[2],
            self.world_bounds[3],
        )
        stream.write(global_header)

        # Write species data in deterministic order by species_id
        for species_id in sorted(self.species_buffers.keys()):
            buf = self.species_buffers[species_id]
            stream.write(buf.to_binary())

        data = stream.getvalue()
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        return data

    @classmethod
    def from_binary(cls, data_or_path: Union[bytes, bytearray, memoryview, str, Path]) -> GPUInstancingScene:
        """
        Decodes a .anmi binary buffer into a GPUInstancingScene.
        """
        if isinstance(data_or_path, (str, Path)):
            raw = Path(data_or_path).read_bytes()
        else:
            raw = bytes(data_or_path)

        if len(raw) < HEADER_LEN:
            raise ValueError(f"Binary data too short for ANMI header: {len(raw)} < {HEADER_LEN}")

        magic, version, species_count, total_instances, b_min_x, b_min_z, b_max_x, b_max_z = (
            HEADER_STRUCT.unpack_from(raw, 0)
        )

        if magic != INSTANCING_MAGIC:
            raise ValueError(f"Invalid magic: expected {INSTANCING_MAGIC}, got {magic}")
        if version != INSTANCING_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        scene = cls(world_bounds=(b_min_x, b_min_z, b_max_x, b_max_z))

        offset = HEADER_LEN
        for _ in range(species_count):
            if offset + SPECIES_HEADER_LEN > len(raw):
                raise ValueError(f"Premature end of buffer at offset {offset}")

            species_id, stride, count, aabb_min_x, aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z = (
                SPECIES_HEADER_STRUCT.unpack_from(raw, offset)
            )
            offset += SPECIES_HEADER_LEN

            byte_len = count * STRIDE_BYTES
            if offset + byte_len > len(raw):
                raise ValueError(
                    f"Buffer underrun reading species {species_id}: need {byte_len} bytes, have {len(raw) - offset}"
                )

            # Read matrices
            matrix_data = np.frombuffer(raw, dtype=np.float32, count=count * 16, offset=offset)
            offset += byte_len

            ft = FloraType.from_any(species_id)
            species_name = ft.name if ft else f"Species_{species_id}"

            buf = SpeciesInstancingBuffer(
                species_id=species_id,
                species_name=species_name,
                mesh_asset=f"vault://assets/flora/{species_name.lower()}_lod1.glb",
                local_aabb=(aabb_min_x, aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z),
            )

            # Reconstruct instances from matrices
            for k in range(count):
                m_slice = matrix_data[k * 16 : (k + 1) * 16]
                inst = InstanceTransform.from_matrix(m_slice)
                buf.add_instance(inst)

            scene.species_buffers[species_id] = buf

        return scene

    def export_json(self, output_path: Optional[Union[str, Path]] = None, indent: int = 2) -> Dict[str, Any]:
        """
        Generates companion JSON manifest directly consumable by Three.js and Bevy ECS.
        """
        current_binary_offset = HEADER_LEN
        species_list = []

        for species_id in sorted(self.species_buffers.keys()):
            buf = self.species_buffers[species_id]
            # Account for species header in binary offset calculation
            data_offset = current_binary_offset + SPECIES_HEADER_LEN
            species_list.append(buf.to_dict(binary_offset=data_offset))
            current_binary_offset += SPECIES_HEADER_LEN + buf.count * STRIDE_BYTES

        manifest = {
            "magic": INSTANCING_MAGIC.decode("ascii"),
            "version": INSTANCING_VERSION,
            "worldBounds": {
                "minX": self.world_bounds[0],
                "minZ": self.world_bounds[1],
                "maxX": self.world_bounds[2],
                "maxZ": self.world_bounds[3],
            },
            "speciesCount": self.species_count,
            "totalInstances": self.total_instances,
            "species": species_list,
        }

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=indent), encoding="utf-8")

        return manifest

    def to_threejs_manifest(self) -> Dict[str, Any]:
        """
        Generates clean metadata for Three.js `WorldVegetation.tsx`.
        Contains per-species draw call info, instance counts, and binary matrix slice ranges.
        """
        manifest = self.export_json()
        three_species = []
        for s in manifest["species"]:
            three_species.append({
                "speciesId": s["speciesId"],
                "name": s["id"],
                "meshAsset": s["meshAsset"],
                "count": s["count"],
                "drawCalls": 1,
                "castShadow": s["isSolid"],  # Tall flora casts shadows, ground cover skips
                "groundLift": 0.0,  # Bottom-pivot guarantee
                "matrixByteOffset": s["binaryBufferOffset"],
                "matrixFloatCount": s["count"] * 16,
            })
        return {
            "format": "threejs_instanced_mesh",
            "worldBounds": manifest["worldBounds"],
            "totalInstances": manifest["totalInstances"],
            "drawCallCount": len(three_species),
            "species": three_species,
        }

    def to_bevy_bundles(self) -> List[Dict[str, Any]]:
        """
        Formats entity bundles for Bevy ECS (`Transform`, `GlobalTransform`, `FloraCollider`).
        """
        bundles = []
        for species_id, buf in self.species_buffers.items():
            for inst in buf.instances:
                bundles.append({
                    "species_id": species_id,
                    "species_name": buf.species_name,
                    "transform": {
                        "translation": list(inst.position),
                        "rotation": list(inst.rotation_quaternion),
                        "scale": list(inst.scale),
                    },
                    "matrix_column_major": inst.to_matrix().tolist(),
                    "collider": {
                        "is_solid": buf.is_solid,
                        "shape": "cylinder" if buf.is_solid else "none",
                        "radius": buf.calculate_collider_radius(inst.scale[0]) if buf.is_solid else 0.0,
                    },
                })
        return bundles
