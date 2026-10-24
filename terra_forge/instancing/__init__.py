"""
GPU Instancing Transformation Matrix & Buffer Management for Terra Forge.

Provides pure-Python/NumPy generation of 4x4 column-major matrix buffers
(Float32Array `count * 16` floats per species) and companion JSON manifests
for Three.js `InstancedMesh` and Bevy ECS.
"""

from terra_forge.instancing.matrices import (
    INSTANCING_MAGIC,
    INSTANCING_VERSION,
    InstanceTransform,
    SpeciesInstancingBuffer,
    GPUInstancingScene,
    compose_transform_matrix,
    decompose_transform_matrix,
    quaternion_to_matrix3x3,
    euler_to_quaternion,
    quaternion_from_matrix3x3,
)

__all__ = [
    "INSTANCING_MAGIC",
    "INSTANCING_VERSION",
    "InstanceTransform",
    "SpeciesInstancingBuffer",
    "GPUInstancingScene",
    "compose_transform_matrix",
    "decompose_transform_matrix",
    "quaternion_to_matrix3x3",
    "euler_to_quaternion",
    "quaternion_from_matrix3x3",
]
