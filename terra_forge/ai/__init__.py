"""
TerraForge AI Module: Meshy v2 Text-to-3D Integration, Local Asset Vault & Cache,
Geometry Normalization, and 3-Tier LOD Generation.
"""

from terra_forge.ai.meshy_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MESHY_API_KEY,
    MeshyClient,
    MeshyTaskResult,
    MeshyTaskStatus,
    mask_api_key,
    resolve_api_key,
)
from terra_forge.ai.vault import (
    DEFAULT_ABIOTIC_PROFILES,
    DEFAULT_FLORA_PROFILES,
    DEFAULT_VAULT_DIR,
    SOLID_FLORA_TYPES,
    TALL_FLORA_TYPES,
    AbioticMetadata,
    AbioticType,
    AssetEntry,
    AssetMetadata,
    BiologicalMetadata,
    FloraType,
    LocalAssetVault,
    compute_asset_hash,
)
from terra_forge.ai.normalizer import (
    AABBPrimitive,
    CollisionPrimitives,
    ConvexHullPrimitive,
    CylinderPrimitive,
    GeometryNormalizer,
    NormalizedMesh,
    build_glb_from_mesh,
    create_procedural_primitive,
    extract_mesh_from_glb,
    normalize_glb_file,
    parse_glb,
    repack_glb,
)
from terra_forge.ai.lod import (
    LODGenerator,
    LODResult,
    find_blender_binary,
)

__all__ = [
    # Client
    "DEFAULT_BASE_URL",
    "DEFAULT_MESHY_API_KEY",
    "MeshyClient",
    "MeshyTaskResult",
    "MeshyTaskStatus",
    "mask_api_key",
    "resolve_api_key",
    # Vault
    "DEFAULT_ABIOTIC_PROFILES",
    "DEFAULT_FLORA_PROFILES",
    "DEFAULT_VAULT_DIR",
    "SOLID_FLORA_TYPES",
    "TALL_FLORA_TYPES",
    "AbioticMetadata",
    "AbioticType",
    "AssetEntry",
    "AssetMetadata",
    "BiologicalMetadata",
    "FloraType",
    "LocalAssetVault",
    "compute_asset_hash",
    # Normalizer
    "AABBPrimitive",
    "CollisionPrimitives",
    "ConvexHullPrimitive",
    "CylinderPrimitive",
    "GeometryNormalizer",
    "NormalizedMesh",
    "build_glb_from_mesh",
    "create_procedural_primitive",
    "extract_mesh_from_glb",
    "normalize_glb_file",
    "parse_glb",
    "repack_glb",
    # LOD
    "LODGenerator",
    "LODResult",
    "find_blender_binary",
]
