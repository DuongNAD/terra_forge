"""
MapManifest Generator & Validator for Terra Forge and Anima-Engine.

Generates and validates `map_manifest.json` compliant with Anima-Engine's
JSON Schema Draft-07 specification (`map_manifest.schema.json`).

Binds a WorldArtifact (.anmw) to the canonical coordinate system ([-100, 100],
elevation [0, 10], 256x256), the 22/11 biome taxonomy, and the 8 canonical
camera viewpoints inspected by the map-vision gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

from terra_forge.core.world_artifact import (
    CANONICAL_GRID_DIM,
    CANONICAL_WORLD_SCALE,
    BIOME_COUNT,
    LEGACY_BIOME_COUNT,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WorldArtifact,
)

# -----------------------------------------------------------------------------
# Canonical View Poses (defined by Anima-Engine mapManifest.ts)
# -----------------------------------------------------------------------------
CANONICAL_VIEW_IDS: List[str] = [
    "overview",
    "navigation",
    "collision",
    "lighting",
    "spawn",
    "water",
    "biome_transition",
    "ecosystem",
]

CANONICAL_VIEW_CAMERAS: Dict[str, Dict[str, Any]] = {
    "overview": {
        "position": [0.0, 112.0, 112.0],
        "target": [0.0, 0.0, 0.0],
        "fov": 45.0,
    },
    "navigation": {
        "position": [7.9, 72.9, -38.1],
        "target": [2.1, 11.8, -8.8],
        "fov": 45.0,
    },
    "collision": {
        "position": [-62.1, 16.9, 6.8],
        "target": [-54.0, 11.1, 6.0],
        "fov": 45.0,
    },
    "lighting": {
        "position": [-95.6, 59.5, -32.2],
        "target": [-59.9, 30.7, -17.4],
        "fov": 45.0,
    },
    "spawn": {
        "position": [-29.5, 23.1, -21.0],
        "target": [-21.4, 15.4, -15.6],
        "fov": 45.0,
    },
    "water": {
        "position": [-85.2, 34.9, 81.9],
        "target": [-67.3, 13.1, 64.0],
        "fov": 45.0,
    },
    "biome_transition": {
        "position": [-64.8, 38.2, -34.5],
        "target": [-45.1, 18.1, -24.0],
        "fov": 45.0,
    },
    "ecosystem": {
        "position": [-52.5, 22.6, -1.7],
        "target": [-38.3, 10.9, -1.7],
        "fov": 45.0,
    },
}

CANONICAL_VIEW_LABELS: Dict[str, str] = {
    "overview": "Isometric Overview (Top-Down)",
    "navigation": "North Angle (Navigation)",
    "collision": "West Angle (Collision / Cliffs)",
    "lighting": "Northwest Angle (Lighting / Sun)",
    "spawn": "East Angle (Spawn Point)",
    "water": "Lake Overview (Water)",
    "biome_transition": "South Angle (Biome Transition)",
    "ecosystem": "Cavern Close-Up (Ecosystem)",
}


# -----------------------------------------------------------------------------
# Manifest Generation
# -----------------------------------------------------------------------------
def generate_map_manifest(
    artifact: Union[WorldArtifact, bytes, Path, str],
    artifact_rel_path: str = "artifacts/world_256.anmw",
    glb_path: Optional[Union[str, Path, bool]] = None,
    glb_rel_path: Optional[str] = "assets/blender_map/ecosystem_map.glb",
    glb_format: str = "gltf-binary",
    view_images_dir: Optional[Union[str, Path]] = None,
    captured_override: Optional[bool] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    map_id: Optional[str] = "genesis_primordial_abiotic",
    version: Optional[str] = "2.0.0",
    biomes: Optional[List[str]] = None,
    spawn_point: Optional[List[float]] = None,
    nav_mesh: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a map manifest dictionary compliant with Anima-Engine draft-07 schema.

    Args:
        artifact: WorldArtifact instance, raw bytes, or path to .anmw file.
        artifact_rel_path: Target relative path written to manifest.
        glb_path: Optional local path to 3D production model (.glb) to hash and record.
        glb_rel_path: Target relative path written to modelAsset block.
        glb_format: Format string for modelAsset (defaults to "gltf-binary").
        view_images_dir: Optional local directory where view images reside to verify capture.
        captured_override: Optional explicit boolean for all view captured states.
        extra_metadata: Optional custom dict added to `_generated`.
        map_id: Identifier for the map (default: "genesis_primordial_abiotic").
        version: Version string for the map (default: "2.0.0").
        biomes: List of biomes present in map (e.g. pure abiotic profile).
        spawn_point: Safe spawn location coordinates [x, y, z].
        nav_mesh: Dictionary containing navigation parameters and reachability score.
    """
    if isinstance(artifact, WorldArtifact):
        art_obj = artifact
        art_bytes = artifact.to_bytes()
    elif isinstance(artifact, bytes):
        art_bytes = artifact
        art_obj = WorldArtifact.from_bytes(artifact)
    elif isinstance(artifact, (str, Path)):
        art_path = Path(artifact)
        art_bytes = art_path.read_bytes()
        art_obj = WorldArtifact.from_bytes(art_bytes)
    else:
        raise TypeError(f"unsupported artifact input type: {type(artifact)}")

    sha256_hash = f"sha256:{hashlib.sha256(art_bytes).hexdigest()}"
    byte_count = len(art_bytes)

    # Build views list
    img_base_dir = Path(view_images_dir) if view_images_dir else None
    views: List[Dict[str, Any]] = []

    for vid in CANONICAL_VIEW_IDS:
        cam = CANONICAL_VIEW_CAMERAS[vid]
        img_rel = f"map-views/{vid}.png"

        # Check capture status
        captured = False
        img_bytes_count: Optional[int] = None
        img_checksum: Optional[str] = None

        if captured_override is not None:
            captured = captured_override
        elif img_base_dir:
            cand = img_base_dir / f"{vid}.png"
            if cand.is_file() and cand.stat().st_size > 0:
                captured = True
                content = cand.read_bytes()
                img_bytes_count = len(content)
                img_checksum = f"sha256:{hashlib.sha256(content).hexdigest()}"

        v_dict: Dict[str, Any] = {
            "id": vid,
            "label": CANONICAL_VIEW_LABELS.get(vid, vid),
            "imagePath": img_rel,
            "camera": {
                "position": [float(x) for x in cam["position"]],
                "target": [float(x) for x in cam["target"]],
                "fov": float(cam.get("fov", 45.0)),
            },
            "captured": captured,
        }
        if img_bytes_count is not None:
            v_dict["bytes"] = img_bytes_count
        if img_checksum is not None:
            v_dict["checksum"] = img_checksum

        views.append(v_dict)

    units_per_cell = float(art_obj.world_scale / art_obj.width)

    manifest: Dict[str, Any] = {
        "$schema": "./map_manifest.schema.json",
        "schemaVersion": 1,
        "_generated": {
            "by": "terra_forge.core.manifest",
            "seed": art_obj.seed,
            "generatorVersion": art_obj.generator_version,
            "worldScale": art_obj.world_scale,
        },
        "worldArtifact": {
            "path": artifact_rel_path,
            "magic": "ANMW",
            "version": art_obj.generator_version if art_obj.generator_version == WORLD_ARTIFACT_VERSION else WORLD_ARTIFACT_VERSION,
            "width": art_obj.width,
            "height": art_obj.height,
            "seaLevel": round(float(art_obj.sea_level), 5),
            "checksum": sha256_hash,
            "bytes": byte_count,
        },
        "coordinateSystem": {
            "worldMinXZ": -100,
            "worldMaxXZ": 100,
            "worldMinY": 0,
            "worldMaxY": 10,
            "gridDim": art_obj.width,
            "unitsPerCell": units_per_cell,
            "worldScale": int(art_obj.world_scale) if art_obj.world_scale == 200.0 else art_obj.world_scale,
        },
        "biomeTaxonomy": {
            "canonicalCount": BIOME_COUNT,
            "legacyCount": LEGACY_BIOME_COUNT,
        },
        "views": views,
    }

    target_glb: Optional[Path] = None
    if glb_path is not None and glb_path is not False:
        target_glb = Path(glb_path)
    elif glb_path is None and glb_rel_path:
        # Auto-detect canonical production GLB on disk
        candidates = [
            Path(glb_rel_path),
            Path(r"e:/Project/01_AI_Agents/Genesis_Zero") / glb_rel_path,
        ]
        if isinstance(artifact, (str, Path)):
            p_art = Path(artifact)
            if p_art.is_file():
                candidates.append(p_art.parent / "blender_map" / Path(glb_rel_path).name)
                candidates.append(p_art.parent.parent / glb_rel_path)
        for cand in candidates:
            if cand.is_file():
                if art_obj.width == CANONICAL_GRID_DIM and art_obj.height == CANONICAL_GRID_DIM:
                    target_glb = cand
                    break

    if target_glb is not None and target_glb.is_file():
        glb_data = target_glb.read_bytes()
        manifest["modelAsset"] = {
            "path": glb_rel_path or str(target_glb),
            "format": glb_format,
            "checksum": f"sha256:{hashlib.sha256(glb_data).hexdigest()}",
            "bytes": len(glb_data),
        }

    if extra_metadata:
        manifest["_generated"].update(extra_metadata)

    if map_id is not None:
        manifest["mapId"] = map_id
    if version is not None:
        manifest["version"] = version
    if biomes is not None:
        manifest["biomes"] = list(biomes)
    if spawn_point is not None:
        manifest["spawnPoint"] = [float(x) for x in spawn_point]
    if nav_mesh is not None:
        manifest["navMesh"] = nav_mesh

    return manifest


# -----------------------------------------------------------------------------
# Manifest Validation
# -----------------------------------------------------------------------------
def validate_map_manifest(manifest: Any) -> Tuple[bool, List[str]]:
    """
    Validates a manifest against the Anima-Engine schema draft-07 contract.
    Returns (is_valid: bool, errors: List[str]).
    """
    errors: List[str] = []

    if not isinstance(manifest, dict):
        return False, ["manifest must be an object"]

    # 1. schemaVersion
    if "schemaVersion" not in manifest:
        errors.append("missing required field: schemaVersion")
    elif not isinstance(manifest["schemaVersion"], (int, float)) or manifest["schemaVersion"] != 1:
        errors.append(f"schemaVersion must be 1, got {manifest.get('schemaVersion')}")

    # 2. worldArtifact
    if "worldArtifact" not in manifest:
        errors.append("missing required field: worldArtifact")
    else:
        wa = manifest["worldArtifact"]
        if not isinstance(wa, dict):
            errors.append("worldArtifact must be an object")
        else:
            if not isinstance(wa.get("path"), str):
                errors.append("worldArtifact.path must be a string")
            if wa.get("magic") != "ANMW":
                errors.append('worldArtifact.magic must be the literal "ANMW"')
            if not isinstance(wa.get("version"), int) or wa["version"] < 1:
                errors.append("worldArtifact.version must be an integer >= 1")
            if not isinstance(wa.get("width"), int) or wa["width"] < 1:
                errors.append("worldArtifact.width must be an integer >= 1")
            if not isinstance(wa.get("height"), int) or wa["height"] < 1:
                errors.append("worldArtifact.height must be an integer >= 1")
            if not isinstance(wa.get("seaLevel"), (int, float)):
                errors.append("worldArtifact.seaLevel must be a number")
            if "checksum" in wa and not isinstance(wa["checksum"], str):
                errors.append("worldArtifact.checksum must be a string")
            if "bytes" in wa and (not isinstance(wa["bytes"], int) or wa["bytes"] <= 0):
                errors.append("worldArtifact.bytes must be a positive integer")

    # 3. coordinateSystem
    if "coordinateSystem" not in manifest:
        errors.append("missing required field: coordinateSystem")
    else:
        cs = manifest["coordinateSystem"]
        if not isinstance(cs, dict):
            errors.append("coordinateSystem must be an object")
        else:
            if cs.get("worldMinXZ") != -100:
                errors.append(f"coordinateSystem.worldMinXZ must be -100, got {cs.get('worldMinXZ')}")
            if cs.get("worldMaxXZ") != 100:
                errors.append(f"coordinateSystem.worldMaxXZ must be 100, got {cs.get('worldMaxXZ')}")
            if cs.get("worldMinY") != 0:
                errors.append(f"coordinateSystem.worldMinY must be 0, got {cs.get('worldMinY')}")
            if cs.get("worldMaxY") != 10:
                errors.append(f"coordinateSystem.worldMaxY must be 10, got {cs.get('worldMaxY')}")
            if not isinstance(cs.get("gridDim"), int) or cs["gridDim"] < 1:
                errors.append("coordinateSystem.gridDim must be an integer >= 1")

    # 4. biomeTaxonomy
    if "biomeTaxonomy" not in manifest:
        errors.append("missing required field: biomeTaxonomy")
    else:
        bt = manifest["biomeTaxonomy"]
        if not isinstance(bt, dict):
            errors.append("biomeTaxonomy must be an object")
        else:
            if bt.get("canonicalCount") != 22:
                errors.append(f"biomeTaxonomy.canonicalCount must be 22, got {bt.get('canonicalCount')}")
            if bt.get("legacyCount") != 11:
                errors.append(f"biomeTaxonomy.legacyCount must be 11, got {bt.get('legacyCount')}")

    # 5. views
    if "views" not in manifest:
        errors.append("missing required field: views")
    else:
        views = manifest["views"]
        if not isinstance(views, list):
            errors.append("views must be an array")
        elif len(views) < 1:
            errors.append("views array must contain at least 1 view")
        else:
            for i, v in enumerate(views):
                prefix = f"views[{i}]"
                if not isinstance(v, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                vid = v.get("id")
                if not isinstance(vid, str) or vid not in CANONICAL_VIEW_IDS:
                    errors.append(
                        f'{prefix}.id "{vid}" is not a canonical view id (expected one of: {", ".join(CANONICAL_VIEW_IDS)})'
                    )
                if not isinstance(v.get("imagePath"), str):
                    errors.append(f"{prefix}.imagePath must be a string")

                cam = v.get("camera")
                if not isinstance(cam, dict):
                    errors.append(f"{prefix}.camera must be an object")
                else:
                    pos = cam.get("position")
                    tgt = cam.get("target")
                    if not (isinstance(pos, list) and len(pos) == 3 and all(isinstance(x, (int, float)) for x in pos)):
                        errors.append(f"{prefix}.camera.position must be a [x, y, z] number triple")
                    if not (isinstance(tgt, list) and len(tgt) == 3 and all(isinstance(x, (int, float)) for x in tgt)):
                        errors.append(f"{prefix}.camera.target must be a [x, y, z] number triple")
                    if "fov" in cam:
                        if not isinstance(cam["fov"], (int, float)) or cam["fov"] <= 0:
                            errors.append(f"{prefix}.camera.fov must be a positive number")

    # 6. modelAsset (optional)
    if "modelAsset" in manifest:
        ma = manifest["modelAsset"]
        if not isinstance(ma, dict):
            errors.append("modelAsset must be an object")
        else:
            if not isinstance(ma.get("path"), str):
                errors.append("modelAsset.path must be a string")
            if not isinstance(ma.get("format"), str):
                errors.append("modelAsset.format must be a string")
            if "checksum" in ma and not isinstance(ma["checksum"], str):
                errors.append("modelAsset.checksum must be a string")
            if "bytes" in ma and (not isinstance(ma["bytes"], int) or ma["bytes"] <= 0):
                errors.append("modelAsset.bytes must be a positive integer")

    return (len(errors) == 0), errors


# -----------------------------------------------------------------------------
# Persistence & Synchronization
# -----------------------------------------------------------------------------
def save_map_manifest(manifest: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> Path:
    """Saves manifest dictionary as JSON, validating before saving."""
    valid, errors = validate_map_manifest(manifest)
    if not valid:
        raise ValueError(f"refusing to save invalid map manifest: {'; '.join(errors)}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=indent), encoding="utf-8")
    return p


def load_map_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a map manifest from JSON file."""
    p = Path(path)
    manifest = json.loads(p.read_text(encoding="utf-8"))
    valid, errors = validate_map_manifest(manifest)
    if not valid:
        raise ValueError(f"manifest validation failed: {'; '.join(errors)}")
    return manifest


def sync_manifest_model_asset(
    manifest_path: Union[str, Path],
    glb_path: Union[str, Path],
    format: str = "gltf-binary",
    glb_rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronizes the modelAsset block and camera FOVs of an existing manifest
    directly from the physical GLB file on disk.

    Calculates actual byte size and SHA-256 hash, sets standardized format,
    and guarantees fov: 45.0 across all canonical views.
    """
    p_manifest = Path(manifest_path)
    p_glb = Path(glb_path)
    if not p_glb.is_file():
        raise FileNotFoundError(f"Physical GLB file not found: {p_glb}")

    glb_bytes = p_glb.read_bytes()
    manifest = json.loads(p_manifest.read_text(encoding="utf-8"))

    current_ma = manifest.get("modelAsset", {})
    resolved_rel_path = glb_rel_path or current_ma.get("path") or "assets/blender_map/ecosystem_map.glb"

    manifest["modelAsset"] = {
        "path": resolved_rel_path,
        "format": format,
        "checksum": f"sha256:{hashlib.sha256(glb_bytes).hexdigest()}",
        "bytes": len(glb_bytes),
    }

    # Ensure all cameras have fov: 45.0
    for v in manifest.get("views", []):
        cam = v.get("camera", {})
        if "fov" not in cam:
            cam["fov"] = 45.0

    save_map_manifest(manifest, p_manifest)
    return manifest
