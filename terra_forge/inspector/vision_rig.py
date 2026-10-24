"""
Vision Inspection Rig & Multi-Angle Image Matrix Composer.
Renders snapshots from standard inspection angles and stitches them into a 2x2 vision grid.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Any
try:
    import bpy
    from terra_forge.inspector.cameras import setup_inspection_cameras
except ImportError:
    bpy = None
    setup_inspection_cameras = None

from terra_forge.inspector.cameras import INSPECTION_PRESETS

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _bypass_world_volume():
    """Disconnects World Volume output socket to eliminate EEVEE Next pitch-black render bug over distance."""
    if not bpy or not bpy.context.scene.world or not bpy.context.scene.world.node_tree:
        return []
    tree = bpy.context.scene.world.node_tree
    volume_links = []
    for link in list(tree.links):
        if link.to_socket.name == "Volume" and link.to_node.type == "OUTPUT_WORLD":
            volume_links.append((link.from_socket, link.to_socket))
            tree.links.remove(link)
    return volume_links


def _restore_world_volume(volume_links):
    """Restores disconnected world volume links."""
    if not bpy or not bpy.context.scene.world or not bpy.context.scene.world.node_tree:
        return
    tree = bpy.context.scene.world.node_tree
    for from_sock, to_sock in volume_links:
        try:
            tree.links.new(from_sock, to_sock)
        except Exception:
            pass


def _configure_render_engine(scene):
    """Configures the optimal render engine (supporting BLENDER_EEVEE_NEXT in Blender 4.5+)."""
    if hasattr(bpy.types, "RenderSettings"):
        engine_enum = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        if "BLENDER_EEVEE_NEXT" in engine_enum:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in engine_enum:
            scene.render.engine = "BLENDER_EEVEE"


def render_inspection_snapshots(
    output_dir: str,
    width: int = 640,
    height: int = 480,
    config: Optional[Any] = None,
    hf: Optional[Any] = None,
) -> Dict[str, str]:
    """Renders snapshots from each of the standard inspection camera angles."""
    if not bpy:
        return {}

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cameras = setup_inspection_cameras(config=config, hf=hf)
    scene = bpy.context.scene

    # Set render resolution for fast inspection
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    _configure_render_engine(scene)

    vol_links = _bypass_world_volume()
    snapshot_paths: Dict[str, str] = {}

    try:
        for name, cam_obj in cameras.items():
            scene.camera = cam_obj
            filepath = str(out_path / f"inspect_{name}.png")
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            snapshot_paths[name] = filepath
    finally:
        _restore_world_volume(vol_links)

    return snapshot_paths


def render_visual_acceptance_renders(
    output_dir: str,
    resolution: Tuple[int, int] = (1280, 720),
    config: Optional[Any] = None,
    hf: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Renders the 4 official visual acceptance renders:
    1. render_iso_overview.png: Isometric overview framing the entire 3D diorama block.
    2. render_ne_angle.png: Northeast angle showing the bay, rear mountain slopes, and massif.
    3. render_top_down.png: Top-down cartographic projection of the 160x160m diorama.
    4. render_closeup_water_cavern.png: Close-up water and cavern perspective showing river cascade, lake, arch entrance, and alpine peak.
    """
    if not bpy:
        return {}

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cameras = setup_inspection_cameras(config=config, hf=hf)
    scene = bpy.context.scene

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    _configure_render_engine(scene)

    vol_links = _bypass_world_volume()

    cam_to_target = {
        "isometric_diorama": "render_iso_overview.png",
        "iso_ne": "render_ne_angle.png",
        "topdown": "render_top_down.png",
        "closeup_water_cavern": "render_closeup_water_cavern.png",
    }

    rendered_files: Dict[str, str] = {}
    try:
        for cam_key, filename in cam_to_target.items():
            cam_obj = cameras.get(cam_key)
            if not cam_obj:
                cam_obj = bpy.data.objects.get(f"CAM_Inspect_{cam_key}")

            if cam_obj:
                scene.camera = cam_obj
                filepath = str(out_path / filename)
                scene.render.filepath = filepath
                bpy.ops.render.render(write_still=True)
                rendered_files[cam_key] = filepath
                print(f"[TerraForge] Rendered acceptance view: {filename} -> {filepath}")
    finally:
        _restore_world_volume(vol_links)

    return rendered_files


def resolve_snapshot_key(key: str, snapshot_paths: Dict[str, str]) -> Optional[str]:
    """Resolves snapshot path with support for legacy aliases."""
    if key in snapshot_paths:
        return snapshot_paths[key]
    alias_map = {
        "isometric_diorama": ["iso_sw"],
        "iso_sw": ["isometric_diorama"],
        "closeup_macro": ["closeup_water"],
        "closeup_water": ["closeup_macro"],
        "iso_ne": ["eye_level_first_person", "isometric_diorama"],
        "eye_level_first_person": ["iso_ne"],
    }
    for alias in alias_map.get(key, []):
        if alias in snapshot_paths:
            return snapshot_paths[alias]
    return None


def build_2x2_vision_grid(snapshot_paths: Dict[str, str], output_file: str) -> Optional[str]:
    """Stitches 4 snapshot images into a single 2x2 grid image with clear labels."""
    if not HAS_PIL:
        print("[TerraForge] PIL (Pillow) is not available in current environment; skipping 2x2 grid stitching.")
        return None

    ordered_keys = ["isometric_diorama", "iso_ne", "topdown", "closeup_macro"]
    images = []

    for key in ordered_keys:
        path = resolve_snapshot_key(key, snapshot_paths)
        if path and os.path.exists(path):
            img = Image.open(path)
        else:
            img = Image.new("RGB", (640, 480), color=(40, 40, 40))
        images.append((key, img))

    w, h = images[0][1].size
    grid_img = Image.new("RGB", (w * 2, h * 2), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    positions = [(0, 0), (w, 0), (0, h), (w, h)]

    for idx, (key, img) in enumerate(images):
        pos = positions[idx]
        grid_img.paste(img, pos)

        label = INSPECTION_PRESETS.get(key, {}).get("label", key)
        draw.rectangle([pos[0], pos[1], pos[0] + 340, pos[1] + 30], fill=(0, 0, 0))
        draw.text((pos[0] + 10, pos[1] + 8), label, fill=(255, 255, 255))

    grid_path = Path(output_file).resolve()
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(str(grid_path), "PNG")
    return str(grid_path)
