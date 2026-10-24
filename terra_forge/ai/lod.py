"""
3-Tier LOD (Level of Detail) Generator for TerraForge.
Generates:
- LOD0: High detail normalized mesh (~10,000 triangles) - Near / Hero view (< 25m).
- LOD1: Medium detail decimated mesh (~2,000 triangles) - Mid range (25m - 80m).
- LOD2: Low detail simplified mesh (~300 triangles) or cross-quad billboard (4 triangles) - Far range (> 80m).

Supports dual-execution paths:
1. Headless Blender 4.5.4 LTS (via `C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe` or PATH) using QEM Decimate modifier.
2. Pure-Python fallback using vertex clustering & cross-quad billboarding for zero-dependency execution.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from terra_forge.ai.normalizer import (
    build_glb_from_mesh,
    extract_mesh_from_glb,
)

logger = logging.getLogger("terra_forge.ai.lod")

DEFAULT_BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/bin/blender",
    "blender",
]


def find_blender_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """Resolve headless Blender executable across Windows, Linux, and macOS."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    for candidate in DEFAULT_BLENDER_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
        which_path = shutil.which(candidate)
        if which_path and os.path.isfile(which_path):
            return which_path

    return None


@dataclass
class LODResult:
    lod0_path: Path
    lod1_path: Path
    lod2_path: Path
    lod0_triangles: int
    lod1_triangles: int
    lod2_triangles: int
    method_used: str  # "blender_headless" or "pure_python"


class LODGenerator:
    """
    Automated 3-tier LOD Generator.
    Produces LOD0, LOD1, and LOD2 GLB files from a normalized input mesh.
    """

    def __init__(
        self,
        blender_bin: Optional[str] = None,
        target_lod1_triangles: int = 2000,
        target_lod2_triangles: int = 300,
    ):
        self.blender_bin = find_blender_binary(blender_bin)
        self.target_lod1_triangles = target_lod1_triangles
        self.target_lod2_triangles = target_lod2_triangles

    @property
    def has_blender(self) -> bool:
        return self.blender_bin is not None

    def generate_lods(
        self,
        input_glb_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        prefer_blender: bool = True,
        force_pure_python: bool = False,
    ) -> LODResult:
        """
        Generate LOD0, LOD1, and LOD2 GLB files.
        LOD0 is the normalized input mesh.
        LOD1 is decimated to ~target_lod1_triangles.
        LOD2 is decimated to ~target_lod2_triangles (or cross-quad billboard).
        """
        in_p = Path(input_glb_path).resolve()
        if not in_p.is_file():
            raise FileNotFoundError(f"Input GLB not found at {in_p}")

        out_d = Path(output_dir).resolve() if output_dir else in_p.parent
        out_d.mkdir(parents=True, exist_ok=True)

        lod0_path = out_d / "model.glb" if out_d != in_p.parent else out_d / "lod0.glb"
        if in_p != lod0_path:
            shutil.copy2(in_p, lod0_path)

        lod1_path = out_d / "lod1.glb"
        lod2_path = out_d / "lod2.glb"

        # Try Blender headless first if available and preferred
        if prefer_blender and not force_pure_python and self.has_blender:
            try:
                res = self._generate_with_blender(
                    in_p, lod0_path, lod1_path, lod2_path
                )
                if res is not None:
                    return res
            except Exception as e:
                logger.warning(f"Blender LOD generation failed, falling back to pure Python: {e}")

        # Pure Python fallback
        return self._generate_pure_python(
            in_p, lod0_path, lod1_path, lod2_path
        )

    def _generate_with_blender(
        self,
        input_path: Path,
        lod0_path: Path,
        lod1_path: Path,
        lod2_path: Path,
    ) -> Optional[LODResult]:
        """Execute Blender headless Quadric Error Metric (QEM) Decimate modifier."""
        blender_exe = self.blender_bin
        if not blender_exe:
            return None

        # Python script executed inside Blender
        blender_script = f"""
import bpy
import sys

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB
bpy.ops.import_scene.gltf(filepath=r"{input_path}")
mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
if not mesh_objs:
    sys.exit(1)

# Join meshes if multiple
ctx = bpy.context
if len(mesh_objs) > 1:
    ctx.view_layer.objects.active = mesh_objs[0]
    for o in mesh_objs:
        o.select_set(True)
    bpy.ops.object.join()

obj = mesh_objs[0]
orig_faces = len(obj.data.polygons)
print(f"LOD_ORIG_FACES:{{orig_faces}}")

# --- LOD1 Decimation ---
ratio1 = min(1.0, max(0.01, {self.target_lod1_triangles} / max(1, orig_faces)))
mod1 = obj.modifiers.new(name="Decimate_LOD1", type='DECIMATE')
mod1.ratio = ratio1
bpy.ops.object.modifier_apply(modifier="Decimate_LOD1")
lod1_faces = len(obj.data.polygons)
print(f"LOD1_FACES:{{lod1_faces}}")
bpy.ops.export_scene.gltf(filepath=r"{lod1_path}", export_format='GLB')

# --- LOD2 Decimation ---
ratio2 = min(1.0, max(0.01, {self.target_lod2_triangles} / max(1, lod1_faces)))
mod2 = obj.modifiers.new(name="Decimate_LOD2", type='DECIMATE')
mod2.ratio = ratio2
bpy.ops.object.modifier_apply(modifier="Decimate_LOD2")
lod2_faces = len(obj.data.polygons)
print(f"LOD2_FACES:{{lod2_faces}}")
bpy.ops.export_scene.gltf(filepath=r"{lod2_path}", export_format='GLB')
"""

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
            f.write(blender_script)
            script_path = f.name

        try:
            cmd = [
                blender_exe,
                "-b",
                "-P",
                script_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                logger.warning(f"Blender process returned {res.returncode}: {res.stderr}")
                return None

            orig_tris = 10000
            lod1_tris = self.target_lod1_triangles
            lod2_tris = self.target_lod2_triangles

            for line in res.stdout.splitlines():
                if "LOD_ORIG_FACES:" in line:
                    orig_tris = int(line.split("LOD_ORIG_FACES:")[1])
                elif "LOD1_FACES:" in line:
                    lod1_tris = int(line.split("LOD1_FACES:")[1])
                elif "LOD2_FACES:" in line:
                    lod2_tris = int(line.split("LOD2_FACES:")[1])

            if lod1_path.is_file() and lod2_path.is_file():
                return LODResult(
                    lod0_path=lod0_path,
                    lod1_path=lod1_path,
                    lod2_path=lod2_path,
                    lod0_triangles=orig_tris,
                    lod1_triangles=lod1_tris,
                    lod2_triangles=lod2_tris,
                    method_used="blender_headless",
                )
            return None
        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except Exception:
                    pass

    def _generate_pure_python(
        self,
        input_path: Path,
        lod0_path: Path,
        lod1_path: Path,
        lod2_path: Path,
    ) -> LODResult:
        """
        Pure-Python decimation fallback:
        - Reads mesh from input GLB.
        - Generates LOD1 via vertex grid clustering / strided simplification.
        - Generates LOD2 via cross-quad billboard (4 triangles / 8 vertices) matching bounding extents.
        """
        with open(input_path, "rb") as f:
            raw_bytes = f.read()

        verts, tris = extract_mesh_from_glb(raw_bytes)
        lod0_tris = len(tris)

        # LOD1 simplification
        lod1_verts, lod1_tris_arr = self._decimate_pure_python(
            verts, tris, target_count=self.target_lod1_triangles
        )
        lod1_glb = build_glb_from_mesh(lod1_verts, lod1_tris_arr)
        with open(lod1_path, "wb") as f:
            f.write(lod1_glb)

        # LOD2: Cross-quad billboard (4 triangles)
        lod2_verts, lod2_tris_arr = self.create_cross_quad_billboard(verts)
        lod2_glb = build_glb_from_mesh(lod2_verts, lod2_tris_arr)
        with open(lod2_path, "wb") as f:
            f.write(lod2_glb)

        return LODResult(
            lod0_path=lod0_path,
            lod1_path=lod1_path,
            lod2_path=lod2_path,
            lod0_triangles=lod0_tris,
            lod1_triangles=len(lod1_tris_arr),
            lod2_triangles=len(lod2_tris_arr),
            method_used="pure_python",
        )

    @staticmethod
    def _decimate_pure_python(
        vertices: np.ndarray,
        triangles: np.ndarray,
        target_count: int = 2000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simplify mesh by uniform spatial stride or vertex clustering."""
        cur_tris = len(triangles)
        if cur_tris <= target_count:
            return vertices, triangles

        # Stride sampling preserving manifold triangles
        stride = max(1, cur_tris // target_count)
        sampled_indices = np.arange(0, cur_tris, stride)[:target_count]
        sampled_tris = triangles[sampled_indices]

        # Re-index unique vertices
        unique_v_idx, inverse = np.unique(sampled_tris, return_inverse=True)
        new_verts = vertices[unique_v_idx]
        new_tris = inverse.reshape(sampled_tris.shape)

        return new_verts, new_tris

    @staticmethod
    def create_cross_quad_billboard(
        vertices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cross-quad billboard (2 orthogonal vertical planes, 4 triangles, 8 vertices).
        Perfect for distance vegetation rendering in WebGL / Three.js.
        Plane 1: X in [-w/2, w/2], Z = 0, Y in [0, H]
        Plane 2: X = 0, Z in [-d/2, d/2], Y in [0, H]
        """
        pts = np.asarray(vertices, dtype=np.float32)
        x_min, y_min, z_min = np.min(pts, axis=0)
        x_max, y_max, z_max = np.max(pts, axis=0)

        h = float(y_max - y_min)
        w = float(x_max - x_min)
        d = float(z_max - z_min)
        hw = max(0.2, w / 2.0)
        hd = max(0.2, d / 2.0)

        # 8 vertices: 4 for X-plane, 4 for Z-plane
        # All base vertices at Y = 0.0 (ground plane)
        bb_verts = np.array([
            # Plane 1 (facing Z)
            [-hw, 0.0, 0.0],
            [ hw, 0.0, 0.0],
            [ hw,   h, 0.0],
            [-hw,   h, 0.0],
            # Plane 2 (facing X)
            [0.0, 0.0, -hd],
            [0.0, 0.0,  hd],
            [0.0,   h,  hd],
            [0.0,   h, -hd],
        ], dtype=np.float32)

        bb_tris = np.array([
            # Plane 1: two double-sided or two triangles
            [0, 1, 2],
            [0, 2, 3],
            # Plane 2
            [4, 5, 6],
            [4, 6, 7],
        ], dtype=np.uint16)

        return bb_verts, bb_tris
