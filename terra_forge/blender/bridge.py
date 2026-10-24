"""
TerraForge Blender Bridge:
Orchestrates communication between external AI/CLI callers and Blender 5.2+.
Supports:
1. Headless subprocess execution (isolated, batch, CI/CD).
2. Live MCP script execution (real-time interactive viewport updates).
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_BLENDER_PATHS = [
    "/Applications/Blender.app/Contents/MacOS/Blender",
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    "blender",
]


class BlenderBridge:
    def __init__(self, blender_bin: Optional[str] = None):
        self.blender_bin = self._find_blender(blender_bin)
        self.runner_script = Path(__file__).parent / "runner.py"

    def _find_blender(self, custom_path: Optional[str]) -> str:
        if custom_path and os.path.exists(custom_path):
            return custom_path

        for candidate in DEFAULT_BLENDER_PATHS:
            if os.path.exists(candidate):
                return candidate

        # Check Windows Program Files dynamically
        win_bf = Path(r"C:\Program Files\Blender Foundation")
        if win_bf.exists():
            for exe in sorted(win_bf.glob("*/blender.exe"), reverse=True):
                if exe.exists():
                    return str(exe)

        return "blender"

    def run_headless(
        self,
        config: Dict[str, Any] | str,
        render_preview: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the TerraForge pipeline inside Blender in background headless mode (-b).
        """
        temp_cfg = None
        if isinstance(config, dict):
            fd, temp_cfg = tempfile.mkstemp(suffix=".json", prefix="terraforge_cfg_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            cfg_path = temp_cfg
        else:
            cfg_path = str(Path(config).resolve())

        cmd = [
            self.blender_bin,
            "-b",
            "-P",
            str(self.runner_script.resolve()),
            "--",
            cfg_path,
        ]
        if render_preview:
            cmd.append("--render")

        print(f"[BlenderBridge] Launching Blender headless: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if temp_cfg and os.path.exists(temp_cfg):
            try:
                os.remove(temp_cfg)
            except OSError:
                pass

        if result.returncode != 0:
            print("[BlenderBridge] Blender error output:\n", result.stderr)
            raise RuntimeError(f"Blender process failed with code {result.returncode}:\n{result.stderr[-1000:]}")

        # Parse RESULT_JSON from stdout
        res = None
        stdout_lines = result.stdout.splitlines()
        for line in stdout_lines:
            if line.startswith("RESULT_JSON:"):
                res = json.loads(line[len("RESULT_JSON:"):])
                break

        if res is None:
            res = {
                "status": "completed",
                "stdout": result.stdout[-500:],
            }

        # If render_preview is enabled, stitch 2x2 grid using PIL in Python runtime
        if render_preview:
            try:
                from terra_forge.inspector.vision_rig import build_2x2_vision_grid
                p_dir = Path("previews").resolve()
                snaps = {
                    "topdown": str(p_dir / "inspect_topdown.png"),
                    "iso_sw": str(p_dir / "inspect_iso_sw.png"),
                    "iso_ne": str(p_dir / "inspect_iso_ne.png"),
                    "closeup_water": str(p_dir / "inspect_closeup_water.png"),
                }
                name = res.get("name", "diorama")
                grid_file = str(p_dir / f"{name}_preview_2x2.png")
                built_path = build_2x2_vision_grid(snaps, grid_file)
                if built_path:
                    res["preview_grid"] = built_path
            except Exception as e:
                print(f"[BlenderBridge] Note: Could not stitch preview grid: {e}")

        return res

    def generate_live_blender_code(self, config_dict: Dict[str, Any]) -> str:
        """
        Generates self-contained Python code to be sent to Blender MCP's `execute_blender_code` tool
        to update the live active 3D viewport instantly!
        """
        cfg_json_str = json.dumps(config_dict)
        return f"""
import sys
import json
from pathlib import Path

terra_forge_dir = "{Path(__file__).resolve().parent.parent.parent}"
if terra_forge_dir not in sys.path:
    sys.path.insert(0, terra_forge_dir)

from terra_forge.blender.runner import run_pipeline

cfg_dict = json.loads({repr(cfg_json_str)})
result = run_pipeline(cfg_dict, render_preview=False)
print("LIVE_RESULT:", json.dumps(result))

# Force redraw 3D Viewport
import bpy
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.tag_redraw()
"""

    @staticmethod
    def generate_modifier_tweak_code(modifier_name: str, socket_name: str, value: Any) -> str:
        """Generates 1-line Python code to tweak exposed Geometry Node socket in active viewport."""
        return f"""
import bpy
obj = bpy.data.objects.get("Diorama_Island_Block")
if obj and "{modifier_name}" in obj.modifiers:
    mod = obj.modifiers["{modifier_name}"]
    tree = mod.node_group
    for s in tree.interface.items_tree:
        if s.in_out == 'INPUT' and s.name == "{socket_name}":
            mod[s.identifier] = {repr(value)}
            break
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.tag_redraw()
print("Tweaked {modifier_name} [{socket_name}] = {value}")
"""
