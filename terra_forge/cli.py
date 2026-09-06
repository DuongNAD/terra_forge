"""
TerraForge CLI Entrypoint.
Provides command-line utilities to generate, inspect, tweak, export 3D terrains,
install Blender addons, and run environment health checks.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict

from terra_forge.schema.map_config import MapConfig, PRESET_CONFIGS
from terra_forge.blender.bridge import BlenderBridge

PRESETS_DIR = Path(__file__).parent / "schema" / "presets"


def list_presets():
    """Prints all built-in terrain presets."""
    print("\nAvailable TerraForge Presets:")
    print("=" * 70)
    discovered: Dict[str, tuple[str, str]] = {}
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            discovered[p.stem] = (data.get("theme", ""), data.get("name", p.stem))
        except Exception:
            pass

    for k, v in PRESET_CONFIGS.items():
        if k not in discovered:
            discovered[k] = (v.theme, v.name)

    for stem, (theme, name) in sorted(discovered.items()):
        print(f"  • {stem:<32} [{theme:<18}] : {name}")
    print("=" * 70)


def load_preset_or_file(target: str) -> dict:
    """Loads preset by name or path."""
    # 1. Check disk presets first
    preset_file = PRESETS_DIR / f"{target}.json"
    if preset_file.exists():
        with open(preset_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Check programmatic PRESET_CONFIGS registry
    if target in PRESET_CONFIGS:
        return PRESET_CONFIGS[target].to_dict()

    # 3. Check direct file path
    file_path = Path(target).resolve()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"Preset or config file not found: {target}")


def find_blender_addon_dir() -> Optional[Path]:
    """Locates Blender's user scripts/addons directory across OS platforms."""
    candidates = []

    # macOS
    mac_base = Path.home() / "Library" / "Application Support" / "Blender"
    if mac_base.exists():
        # Search version folders (e.g. 5.2, 5.1, 4.2)
        for ver_dir in sorted(mac_base.glob("*"), reverse=True):
            addon_dir = ver_dir / "scripts" / "addons"
            if addon_dir.exists() or ver_dir.is_dir():
                candidates.append(addon_dir)

    # Linux
    linux_base = Path.home() / ".config" / "blender"
    if linux_base.exists():
        for ver_dir in sorted(linux_base.glob("*"), reverse=True):
            addon_dir = ver_dir / "scripts" / "addons"
            candidates.append(addon_dir)

    # Windows
    appdata = os.environ.get("APPDATA")
    if appdata:
        win_base = Path(appdata) / "Blender Foundation" / "Blender"
        if win_base.exists():
            for ver_dir in sorted(win_base.glob("*"), reverse=True):
                addon_dir = ver_dir / "scripts" / "addons"
                candidates.append(addon_dir)

    for c in candidates:
        if c.exists():
            return c

    # Fallback to creating highest version found
    if candidates:
        return candidates[0]

    return None


def cmd_install_addon():
    """Installs the TerraForge sidebar addon into Blender's user addons folder."""
    print("\nInstalling TerraForge Blender Add-on...")
    print("-" * 55)

    addon_dir = find_blender_addon_dir()
    if not addon_dir:
        print("❌ Could not locate Blender's scripts/addons directory automatically.")
        print("   Please install manually by copying:")
        print(f"   {Path(__file__).parent / 'blender' / 'addon.py'} -> <Blender_Addons_Folder>/terra_forge_addon.py")
        return False

    addon_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(__file__).parent / "blender" / "addon.py"
    target_file = addon_dir / "terra_forge_addon.py"

    try:
        shutil.copy2(source_file, target_file)
        print("✅ Add-on copied successfully to:")
        print(f"   {target_file}")
        print("\n👉 To activate in Blender:")
        print("   1. Open Blender")
        print("   2. Go to Edit > Preferences > Add-ons")
        print("   3. Search for 'TerraForge World Generator' and check the box [✓]")
        print("   4. Press 'N' in 3D Viewport to open the 'TerraForge' tab!\n")
        return True
    except Exception as e:
        print(f"❌ Failed to copy add-on: {e}")
        return False


def cmd_doctor():
    """Comprehensive environment and installation diagnostic test."""
    print("\n🔍 TerraForge Health Check & Doctor")
    print("=" * 65)

    all_ok = True

    # 1. Python Check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        print(f"  [✓] Python Runtime       : {py_ver} ({sys.executable})")
    else:
        print(f"  [❌] Python Runtime      : {py_ver} (Python 3.10+ required)")
        all_ok = False

    # 2. Required Libraries Check
    modules = ["numpy", "pydantic", "PIL"]
    for mod in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  [✓] Package {mod:<12} : {ver}")
        except ImportError:
            print(f"  [❌] Package {mod:<12} : NOT INSTALLED")
            all_ok = False

    # 3. Blender Executable Check
    bridge = BlenderBridge()
    blender_bin = bridge.blender_bin
    if os.path.exists(blender_bin) or shutil.which(blender_bin):
        # Query Blender version
        try:
            res = subprocess.run(
                [blender_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = res.stdout.splitlines()[0] if res.stdout else "Detected"
            print(f"  [✓] Blender Executable   : {blender_bin} ({first_line})")
        except Exception as e:
            print(f"  [!] Blender Executable   : {blender_bin} (detected but could not run --version: {e})")
    else:
        print(f"  [❌] Blender Executable  : NOT FOUND (Checked: {blender_bin})")
        all_ok = False

    # 4. Blender Add-on Check
    addon_dir = find_blender_addon_dir()
    if addon_dir:
        addon_installed = (addon_dir / "terra_forge_addon.py").exists()
        status_icon = "[✓]" if addon_installed else "[!]"
        status_text = "Installed" if addon_installed else "Not installed (run: terraforge install-addon)"
        print(f"  {status_icon} Blender Add-on Folder: {addon_dir}")
        print(f"  {status_icon} Add-on Status        : {status_text}")
    else:
        print("  [!] Blender Add-on Folder: Not located automatically")

    # 5. Presets Schema Check
    preset_files = list(PRESETS_DIR.glob("*.json"))
    valid_presets = 0
    for p in preset_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            MapConfig.model_validate(data)
            valid_presets += 1
        except Exception as e:
            print(f"  [❌] Preset error in {p.name}: {e}")
            all_ok = False

    print(f"  [✓] Terrain Presets      : {valid_presets}/{len(preset_files)} valid presets loaded")

    # 6. Workspace Write Permissions
    for folder in ["models", "previews"]:
        p = Path(folder).resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.touch()
            test_file.unlink()
            print(f"  [✓] Directory Writeable  : {folder}/")
        except Exception as e:
            print(f"  [❌] Directory Not Writeable : {folder}/ ({e})")
            all_ok = False

    print("=" * 65)
    if all_ok:
        print("🎉 System Status: ALL CHECKS PASSED! TerraForge is ready for production.\n")
    else:
        print("⚠️  System Status: Some checks failed or require attention. See above.\n")

    return all_ok


def cmd_inspect(args):
    """Inspects preset geometry, topology, and physical invariants."""
    print(f"\n🔬 Inspecting Preset: '{args.preset}'")
    print("=" * 70)
    cfg_dict = load_preset_or_file(args.preset)
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.res is not None:
        cfg_dict["diorama"]["resolution"] = args.res

    cfg = MapConfig.model_validate(cfg_dict)

    from terra_forge.core.heightfield import Heightfield2D
    from terra_forge.core.erosion import ErosionSimulator
    from terra_forge.core.hydrology import HydrologyEngine
    from terra_forge.core.biomes import BiomeEngine
    from terra_forge.core.diorama_slab import DioramaSlabBuilder
    from terra_forge.inspector.validator import QualityValidator

    print(f"  Configuration Name: {cfg.name} (Theme: {cfg.theme})")
    print(f"  Diorama Dimensions: {cfg.diorama.width}m x {cfg.diorama.depth}m, z_base={cfg.diorama.z_base}m")
    print(f"  Grid Resolution   : {cfg.diorama.resolution}x{cfg.diorama.resolution}")
    print(f"  Peak Count        : {len(cfg.peaks)}")
    print(f"  Biome Tiers       : {len(cfg.biomes)}")

    # 1. Base topography
    res = min(cfg.diorama.resolution, 41) if getattr(args, "fast", False) else cfg.diorama.resolution
    hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=res)
    for p in cfg.peaks:
        hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

    z_min, z_max = float(hf.Z.min()), float(hf.Z.max())
    print("\n  [Geomorphology]")
    print(f"    Elevation Range  : {z_min:.2f}m to {z_max:.2f}m")

    # 2. Erosion
    if cfg.erosion.enabled:
        if getattr(args, "fast", False):
            cfg.erosion.droplet_count = min(cfg.erosion.droplet_count, 350)
            cfg.erosion.thermal_iterations = min(cfg.erosion.thermal_iterations, 6)
        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(seed=cfg.seed)
        ero.run_thermal_erosion()
        print(f"    Sediment Fan Max : {float(hf.sediment.max()):.4f}m")
        print(f"    Scree Talus Max  : {float(hf.scree_talus.max()):.4f}m")

    # 3. Hydrology
    if cfg.hydrology.lake_radius > 0 and abs(cfg.hydrology.lake_center[0]) < cfg.diorama.width:
        hf.carve_lake_basin(
            cfg.hydrology.lake_center[0],
            cfg.hydrology.lake_center[1],
            cfg.hydrology.lake_radius,
            cfg.hydrology.lake_level,
            cfg.hydrology.lake_depth,
            berm_height=1.5,
        )
    hydro = HydrologyEngine(hf, cfg.hydrology)
    if cfg.hydrology.river_enabled:
        hydro.carve_river_channel()
    dist_water = hydro.compute_water_distance_field()
    print("\n  [Hydrology]")
    print(f"    Water Proximity  : min={float(dist_water.min()):.2f}m, max={float(dist_water.max()):.2f}m")

    # 4. Biomes
    be = BiomeEngine(hf, dist_water)
    print("\n  [Ecological Distribution]")
    for b in cfg.biomes:
        pts = be.generate_poisson_scatter(b, max_samples=40)
        print(f"    • {b.name:<26} ({b.prototype:<12}) : {len(pts)} instances spawned")

    # 5. Diorama Slab Topology & Quality Invariants
    slab = DioramaSlabBuilder(hf, cfg.diorama)
    mesh_data = slab.build_mesh_data()

    validator = QualityValidator(cfg, hf)
    results = validator.validate_all(mesh_data=mesh_data)
    metrics = results.get("metrics", {})

    print("\n  [Topology Invariants]")
    print(f"    Vertices         : {metrics.get('num_vertices', len(mesh_data['vertices']))}")
    print(f"    Faces            : {metrics.get('num_faces', len(mesh_data['faces']))}")
    print(f"    Edges            : {metrics.get('num_edges', 0)}")
    print(f"    Euler \u03c7          : {metrics.get('euler_characteristic', 0)} (Target: 2)")
    print(f"    Boundary Edges   : {metrics.get('boundary_edges', 0)} (Target: 0)")
    print(f"    Non-manifold     : {metrics.get('non_manifold_edges', 0)} (Target: 0)")
    status_str = "PASS [✓]" if results["passed"] else "FAIL [❌]"
    print(f"    Watertight       : {status_str}")
    if results.get("errors"):
        print(f"    Errors           : {results['errors']}")
    if results.get("warnings"):
        print(f"    Warnings         : {results['warnings']}")
    print("=" * 70 + "\n")
    return results["passed"]


def cmd_generate(args):
    """Generates a complete 3D terrain diorama."""
    cfg_dict = load_preset_or_file(args.preset)

    if args.seed is not None:
        cfg_dict["seed"] = args.seed

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name = cfg_dict.get("name", "diorama").lower()
        cfg_dict["export"]["glb_path"] = str(out_dir / f"{name}.glb")
        cfg_dict["export"]["blend_path"] = str(out_dir / f"{name}.blend")
        cfg_dict["export"]["heightmap_path"] = str(out_dir / f"{name}_heightmap.json")

    bridge = BlenderBridge()

    if args.mode == "live":
        print(f"\n[TerraForge] Generating live update script for '{args.preset}'...")
        live_code = bridge.generate_live_blender_code(cfg_dict)
        print("-" * 55)
        print("Copy or send the following snippet to Blender MCP tool 'execute_blender_code':")
        print(live_code)
        print("-" * 55)
        return live_code
    else:
        print(f"\n[TerraForge] Executing headless generation for preset '{args.preset}'...")
        results = bridge.run_headless(cfg_dict, render_preview=args.render)
        print("\n" + "=" * 65)
        print(f"  Map Name          : {results.get('name', 'N/A')} [{results.get('theme', '')}]")
        print(f"  Status            : {results.get('status')}")
        print(f"  Execution Time    : {results.get('execution_time_sec', 'N/A')}s")
        if "validation" in results:
            val = results["validation"]
            print(f"  Validation Passed : {val.get('passed')}")
            if val.get("errors"):
                print(f"  Errors            : {val.get('errors')}")
            if val.get("warnings"):
                print(f"  Warnings          : {val.get('warnings')}")
            print(f"  Metrics           : {val.get('metrics')}")
        if "exports" in results:
            print("  Exported Assets   :")
            for k, v in results["exports"].items():
                print(f"    - {k:<15}: {v}")
        if results.get("preview_grid"):
            print(f"  Vision 2x2 Grid   : {results.get('preview_grid')}")
        print("=" * 65 + "\n")
        return results


def main():
    parser = argparse.ArgumentParser(
        prog="terraforge",
        description="TerraForge: Next-Generation Procedural 3D World Generation for Blender & AI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: presets
    subparsers.add_parser("presets", help="List all available landscape presets")

    # Command: doctor
    subparsers.add_parser("doctor", help="Run system healthcheck & diagnostic test")

    # Command: install-addon
    subparsers.add_parser("install-addon", help="Install TerraForge GUI add-on into Blender's user addons folder")

    # Command: inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect terrain preset geometry and topology")
    p_insp.add_argument("-p", "--preset", default="alpine_peak_glacial_lake", help="Preset name or path to config JSON")
    p_insp.add_argument("-s", "--seed", type=int, default=None, help="Random seed override")
    p_insp.add_argument("--res", type=int, default=None, help="Resolution override for inspection")
    p_insp.add_argument("--fast", action="store_true", help="Use fast grid resolution (41x41) for quick testing")

    # Command: generate
    p_gen = subparsers.add_parser("generate", help="Generate 3D terrain diorama")
    p_gen.add_argument("-p", "--preset", default="alpine_peak_glacial_lake", help="Preset name or path to config JSON")
    p_gen.add_argument("-s", "--seed", type=int, default=None, help="Random seed override")
    p_gen.add_argument("-m", "--mode", choices=["headless", "live"], default="headless", help="Execution mode (headless or live)")
    p_gen.add_argument("-r", "--render", action="store_true", help="Render 2x2 multi-angle vision preview grid")
    p_gen.add_argument("-o", "--out-dir", default="models", help="Directory to save exported assets")

    args = parser.parse_args()

    if args.command == "presets":
        list_presets()
    elif args.command == "doctor":
        cmd_doctor()
    elif args.command == "install-addon":
        cmd_install_addon()
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
