"""
Milestone 4 Verification Test Suite: Feature 16 (5-Class Triplanar PBR Strata Shader)
and Feature 17 (Strict 100% Abiotic Verification).

Test Categories:
1. Feature 16:
   - Object-space normal-exponent triplanar projection (exponent p=6.0, normalized weights)
   - Dual noise perturbation (+0.18 on slope normal, +3.5m on altitude)
   - 5 PBR geological classes:
     * 1) Cliff Bedrock: base color (0.19, 0.18, 0.17), roughness 0.85, slope > 30 deg (cos(30 deg) ≈ 0.866)
     * 2) Scree & Talus: base color (0.34, 0.31, 0.27), roughness 0.90, intermediate slopes & scree_talus attribute
     * 3) Humus Soil & Valley Moss: rich dark loam (0.22, 0.17, 0.11), lush damp moss (0.11, 0.24, 0.08), alluvial silt (0.38, 0.32, 0.22), roughness 0.50 - 0.70
     * 4) Coastal & River Sand: quartz sand (0.74, 0.66, 0.48), roughness 0.60, Z < 3.8m and gentle slope
     * 5) Alpine Snow: crystalline snow (0.92, 0.95, 0.98), roughness 0.30, Z > 22.0m and slope <= 58 deg
   - Geological strata multi-layering: multiplies COLOR_0 vertex color attribute (factor 0.82) into Base Color
   - Micro-roughness bump: procedural Noise (scale 6.0, detail 5.0, roughness 0.6) feeding Bump (strength 0.12, distance 0.20)

2. Feature 17:
   - genesis_primordial_abiotic.json specifies "biomes": []
   - runner.py verify_strict_abiotic_scene enforces:
     * exactly 0 Geometry Nodes scatter modifiers
     * TerraForge_Prototypes contains 0 objects
     * all scene objects match whitelisted prefixes: ("Diorama_", "Water_", "Cave_", "CAM_", "Sun_")
     * strictly 0% flora, 0% fauna, 0% architecture across all meshes, manifests, presets, and scene objects
     * adversarial detection: correctly rejects unauthorized biological or architectural objects
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from terra_forge.schema.map_config import MapConfig, ShaderConfig

# Detect Blender across Windows, macOS, and Linux
BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]
BLENDER_BIN = None
for cand in BLENDER_CANDIDATES:
    if os.path.exists(cand) or shutil.which(cand) is not None:
        BLENDER_BIN = cand
        break

HAS_BLENDER = BLENDER_BIN is not None


# =============================================================================
# Feature 16: 5-Class Triplanar PBR Strata Shader Tests
# =============================================================================

@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestFeature16PBRStrataShader:
    """Verifies complete 5-class triplanar PBR strata shader construction and properties."""

    def test_triplanar_exponent_sharpness_and_normalization(self):
        """Verify normal-exponent power sharpness (p=6.0), weight normalization, active outgoing links, and full graph connectivity."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader

cfg = ShaderConfig(snow_altitude=22.0)
mat = create_triplanar_terrain_shader(cfg)
assert mat is not None
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Verify power nodes have exponent p=6.0
power_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "POWER"]
assert len(power_nodes) >= 3, f"Expected >= 3 POWER nodes, found {len(power_nodes)}"
for pn in power_nodes:
    assert pn.inputs[1].default_value == 6.0, f"Expected power exponent 6.0, got {pn.inputs[1].default_value}"

# Verify normalization divide nodes exist
divide_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "DIVIDE"]
assert len(divide_nodes) >= 3, f"Expected >= 3 DIVIDE nodes for normalized triplanar weights, got {len(divide_nodes)}"

# Verify weight_x, weight_y, weight_z have active outgoing links (> 0)
for dn in divide_nodes:
    out_links = [l for l in links if l.from_node == dn]
    assert len(out_links) > 0, f"Divide node {dn.name} has no outgoing links (len={len(out_links)})"
    assert len(dn.outputs[0].links) > 0, f"Divide node {dn.name} output socket has 0 links"

# Verify full backward reachability from ShaderNodeOutputMaterial with 0 dead/orphan nodes
node_out = [n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
visited = set()
def walk_back(node):
    if node in visited:
        return
    visited.add(node)
    for l in links:
        if l.to_node == node:
            walk_back(l.from_node)

walk_back(node_out)

for dn in divide_nodes:
    assert dn in visited, f"Triplanar divide weight node {dn.name} is not reachable from ShaderNodeOutputMaterial!"

unreachable = [n for n in nodes if n not in visited]
assert len(unreachable) == 0, f"Found {len(unreachable)} unreachable dead nodes in triplanar shader: {[n.name for n in unreachable]}"

print("TRIPLANAR_EXPONENT_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "TRIPLANAR_EXPONENT_VERIFIED" in res.stdout

    def test_dual_noise_perturbation_coarse_perlin(self):
        """Verify coarse Perlin noise (+0.18 on slope normal, +3.5m on altitude)."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader

cfg = ShaderConfig()
mat = create_triplanar_terrain_shader(cfg)
nodes = mat.node_tree.nodes

# Find coarse noise texture node
noise_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeTexNoise"]
coarse_noise = [n for n in noise_nodes if abs(n.inputs["Scale"].default_value - 0.35) < 1e-3]
assert len(coarse_noise) >= 1, "Coarse noise node (scale 0.35) not found"
cn = coarse_noise[0]
assert abs(cn.inputs["Detail"].default_value - 4.0) < 1e-3

# Find multiply-add nodes for slope and altitude perturbation
madd_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "MULTIPLY_ADD"]
assert len(madd_nodes) >= 2, f"Expected >= 2 MULTIPLY_ADD perturbation nodes, found {len(madd_nodes)}"

factors = [round(float(n.inputs[1].default_value), 2) for n in madd_nodes]
assert 0.18 in factors, f"Slope perturbation factor +0.18 not found in {factors}"
assert 3.5 in factors, f"Altitude perturbation factor +3.5m not found in {factors}"

print("DUAL_NOISE_PERTURBATION_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "DUAL_NOISE_PERTURBATION_VERIFIED" in res.stdout

    def test_5_pbr_geological_material_classes(self):
        """Verify 5 PBR geological classes: Bedrock, Scree/Talus, Soil/Moss, Sand, and Snow."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader, _get_mix_rgba_sockets, _get_mix_float_sockets

cfg = ShaderConfig(snow_altitude=22.0)
mat = create_triplanar_terrain_shader(cfg)
nodes = mat.node_tree.nodes

# 1. Verify Cliff Bedrock: base color (0.19, 0.18, 0.17), roughness 0.85, slope threshold cos(30 deg) ≈ 0.866
ramp_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeValToRGB"]
cliff_ramps = [n for n in ramp_nodes if any(abs(e.position - 0.866) < 1e-2 for e in n.color_ramp.elements)]
assert len(cliff_ramps) >= 1, "Cliff bedrock slope threshold ramp (cos(30 deg) ≈ 0.866) not found"

# Check Bedrock Base Color (0.19, 0.18, 0.17)
rgba_mix_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMix" and n.data_type == "RGBA"]
colors_a = [tuple(round(float(c), 2) for c in _get_mix_rgba_sockets(n)[1].default_value[:3]) for n in rgba_mix_nodes]
colors_b = [tuple(round(float(c), 2) for c in _get_mix_rgba_sockets(n)[2].default_value[:3]) for n in rgba_mix_nodes]
all_colors = colors_a + colors_b

# 1) Bedrock: (0.19, 0.18, 0.17)
assert (0.19, 0.18, 0.17) in all_colors, f"Bedrock color (0.19, 0.18, 0.17) not found in {all_colors}"

# 2) Scree & Talus: (0.34, 0.31, 0.27)
assert (0.34, 0.31, 0.27) in all_colors, f"Scree/talus color (0.34, 0.31, 0.27) not found in {all_colors}"

# 3) Humus Soil & Valley Moss: (0.22, 0.17, 0.11), (0.11, 0.24, 0.08), (0.38, 0.32, 0.22)
assert (0.22, 0.17, 0.11) in all_colors, f"Humus loam color (0.22, 0.17, 0.11) not found in {all_colors}"
assert (0.11, 0.24, 0.08) in all_colors, f"Valley moss color (0.11, 0.24, 0.08) not found in {all_colors}"
assert (0.38, 0.32, 0.22) in all_colors, f"Alluvial silt color (0.38, 0.32, 0.22) not found in {all_colors}"

# 4) Coastal & River Sand: (0.74, 0.66, 0.48)
assert (0.74, 0.66, 0.48) in all_colors, f"Quartz sand color (0.74, 0.66, 0.48) not found in {all_colors}"

# 5) Alpine Snow: (0.92, 0.95, 0.98)
assert (0.92, 0.95, 0.98) in all_colors, f"Alpine snow color (0.92, 0.95, 0.98) not found in {all_colors}"

# Check 5-Class PBR Roughness Values in float mix chain
float_mix_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMix" and n.data_type == "FLOAT"]
floats_a = [round(float(_get_mix_float_sockets(n)[1].default_value), 2) for n in float_mix_nodes]
floats_b = [round(float(_get_mix_float_sockets(n)[2].default_value), 2) for n in float_mix_nodes]
all_roughness = floats_a + floats_b

assert 0.85 in all_roughness, f"Bedrock roughness 0.85 not found in {all_roughness}"
assert 0.90 in all_roughness, f"Scree roughness 0.90 not found in {all_roughness}"
assert 0.60 in all_roughness, f"Sand roughness 0.60 not found in {all_roughness}"
assert 0.30 in all_roughness, f"Snow roughness 0.30 not found in {all_roughness}"
assert (0.50 in all_roughness or 0.70 in all_roughness), f"Soil/moss roughness not found in {all_roughness}"

print("5_PBR_CLASSES_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "5_PBR_CLASSES_VERIFIED" in res.stdout

    def test_strata_vertex_color_multiplication_f082(self):
        """Verify strata vertex color attribute (COLOR_0) multiplies into Base Color with f=0.82."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader, _get_mix_rgba_sockets

cfg = ShaderConfig()
mat = create_triplanar_terrain_shader(cfg)
nodes = mat.node_tree.nodes

# Find attribute node for COLOR_0
attr_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == "COLOR_0"]
assert len(attr_nodes) >= 1, "COLOR_0 attribute node not found"

# Find MULTIPLY mix node with factor 0.82
mult_nodes = [
    n for n in nodes
    if n.bl_idname == "ShaderNodeMix"
    and n.data_type == "RGBA"
    and getattr(n, "blend_type", "") == "MULTIPLY"
]
assert len(mult_nodes) >= 1, "MULTIPLY RGBA mix node not found"
mult_node = mult_nodes[0]
fac, _, _, out_sock = _get_mix_rgba_sockets(mult_node)
assert abs(fac.default_value - 0.82) < 1e-2, f"Expected multiplication factor 0.82, got {fac.default_value}"

# Verify connected to Principled BSDF Base Color
bsdf = [n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"][0]
bc_links = [lnk for lnk in mat.node_tree.links if lnk.to_socket == bsdf.inputs["Base Color"]]
assert len(bc_links) >= 1
assert bc_links[0].from_socket == out_sock, "Base Color not fed by strata vertex color multiplication node"

print("STRATA_COLOR_0_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "STRATA_COLOR_0_VERIFIED" in res.stdout

    def test_micro_roughness_bump_parameters(self):
        """Verify micro-roughness bump: procedural Noise feeding Bump (strength 0.12, distance 0.20)."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader

cfg = ShaderConfig()
mat = create_triplanar_terrain_shader(cfg)
nodes = mat.node_tree.nodes

# Find Bump node
bump_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeBump"]
assert len(bump_nodes) >= 1, "ShaderNodeBump not found"
bn = bump_nodes[0]

assert abs(bn.inputs["Strength"].default_value - 0.12) < 1e-3, f"Expected strength 0.12, got {bn.inputs['Strength'].default_value}"
assert abs(bn.inputs["Distance"].default_value - 0.20) < 1e-3, f"Expected distance 0.20, got {bn.inputs['Distance'].default_value}"

# Find high-frequency micro bump noise
noise_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeTexNoise"]
bump_noise = [n for n in noise_nodes if abs(n.inputs["Scale"].default_value - 6.0) < 1e-3]
assert len(bump_noise) >= 1, "Micro bump noise node (scale 6.0) not found"
b_noise = bump_noise[0]
assert abs(b_noise.inputs["Detail"].default_value - 5.0) < 1e-3
assert abs(b_noise.inputs["Roughness"].default_value - 0.6) < 1e-3

# Verify bump connects to Principled BSDF Normal
bsdf = [n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"][0]
norm_links = [lnk for lnk in mat.node_tree.links if lnk.to_socket == bsdf.inputs["Normal"]]
assert len(norm_links) >= 1
assert norm_links[0].from_node == bn

print("MICRO_BUMP_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "MICRO_BUMP_VERIFIED" in res.stdout


# =============================================================================
# Feature 17: Strict 100% Abiotic Verification Tests
# =============================================================================

class TestFeature17StrictAbiotic:
    """Verifies strict 100% abiotic invariants across presets, runners, and scene objects."""

    def test_genesis_primordial_abiotic_preset_biomes_empty(self):
        """Verify genesis_primordial_abiotic.json defines biomes: []."""
        preset_path = Path("terra_forge/schema/presets/genesis_primordial_abiotic.json")
        assert preset_path.is_file(), f"Preset file not found: {preset_path}"

        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "biomes" in data, "biomes key missing from preset"
        assert isinstance(data["biomes"], list)
        assert len(data["biomes"]) == 0, f"biomes must be strictly empty [], got: {data['biomes']}"

        # Validate through Pydantic MapConfig schema
        cfg = MapConfig.model_validate(data)
        assert len(cfg.biomes) == 0

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_runner_abiotic_verification_zero_scatter_and_zero_prototypes(self):
        """Verify runner verifies 0 scatter modifiers and 0 prototype objects in abiotic mode."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import MapConfig
from terra_forge.blender.runner import verify_strict_abiotic_scene

bpy.ops.wm.read_factory_settings(use_empty=True)

# Create abiotic mock terrain
mesh = bpy.data.meshes.new("Diorama_Island_Mesh")
obj = bpy.data.objects.new("Diorama_Island_Block", mesh)
bpy.context.scene.collection.objects.link(obj)

# Create whitelisted abiotic objects
water_mesh = bpy.data.meshes.new("Water_Lake_Central_Mesh")
water_obj = bpy.data.objects.new("Water_Lake_Central", water_mesh)
bpy.context.scene.collection.objects.link(water_obj)

cave_mesh = bpy.data.meshes.new("Cave_Cavern_Chamber_Mesh")
cave_obj = bpy.data.objects.new("Cave_Cavern_Chamber", cave_mesh)
bpy.context.scene.collection.objects.link(cave_obj)

cam_data = bpy.data.cameras.new("CAM_Inspect_iso_Data")
cam_obj = bpy.data.objects.new("CAM_Inspect_iso", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)

sun_data = bpy.data.lights.new("Sun_Key_Light_Data", "SUN")
sun_obj = bpy.data.objects.new("Sun_Key_Light", sun_data)
bpy.context.scene.collection.objects.link(sun_obj)

# Ensure TerraForge_Prototypes collection is empty
proto_col = bpy.data.collections.new("TerraForge_Prototypes")
bpy.context.scene.collection.children.link(proto_col)

cfg = MapConfig(name="Abiotic_Test", biomes=[])
res = verify_strict_abiotic_scene(obj, cfg)

assert res["passed"] is True, f"Abiotic verification failed: {res['errors']}"
assert res["metrics"]["scatter_modifiers"] == 0
assert res["metrics"]["prototype_objects"] == 0
assert res["metrics"]["biological_objects_count"] == 0
assert res["metrics"]["whitelisted_objects_count"] == 5

print("RUNNER_ABIOTIC_INVARIANTS_PASSED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "RUNNER_ABIOTIC_INVARIANTS_PASSED" in res.stdout

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_runner_abiotic_verification_catches_unauthorized_biological_objects(self):
        """Verify verify_strict_abiotic_scene catches unauthorized flora/fauna objects."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import MapConfig
from terra_forge.blender.runner import verify_strict_abiotic_scene

bpy.ops.wm.read_factory_settings(use_empty=True)

# Create terrain
mesh = bpy.data.meshes.new("Diorama_Island_Mesh")
obj = bpy.data.objects.new("Diorama_Island_Block", mesh)
bpy.context.scene.collection.objects.link(obj)

# Inject an unauthorized tree object
tree_mesh = bpy.data.meshes.new("Proto_Pine_Mesh")
tree_obj = bpy.data.objects.new("Diorama_Pine_Tree_01", tree_mesh)
bpy.context.scene.collection.objects.link(tree_obj)

cfg = MapConfig(name="Abiotic_Adversarial", biomes=[])
res = verify_strict_abiotic_scene(obj, cfg)

assert res["passed"] is False, "Expected abiotic verification to fail on unauthorized pine tree"
assert any("pine" in err.lower() or "tree" in err.lower() for err in res["errors"])

print("ABIOTIC_REJECTION_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "ABIOTIC_REJECTION_VERIFIED" in res.stdout

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_runner_abiotic_verification_catches_non_whitelisted_prefixes(self):
        """Verify verify_strict_abiotic_scene catches objects not in whitelisted prefixes."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import MapConfig
from terra_forge.blender.runner import verify_strict_abiotic_scene

bpy.ops.wm.read_factory_settings(use_empty=True)

# Create terrain
mesh = bpy.data.meshes.new("Diorama_Island_Mesh")
obj = bpy.data.objects.new("Diorama_Island_Block", mesh)
bpy.context.scene.collection.objects.link(obj)

# Inject an object with illegal prefix
cube_mesh = bpy.data.meshes.new("RandomCube_Mesh")
cube_obj = bpy.data.objects.new("Alien_Monolith_01", cube_mesh)
bpy.context.scene.collection.objects.link(cube_obj)

cfg = MapConfig(name="Abiotic_Prefix_Test", biomes=[])
res = verify_strict_abiotic_scene(obj, cfg)

assert res["passed"] is False, "Expected abiotic verification to fail on non-whitelisted prefix 'Alien_'"
assert any("Alien_Monolith_01" in err for err in res["errors"])

print("ABIOTIC_PREFIX_REJECTION_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender execution failed:\n{res.stderr}"
        assert "ABIOTIC_PREFIX_REJECTION_VERIFIED" in res.stdout

    def test_production_manifest_and_glb_strictly_abiotic(self):
        """Verify production map_manifest.json and ecosystem_map.glb contain 0 biological elements."""
        manifest_candidates = [
            Path("assets/map_manifest.json"),
            Path("assets/blender_map/map_manifest.json"),
            Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/map_manifest.json"),
            Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/map_manifest.json"),
        ]
        manifest_path = next((p for p in manifest_candidates if p.is_file()), None)
        if manifest_path is None:
            pytest.skip("Production manifest not present on disk")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_str = json.dumps(manifest).lower()
        forbidden = [
            "tree", "pine", "palm", "oak", "shrub", "bush", "grass", "flower",
            "fern", "mushroom", "moss", "log", "stump", "leaf", "flora",
            "fauna", "animal", "creature", "deer", "bird", "fish", "coral",
            "house", "building", "ruin", "bridge", "fence", "wall", "plank",
            "cabin", "temple", "tower", "castle", "road", "pavement"
        ]
        for term in forbidden:
            assert term not in manifest_str, f"Forbidden term '{term}' found in production manifest!"

        # Inspect ecosystem_map.glb binary directly (unpack chunk 0 JSON)
        glb_candidates = [
            Path("assets/blender_map/ecosystem_map.glb"),
            Path("e:/Project/01_AI_Agents/Genesis_Zero/assets/blender_map/ecosystem_map.glb"),
        ]
        glb_path = next((p for p in glb_candidates if p.is_file()), None)
        assert glb_path is not None, "ecosystem_map.glb not found on disk for abiotic verification!"

        with open(glb_path, "rb") as f:
            magic, version, length = struct.unpack("<4sII", f.read(12))
            assert magic == b"glTF", f"Invalid GLB magic header: {magic}"
            assert version == 2, f"Expected glTF version 2, got {version}"
            chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
            assert chunk_type == b"JSON", f"Expected JSON chunk 0, got {chunk_type}"
            chunk_data = f.read(chunk_len)
            gltf = json.loads(chunk_data.decode("utf-8"))

        nodes = gltf.get("nodes", [])
        meshes = gltf.get("meshes", [])
        assert len(nodes) > 0, "GLB contains 0 nodes"
        assert len(meshes) > 0, "GLB contains 0 meshes"

        detected_forbidden = []
        for n in nodes:
            name = n.get("name", "")
            name_l = name.lower()
            for tok in forbidden:
                if tok in name_l:
                    detected_forbidden.append(f"Node:{name}({tok})")

        for m in meshes:
            name = m.get("name", "")
            name_l = name.lower()
            for tok in forbidden:
                if tok in name_l:
                    detected_forbidden.append(f"Mesh:{name}({tok})")

        assert len(detected_forbidden) == 0, (
            f"Production GLB contains forbidden biological/architectural elements: {detected_forbidden}"
        )
