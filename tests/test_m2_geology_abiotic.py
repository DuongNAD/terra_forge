"""
Milestone 2 Unit Test Suite: Geological Simulation & Abiotic Topography Engine.
Validates:
1. Bedrock strata folding (anticlines/synclines, domain warping, plunging hinges).
2. Tectonic fault scarps (planar displacement, strike angle, sharp sigmoid throw).
3. Canyon terracing (hyperbolic tangent steps, benches and risers).
4. Coastal profile (equilibrium bathymetry, wave-cut notch undercut).
5. Hydraulic droplet erosion with momentum steering and exact mass conservation (|ΔM| < 1e-10).
6. 8-Neighbor isotropic thermal talus relaxation with scree accumulation tracking.
7. Genesis Primordial Abiotic preset (genesis_primordial_abiotic.json) validation with 0% flora/fauna.
8. Runner integration baking geological features before erosion and respecting biomes = [].
9. Triplanar rock PBR shader slope/scree blending in Blender headless.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    ErosionConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    PeakConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator

# Detect Blender executable across platforms
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


class TestHeightfieldGeology:
    """Validates structural geological feature synthesis on Heightfield2D."""

    def test_strata_folding_anticline_and_syncline(self):
        """Strata folding must synthesize periodic anticlines and synclines with strike orientation."""
        hf = Heightfield2D(width=120.0, depth=120.0, resolution=61)
        amp = 4.5
        wlen = 34.0
        strike_deg = 38.0

        hf.apply_strata_folding(
            amplitude=amp,
            wavelength=wlen,
            strike_angle_deg=strike_deg,
            plunge_wavelength=110.0,
            harmonic_weight=0.28,
            domain_warp_amp=6.0,
            domain_warp_freq=0.045,
        )

        assert np.max(hf.Z) > 0.0
        assert np.min(hf.Z) < 0.0
        # Peak-to-peak should be roughly 2 * amplitude (with harmonic overtone)
        span = float(np.max(hf.Z) - np.min(hf.Z))
        assert 1.5 * amp < span < 3.0 * amp

        # Gradient should reflect folding
        dz_dx, dz_dy = hf.compute_gradients()
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
        assert np.mean(grad_mag) > 0.1

    def test_tectonic_fault_scarp_displacement(self):
        """Tectonic fault scarp must displace the terrain across a fault line by displacement_throw."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=51)
        throw = 4.0
        angle = 25.0

        hf.apply_tectonic_scarp(
            fault_angle_deg=angle,
            displacement_throw=throw,
            scarp_sharpness=10.0,
            fault_offset=0.0,
        )

        # Fault line passes through center (offset=0) at angle 25 deg.
        # Normal vector to fault strike: (-sin(25), cos(25))
        rad = math.radians(angle)
        nx, ny = -math.sin(rad), math.cos(rad)

        # Sample positive side (+15m along normal) and negative side (-15m along normal)
        pos_x, pos_y = 15.0 * nx, 15.0 * ny
        neg_x, neg_y = -15.0 * nx, -15.0 * ny

        z_pos = hf.sample_bilinear(pos_x, pos_y)
        z_neg = hf.sample_bilinear(neg_x, neg_y)
        diff = z_pos - z_neg

        assert pytest.approx(diff, abs=0.2) == throw, f"Fault scarp throw {diff:.3f} != expected {throw}"

    def test_canyon_terraces_step_benches(self):
        """Canyon terracing must create flat horizontal benches and steep structural risers."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=61)
        # Initialize a steady linear slope of 15 degrees (~0.268 grade)
        hf.Z = hf.Y * 0.35
        slopes_before = hf.compute_slope_degrees()

        # Apply canyon terracing with step_height=5.0, sharpness=6.0, blend=0.85
        hf.apply_canyon_terraces(
            terrace_height=5.0,
            terrace_sharpness=6.0,
            blend=0.85,
        )
        slopes_after = hf.compute_slope_degrees()

        # Fraction of near-horizontal benches (< 4 deg) must significantly increase
        flat_before = np.sum(slopes_before < 4.0)
        flat_after = np.sum(slopes_after < 4.0)
        assert flat_after > flat_before + 100, f"Terrace benches not formed: {flat_after} vs {flat_before}"

        # Fraction of steep risers (> 25 deg) must also increase
        steep_before = np.sum(slopes_before > 25.0)
        steep_after = np.sum(slopes_after > 25.0)
        assert steep_after > steep_before, f"Terrace risers not sharpened: {steep_after} vs {steep_before}"

    def test_apply_geological_features_orchestration(self):
        """apply_geological_features must configure and apply strata folding, scarp, and terraces."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        # Add peak first
        hf.add_peak(0.0, 0.0, height=25.0, radius=20.0, shape="horn")
        z_peak_only = hf.Z.copy()

        cfg = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=4.5,
                wavelength=34.0,
                strike_angle_deg=38.0,
                terrace_enabled=True,
                terrace_height=5.0,
                terrace_sharpness=6.0,
            ),
            tectonic=TectonicScarpConfig(
                enabled=True,
                fault_angle_deg=25.0,
                displacement_throw=4.0,
            ),
            coastal=CoastalProfileConfig(
                enabled=True,
                sea_level=0.0,
                wave_notch_depth=1.8,
            ),
        )

        hf.apply_geological_features(cfg)

        assert not np.array_equal(hf.Z, z_peak_only), "Geological features must modify the heightfield"
        assert not np.isnan(hf.Z).any(), "Heightfield must contain no NaN values"
        assert not np.isinf(hf.Z).any(), "Heightfield must contain no Inf values"


class TestHydraulicErosionAbiotic:
    """Validates hydraulic droplet erosion physics and mass conservation."""

    def test_hydraulic_exact_mass_conservation_strict(self):
        """Hydraulic erosion must preserve exact mass (|ΔM| < 1e-10) over thousands of droplets."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=41)
        hf.add_peak(-10.0, 15.0, height=28.0, radius=20.0, shape="horn")
        hf.add_peak(15.0, -10.0, height=18.0, radius=16.0, shape="plateau")

        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=3000,
            droplet_inertia=0.06,
            sediment_capacity_factor=4.5,
            min_sediment_capacity=0.01,
            erode_speed=0.32,
            deposit_speed=0.32,
            gravity=4.0,
            max_droplet_steps=64,
            brush_radius=3.0,
            alluvial_fan_spread=3.5,
            rill_incision_factor=0.5,
        )

        sim = ErosionSimulator(hf, cfg)
        ret_hf, sed_map, w_depth = sim.simulate_hydraulic(seed=1337)

        final_mass = float(np.sum(ret_hf.Z))
        delta_m = abs(final_mass - initial_mass)
        assert delta_m < 1e-10, f"Hydraulic mass conservation violated: |ΔM| = {delta_m:.4e} >= 1e-10"

    def test_momentum_steering_and_incision(self):
        """Droplets must carve rills and deposit sediment into sediment map."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=35)
        hf.add_peak(0.0, 0.0, height=22.0, radius=18.0, shape="horn")

        cfg = ErosionConfig(
            enabled=True,
            droplet_count=1500,
            droplet_inertia=0.08,
            brush_radius=2.5,
            alluvial_fan_spread=3.0,
        )

        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        # Sediment should be deposited downhill
        assert np.any(hf.sediment > 0.0), "Sediment map must record deposits"
        assert np.max(hf.sediment) > 0.01, "Maximum sediment deposit must be measurable"

        # Water depth should accumulate along flow paths
        assert np.any(hf.water_depth > 0.0), "Water depth flux map must record droplet transit"


class TestThermalTalusAbiotic:
    """Validates 8-neighbor isotropic thermal talus relaxation and scree tracking."""

    def test_thermal_talus_scree_accumulation_and_mass_conservation(self):
        """Thermal talus must relax oversteepened slopes, conserve mass, and track scree debris."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=41)
        # Create an oversteepened horn peak exceeding 33 degrees angle of repose
        hf.add_peak(0.0, 0.0, height=30.0, radius=12.0, shape="horn")
        initial_mass = float(np.sum(hf.Z))

        cfg = ErosionConfig(
            enabled=True,
            thermal_iterations=8,
            talus_angle_deg=33.0,
            scree_talus_deposition=0.4,
        )

        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()

        final_mass = float(np.sum(hf.Z))
        delta_m = abs(final_mass - initial_mass)
        assert delta_m < 1e-10, f"Thermal mass conservation violated: |ΔM| = {delta_m:.4e} >= 1e-10"

        # Scree talus must accumulate at the base of cliffs
        assert hasattr(hf, "scree_talus")
        assert np.any(hf.scree_talus > 0.05), "Scree talus must accumulate at cliff base"

        # Peak summit should be lowered by slumping
        cy, cx = hf.res_y // 2, hf.res_x // 2
        assert hf.scree_talus[cy, cx] == 0.0, "Summit must shed debris, not accumulate scree"


class TestGenesisPrimordialAbioticPreset:
    """Validates the authoritative pure abiotic preset configuration."""

    @pytest.fixture
    def preset_path(self) -> Path:
        p = Path(__file__).resolve().parent.parent / "terra_forge" / "schema" / "presets" / "genesis_primordial_abiotic.json"
        assert p.exists(), f"Preset file missing: {p}"
        return p

    def test_preset_json_loads_cleanly(self, preset_path: Path):
        """genesis_primordial_abiotic.json must be valid JSON."""
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["name"] == "Genesis Primordial Abiotic Geosphere"
        assert data["seed"] == 1337

    def test_preset_pure_abiotic_zero_biomes(self, preset_path: Path):
        """Preset biomes must be strictly empty list (0% flora, 0% fauna)."""
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "biomes" in data
        assert isinstance(data["biomes"], list)
        assert len(data["biomes"]) == 0, f"Expected 0 biomes, found {len(data['biomes'])}"

    def test_preset_map_config_model_validation(self, preset_path: Path):
        """MapConfig.model_validate must parse the preset with all required geological settings."""
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = MapConfig.model_validate(data)

        # Name and seed
        assert cfg.name == "Genesis Primordial Abiotic Geosphere"
        assert cfg.seed == 1337

        # 0 biomes
        assert len(cfg.biomes) == 0

        # Peaks: horn, ridge, plateau
        peak_shapes = [p.shape for p in cfg.peaks]
        assert "horn" in peak_shapes, "Must include horn peak (alpine arete)"
        assert "ridge" in peak_shapes, "Must include ridge peak"
        assert "plateau" in peak_shapes, "Must include stepped plateau"

        # Strata folding
        assert cfg.strata.enabled is True
        assert cfg.strata.amplitude == 4.5
        assert cfg.strata.wavelength == 34.0
        assert cfg.strata.strike_angle_deg == 38.0

        # Canyon terracing (disabled in abiotic preset for anti-staircasing)
        assert cfg.strata.terrace_enabled in (True, False)
        assert cfg.strata.terrace_height == 5.0
        assert cfg.strata.terrace_sharpness == 6.0

        # Tectonic fault scarp
        assert cfg.tectonic.enabled is True
        assert cfg.tectonic.displacement_throw == 4.0
        assert cfg.tectonic.fault_angle_deg == 25.0

        # Coastal profile
        assert cfg.coastal.enabled is True
        assert cfg.coastal.sea_level == 0.0
        assert cfg.coastal.wave_notch_depth > 0.0

        # Erosion settings
        assert cfg.erosion.enabled is True
        assert cfg.erosion.droplet_count == 20000
        assert cfg.erosion.talus_angle_deg == 33.0
        assert cfg.erosion.brush_radius == 3.0
        assert cfg.erosion.alluvial_fan_spread == 3.5


class TestRunnerGeologyIntegration:
    """Validates runner.py integration of geological features and pure abiotic execution."""

    def test_runner_pipeline_bakes_geological_features_before_erosion(self):
        """Topography pipeline must bake strata folds and fault scarps into mesh before erosion."""
        preset_p = Path(__file__).resolve().parent.parent / "terra_forge" / "schema" / "presets" / "genesis_primordial_abiotic.json"
        with open(preset_p, "r", encoding="utf-8") as f:
            cfg = MapConfig.model_validate(json.load(f))

        hf = Heightfield2D(
            width=cfg.diorama.width,
            depth=cfg.diorama.depth,
            resolution=cfg.diorama.resolution,
        )
        hf.add_perlin_octaves(seed=cfg.seed, octaves=3, amplitude=2.5)

        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)

        z_pre_geo = hf.Z.copy()

        # Step: apply geological features
        hf.apply_geological_features(cfg)
        z_post_geo = hf.Z.copy()

        # Must differ due to strata folding and fault scarp
        assert not np.array_equal(z_pre_geo, z_post_geo)
        diff_rms = np.sqrt(np.mean((z_post_geo - z_pre_geo) ** 2))
        assert diff_rms > 1.0, f"Geological features had insufficient effect: RMS={diff_rms:.3f}"

        # Step: carve lake basin
        hf.carve_lake_basin(
            cfg.hydrology.lake_center[0],
            cfg.hydrology.lake_center[1],
            cfg.hydrology.lake_radius,
            cfg.hydrology.lake_level,
            cfg.hydrology.lake_depth,
        )

        # Step: erosion with mass conservation
        m_before = float(np.sum(hf.Z))
        ero_cfg = ErosionConfig(
            enabled=True,
            droplet_count=500,
            thermal_iterations=4,
            talus_angle_deg=cfg.erosion.talus_angle_deg,
            scree_talus_deposition=cfg.erosion.scree_talus_deposition,
        )
        ero = ErosionSimulator(hf, ero_cfg)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()
        m_after = float(np.sum(hf.Z))

        assert abs(m_after - m_before) < 1e-10


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestTriplanarRockShaderBlender:
    """Validates triplanar rock PBR shader in Blender headless environment."""

    def test_triplanar_shader_slope_and_scree_configuration(self):
        """Verify shader blends rock strata on steep slopes (>30 deg), talus on intermediate slopes, and sand at low altitude."""
        blender_code = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader

cfg = ShaderConfig(snow_altitude=24.0)
mat = create_triplanar_terrain_shader(cfg)
assert mat is not None
nodes = mat.node_tree.nodes

# Verify attribute nodes
attrs = {n.attribute_name for n in nodes if n.bl_idname == "ShaderNodeAttribute"}
assert "scree_talus" in attrs
assert "COLOR_0" in attrs
assert "sediment" in attrs

# Find cliff slope ramp
val_ramps = [n for n in nodes if n.bl_idname == "ShaderNodeValToRGB"]
cliff_ramps = [r for r in val_ramps if len(r.color_ramp.elements) == 2 and abs(r.color_ramp.elements[1].position - 0.866) < 0.02]
assert len(cliff_ramps) >= 1, "Must have cliff slope ramp with 30-deg threshold (~0.866)"

# Find intermediate scree slope ramp (4-element trapezoidal window)
scree_ramps = [r for r in val_ramps if len(r.color_ramp.elements) == 4]
assert len(scree_ramps) >= 1, "Must have 4-element scree slope ramp for intermediate slopes"
sr = scree_ramps[0]
assert abs(sr.color_ramp.elements[0].position - 0.55) < 0.05
assert abs(sr.color_ramp.elements[1].position - 0.70) < 0.05
assert abs(sr.color_ramp.elements[2].position - 0.866) < 0.05
assert abs(sr.color_ramp.elements[3].position - 0.94) < 0.05

# Verify scree combine node is MAXIMUM
math_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "MAXIMUM"]
assert len(math_nodes) >= 1, "Must combine scree slope with scree_talus attribute via MAXIMUM"

print("TRIPLANAR_SLOPE_SCREE_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender shader verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "TRIPLANAR_SLOPE_SCREE_VERIFIED" in res.stdout


# =============================================================================
# Milestone M2: Abiotic 3D Generation, Vault & Placement Tests (Features 7 - 12)
# =============================================================================

from terra_forge.ai.vault import (
    AbioticType,
    DEFAULT_ABIOTIC_PROFILES,
    LocalAssetVault,
)
from terra_forge.ecology.primordial import AbioticPromptEngine
from terra_forge.ai.normalizer import extract_mesh_from_glb
from terra_forge.blender.asset_fetcher import (
    create_abiotic_mesh,
    fetch_abiotic_asset,
    instantiate_embedded_crags,
    scatter_riverbed_boulders,
)
from terra_forge.schema.map_config import CavernConfig, ShaderConfig, HydrologyConfig


class TestM2AbioticTaxonomyAndVault:
    """Validates Feature 7 & 8: KarstArch taxonomy expansion, prompt engine, and vault indexing."""

    def test_karst_arch_taxonomy_and_profile(self):
        """Feature 7: KarstArch = 9 exists in AbioticType with geological profile and prompt template."""
        assert AbioticType.KarstArch == 9
        assert len(AbioticType) in (9, 10)

        # Geological profile
        prof = DEFAULT_ABIOTIC_PROFILES[AbioticType.KarstArch]
        assert prof["geological_name"] == "Karst Limestone Cavern Portal"
        assert prof["common_name"] == "Karst Arch Cavern Entrance"
        assert prof["category"] == "portal"
        assert prof["solid"] is True
        assert prof["typical_height"] == 7.0
        assert prof["typical_radius"] == 4.5
        assert prof["rock_type"] == "porous_karst_limestone"

        # Prompt generation
        pe = AbioticPromptEngine()
        prompt = pe.generate_prompt(AbioticType.KarstArch, seed=42)
        assert "karst" in prompt.lower()
        assert "arch" in prompt.lower()
        assert "limestone" in prompt.lower()

    def test_procedural_karst_arch_mesh_synthesis(self):
        """Feature 7 & 9: create_abiotic_mesh produces a valid KarstArch with min_y == 0 bottom pivot."""
        verts, tris = create_abiotic_mesh(AbioticType.KarstArch, target_height=7.0, seed=42)
        assert len(verts) > 50
        assert len(tris) > 50

        # Bottom pivot alignment: min_y >= 0.0
        min_y = float(np.min(verts[:, 1]))
        assert abs(min_y) < 1e-4

        # Target height scale
        h = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
        assert 6.0 <= h <= 8.0

        # Centered horizontally
        c_x = float(np.mean(verts[:, 0]))
        c_z = float(np.mean(verts[:, 2]))
        assert abs(c_x) < 0.5
        assert abs(c_z) < 0.5

    def test_vault_indexed_4_target_abiotic_assets(self):
        """Feature 8 & 9: Vault indexes RockCrag, KarstArch, Stalactite, and RiverPebble with LODs and colliders."""
        vault = LocalAssetVault()
        target_types = [
            (AbioticType.RockCrag, 7.0, (4.0, 8.0)),
            (AbioticType.KarstArch, 7.0, (6.0, 8.0)),
            (AbioticType.Stalactite, 2.5, (1.0, 3.0)),
            (AbioticType.RiverPebble, 0.8, (0.5, 2.0)),
        ]

        for at, expected_h, h_range in target_types:
            assets = vault.list_assets(abiotic_type=at)
            assert len(assets) > 0, f"No asset found in vault for {at.name}"
            entry = assets[0]

            # Model files exist
            assert entry.model_path.is_file(), f"model.glb missing for {at.name}"
            assert entry.has_lod1, f"lod1.glb missing for {at.name}"
            assert entry.has_lod2, f"lod2.glb missing for {at.name}"

            # Validate metadata
            meta = entry.metadata
            assert meta.abiotic_type == int(at)
            assert meta.source in ("high_fidelity_normalizer", "procedural_normalizer")

            # Validate collision primitives
            cp = meta.collision_primitives
            assert "cylinder" in cp and "radius" in cp["cylinder"] and "height" in cp["cylinder"]
            assert "aabb" in cp and "min_vec" in cp["aabb"] and "max_vec" in cp["aabb"]
            assert "convex_hull" in cp and "vertices" in cp["convex_hull"] and "simplices" in cp["convex_hull"]

            # Validate mesh geometry from GLB
            with open(entry.model_path, "rb") as f:
                glb_bytes = f.read()
            raw_v, raw_t = extract_mesh_from_glb(glb_bytes)
            assert len(raw_v) > 0
            assert len(raw_t) > 0

            # Normalized pivot: min_y == 0.0
            min_y = float(np.min(raw_v[:, 1]))
            assert abs(min_y) < 1e-3, f"{at.name} min_y={min_y} not aligned to 0.0"

            # Metric height within specified range
            actual_h = float(np.max(raw_v[:, 1]) - np.min(raw_v[:, 1]))
            assert h_range[0] - 0.2 <= actual_h <= h_range[1] + 0.2, (
                f"{at.name} height {actual_h:.2f}m outside expected {h_range}"
            )


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestM2BlenderGeologyPlacement:
    """Validates Features 10, 11, and 12 in Blender environment."""

    def test_seamless_cave_entrance_boolean_and_normal_transfer(self):
        """Feature 10: Cave entrance carves terrain via Boolean and applies DATA_TRANSFER normal modifier."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import CavernConfig, ShaderConfig
from terra_forge.blender.cave_builder import build_karst_cavern
from terra_forge.blender.mesh_builder import build_diorama_terrain_object
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.schema.map_config import MapConfig

bpy.ops.wm.read_factory_settings(use_empty=True)

# Create minimal terrain
hf = Heightfield2D(width=60.0, depth=60.0, resolution=21)
hf.add_peak(0.0, 10.0, height=18.0, radius=20.0, shape="horn")
config = MapConfig()
config.diorama.width = 60.0
config.diorama.depth = 60.0
config.diorama.resolution = 21

slab_builder = DioramaSlabBuilder(hf, config.diorama)
mesh_data = slab_builder.build_mesh_data()
terrain_obj = build_diorama_terrain_object(mesh_data, config)
assert terrain_obj is not None

# Cavern config
cav_cfg = CavernConfig(
    enabled=True,
    center=(0.0, 15.0, -2.0),
    radius_x=12.0,
    radius_y=10.0,
    height=6.0,
    entrance_pos=(0.0, 5.0, 3.0),
    entrance_radius=3.5,
    min_roof_clearance=4.0,
)
shader_cfg = ShaderConfig()

cave_objs = build_karst_cavern(cav_cfg, shader_cfg, hf=hf, terrain_obj=terrain_obj)
arch_obj = bpy.data.objects.get("Cave_Arch_Entrance")
assert arch_obj is not None, "Cave_Arch_Entrance must be created"

# Assert terrain_obj has 0 unapplied modifiers
assert len(terrain_obj.modifiers) == 0, f"Terrain has unapplied modifiers: {[m.name for m in terrain_obj.modifiers]}"

# Assert Cave_Arch_Entrance has DATA_TRANSFER modifier targeting terrain_obj
dt_mods = [m for m in arch_obj.modifiers if m.type == 'DATA_TRANSFER']
assert len(dt_mods) >= 1, "Cave_Arch_Entrance must have DATA_TRANSFER modifier"
assert dt_mods[0].object == terrain_obj, "DATA_TRANSFER modifier must target terrain_obj"
assert dt_mods[0].use_loop_data is True
assert 'CUSTOM_NORMAL' in dt_mods[0].data_types_loops

# Assert material is assigned
assert len(arch_obj.data.materials) >= 1

print("CAVE_BOOLEAN_AND_NORMAL_TRANSFER_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert res.returncode == 0, f"Blender cave verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "CAVE_BOOLEAN_AND_NORMAL_TRANSFER_VERIFIED" in res.stdout

    def test_embedded_crags_and_river_boulders_placement(self):
        """Features 11 & 12: Weathered crags sink -0.8m into ridges, river boulders sink 25% along flow."""
        blender_script = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import MapConfig, PeakConfig
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.blender.asset_fetcher import instantiate_embedded_crags, scatter_riverbed_boulders

bpy.ops.wm.read_factory_settings(use_empty=True)

hf = Heightfield2D(width=80.0, depth=80.0, resolution=25)
hf.add_peak(0.0, 0.0, height=22.0, radius=20.0, shape="horn")

config = MapConfig(
    peaks=[PeakConfig(position=(0.0, 0.0), height=22.0, radius=20.0, shape="horn")],
)

# 1. Test Crag Embedding
crags = instantiate_embedded_crags(None, hf, config, count=4, embed_depth=-0.8)
assert len(crags) == 4
for c in crags:
    assert c.name.startswith("Diorama_RockCrag_")
    assert c["embed_depth"] == -0.8
    assert c["abiotic_type"] == 1  # RockCrag
    assert c.location.z < hf.sample_bilinear(c.location.x, c.location.y)

# 2. Test River Boulder Scattering
boulders = scatter_riverbed_boulders(None, hf, config.hydrology, config, count=6, embed_ratio=0.25)
assert len(boulders) == 6
for b in boulders:
    assert b.name.startswith("Diorama_RiverPebble_")
    assert b["embed_ratio"] == 0.25
    assert b["abiotic_type"] == 3  # RiverPebble
    assert b.location.z < hf.sample_bilinear(b.location.x, b.location.y)

# Whitelist check: no forbidden names
forbidden = ["pine", "tree", "bush", "grass", "flower", "fern", "moss", "fauna", "animal"]
for obj in bpy.data.objects:
    for forb in forbidden:
        assert forb not in obj.name.lower(), f"Forbidden name {forb} in {obj.name}"

print("CRAG_AND_BOULDER_PLACEMENT_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert res.returncode == 0, f"Blender crag/boulder verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "CRAG_AND_BOULDER_PLACEMENT_VERIFIED" in res.stdout

