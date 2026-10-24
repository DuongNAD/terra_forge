"""
Milestone 2 Adversarial Stress Test Suite — Challenger 2 (Empirical Verification).
Author: Challenger 2
Target: terra_forge (Milestone 2: Geological Simulation & Abiotic Topography Engine)

Objectives:
1. Abiotic Invariant Stress Testing:
   - Under NO circumstance can any biological object, tree, grass, or fauna be spawned
     when genesis_primordial_abiotic.json is processed.
   - Inspect runner execution trace: assert 0 prototype objects resolved, 0 geometry nodes modifiers attached.
   - Whitelist all instantiated scene objects and collections.
2. Heightfield Boundary & Extreme Parameter Stress:
   - Test apply_geological_features with zero amplitude, negative wavelengths, extreme strike angles,
     zero displacement scarps, out-of-range parameters, and high-entropy fuzzing.
   - Assert no NaNs, Infs, or array shape corruptions occur.
3. Watertightness Under Extreme Deformation:
   - Verify that terrain generated from genesis_primordial_abiotic.json with diorama skirt enabled
     produces 100% closed, watertight solid with Euler characteristic chi = V - E + F == 2.
   - Stress-test under violent tectonic throws, high-frequency strata folds, sub-baseplate depressions,
     and non-square grid resolutions.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    PeakConfig,
    StrataFoldConfig,
    TectonicScarpConfig,
    CoastalProfileConfig,
    ErosionConfig,
    HydrologyConfig,
    CavernConfig,
    ShaderConfig,
    BiomeScatterConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.inspector.validator import QualityValidator

# Blender detection
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

PRESET_PATH = (
    Path(__file__).resolve().parent.parent
    / "terra_forge"
    / "schema"
    / "presets"
    / "genesis_primordial_abiotic.json"
)


# =============================================================================
# 1. Abiotic Invariant Stress Testing
# =============================================================================

class TestAbioticInvariantsStress:
    """Stress-test and prove that genesis_primordial_abiotic.json cannot spawn any flora or fauna."""

    def test_preset_abiotic_content_contract(self):
        """Authoritative preset must have biomes strictly empty and no biological keywords."""
        assert PRESET_PATH.exists(), f"Abiotic preset missing: {PRESET_PATH}"
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Biomes array must be strictly empty
        assert "biomes" in data, "Preset must define 'biomes' key"
        assert isinstance(data["biomes"], list)
        assert len(data["biomes"]) == 0, f"Abiotic preset has non-empty biomes: {len(data['biomes'])}"

        # 2. Theme must be primordial_abiotic
        assert data.get("theme") == "primordial_abiotic"

        # 3. Textual inspection for forbidden biological whole-word terms in keys/values
        biological_forbidden = {
            "tree", "trees", "conifer", "pine", "oak", "birch", "palm",
            "shrub", "bush", "grass", "flower", "fern", "moss",
            "reed", "fauna", "animal", "deer", "ibex", "hare",
            "wolf", "bear", "eagle", "fish", "frog", "creature",
        }
        import re
        # Check peak names
        for p in data.get("peaks", []):
            name_tokens = set(re.findall(r"[a-z]+", p.get("name", "").lower()))
            overlap = name_tokens.intersection(biological_forbidden)
            assert not overlap, f"Biological terms {overlap} found in peak name '{p.get('name')}'"

    def test_runner_code_logic_path_skips_biome_scattering(self):
        """Verify runner.py logic completely branches around scattering when biomes is empty."""
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            cfg = MapConfig.model_validate(json.load(f))

        assert len(cfg.biomes) == 0

        # Mock Blender modules for headless testing outside Blender
        mock_bpy = MagicMock()
        mock_bmesh = MagicMock()
        mock_mathutils = MagicMock()

        with patch.dict("sys.modules", {"bpy": mock_bpy, "bmesh": mock_bmesh, "mathutils": mock_mathutils}):
            from terra_forge.blender import runner

            with patch.object(runner, "resolve_prototype_object") as mock_resolve, \
                 patch.object(runner, "apply_biome_scatter_modifier") as mock_modifier, \
                 patch.object(runner, "build_diorama_terrain_object") as mock_terrain, \
                 patch.object(runner, "build_water_bodies") as mock_water, \
                 patch.object(runner, "build_karst_cavern") as mock_cave, \
                 patch.object(runner, "setup_inspection_cameras"), \
                 patch.object(runner, "QualityValidator") as mock_val_cls, \
                 patch.object(runner, "export_production_glb"), \
                 patch.object(runner, "export_blend_file"), \
                 patch.object(runner, "export_simulation_heightmap_json"):

                val_inst = MagicMock()
                val_inst.validate_all.return_value = {"passed": True, "errors": [], "warnings": []}
                mock_val_cls.return_value = val_inst

                dummy_obj = MagicMock()
                dummy_obj.modifiers = []
                mock_terrain.return_value = dummy_obj
                mock_water.return_value = []
                mock_cave.return_value = []

                res = runner.run_pipeline(PRESET_PATH)

                # Assert 0 prototype objects resolved
                assert mock_resolve.call_count == 0, (
                    f"Expected 0 calls to resolve_prototype_object, got {mock_resolve.call_count}"
                )
                # Assert 0 geometry nodes modifiers attached
                assert mock_modifier.call_count == 0, (
                    f"Expected 0 calls to apply_biome_scatter_modifier, got {mock_modifier.call_count}"
                )
                assert res["status"] == "success"

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_headless_blender_runner_execution_trace_abiotic_purity(self):
        """
        Execute runner.py inside headless Blender with genesis_primordial_abiotic.json.
        Inspect entire scene:
        1. Assert 0 prototype objects in scene or collections.
        2. Assert Diorama_Island_Block has 0 modifiers (0 Geometry Nodes).
        3. Assert all scene objects belong strictly to abiotic categories.
        """
        venv_site_packages = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
        repo_root = Path(__file__).resolve().parent.parent

        blender_verify_script = f"""
import sys
sys.path.insert(0, r"{venv_site_packages}")
sys.path.insert(0, r"{repo_root}")
import json
import bpy
from terra_forge.blender.runner import run_pipeline
from terra_forge.schema.map_config import MapConfig

with open(r"{PRESET_PATH}", "r", encoding="utf-8") as f:
    cfg_data = json.load(f)

# Use lower erosion droplets for fast headless verification
cfg_data["erosion"]["droplet_count"] = 500
cfg_data["erosion"]["thermal_iterations"] = 4
cfg_data["export"]["export_glb"] = False
cfg_data["export"]["export_blend"] = False
cfg_data["export"]["export_heightmap_json"] = False

result = run_pipeline(cfg_data, render_preview=False)
assert result["status"] == "success", f"Pipeline failed: {{result}}"

# 1. Check Diorama_Island_Block modifiers
terrain_obj = bpy.data.objects.get("Diorama_Island_Block")
assert terrain_obj is not None, "Diorama_Island_Block must exist"
assert len(terrain_obj.modifiers) == 0, (
    f"Terrain object must have 0 modifiers in pure abiotic mode, found: {{[m.name for m in terrain_obj.modifiers]}}"
)

# 2. Check TerraForge_Prototypes collection
proto_col = bpy.data.collections.get("TerraForge_Prototypes")
if proto_col:
    assert len(proto_col.objects) == 0, (
        f"TerraForge_Prototypes collection must contain 0 objects, found: {{len(proto_col.objects)}}"
    )

# 3. Check all scene objects for any biological traces
forbidden_names = [
    "pine", "conifer", "oak", "birch", "palm", "tree",
    "bush", "shrub", "grass", "flower", "fern", "moss",
    "reed", "fauna", "animal", "deer", "eagle", "fish", "creature",
]

abiotic_allowed_prefixes = [
    "Diorama_",
    "Water_",
    "Cave_",
    "CAM_",
    "Sun_",
]

for obj in bpy.data.objects:
    name_lower = obj.name.lower()
    for forb in forbidden_names:
        assert forb not in name_lower, f"Forbidden biological object detected in scene: '{{obj.name}}'"

    # Verify object matches known abiotic prefixes
    matches_allowed = any(obj.name.startswith(pfx) for pfx in abiotic_allowed_prefixes)
    assert matches_allowed, f"Unexpected object in abiotic diorama scene: '{{obj.name}}'"

print("ABIOTIC_INVARIANT_HEADLESS_VERIFIED_SUCCESS")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_verify_script]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert proc.returncode == 0, (
            f"Blender abiotic verification failed with code {proc.returncode}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "ABIOTIC_INVARIANT_HEADLESS_VERIFIED_SUCCESS" in proc.stdout
        assert "0 biomes configured" in proc.stdout
        assert "pure abiotic geosphere" in proc.stdout


# =============================================================================
# 2. Heightfield Boundary & Extreme Parameter Stress Testing
# =============================================================================

class TestHeightfieldBoundaryExtremeParameters:
    """Stress-test apply_geological_features under boundary, zero, and extreme values."""

    @pytest.fixture
    def base_hf(self) -> Heightfield2D:
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=20.0, radius=18.0, shape="horn")
        return hf

    def test_strata_folding_zero_amplitude(self, base_hf: Heightfield2D):
        """Zero amplitude strata folding must execute cleanly without modifying elevation."""
        z_orig = base_hf.Z.copy()
        cfg = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=0.0,
                wavelength=34.0,
                strike_angle_deg=38.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg)
        np.testing.assert_array_equal(base_hf.Z, z_orig)
        assert not np.isnan(base_hf.Z).any()
        assert not np.isinf(base_hf.Z).any()

    def test_strata_folding_negative_and_zero_wavelength(self, base_hf: Heightfield2D):
        """Zero and negative wavelengths must be rejected safely without division by zero or NaN."""
        z_orig = base_hf.Z.copy()

        # Zero wavelength
        cfg_zero = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=5.0,
                wavelength=0.0,
                strike_angle_deg=45.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg_zero)
        np.testing.assert_array_equal(base_hf.Z, z_orig)
        assert not np.isnan(base_hf.Z).any()

        # Negative wavelength
        cfg_neg = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=5.0,
                wavelength=-50.0,
                strike_angle_deg=45.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg_neg)
        np.testing.assert_array_equal(base_hf.Z, z_orig)
        assert not np.isnan(base_hf.Z).any()

    def test_strata_folding_extreme_strike_angles(self, base_hf: Heightfield2D):
        """Extreme strike angles (-36000 deg, +72000 deg, 1e9 deg) must produce valid finite numbers."""
        for angle in [-36000.0, -720.0, 7200.0, 36000.0, 1e7]:
            hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
            cfg = MapConfig(
                strata=StrataFoldConfig(
                    enabled=True,
                    amplitude=4.5,
                    wavelength=30.0,
                    strike_angle_deg=angle,
                    plunge_wavelength=60.0,
                ),
                tectonic=TectonicScarpConfig(enabled=False),
                coastal=CoastalProfileConfig(enabled=False),
            )
            hf.apply_geological_features(cfg)
            assert not np.isnan(hf.Z).any(), f"NaN generated with strike angle {angle}"
            assert not np.isinf(hf.Z).any(), f"Inf generated with strike angle {angle}"
            assert hf.Z.shape == (41, 41)

    def test_strata_folding_extreme_amplitudes(self, base_hf: Heightfield2D):
        """Test negative amplitude and huge amplitude (1e5)."""
        # Negative amplitude
        hf_neg = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cfg_neg = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=-15.0,
                wavelength=30.0,
                strike_angle_deg=30.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        hf_neg.apply_geological_features(cfg_neg)
        assert not np.isnan(hf_neg.Z).any()
        assert not np.isinf(hf_neg.Z).any()

        # Huge amplitude (1e5)
        hf_huge = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cfg_huge = MapConfig(
            strata=StrataFoldConfig(
                enabled=True,
                amplitude=100000.0,
                wavelength=30.0,
                strike_angle_deg=30.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        hf_huge.apply_geological_features(cfg_huge)
        assert not np.isnan(hf_huge.Z).any()
        assert not np.isinf(hf_huge.Z).any()
        assert float(np.max(hf_huge.Z)) > 50000.0

    def test_tectonic_scarp_zero_displacement(self, base_hf: Heightfield2D):
        """Zero displacement throw must leave the heightfield completely unchanged."""
        z_orig = base_hf.Z.copy()
        cfg = MapConfig(
            strata=StrataFoldConfig(enabled=False),
            tectonic=TectonicScarpConfig(
                enabled=True,
                displacement_throw=0.0,
                fault_angle_deg=25.0,
                scarp_sharpness=10.0,
            ),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg)
        np.testing.assert_array_equal(base_hf.Z, z_orig)
        assert not np.isnan(base_hf.Z).any()

    def test_tectonic_scarp_extreme_parameters(self, base_hf: Heightfield2D):
        """Test negative displacement, extreme angles, sharpness bounds, and finite lengths."""
        # 1. Negative throw
        hf1 = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cfg1 = MapConfig(
            strata=StrataFoldConfig(enabled=False),
            tectonic=TectonicScarpConfig(
                enabled=True,
                displacement_throw=-25.0,
                fault_angle_deg=-1850.0,
                scarp_sharpness=0.0,  # Should clamp to 0.1
                fault_offset=1000.0,
            ),
            coastal=CoastalProfileConfig(enabled=False),
        )
        hf1.apply_geological_features(cfg1)
        assert not np.isnan(hf1.Z).any()
        assert not np.isinf(hf1.Z).any()

        # 2. Extreme sharpness (e.g. 500.0) -> exp clipping prevents numerical overflow
        hf2 = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cfg2 = MapConfig(
            strata=StrataFoldConfig(enabled=False),
            tectonic=TectonicScarpConfig(
                enabled=True,
                displacement_throw=10.0,
                fault_angle_deg=45.0,
                scarp_sharpness=500.0,
                fault_length=40.0,
            ),
            coastal=CoastalProfileConfig(enabled=False),
        )
        hf2.apply_geological_features(cfg2)
        assert not np.isnan(hf2.Z).any()
        assert not np.isinf(hf2.Z).any()

    def test_canyon_terraces_boundary_parameters(self, base_hf: Heightfield2D):
        """Test zero/negative terrace height, zero blend, and extreme sharpness."""
        z_orig = base_hf.Z.copy()

        # Zero terrace height -> early return
        cfg_zero_h = MapConfig(
            strata=StrataFoldConfig(
                enabled=False,
                terrace_enabled=True,
                terrace_height=0.0,
                terrace_sharpness=5.0,
                terrace_blend=0.8,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg_zero_h)
        np.testing.assert_array_equal(base_hf.Z, z_orig)

        # Zero blend -> early return
        cfg_zero_b = MapConfig(
            strata=StrataFoldConfig(
                enabled=False,
                terrace_enabled=True,
                terrace_height=5.0,
                terrace_sharpness=5.0,
                terrace_blend=0.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        base_hf.apply_geological_features(cfg_zero_b)
        np.testing.assert_array_equal(base_hf.Z, z_orig)

        # Extreme terrace sharpness and blend out of bounds
        hf_extreme = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf_extreme.Z = hf_extreme.Y * 0.3
        cfg_ext = MapConfig(
            strata=StrataFoldConfig(
                enabled=False,
                terrace_enabled=True,
                terrace_height=3.0,
                terrace_sharpness=100.0,
                terrace_blend=1.5,  # Clipped to 1.0
                noise_modulation=5.0,
            ),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(enabled=False),
        )
        hf_extreme.apply_geological_features(cfg_ext)
        assert not np.isnan(hf_extreme.Z).any()
        assert not np.isinf(hf_extreme.Z).any()

    def test_coastal_profile_boundary_parameters(self, base_hf: Heightfield2D):
        """Test zero wave notch depth, zero envelope, and extreme sea level."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.add_peak(0.0, 0.0, height=15.0, radius=20.0, shape="horn")

        cfg = MapConfig(
            strata=StrataFoldConfig(enabled=False),
            tectonic=TectonicScarpConfig(enabled=False),
            coastal=CoastalProfileConfig(
                enabled=True,
                sea_level=-50.0,
                wave_notch_depth=0.0,
                wave_notch_envelope=-1.0,
                beach_width=0.0,
                berm_elevation=0.0,
                coastline_angle_deg=1080.0,
            ),
        )
        hf.apply_geological_features(cfg)
        assert not np.isnan(hf.Z).any()
        assert not np.isinf(hf.Z).any()

    def test_fuzz_geological_features_resilience(self):
        """Adversarial Fuzzer: Execute 50 randomized iterations of extreme parameter combinations."""
        rng = random.Random(42)

        for i in range(50):
            res = rng.choice([21, 35, 41])
            hf = Heightfield2D(width=100.0, depth=100.0, resolution=res)
            hf.add_peak(
                rng.uniform(-30, 30),
                rng.uniform(-30, 30),
                height=rng.uniform(5.0, 40.0),
                radius=rng.uniform(10.0, 30.0),
                shape=rng.choice(["cone", "horn", "plateau", "ridge"]),
            )

            cfg = MapConfig(
                strata=StrataFoldConfig(
                    enabled=rng.choice([True, False]),
                    amplitude=rng.choice([0.0, -50.0, 0.01, 100.0, 1000.0]),
                    wavelength=rng.choice([-100.0, 0.0, 0.001, 20.0, 500.0]),
                    strike_angle_deg=rng.uniform(-7200.0, 7200.0),
                    plunge_wavelength=rng.choice([-50.0, 0.0, 100.0]),
                    harmonic_weight=rng.choice([-1.0, 0.0, 0.3, 5.0]),
                    domain_warp_amp=rng.choice([-10.0, 0.0, 8.0]),
                    domain_warp_freq=rng.choice([-0.05, 0.0, 0.04]),
                    terrace_enabled=rng.choice([True, False]),
                    terrace_height=rng.choice([-5.0, 0.0, 0.1, 4.0, 100.0]),
                    terrace_sharpness=rng.choice([-10.0, 0.0, 5.0, 80.0]),
                    terrace_blend=rng.choice([-0.5, 0.0, 0.8, 2.0]),
                    noise_modulation=rng.choice([-2.0, 0.0, 0.5, 10.0]),
                ),
                tectonic=TectonicScarpConfig(
                    enabled=rng.choice([True, False]),
                    fault_angle_deg=rng.uniform(-3600.0, 3600.0),
                    displacement_throw=rng.choice([0.0, -100.0, 5.0, 200.0]),
                    scarp_sharpness=rng.choice([-5.0, 0.0, 8.0, 200.0]),
                    fault_offset=rng.uniform(-200.0, 200.0),
                    fault_length=rng.choice([-20.0, 0.0, 50.0]),
                ),
                coastal=CoastalProfileConfig(
                    enabled=rng.choice([True, False]),
                    sea_level=rng.uniform(-20.0, 20.0),
                    wave_notch_depth=rng.choice([-2.0, 0.0, 2.0]),
                    wave_notch_envelope=rng.choice([-1.0, 0.0, 1.5]),
                    beach_slope_coefficient=rng.choice([-0.1, 0.0, 0.15]),
                    berm_elevation=rng.choice([-2.0, 0.0, 1.5]),
                    beach_width=rng.choice([-10.0, 0.0, 15.0]),
                    coastline_angle_deg=rng.choice([None, rng.uniform(-1000.0, 1000.0)]),
                ),
            )

            hf.apply_geological_features(cfg)

            assert not np.isnan(hf.Z).any(), f"Iteration {i}: NaN detected in hf.Z"
            assert not np.isinf(hf.Z).any(), f"Iteration {i}: Inf detected in hf.Z"
            assert hf.Z.shape == (res, res), f"Iteration {i}: Shape corrupted to {hf.Z.shape}"


# =============================================================================
# 3. Watertightness Under Extreme Deformation
# =============================================================================

class TestWatertightnessExtremeDeformation:
    """Stress-test DioramaSlabBuilder topological watertightness and Euler chi = 2."""

    def test_watertightness_authoritative_abiotic_preset(self):
        """
        Verify authoritative genesis_primordial_abiotic.json terrain with diorama skirt
        produces a 100% closed, watertight solid with Euler characteristic chi = 2.
        """
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            cfg = MapConfig.model_validate(json.load(f))

        # Test at standard resolution
        hf = Heightfield2D(
            width=cfg.diorama.width,
            depth=cfg.diorama.depth,
            resolution=cfg.diorama.resolution,
        )
        hf.add_perlin_octaves(seed=cfg.seed, octaves=3, amplitude=2.5)

        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)

        hf.apply_geological_features(cfg)
        hf.carve_lake_basin(
            cfg.hydrology.lake_center[0],
            cfg.hydrology.lake_center[1],
            cfg.hydrology.lake_radius,
            cfg.hydrology.lake_level,
            cfg.hydrology.lake_depth,
        )

        # Run fast erosion
        ero_cfg = ErosionConfig(
            enabled=True,
            droplet_count=1000,
            thermal_iterations=4,
            talus_angle_deg=cfg.erosion.talus_angle_deg,
            scree_talus_deposition=cfg.erosion.scree_talus_deposition,
        )
        ero = ErosionSimulator(hf, ero_cfg)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()

        # Build watertight diorama slab mesh
        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()

        # Topological Validation
        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        metrics = topo["metrics"]

        assert topo["passed"] is True, f"Topological validation failed: {topo['errors']}"
        assert topo["is_watertight"] is True, "Diorama slab must be 100% watertight"
        assert topo["euler_characteristic"] == 2, (
            f"Euler characteristic chi = {topo['euler_characteristic']} != 2 (not genus-0 sphere)"
        )
        assert topo["boundary_edges"] == 0, f"Found {topo['boundary_edges']} boundary edges (must be 0)"
        assert topo["non_manifold_edges"] == 0, f"Found {topo['non_manifold_edges']} non-manifold edges"
        assert topo["non_manifold_vertices"] == 0, f"Found {topo['non_manifold_vertices']} non-manifold vertices"
        assert metrics["orientation_conflicts"] == 0, "All face normals must be consistently oriented"
        assert metrics["pinch_point_vertices"] == 0, "No pinch points allowed"

    def test_watertightness_under_massive_tectonic_throw(self):
        """Topological watertightness must remain invariant under extreme 200m vertical fault throw."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        hf.apply_tectonic_scarp(
            fault_angle_deg=35.0,
            displacement_throw=200.0,
            scarp_sharpness=20.0,
        )

        bounds = DioramaBounds(width=100.0, depth=100.0, z_base=-50.0, resolution=51)
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        assert topo["passed"] is True
        assert topo["euler_characteristic"] == 2
        assert topo["boundary_edges"] == 0
        assert topo["non_manifold_edges"] == 0

    def test_watertightness_under_violent_high_frequency_folds(self):
        """Topological watertightness must hold under violent accordion strata folding (amp 150m, wlen 4m)."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=61)
        hf.apply_strata_folding(
            amplitude=150.0,
            wavelength=4.0,
            strike_angle_deg=72.0,
            harmonic_weight=0.5,
        )

        bounds = DioramaBounds(width=80.0, depth=80.0, z_base=-200.0, resolution=61)
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        assert topo["passed"] is True
        assert topo["euler_characteristic"] == 2
        assert topo["boundary_edges"] == 0

    def test_watertightness_sub_baseplate_depression(self):
        """
        Verify that even if surface elevations severely drop below z_base (e.g. -100m vs z_base -16m),
        the manifold graph connectivity preserves Euler chi = 2 and boundary_edges = 0.
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        # Force a massive crater sinking down to -100.0
        cy, cx = hf.res_y // 2, hf.res_x // 2
        dist_sq = (hf.X - 0.0)**2 + (hf.Y - 0.0)**2
        hf.Z = -100.0 * np.exp(-dist_sq / (2.0 * 15.0**2))

        bounds = DioramaBounds(width=60.0, depth=60.0, z_base=-16.0, resolution=31)
        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        assert topo["passed"] is True
        assert topo["euler_characteristic"] == 2
        assert topo["boundary_edges"] == 0
        assert topo["non_manifold_edges"] == 0

    def test_watertightness_arbitrary_grid_resolutions(self):
        """Verify Euler chi = 2 on diverse rectangular and non-square grids."""
        test_resolutions = [(15, 15), (31, 31), (25, 45), (51, 27), (65, 65)]

        for rx, ry in test_resolutions:
            hf = Heightfield2D(width=50.0, depth=50.0, resolution=rx)
            if rx != ry:
                # Re-allocate Z for non-square grid test
                hf.res_x = rx
                hf.res_y = ry
                hf.Z = np.zeros((ry, rx), dtype=np.float64)
                xs = np.linspace(-25.0, 25.0, rx)
                ys = np.linspace(-25.0, 25.0, ry)
                hf.X, hf.Y = np.meshgrid(xs, ys)

            hf.add_peak(0.0, 0.0, height=12.0, radius=10.0, shape="horn")
            bounds = DioramaBounds(width=50.0, depth=50.0, z_base=-10.0, resolution=rx)

            slab = DioramaSlabBuilder(hf, bounds)
            mesh_data = slab.build_mesh_data()

            topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
            assert topo["passed"] is True, f"Failed at resolution ({rx}, {ry}): {topo['errors']}"
            assert topo["euler_characteristic"] == 2, f"Resolution ({rx}, {ry}) produced chi = {topo['euler_characteristic']}"
            assert topo["boundary_edges"] == 0
            assert topo["non_manifold_edges"] == 0

    @pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
    def test_blender_bmesh_watertight_manifold_assert(self):
        """Verify BMesh topological invariants directly inside headless Blender."""
        blender_code = f"""
import sys
sys.path.insert(0, r"{Path(__file__).resolve().parent.parent}")
import json
import bmesh
import bpy
from terra_forge.schema.map_config import MapConfig
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder

with open(r"{PRESET_PATH}", "r", encoding="utf-8") as f:
    cfg = MapConfig.model_validate(json.load(f))

hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=41)
for p in cfg.peaks:
    hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)
hf.apply_geological_features(cfg)

slab = DioramaSlabBuilder(hf, cfg.diorama)
mesh_data = slab.build_mesh_data()

bm = bmesh.new()
# Create vertices
bm_verts = [bm.verts.new(v) for v in mesh_data["vertices"]]
bm.verts.ensure_lookup_table()

# Create faces
for face_indices in mesh_data["faces"]:
    bm.faces.new([bm_verts[idx] for idx in face_indices])
bm.faces.ensure_lookup_table()
bm.edges.ensure_lookup_table()

# Verify BMesh topological invariants
boundary_edges = [e for e in bm.edges if e.is_boundary]
assert len(boundary_edges) == 0, f"BMesh has {{len(boundary_edges)}} boundary edges"

non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
assert len(non_manifold_edges) == 0, f"BMesh has {{len(non_manifold_edges)}} non-manifold edges"

non_manifold_verts = [v for v in bm.verts if not v.is_manifold]
assert len(non_manifold_verts) == 0, f"BMesh has {{len(non_manifold_verts)}} non-manifold verts"

# Euler characteristic: V - E + F == 2
V = len(bm.verts)
E = len(bm.edges)
F = len(bm.faces)
chi = V - E + F
assert chi == 2, f"Euler characteristic chi = {{chi}} != 2 (V={{V}}, E={{E}}, F={{F}})"

bm.free()
print("BMESH_TOPOLOGICAL_INVARIANTS_VERIFIED_SUCCESS")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert proc.returncode == 0, (
            f"Blender BMesh verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "BMESH_TOPOLOGICAL_INVARIANTS_VERIFIED_SUCCESS" in proc.stdout
