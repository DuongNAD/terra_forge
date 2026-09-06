"""
Unit test suite for Milestone 4 Presets Quality Assurance.
Verifies:
1. All 4 natural preset JSON files exist on disk in terra_forge/schema/presets/.
2. MapConfig.model_validate and from_dict succeed on all preset files.
3. PRESET_CONFIGS dictionary in map_config.py contains valid MapConfig instances.
4. CLI preset resolution via load_preset_or_file.
5. End-to-end topological 2-manifold validation on diorama slabs synthesized for each preset.
"""

import json
import pytest

from terra_forge.schema.map_config import (
    MapConfig,
    PRESET_CONFIGS,
    AtmosphereConfig,
)
from terra_forge.cli import PRESETS_DIR, load_preset_or_file
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.inspector.validator import QualityValidator

M4_PRESET_NAMES = [
    "alpine_peak_glacial_lake",
    "temperate_forest_river_basin",
    "tropical_island_coastal_karst",
    "canyon_semiarid_plateau",
]


class TestPresetFilesOnDisk:
    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_file_exists_and_validates(self, preset_stem):
        """Verify each preset file exists and parses cleanly into MapConfig."""
        path = PRESETS_DIR / f"{preset_stem}.json"
        assert path.exists(), f"Missing preset file: {path}"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = MapConfig.model_validate(data)
        assert cfg.diorama.width == 160.0
        assert cfg.diorama.depth == 160.0
        assert cfg.diorama.resolution == 256
        assert len(cfg.peaks) >= 3
        assert len(cfg.biomes) == 4
        assert cfg.shaders.triplanar is True
        assert cfg.export.glb_path is not None
        assert isinstance(cfg.atmosphere, AtmosphereConfig)
        assert cfg.atmosphere.enabled is True


class TestPresetConfigsRegistry:
    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_configs_registry(self, preset_stem):
        """Verify PRESET_CONFIGS dictionary exports each preset with consistent data."""
        assert preset_stem in PRESET_CONFIGS
        cfg = PRESET_CONFIGS[preset_stem]
        assert isinstance(cfg, MapConfig)
        assert cfg.diorama.width > 0
        assert len(cfg.peaks) >= 3
        assert len(cfg.biomes) >= 3


class TestCliPresetLoading:
    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_load_preset_or_file(self, preset_stem):
        """Verify load_preset_or_file resolves both by stem name and direct file path."""
        data_by_name = load_preset_or_file(preset_stem)
        assert isinstance(data_by_name, dict)
        assert data_by_name["name"] is not None

        file_path = str(PRESETS_DIR / f"{preset_stem}.json")
        data_by_path = load_preset_or_file(file_path)
        assert data_by_path["name"] == data_by_name["name"]


class TestPresetTopologySynthesis:
    @pytest.mark.parametrize("preset_stem", M4_PRESET_NAMES)
    def test_preset_slab_topology_manifold(self, preset_stem):
        """Verify diorama slab produced from preset configuration satisfies 2-manifold invariants."""
        cfg = PRESET_CONFIGS[preset_stem]
        res = 21
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=res)

        for p in cfg.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        if cfg.hydrology.lake_radius > 0 and abs(cfg.hydrology.lake_center[0]) < cfg.diorama.width:
            hf.carve_lake_basin(
                cfg.hydrology.lake_center[0],
                cfg.hydrology.lake_center[1],
                cfg.hydrology.lake_radius,
                cfg.hydrology.lake_level,
                cfg.hydrology.lake_depth,
            )

        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()

        res_topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        assert res_topo["passed"] is True
        assert res_topo["is_watertight"] is True
        assert res_topo["metrics"]["euler_characteristic"] == 2
        assert res_topo["metrics"]["boundary_edges"] == 0
        assert res_topo["metrics"]["non_manifold_edges"] == 0
        assert res_topo["metrics"]["non_manifold_vertices"] == 0
