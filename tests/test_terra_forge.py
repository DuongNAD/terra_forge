"""
Comprehensive test suite for TerraForge.
Covers schemas, core terrain math, erosion, hydrology, biomes, diorama builder, and CLI bridge.
"""

import json
import math
import os
import pytest
import numpy as np
from pathlib import Path

from terra_forge.schema.map_config import (
    MapConfig,
    DioramaBounds,
    PeakConfig,
    ErosionConfig,
    HydrologyConfig,
    CavernConfig,
    BiomeScatterConfig,
    ShaderConfig,
    ExportConfig,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.core.biomes import BiomeEngine
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.blender.bridge import BlenderBridge
from terra_forge.cli import PRESETS_DIR, list_presets, load_preset_or_file


class TestSchemas:
    def test_default_map_config(self):
        cfg = MapConfig()
        assert cfg.name == "Genesis_Diorama"
        assert cfg.diorama.resolution == 256
        assert isinstance(cfg.peaks, list)
        assert cfg.erosion.enabled is True
        assert cfg.hydrology.river_enabled is True
        assert isinstance(cfg.biomes, list)

    @pytest.mark.parametrize("preset_file", ["alpine_fjord.json", "karst_cavern.json", "volcanic_marsh.json"])
    def test_builtin_presets_valid(self, preset_file):
        path = PRESETS_DIR / preset_file
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MapConfig.model_validate(data)
        assert cfg.diorama.width > 0
        assert cfg.diorama.depth > 0
        assert len(cfg.peaks) > 0


class TestHeightfield:
    def test_grid_world_transforms(self):
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=51)
        gx, gy = hf.world_to_grid(0.0, 0.0)
        assert pytest.approx(gx, 0.01) == 25.0
        assert pytest.approx(gy, 0.01) == 25.0

        wx, wy = hf.grid_to_world(25.0, 25.0)
        assert pytest.approx(wx, 0.01) == 0.0
        assert pytest.approx(wy, 0.01) == 0.0

    def test_bilinear_interpolation(self):
        hf = Heightfield2D(width=10.0, depth=10.0, resolution=3)
        hf.Z[:, :] = 5.0
        val = hf.sample_bilinear(0.0, 0.0)
        assert pytest.approx(val, 0.001) == 5.0

    def test_peak_synthesis_all_shapes(self):
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        for shape in ["horn", "ridge", "plateau", "dome"]:
            hf.Z.fill(0.0)
            hf.add_peak(center_x=0.0, center_y=0.0, height=20.0, radius=15.0, shape=shape)
            assert hf.Z.max() > 0.0
            center_val = hf.sample_bilinear(0.0, 0.0)
            assert center_val > 10.0

    def test_normals_and_slopes(self):
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(0.0)
        normals = hf.compute_normals()
        assert np.allclose(normals[:, :, 2], 1.0)
        slopes = hf.compute_slope_degrees()
        assert np.allclose(slopes, 0.0)

    def test_lake_basin_carving(self):
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(10.0)
        hf.carve_lake_basin(center_x=0.0, center_y=0.0, radius=10.0, water_level=5.0, lake_depth=8.0)
        center_z = hf.sample_bilinear(0.0, 0.0)
        assert center_z < 0.0

    def test_lake_berm_containment(self):
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(0.0)
        hf.carve_lake_basin(center_x=0.0, center_y=0.0, radius=15.0, water_level=4.5, lake_depth=10.0, berm_height=1.2)
        # Check that outer rim forms a positive retaining berm holding the water
        rim_z = hf.sample_bilinear(15.0, 0.0)
        assert rim_z > 4.5
        berm_crest = hf.sample_bilinear(16.0, 0.0)
        assert berm_crest >= 4.5 + 0.35  # Satisfies validator containment requirement >= 0.35m



class TestErosion:
    def test_hydraulic_erosion_runs(self):
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 0.0, 15.0, 10.0, shape="horn")
        initial_sum = np.sum(hf.Z)

        cfg = ErosionConfig(enabled=True, droplet_count=200, max_droplet_steps=16)
        sim = ErosionSimulator(hf, cfg)
        sim.run_hydraulic_erosion(seed=42)

        assert np.any(hf.sediment != 0.0)

    def test_thermal_erosion_relaxes_steep_cliffs(self):
        hf = Heightfield2D(width=20.0, depth=20.0, resolution=11)
        hf.Z.fill(0.0)
        hf.Z[5, 5] = 50.0
        cfg = ErosionConfig(thermal_iterations=5, talus_angle_deg=30.0, scree_talus_deposition=0.5)
        sim = ErosionSimulator(hf, cfg)
        sim.run_thermal_erosion()
        assert hf.Z[5, 5] < 50.0


class TestHydrology:
    def test_river_spline_endpoints(self):
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=21)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        rx0, ry0, rz0, rw0 = he.evaluate_river_spline(0.0)
        assert pytest.approx(rx0[0], 0.1) == -8.0
        assert pytest.approx(ry0[0], 0.1) == 48.0
        assert pytest.approx(rz0[0], 0.1) == 21.5

        rx1, ry1, rz1, rw1 = he.evaluate_river_spline(1.0)
        assert pytest.approx(rx1[0], 0.1) == 26.0
        assert pytest.approx(ry1[0], 0.1) == -36.0
        assert pytest.approx(rz1[0], 0.1) == 0.0

    def test_water_distance_field(self):
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=5.0)
        he = HydrologyEngine(hf, cfg)
        dist = he.compute_water_distance_field()

        center_gx, center_gy = hf.world_to_grid(0.0, 0.0)
        assert dist[int(center_gy), int(center_gx)] == 0.0
        assert dist[0, 0] > 10.0

    def test_river_multi_stage_continuity(self):
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=21)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)
        t_vals = np.linspace(0.0, 1.0, 50)
        rx, ry, rz, rw = he.evaluate_river_spline(t_vals)
        # Verify continuous elevations declining from mountain peak to sea level
        assert rz[0] > 20.0
        assert rz[-1] == pytest.approx(0.0, 0.01)
        assert np.all(np.diff(rz) <= 0.01)  # Monotonically downhill



class TestBiomes:
    def test_biome_suitability_and_poisson_scatter(self):
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(10.0)
        dist_water = np.full_like(hf.Z, 5.0)

        bio_cfg = BiomeScatterConfig(
            name="Test_Forest",
            prototype="pine_tall",
            min_altitude=5.0,
            max_altitude=15.0,
            min_slope_deg=0.0,
            max_slope_deg=20.0,
            min_water_dist=1.0,
            max_water_dist=20.0,
            density=0.1,
            poisson_dist_min=3.0,
            seed=42,
        )

        be = BiomeEngine(hf, dist_water)
        suitability = be.evaluate_suitability(bio_cfg)
        assert np.all(suitability == 1.0)

        points = be.generate_poisson_scatter(bio_cfg, max_samples=100)
        assert len(points) > 0

        r_min = bio_cfg.poisson_dist_min
        coords = np.array([(p[0], p[1]) for p in points])
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[j])
                assert d >= r_min * 0.95


class TestDioramaSlab:
    def test_watertight_slab_topology(self):
        res = 16
        bounds = DioramaBounds(width=40.0, depth=40.0, z_base=-10.0, resolution=res)
        hf = Heightfield2D(width=bounds.width, depth=bounds.depth, resolution=res)
        hf.Z.fill(2.0)

        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()

        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]
        colors = mesh_data["vertex_colors"]

        assert len(vertices) == 2 * res * res
        assert len(colors) == len(vertices)

        expected_faces = 2 * (res - 1) * (res - 1) + 4 * (res - 1)
        assert len(faces) == expected_faces

        max_idx = len(vertices)
        for face in faces:
            assert len(face) == 4
            for v_idx in face:
                assert 0 <= v_idx < max_idx


class TestCLI:
    def test_load_presets(self):
        data = load_preset_or_file("alpine_fjord")
        assert data["name"] == "Alpine_Fjord_Diorama"

    def test_blender_bridge_code_generation(self):
        bridge = BlenderBridge()
        cfg_dict = {"name": "Test", "theme": "alpine_fjord"}
        code = bridge.generate_live_blender_code(cfg_dict)
        assert "run_pipeline" in code
        assert "VIEW_3D" in code

        tweak_code = bridge.generate_modifier_tweak_code("Scatter_Alpine", "Density", 0.15)
        assert "Density" in tweak_code
        assert "0.15" in tweak_code


class TestExporter:
    def test_simulation_passability_spatial_accuracy(self, tmp_path):
        from terra_forge.blender.exporter import export_simulation_heightmap_json

        cfg = MapConfig(name="Passability_Test")
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.add_peak(0.0, 20.0, 25.0, 15.0, shape="horn")
        hf.carve_lake_basin(-10.0, -10.0, 12.0, 4.0, 8.0)

        out_json = str(tmp_path / "test_sim.json")
        res_path = export_simulation_heightmap_json(out_json, hf, cfg)
        assert Path(res_path).exists()

        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        p_mat = np.array(data["passability_matrix"])
        walkable_ratio = float(np.sum(p_mat == 1) / p_mat.size)
        water_ratio = float(np.sum(p_mat == 0) / p_mat.size)

        # Walkable terrain must be majority (> 50%) and not over-classified as deep water
        assert walkable_ratio > 0.50
        assert water_ratio < 0.45

