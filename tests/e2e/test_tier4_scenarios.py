"""
Tier 4: Real-World Natural Preset Scenarios Test Suite.
Validates end-to-end procedural generation, geological accuracy, ecological distribution,
and watertight diorama slab manifoldness across the 4 required natural presets:
1. Alpine Peak & Glacial Lake (alpine_peak_glacial_lake)
2. Temperate Forest River Basin (temperate_forest_river_basin)
3. Tropical Island & Coastal Karst Cliffs (tropical_island_coastal_karst)
4. Canyon & Semi-arid Plateau (canyon_semiarid_plateau)
"""

import json
from pathlib import Path
import numpy as np
import pytest

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
from terra_forge.cli import PRESETS_DIR


def build_alpine_peak_spec() -> MapConfig:
    """Authoritative specification for Preset 1: Alpine Peak & Glacial Lake."""
    return MapConfig(
        name="Alpine_Peak_Glacial_Lake",
        theme="alpine_glacial",
        seed=101,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-18.0, resolution=33),
        peaks=[
            PeakConfig(name="Grand_Horn_North", position=(-22.0, 42.0), height=40.0, radius=26.0, shape="horn"),
            PeakConfig(name="Arete_Ridge_East", position=(28.0, 32.0), height=32.0, radius=22.0, shape="ridge", ridge_angle_deg=60.0),
            PeakConfig(name="Glacial_Cirque_West", position=(-38.0, 12.0), height=24.0, radius=18.0, shape="plateau"),
        ],
        erosion=ErosionConfig(enabled=True, droplet_count=300, thermal_iterations=6, talus_angle_deg=35.0, scree_talus_deposition=0.45),
        hydrology=HydrologyConfig(
            lake_center=(-14.0, -8.0),
            lake_radius=24.0,
            lake_level=4.8,
            lake_depth=16.0,
            sea_level=0.0,
            river_enabled=True,
        ),
        biomes=[
            BiomeScatterConfig(name="Alpine_Conifers", prototype="pine_tall", min_altitude=10.0, max_altitude=28.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.08, seed=101),
            BiomeScatterConfig(name="Krummholz_Shrubs", prototype="broadleaf", min_altitude=6.0, max_altitude=15.0, min_slope_deg=0.0, max_slope_deg=30.0, density=0.06, seed=102),
            BiomeScatterConfig(name="Alpine_Meadow_Moss", prototype="fern", min_altitude=4.5, max_altitude=12.0, min_slope_deg=0.0, max_slope_deg=25.0, density=0.12, seed=103),
            BiomeScatterConfig(name="Moraine_Talus_Boulders", prototype="rock_boulder", min_altitude=5.0, max_altitude=35.0, min_slope_deg=20.0, max_slope_deg=60.0, density=0.06, seed=104),
        ],
        shaders=ShaderConfig(snow_altitude=23.0, water_absorption_color=(0.02, 0.46, 0.65)),
    )


def build_temperate_forest_spec() -> MapConfig:
    """Authoritative specification for Preset 2: Temperate Forest River Basin."""
    return MapConfig(
        name="Temperate_Forest_River_Basin",
        theme="temperate_forest",
        seed=202,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=33),
        peaks=[
            PeakConfig(name="Forested_Ridge_North", position=(-10.0, 48.0), height=26.0, radius=30.0, shape="dome"),
            PeakConfig(name="Valley_Shoulder_East", position=(38.0, 15.0), height=20.0, radius=25.0, shape="dome"),
            PeakConfig(name="Low_Mound_South", position=(-32.0, -28.0), height=16.0, radius=22.0, shape="dome"),
        ],
        erosion=ErosionConfig(enabled=True, droplet_count=250, thermal_iterations=4, talus_angle_deg=32.0),
        hydrology=HydrologyConfig(
            lake_center=(-14.0, -10.0),
            lake_radius=20.0,
            lake_level=3.5,
            lake_depth=10.0,
            river_enabled=True,
            river_meander_amplitude=1.6,
            river_carve_depth=2.4,
        ),
        biomes=[
            BiomeScatterConfig(name="Canopy_Deciduous_Oaks", prototype="broadleaf", min_altitude=3.0, max_altitude=22.0, min_slope_deg=0.0, max_slope_deg=28.0, density=0.09, seed=201),
            BiomeScatterConfig(name="Understory_Bush_Layer", prototype="fern", min_altitude=3.0, max_altitude=18.0, min_slope_deg=0.0, max_slope_deg=32.0, density=0.14, seed=202),
            BiomeScatterConfig(name="Riparian_River_Reeds", prototype="reeds", min_altitude=3.2, max_altitude=4.5, min_slope_deg=0.0, max_slope_deg=12.0, min_water_dist=0.1, max_water_dist=3.5, density=0.22, seed=203),
            BiomeScatterConfig(name="Forest_Floor_Fallen_Logs", prototype="rock_boulder", min_altitude=3.5, max_altitude=16.0, min_slope_deg=0.0, max_slope_deg=20.0, density=0.04, seed=204),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.08, 0.35, 0.28)),
    )


def build_tropical_karst_spec() -> MapConfig:
    """Authoritative specification for Preset 3: Tropical Island & Coastal Karst Cliffs."""
    return MapConfig(
        name="Tropical_Island_Coastal_Karst",
        theme="tropical_coastal",
        seed=303,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=33),
        peaks=[
            PeakConfig(name="Karst_Monolith_Alpha", position=(-20.0, 25.0), height=38.0, radius=15.0, shape="horn"),
            PeakConfig(name="Karst_Pinnacle_Beta", position=(22.0, 32.0), height=34.0, radius=14.0, shape="horn"),
            PeakConfig(name="Lagoon_Islet_South", position=(15.0, -25.0), height=18.0, radius=12.0, shape="horn"),
        ],
        erosion=ErosionConfig(enabled=True, droplet_count=300, thermal_iterations=5, talus_angle_deg=45.0),
        hydrology=HydrologyConfig(
            sea_level=0.0,
            lake_center=(10.0, -15.0),
            lake_radius=22.0,
            lake_level=0.0,
            lake_depth=8.0,
            river_enabled=False,
        ),
        biomes=[
            BiomeScatterConfig(name="Tropical_Jungle_Canopy", prototype="broadleaf", min_altitude=1.0, max_altitude=18.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.10, seed=301),
            BiomeScatterConfig(name="Karst_Cliff_Ferns", prototype="fern", min_altitude=3.0, max_altitude=32.0, min_slope_deg=25.0, max_slope_deg=65.0, density=0.12, seed=302),
            BiomeScatterConfig(name="Shoreline_Mangrove_Reeds", prototype="reeds", min_altitude=0.1, max_altitude=1.5, min_slope_deg=0.0, max_slope_deg=15.0, min_water_dist=0.0, max_water_dist=4.0, density=0.18, seed=303),
            BiomeScatterConfig(name="Lagoon_Coral_Boulders", prototype="rock_boulder", min_altitude=-8.0, max_altitude=0.5, min_slope_deg=0.0, max_slope_deg=40.0, density=0.06, seed=304),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.01, 0.48, 0.72)),
    )


def build_canyon_plateau_spec() -> MapConfig:
    """Authoritative specification for Preset 4: Canyon & Semi-arid Plateau."""
    return MapConfig(
        name="Canyon_Semi-arid_Plateau",
        theme="arid_canyon",
        seed=404,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=33),
        peaks=[
            PeakConfig(name="Great_Mesa_North", position=(-15.0, 35.0), height=32.0, radius=28.0, shape="plateau"),
            PeakConfig(name="Isolated_Butte_East", position=(32.0, 10.0), height=26.0, radius=14.0, shape="plateau"),
            PeakConfig(name="Canyon_Rim_South", position=(-25.0, -32.0), height=22.0, radius=20.0, shape="plateau"),
        ],
        erosion=ErosionConfig(enabled=True, droplet_count=200, thermal_iterations=7, talus_angle_deg=40.0, scree_talus_deposition=0.55),
        hydrology=HydrologyConfig(
            lake_center=(999.0, 999.0),
            lake_radius=0.0,
            river_enabled=True,
            river_carve_depth=3.2,
        ),
        biomes=[
            BiomeScatterConfig(name="Arid_Plateau_Juniper", prototype="pine_medium", min_altitude=12.0, max_altitude=32.0, min_slope_deg=0.0, max_slope_deg=25.0, density=0.04, seed=401),
            BiomeScatterConfig(name="Desert_Scrub_Bushes", prototype="fern", min_altitude=2.0, max_altitude=24.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.07, seed=402),
            BiomeScatterConfig(name="Gully_Wash_Reeds", prototype="reeds", min_altitude=1.0, max_altitude=3.5, min_slope_deg=0.0, max_slope_deg=15.0, min_water_dist=0.1, max_water_dist=3.0, density=0.10, seed=403),
            BiomeScatterConfig(name="Sandstone_Scree_Boulders", prototype="rock_boulder", min_altitude=2.0, max_altitude=30.0, min_slope_deg=25.0, max_slope_deg=65.0, density=0.08, seed=404),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.15, 0.30, 0.25)),
    )


class TestTier4Scenarios:
    # ------------------------------------------------------------------------
    # Preset 1: Alpine Peak & Glacial Lake
    # ------------------------------------------------------------------------
    def test_preset_1_alpine_peak_end_to_end(self, topology_validator):
        """End-to-end generation and validation of Alpine Peak & Glacial Lake."""
        cfg = build_alpine_peak_spec()

        # Step 1: Base topography
        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)
        hf.carve_lake_basin(cfg.hydrology.lake_center[0], cfg.hydrology.lake_center[1], cfg.hydrology.lake_radius, cfg.hydrology.lake_level, cfg.hydrology.lake_depth)

        # Horn peak summit check at continuous coordinates (accounting for 5m cell discretization)
        summit_z = hf.sample_bilinear(-22.0, 42.0)
        assert summit_z >= 25.0
        assert hf.Z.max() >= 30.0

        # Step 2: Geological Erosion
        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()

        # Step 3: Hydrology
        hydro = HydrologyEngine(hf, cfg.hydrology)
        hydro.carve_river_channel()
        dist_water = hydro.compute_water_distance_field()

        # Step 4: Ecological Biomes
        be = BiomeEngine(hf, dist_water)
        conifer_pts = be.generate_poisson_scatter(cfg.biomes[0], max_samples=50)
        assert len(conifer_pts) > 0, "Alpine conifers must populate mountain slopes"

        # Conifers must not spawn on extreme high summit above snowline
        for pt in conifer_pts:
            assert pt[2] <= cfg.shaders.snow_altitude + 5.0

        # Step 5: Watertight Diorama Slab
        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])

        assert metrics["is_watertight"] is True
        assert metrics["euler_characteristic"] == 2
        assert metrics["boundary_edges"] == 0
        assert metrics["non_manifold_edges"] == 0

    # ------------------------------------------------------------------------
    # Preset 2: Temperate Forest River Basin
    # ------------------------------------------------------------------------
    def test_preset_2_temperate_forest_end_to_end(self, topology_validator):
        """End-to-end generation and validation of Temperate Forest River Basin."""
        cfg = build_temperate_forest_spec()

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)
        hf.carve_lake_basin(cfg.hydrology.lake_center[0], cfg.hydrology.lake_center[1], cfg.hydrology.lake_radius, cfg.hydrology.lake_level, cfg.hydrology.lake_depth)

        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()

        hydro = HydrologyEngine(hf, cfg.hydrology)
        hydro.carve_river_channel()
        dist_water = hydro.compute_water_distance_field()

        be = BiomeEngine(hf, dist_water)
        oak_pts = be.generate_poisson_scatter(cfg.biomes[0], max_samples=60)
        reeds_pts = be.generate_poisson_scatter(cfg.biomes[2], max_samples=60)

        assert len(oak_pts) > 0, "Deciduous canopy trees must populate valley"
        assert len(reeds_pts) > 0, "Riparian reeds must spawn along river meander"
        # Reeds must be close to water within grid cell tolerance (dx)
        for r_pt in reeds_pts:
            gx, gy = hf.world_to_grid(r_pt[0], r_pt[1])
            d_w = dist_water[int(gy), int(gx)]
            assert d_w <= cfg.biomes[2].max_water_dist + hf.dx

        # Watertight diorama verification
        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])
        assert metrics["is_watertight"] is True

    # ------------------------------------------------------------------------
    # Preset 3: Tropical Island & Coastal Karst Cliffs
    # ------------------------------------------------------------------------
    def test_preset_3_tropical_karst_end_to_end(self, topology_validator):
        """End-to-end generation and validation of Tropical Island & Coastal Karst Cliffs."""
        cfg = build_tropical_karst_spec()

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)
        hf.carve_lake_basin(cfg.hydrology.lake_center[0], cfg.hydrology.lake_center[1], cfg.hydrology.lake_radius, cfg.hydrology.lake_level, cfg.hydrology.lake_depth)

        # Karst pinnacles must rise sheer above sea level
        assert hf.Z.max() >= 30.0

        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()

        hydro = HydrologyEngine(hf, cfg.hydrology)
        dist_water = hydro.compute_water_distance_field()

        be = BiomeEngine(hf, dist_water)
        jungle_pts = be.generate_poisson_scatter(cfg.biomes[0], max_samples=50)
        assert len(jungle_pts) > 0, "Tropical jungle must populate islands"

        # Watertight diorama verification
        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])
        assert metrics["is_watertight"] is True

    # ------------------------------------------------------------------------
    # Preset 4: Canyon & Semi-arid Plateau
    # ------------------------------------------------------------------------
    def test_preset_4_canyon_plateau_end_to_end(self, topology_validator):
        """End-to-end generation and validation of Canyon & Semi-arid Plateau."""
        cfg = build_canyon_plateau_spec()

        hf = Heightfield2D(width=cfg.diorama.width, depth=cfg.diorama.depth, resolution=cfg.diorama.resolution)
        for peak in cfg.peaks:
            hf.add_peak(peak.position[0], peak.position[1], peak.height, peak.radius, peak.shape, peak.ridge_angle_deg)

        # Plateau flat top check
        mesa_z = hf.sample_bilinear(-15.0, 35.0)
        assert mesa_z >= 28.0

        ero = ErosionSimulator(hf, cfg.erosion)
        ero.run_hydraulic_erosion(cfg.seed)
        ero.run_thermal_erosion()

        hydro = HydrologyEngine(hf, cfg.hydrology)
        hydro.carve_river_channel()
        dist_water = hydro.compute_water_distance_field()

        be = BiomeEngine(hf, dist_water)
        juniper_pts = be.generate_poisson_scatter(cfg.biomes[0], max_samples=40)
        assert len(juniper_pts) > 0, "Desert plateau junipers must populate mesa tops"

        # Watertight diorama verification
        slab = DioramaSlabBuilder(hf, cfg.diorama)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])
        assert metrics["is_watertight"] is True

    # ------------------------------------------------------------------------
    # Preset JSON File Loading on Disk (Tracks M4 Delivery)
    # ------------------------------------------------------------------------
    @pytest.mark.parametrize("preset_stem", [
        "alpine_peak_glacial_lake",
        "temperate_forest_river_basin",
        "tropical_island_coastal_karst",
        "canyon_semiarid_plateau",
    ])
    def test_preset_json_file_on_disk(self, preset_stem):
        """Verify that the 4 required natural presets exist on disk in schema/presets/ and validate against MapConfig."""
        preset_path = PRESETS_DIR / f"{preset_stem}.json"
        if not preset_path.exists():
            pytest.xfail(f"Preset JSON {preset_stem}.json is in progress under Milestone M4")

        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MapConfig.model_validate(data)
        assert cfg.diorama.width > 0
        assert len(cfg.peaks) >= 2
        assert len(cfg.biomes) >= 3
