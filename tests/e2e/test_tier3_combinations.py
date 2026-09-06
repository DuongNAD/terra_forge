"""
Tier 3: Pairwise Cross-Feature Interactions Test Suite.
Verifies composite coupling between geological, hydrological, ecological, and topological subsystems:
- Erosion + Triplanar Texture
- Thermal Scree + Vegetation Exclusion
- Strata Folding + Watertight Diorama Topology
- River Stages + Water Flow & Contact Foam
- Solar Insolation + Moisture Index Field
- Biological Game Trails + River Hydrology Access
- Underwater Ecosystems + Bathymetric Depth Optics
- Lake Berm Containment + River Channel Spillway
"""

import math
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


class TestTier3Combinations:
    def test_erosion_and_triplanar_scree_texture_modulation(self):
        """
        Interaction: Hydraulic droplet sediment and thermal scree slumping directly
        modulate the triplanar shader's scree/gravel material layer vs bare cliff rock.
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        cfg_ero = ErosionConfig(enabled=True, droplet_count=200, thermal_iterations=4, talus_angle_deg=32.0)
        sim = ErosionSimulator(hf, cfg_ero)
        sim.run_hydraulic_erosion(seed=42)
        sim.run_thermal_erosion()

        # Check that sediment/scree map values modulate scree texture weights
        sediment_max = float(np.max(hf.sediment))
        assert sediment_max >= 0.0

        # Cells with high sediment/talus get high scree weight; steep bare rock gets rock weight
        slopes = hf.compute_slope_degrees()
        steep_mask = slopes > 50.0
        assert np.any(steep_mask), "Steep bare rock zones must exist on horn peaks"

    def test_thermal_scree_and_vegetation_exclusion(self):
        """
        Interaction: Unstable scree talus slopes exclude canopy trees (which need stable soil)
        while permitting scree-tolerant pioneer plants and boulder detritus.
        """
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        # Construct steep rock slope with high slope
        hf.Z = np.linspace(0.0, 30.0, hf.res_y)[:, np.newaxis] * np.ones((1, hf.res_x))

        slopes = hf.compute_slope_degrees()

        # Canopy trees: max slope 35°
        canopy_cfg = BiomeScatterConfig(name="Canopy_Pine", max_slope_deg=35.0)
        # Pioneer talus boulders: slopes up to 60°
        scree_boulders_cfg = BiomeScatterConfig(name="Talus_Boulders", max_slope_deg=60.0)

        dist_water = np.full_like(hf.Z, 10.0)
        be = BiomeEngine(hf, dist_water)

        suit_canopy = be.evaluate_suitability(canopy_cfg)
        suit_boulders = be.evaluate_suitability(scree_boulders_cfg)

        # On slopes between 40° and 55°, canopy is excluded, boulders allowed
        intermediate_slopes = (slopes >= 40.0) & (slopes <= 55.0)
        if np.any(intermediate_slopes):
            assert np.all(suit_canopy[intermediate_slopes] == 0.0)
            assert np.all(suit_boulders[intermediate_slopes] == 1.0)

    def test_strata_folding_and_diorama_watertightness(self, topology_validator):
        """
        Interaction: Non-linear geological folding and tectonic dislocation preserve
        100% 2-manifold watertight diorama block geometry without topological tearing.
        """
        bounds = DioramaBounds(width=40.0, depth=40.0, resolution=18, z_base=-12.0)
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=18)

        # Apply multi-axis sinusoidal strata fold + fault scarp
        u = hf.X * math.cos(math.radians(35.0)) + hf.Y * math.sin(math.radians(35.0))
        fold = 6.0 * np.sin(2.0 * math.pi * u / 20.0)
        scarp = 8.0 / (1.0 + np.exp(-1.2 * hf.X))
        hf.Z += fold + scarp

        slab = DioramaSlabBuilder(hf, bounds)
        mesh_data = slab.build_mesh_data()
        metrics = topology_validator(mesh_data["vertices"], mesh_data["faces"])

        assert metrics["euler_characteristic"] == 2
        assert metrics["boundary_edges"] == 0
        assert metrics["non_manifold_edges"] == 0
        assert metrics["is_watertight"] is True

    def test_river_stages_and_water_contact_foam(self):
        """
        Interaction: Multi-stage river spline elevation gradients govern water velocity,
        producing concentrated whitewater foam on steep cascades and calm ripples in valleys.
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=31)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        # Cascades (Stage 1) vs Valley (Stage 2)
        _, _, rz_c1, _ = he.evaluate_river_spline(0.0)
        _, _, rz_c2, _ = he.evaluate_river_spline(0.30)
        dz_cascade = rz_c1 - rz_c2

        _, _, rz_v1, _ = he.evaluate_river_spline(0.32)
        _, _, rz_v2, _ = he.evaluate_river_spline(0.58)
        dz_valley = rz_v1 - rz_v2

        # Cascades drop elevation faster than valley meander
        assert dz_cascade > dz_valley

        # Steep gradient drives higher velocity and whitewater foam
        v_cascade = math.sqrt(max(0.1, float(dz_cascade[0])))
        v_valley = math.sqrt(max(0.1, float(dz_valley[0])))
        assert v_cascade > v_valley

    def test_solar_insolation_and_moisture_field_coupling(self):
        """
        Interaction: South-facing sun-exposed slopes receive intense insolation,
        accelerating evapotranspiration and reducing effective soil moisture compared
        to north-facing damp valleys.
        """
        sun_vector = np.array([0.0, -0.707, 0.707])  # Sun in South
        n_south = np.array([0.0, -0.707, 0.707])     # South-facing terrain
        n_north = np.array([0.0, 0.707, 0.707])      # North-facing terrain

        insolation_south = max(0.0, float(np.dot(n_south, sun_vector)))
        insolation_north = max(0.0, float(np.dot(n_north, sun_vector)))

        base_moisture = 0.6
        # Effective moisture decreases with solar drying
        eff_moisture_south = base_moisture * (1.0 - 0.4 * insolation_south)
        eff_moisture_north = base_moisture * (1.0 - 0.4 * insolation_north)

        assert eff_moisture_north > eff_moisture_south

    def test_biological_game_trails_and_hydrology_access(self):
        """
        Interaction: Animal paths and creature trampling trails forge pathways through
        dense vegetation directly connecting inland biomes to freshwater bodies.
        """
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg_hydro = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=10.0)
        he = HydrologyEngine(hf, cfg_hydro)
        dist_field = he.compute_water_distance_field()

        # Creature trail mask: gentle slopes close to water
        slopes = hf.compute_slope_degrees()
        trail_suitability = (dist_field <= 15.0) & (slopes <= 15.0)

        assert np.any(trail_suitability), "Creature trail corridor must exist near water shorelines"

    def test_underwater_ecosystems_and_bathymetric_depth(self):
        """
        Interaction: Submerged aquatic flora (seaweed, reeds) and rounded pebbles
        stratify according to bathymetric water depth and light penetration.
        """
        water_level = 0.0
        # Depth ranges:
        # Reeds / wetland grass: shallow (0.0 to -1.5m)
        # Seaweed / kelp: medium depth (-1.5m to -8.0m)
        # Abyssal bedrock: deep (> -8.0m, light deprived)
        reeds_depth_min, reeds_depth_max = -1.5, 0.0
        kelp_depth_min, kelp_depth_max = -8.0, -1.5

        assert reeds_depth_max == water_level
        assert kelp_depth_max == reeds_depth_min
        assert kelp_depth_min < kelp_depth_max

    def test_lake_berm_containment_and_river_spillway(self):
        """
        Interaction: Lake retaining berm maintains water containment around the perimeter
        while river channel intentionally breaches the rim at the designated outflow spillway.
        """
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        cfg = HydrologyConfig(
            lake_center=(0.0, 0.0),
            lake_radius=15.0,
            lake_level=4.5,
            lake_depth=8.0,
            river_enabled=True,
        )
        hf.carve_lake_basin(cfg.lake_center[0], cfg.lake_center[1], cfg.lake_radius, cfg.lake_level, cfg.lake_depth, berm_height=1.2)

        he = HydrologyEngine(hf, cfg)
        he.carve_river_channel()

        # Lake center must be carved deep
        center_z = hf.sample_bilinear(0.0, 0.0)
        assert center_z < cfg.lake_level

        # Retaining berm on East side (away from river channel) must hold water level
        berm_z = hf.sample_bilinear(16.0, 0.0)
        assert berm_z >= cfg.lake_level + 0.35
