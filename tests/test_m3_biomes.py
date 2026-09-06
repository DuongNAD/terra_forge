"""
Comprehensive Test Suite for Milestone 3:
Multi-Tier Biomes, Natural Ecosystems, Microclimates & Procedural Vegetation.

Covers:
1. Microclimate Physics (Lapse Rate, Soil Moisture Index, Solar Insolation Aspect).
2. Multi-Tier Ecological Stratification (Canopy, Understory, Ground Flora, Detritus Decay, Creature Trails, Underwater).
3. Fast O(N) Blue-Noise Poisson-Disk Clustering with Biological Density Maps & Strict Boundary Clamping.
4. 3D Procedural Prototype Generation & Taxonomy (asset_fetcher.py).
5. Geometry Nodes Scattering Modifiers & Exposed Sockets (geonodes.py).
6. Edge Cases & Boundary Conditions.
"""

from __future__ import annotations
import math
import subprocess
import shutil
import numpy as np
import pytest

from terra_forge.schema.map_config import (
    BiomeScatterConfig,
    DioramaBounds,
)
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.biomes import (
    BiomeEngine,
    EcologicalTier,
    PROTOTYPE_TAXONOMY,
    bridson_poisson_disk_sampling,
)
from terra_forge.blender import asset_fetcher, geonodes
from terra_forge.blender.asset_fetcher import PROTOTYPE_FACTORIES

BLENDER_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"
HAS_BLENDER = shutil.which(BLENDER_BIN) is not None or shutil.which("blender") is not None


# =============================================================================
# 1. Microclimate Simulation Tests
# =============================================================================

class TestMicroclimateSimulation:
    def test_elevation_temperature_lapse_rate(self):
        """Verify temperature decreases linearly with elevation: T(z) = T0 - Gamma * z."""
        elevations = np.array([0.0, 100.0, 500.0, 1000.0])
        sea_level = 0.0
        lapse_rate = 0.0065  # 0.65 C per 100m
        base_temp = 20.0

        temps = BiomeEngine.compute_temperature(elevations, sea_level, lapse_rate, base_temp)
        assert temps[0] == pytest.approx(20.0, abs=1e-3)
        assert temps[1] == pytest.approx(20.0 - 0.65, abs=1e-3)
        assert temps[2] == pytest.approx(20.0 - 3.25, abs=1e-3)
        assert temps[3] == pytest.approx(20.0 - 6.50, abs=1e-3)
        # Monotonically declining with altitude
        assert np.all(np.diff(temps) < 0.0)

    def test_solar_insolation_aspect_normals(self):
        """Verify solar radiation insolation aspect calculates correct dot product from surface normals."""
        # Flat surface normal (0, 0, 1) with sun directly overhead (0, 0, 1)
        normals = np.zeros((5, 5, 3), dtype=np.float64)
        normals[:, :, 2] = 1.0
        sun_overhead = np.array([0.0, 0.0, 1.0])
        insolation_zenith = BiomeEngine.compute_insolation(normals, sun_overhead)
        assert np.allclose(insolation_zenith, 1.0)

        # South-facing slope (normal pointing +Y) with sun at +Y (SE)
        normals_south = np.zeros((5, 5, 3), dtype=np.float64)
        normals_south[:, :, 1] = 0.7071
        normals_south[:, :, 2] = 0.7071

        sun_south = np.array([0.0, 0.7071, 0.7071])
        insolation_south = BiomeEngine.compute_insolation(normals_south, sun_south)
        assert np.allclose(insolation_south, 1.0, atol=0.01)

        # North-facing slope (normal pointing -Y) facing away from south sun
        normals_north = np.zeros((5, 5, 3), dtype=np.float64)
        normals_north[:, :, 1] = -0.7071
        normals_north[:, :, 2] = 0.7071
        insolation_north = BiomeEngine.compute_insolation(normals_north, sun_south)
        # Should be significantly lower than south-facing slope
        assert np.all(insolation_north < insolation_south)

    def test_continuous_soil_moisture_field(self):
        """Verify continuous soil moisture index field M(x, y) synthesizes water proximity and slopes."""
        water_dist = np.array([[0.0, 5.0], [15.0, 50.0]], dtype=np.float64)
        slopes = np.array([[0.0, 10.0], [25.0, 60.0]], dtype=np.float64)

        moisture = BiomeEngine.compute_moisture(water_dist, slopes=slopes)
        assert moisture.shape == (2, 2)
        assert np.all(moisture >= 0.0)
        assert np.all(moisture <= 1.0)

        # Point at shoreline with flat slope has highest moisture
        assert moisture[0, 0] > moisture[1, 1]
        assert moisture[0, 0] > 0.5
        assert moisture[1, 1] < 0.4

    def test_microclimate_synthesis_dictionary(self):
        """Verify compute_microclimate returns all expected fields with valid physical bounds."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")
        dist_water = np.full_like(hf.Z, 12.0)

        be = BiomeEngine(hf, dist_water)
        climate = be.compute_microclimate()

        assert "temperature" in climate
        assert "insolation" in climate
        assert "moisture" in climate
        assert "slopes" in climate

        temp = climate["temperature"]
        assert temp.shape == hf.Z.shape
        # Mountain peak should be colder than base
        assert temp.min() < temp.max()

        insolation = climate["insolation"]
        assert np.all((insolation >= 0.0) & (insolation <= 1.0))

        moisture = climate["moisture"]
        assert np.all((moisture >= 0.0) & (moisture <= 1.0))


# =============================================================================
# 2. Multi-Tier Ecological Stratification Tests
# =============================================================================

class TestMultiTierStratification:
    def test_ecological_tier_enumeration_and_taxonomy(self):
        """Verify all 6 required ecological tiers are defined in PROTOTYPE_TAXONOMY."""
        tiers_found = set()
        for proto, props in PROTOTYPE_TAXONOMY.items():
            assert "tier" in props
            assert "min_alt" in props
            assert "max_alt" in props
            assert "max_slope" in props
            assert "density" in props
            assert "poisson_dist" in props
            tiers_found.add(props["tier"])

        assert EcologicalTier.CANOPY in tiers_found
        assert EcologicalTier.UNDERSTORY in tiers_found
        assert EcologicalTier.GROUND_FLORA in tiers_found
        assert EcologicalTier.DECAY_WINDTHROW in tiers_found
        assert EcologicalTier.BIOLOGICAL_IMPACT in tiers_found
        assert EcologicalTier.UNDERWATER in tiers_found

    def test_canopy_trees_stratification(self):
        """Verify canopy conifers, broadleafs, and tropical palms have appropriate altitudinal niches."""
        pines = PROTOTYPE_TAXONOMY["pine_tall"]
        broadleaf = PROTOTYPE_TAXONOMY["broadleaf"]
        palm = PROTOTYPE_TAXONOMY["palm_tropical"]

        assert pines["max_alt"] >= 30.0, "Alpine pines must reach subalpine heights"
        assert broadleaf["max_slope"] <= 35.0, "Deciduous canopy avoids extreme cliffs"
        assert palm["min_alt"] >= 0.0, "Palms grow in coastal and lowland zones"

    def test_understory_shrubs_and_bushes(self):
        """Verify understory shrubs occupy intermediate height and slope brackets."""
        shrub = PROTOTYPE_TAXONOMY["shrub"]
        bush = PROTOTYPE_TAXONOMY["bush"]
        krummholz = PROTOTYPE_TAXONOMY["krummholz"]

        assert shrub["poisson_dist"] < PROTOTYPE_TAXONOMY["broadleaf"]["poisson_dist"]
        assert krummholz["min_alt"] > bush["min_alt"]

    def test_ground_flora_wildflowers_ferns_moss(self):
        """Verify ground flora (wildflowers, ferns, mushrooms, rock moss) occupy low biomass niches."""
        moss = PROTOTYPE_TAXONOMY["rock_moss"]
        fern = PROTOTYPE_TAXONOMY["fern"]
        flower = PROTOTYPE_TAXONOMY["wildflower"]

        assert moss["max_slope"] >= 60.0, "Rock moss must cling to steep cliff faces"
        assert flower["density"] >= 0.15, "Wildflowers form dense carpet patches"
        assert fern["min_alt"] >= 0.0

    def test_natural_tree_decay_and_windthrow(self):
        """Verify fallen decaying logs, snapped trunks, and weathered stumps have realistic detritus densities."""
        log = PROTOTYPE_TAXONOMY["fallen_log"]
        stump = PROTOTYPE_TAXONOMY["weathered_stump"]

        assert log["density"] <= 0.06, "Fallen log detritus must be naturally sparse"
        assert stump["density"] <= 0.05, "Weathered stumps must have low density"
        assert log["poisson_dist"] >= 4.0, "Logs need large spacing to avoid clumping"

    def test_biological_game_trails_and_disturbed_soil(self):
        """Verify animal trails and naturally churned soil patches generate valid continuous masks."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(6.0)
        dist_water = np.full_like(hf.Z, 5.0)
        be = BiomeEngine(hf, dist_water)

        trails = be.generate_game_trails(count=3, seed=42)
        assert len(trails) > 0
        for trail in trails:
            assert len(trail) >= 4
            # Trail points within bounds
            assert np.all(trail[:, 0] >= -25.0) and np.all(trail[:, 0] <= 25.0)
            assert np.all(trail[:, 1] >= -25.0) and np.all(trail[:, 1] <= 25.0)

        trail_mask = be.compute_game_trail_mask(trails, trail_width=2.5)
        assert trail_mask.shape == hf.Z.shape
        assert np.any(trail_mask > 0.0)
        assert np.all((trail_mask >= 0.0) & (trail_mask <= 1.0))

        soil_mask = be.compute_disturbed_soil_mask(trail_mask, churn_patches=4, seed=42)
        assert soil_mask.shape == hf.Z.shape
        assert np.any(soil_mask > 0.0)

    def test_underwater_ecosystems_mask(self):
        """Verify submerged aquatic flora (seaweed, reeds, pebbles, reefs) inhabit areas below water level."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(5.0)
        hf.Z[15:, :] = -5.0  # Submerged marine trench

        be = BiomeEngine(hf, dist_to_water=np.zeros_like(hf.Z), sea_level=0.0)
        aqua_mask = be.compute_underwater_ecosystem_mask(max_depth=15.0)

        # Above sea level (5.0m) must be 0
        assert np.all(aqua_mask[:14, :] == 0.0)
        # Flat submerged basin floor (-5.0m) must be 1.0
        assert np.all(aqua_mask[16:, :] == 1.0)

    def test_stratified_biome_configs_factory(self):
        """Verify create_stratified_biome_configs generates configs covering all requested biomes."""
        configs = BiomeEngine.create_stratified_biome_configs()
        assert len(configs) >= 15

        prototypes = [c.prototype for c in configs]
        assert "pine_tall" in prototypes
        assert "broadleaf" in prototypes
        assert "palm_tropical" in prototypes
        assert "shrub" in prototypes
        assert "wildflower" in prototypes
        assert "fern" in prototypes
        assert "fallen_log" in prototypes
        assert "weathered_stump" in prototypes
        assert "seaweed" in prototypes
        assert "riverbed_pebble" in prototypes


# =============================================================================
# 3. Fast O(N) Blue-Noise Poisson-Disk Clustering & Clamping Tests
# =============================================================================

class TestPoissonClusteringAndClamping:
    def test_poisson_minimum_distance_guarantee(self):
        """Verify all scattered points strictly respect the minimum radius r_min."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        hf.Z.fill(10.0)
        dist_water = np.full_like(hf.Z, 5.0)

        r_min = 3.5
        cfg = BiomeScatterConfig(
            min_altitude=0.0,
            max_altitude=25.0,
            density=0.2,
            poisson_dist_min=r_min,
            seed=42,
        )
        be = BiomeEngine(hf, dist_water)
        points = be.generate_poisson_scatter(cfg, max_samples=100)

        assert len(points) > 10
        coords = np.array([(p[0], p[1]) for p in points])
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[j])
                assert d >= r_min * 0.95, f"Pairwise distance {d:.3f} violated r_min {r_min}"

    def test_strict_coordinate_clamping_no_perimeter_overshoot(self):
        """Verify all coordinates are clamped strictly within [-W/2, W/2] and [-D/2, D/2]."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(5.0)
        dist_water = np.full_like(hf.Z, 5.0)

        be = BiomeEngine(hf, dist_water)
        cfg = BiomeScatterConfig(
            min_altitude=0.0,
            max_altitude=20.0,
            density=0.25,
            poisson_dist_min=1.8,
            seed=999,
        )
        points = be.generate_poisson_scatter(cfg, max_samples=250)

        half_w = hf.width / 2.0
        half_d = hf.depth / 2.0
        for px, py, pz, scale, rot_z in points:
            assert -half_w <= px <= half_w, f"Perimeter overshoot on X: {px} not in [{-half_w}, {half_w}]"
            assert -half_d <= py <= half_d, f"Perimeter overshoot on Y: {py} not in [{-half_d}, {half_d}]"
            assert scale > 0.0
            assert 0.0 <= rot_z <= 2.0 * math.pi + 1e-4

    def test_poisson_density_modulation(self):
        """Verify higher density generates more candidate instances."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(6.0)
        dist_water = np.full_like(hf.Z, 5.0)
        be = BiomeEngine(hf, dist_water)

        cfg_low = BiomeScatterConfig(density=0.02, poisson_dist_min=2.0, seed=12)
        cfg_high = BiomeScatterConfig(density=0.15, poisson_dist_min=2.0, seed=12)

        pts_low = be.generate_poisson_scatter(cfg_low, max_samples=150)
        pts_high = be.generate_poisson_scatter(cfg_high, max_samples=150)

        assert len(pts_high) > len(pts_low)

    def test_poisson_seed_reproducibility(self):
        """Verify deterministic seed produces identical coordinates."""
        hf = Heightfield2D(width=40.0, depth=40.0, resolution=21)
        hf.Z.fill(8.0)
        dist_water = np.full_like(hf.Z, 5.0)
        be = BiomeEngine(hf, dist_water)

        cfg1 = BiomeScatterConfig(seed=4321)
        cfg2 = BiomeScatterConfig(seed=4321)

        pts1 = be.generate_poisson_scatter(cfg1, max_samples=60)
        pts2 = be.generate_poisson_scatter(cfg2, max_samples=60)

        assert len(pts1) == len(pts2)
        assert pts1 == pts2

    def test_poisson_scatter_interface_contract(self):
        """Verify M1/M2/M4 interface contract: poisson_scatter returns (x, y, z) 3-tuples."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=25)
        hf.Z.fill(12.0)
        suitability = np.ones_like(hf.Z)

        be = BiomeEngine(hf, dist_to_water=np.zeros_like(hf.Z))
        pts = be.poisson_scatter(suitability, min_radius=3.0, max_count=50)

        assert len(pts) > 0
        for pt in pts:
            assert len(pt) == 3
            assert -25.0 <= pt[0] <= 25.0
            assert -25.0 <= pt[1] <= 25.0
            assert pt[2] == pytest.approx(12.0, abs=1e-3)

    def test_standalone_bridson_sampling(self):
        """Verify bridson_poisson_disk_sampling function adheres to r_min and boundaries."""
        pts = bridson_poisson_disk_sampling(width=40.0, depth=40.0, r_min=4.0, seed=123)
        assert len(pts) > 5
        coords = np.array(pts)
        assert np.all(coords[:, 0] >= -20.0) and np.all(coords[:, 0] <= 20.0)
        assert np.all(coords[:, 1] >= -20.0) and np.all(coords[:, 1] <= 20.0)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[j])
                assert d >= 4.0 * 0.95


# =============================================================================
# 4. Blender 3D Procedural Prototypes & Geometry Nodes Tests
# =============================================================================

@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestBlenderPrototypesAndGeoNodes:
    def test_all_17_prototypes_generation_headless_blender(self):
        """Verify all 17 natural prototypes generate valid 3D meshes in Blender."""
        blender_code = """
import sys
sys.path.insert(0, '.')
from terra_forge.blender import asset_fetcher

prototypes = [
    'pine_tall', 'broadleaf', 'palm_tropical', 'shrub',
    'wildflower', 'grass', 'fern', 'mushroom', 'fungus',
    'rock_moss', 'fallen_log', 'snapped_stump', 'seaweed',
    'reeds', 'riverbed_pebble', 'coral_reef', 'rock_boulder'
]
for p in prototypes:
    obj = asset_fetcher.resolve_prototype_object(p)
    assert obj is not None, f"Failed to generate {p}"
    assert len(obj.data.vertices) > 0, f"No vertices for {p}"
    assert len(obj.data.polygons) > 0, f"No polygons for {p}"
    assert len(obj.data.materials) > 0, f"No material for {p}"
print("ALL_PROTOTYPES_OK")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        assert res.returncode == 0
        assert "ALL_PROTOTYPES_OK" in res.stdout

    def test_geonodes_tree_and_modifier_headless_blender(self):
        """Verify Geometry Nodes scatter nodegroups, exposed sockets, and modifier application in Blender."""
        blender_code = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import BiomeScatterConfig
from terra_forge.blender import asset_fetcher, geonodes

bpy.ops.mesh.primitive_plane_add(size=40.0)
terrain = bpy.context.active_object

pine = asset_fetcher.create_pine_tree()
cfg = BiomeScatterConfig(name="Alpine_Pine", prototype="pine_tall", max_slope_deg=40.0)
mod = geonodes.apply_biome_scatter_modifier(terrain, pine, cfg)
assert mod is not None

ng = mod.node_group
socket_names = [s.name for s in ng.interface.items_tree if s.in_out == 'INPUT']
required_sockets = ['Density', 'Distance_Min', 'Min_Altitude', 'Max_Altitude', 'Max_Slope_Deg', 'Water_Level', 'Water_Clearance', 'Is_Aquatic', 'Seed']
for req in required_sockets:
    assert req in socket_names, f"Missing socket {req}"

print("GEONODES_OK")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        assert res.returncode == 0
        assert "GEONODES_OK" in res.stdout
