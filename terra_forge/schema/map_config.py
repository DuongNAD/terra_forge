"""
Declarative schema for TerraForge terrain and world generation profiles.
Built on Python dataclasses with zero external dependencies,
ensuring native compatibility in both standard Python and Blender's bundled Python.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Any, Dict


@dataclass
class DioramaBounds:
    width: float = 160.0
    depth: float = 160.0
    z_base: float = -16.0
    resolution: int = 256


@dataclass
class PeakConfig:
    name: str = "massif_peak"
    position: Tuple[float, float] = (0.0, 0.0)
    height: float = 28.0
    radius: float = 20.0
    shape: str = "horn"
    ridge_angle_deg: float = 0.0


@dataclass
class StrataFoldConfig:
    enabled: bool = False
    amplitude: float = 6.0
    wavelength: float = 45.0
    strike_angle_deg: float = 35.0
    plunge_wavelength: float = 120.0
    harmonic_weight: float = 0.25
    domain_warp_amp: float = 8.0
    domain_warp_freq: float = 0.04
    # Canyon step terracing options
    terrace_enabled: bool = False
    terrace_height: float = 6.0
    terrace_sharpness: float = 6.0
    terrace_blend: float = 0.85
    noise_modulation: float = 0.0


@dataclass
class TectonicScarpConfig:
    enabled: bool = False
    fault_angle_deg: float = -25.0
    displacement_throw: float = 14.0
    scarp_sharpness: float = 10.0
    fault_offset: float = 12.0
    fault_length: float = 0.0


@dataclass
class CoastalProfileConfig:
    enabled: bool = False
    sea_level: float = 0.0
    wave_notch_depth: float = 1.8
    wave_notch_envelope: float = 1.2
    beach_slope_coefficient: float = 0.12
    berm_elevation: float = 1.2
    beach_width: float = 15.0
    shelf_transition_dist: float = 35.0
    deep_water_depth: float = 12.0
    coastline_angle_deg: Optional[float] = None
    coastline_offset: float = 0.0


@dataclass
class ErosionConfig:
    enabled: bool = True
    droplet_count: int = 25000
    droplet_inertia: float = 0.05
    sediment_capacity_factor: float = 4.0
    min_sediment_capacity: float = 0.01
    erode_speed: float = 0.35
    deposit_speed: float = 0.35
    evaporate_speed: float = 0.015
    gravity: float = 4.0
    max_droplet_steps: int = 64
    brush_radius: float = 3.0
    alluvial_fan_spread: float = 3.5
    rill_incision_factor: float = 0.5
    thermal_iterations: int = 8
    talus_angle_deg: float = 38.0
    scree_talus_deposition: float = 0.4


@dataclass
class HydrologyConfig:
    lake_center: Tuple[float, float] = (-14.0, -10.0)
    lake_radius: float = 24.0
    lake_level: float = 4.5
    lake_depth: float = 12.0
    sea_level: float = 0.0
    river_enabled: bool = True
    river_stages: List[dict] = field(default_factory=lambda: [
        {"name": "Mountain Cascades", "t_end": 0.32, "start": (-8.0, 48.0, 21.5, 3.2), "end": (0.0, 26.0, 8.0, 4.5)},
        {"name": "Valley Meander", "t_end": 0.58, "start": (0.0, 26.0, 8.0, 4.5), "end": (-18.0, 6.0, 4.5, 7.5)},
        {"name": "Central Lake Transit", "t_end": 0.74, "start": (-18.0, 6.0, 4.5, 7.5), "end": (-10.0, -18.0, 4.5, 8.0)},
        {"name": "Outlet Gorge & Bay", "t_end": 1.00, "start": (-10.0, -18.0, 4.5, 8.0), "end": (26.0, -36.0, 0.0, 10.0)},
    ])
    river_carve_depth: float = 2.0
    river_meander_amplitude: float = 1.0
    bay_enabled: bool = True


@dataclass
class CavernConfig:
    enabled: bool = True
    center: Tuple[float, float, float] = (14.0, 16.0, -4.5)
    radius_x: float = 18.0
    radius_y: float = 14.0
    height: float = 7.0
    entrance_pos: Tuple[float, float, float] = (0.0, 12.0, 1.5)
    entrance_radius: float = 4.5
    min_roof_clearance: float = 5.0
    stalactite_count: int = 16
    stalagmite_count: int = 14
    column_count: int = 4
    subterranean_pool: bool = True
    pool_level: float = -7.2
    bioluminescent: bool = True
    glow_color: Tuple[float, float, float] = (0.1, 0.9, 0.4)


@dataclass
class BiomeScatterConfig:
    name: str = "Scatter_Biome"
    prototype: str = "pine_tall"
    min_altitude: float = -10.0
    max_altitude: float = 40.0
    min_slope_deg: float = 0.0
    max_slope_deg: float = 45.0
    min_water_dist: float = 0.0
    max_water_dist: float = 999.0
    density: float = 0.05
    poisson_dist_min: float = 2.5
    scale_min: float = 0.8
    scale_max: float = 1.3
    seed: int = 42


@dataclass
class ShaderConfig:
    triplanar: bool = True
    rock_texture: str = "cliff_rock"
    soil_texture: str = "forest_ground"
    sand_texture: str = "wet_sand"
    snow_altitude: float = 22.0
    water_absorption_color: Tuple[float, float, float] = (0.05, 0.45, 0.65)
    water_roughness: float = 0.04
    water_contact_foam: bool = True


@dataclass
class AtmosphereConfig:
    enabled: bool = True
    sun_time_of_day: float = 10.5        # Solar hour (0.0 - 24.0)
    max_elevation_deg: float = 68.0      # Maximum solar elevation at noon
    sun_intensity_max: float = 5.5       # Extraterrestrial baseline I0
    optical_depth: float = 0.10          # Rayleigh/aerosol optical depth tau_R
    mist_density: float = 0.0015         # Atmospheric volume scatter density (1/m)
    mist_anisotropy: float = 0.70        # Henyey-Greenstein forward scattering g
    mist_absorption_color: Tuple[float, float, float] = (0.75, 0.85, 0.95)
    use_volumetrics: bool = True         # Build World VolumeScatter/Absorption
    use_mist_pass: bool = True           # Enable Scene ViewLayer mist pass


@dataclass
class ExportConfig:
    export_glb: bool = True
    export_blend: bool = True
    export_heightmap_json: bool = True
    glb_path: str = "models/genesis_diorama.glb"
    blend_path: str = "models/genesis_diorama_master.blend"
    heightmap_path: str = "models/terrain_heightmap.json"


@dataclass
class MapConfig:
    name: str = "Genesis_Diorama"
    theme: str = "alpine_valley"
    seed: int = 42
    diorama: DioramaBounds = field(default_factory=DioramaBounds)
    peaks: List[PeakConfig] = field(default_factory=list)
    strata: StrataFoldConfig = field(default_factory=StrataFoldConfig)
    tectonic: TectonicScarpConfig = field(default_factory=TectonicScarpConfig)
    coastal: CoastalProfileConfig = field(default_factory=CoastalProfileConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    cavern: CavernConfig = field(default_factory=CavernConfig)
    biomes: List[BiomeScatterConfig] = field(default_factory=list)
    shaders: ShaderConfig = field(default_factory=ShaderConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> MapConfig:
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MapConfig:
        diorama = (
            data["diorama"] if isinstance(data.get("diorama"), DioramaBounds)
            else DioramaBounds(**data["diorama"]) if "diorama" in data
            else DioramaBounds()
        )
        peaks = [
            p if isinstance(p, PeakConfig) else PeakConfig(**p)
            for p in data.get("peaks", [])
        ]
        strata = (
            data["strata"] if isinstance(data.get("strata"), StrataFoldConfig)
            else StrataFoldConfig(**data["strata"]) if "strata" in data
            else StrataFoldConfig()
        )
        tectonic = (
            data["tectonic"] if isinstance(data.get("tectonic"), TectonicScarpConfig)
            else TectonicScarpConfig(**data["tectonic"]) if "tectonic" in data
            else TectonicScarpConfig()
        )
        coastal = (
            data["coastal"] if isinstance(data.get("coastal"), CoastalProfileConfig)
            else CoastalProfileConfig(**data["coastal"]) if "coastal" in data
            else CoastalProfileConfig()
        )
        erosion = (
            data["erosion"] if isinstance(data.get("erosion"), ErosionConfig)
            else ErosionConfig(**data["erosion"]) if "erosion" in data
            else ErosionConfig()
        )
        hydrology = (
            data["hydrology"] if isinstance(data.get("hydrology"), HydrologyConfig)
            else HydrologyConfig(**data["hydrology"]) if "hydrology" in data
            else HydrologyConfig()
        )
        cavern = (
            data["cavern"] if isinstance(data.get("cavern"), CavernConfig)
            else CavernConfig(**data["cavern"]) if "cavern" in data
            else CavernConfig()
        )
        biomes = [
            b if isinstance(b, BiomeScatterConfig) else BiomeScatterConfig(**b)
            for b in data.get("biomes", [])
        ]
        shaders = (
            data["shaders"] if isinstance(data.get("shaders"), ShaderConfig)
            else ShaderConfig(**data["shaders"]) if "shaders" in data
            else ShaderConfig()
        )
        atmosphere = (
            data["atmosphere"] if isinstance(data.get("atmosphere"), AtmosphereConfig)
            else AtmosphereConfig(**data["atmosphere"]) if "atmosphere" in data
            else AtmosphereConfig()
        )
        export = (
            data["export"] if isinstance(data.get("export"), ExportConfig)
            else ExportConfig(**data["export"]) if "export" in data
            else ExportConfig()
        )

        return cls(
            name=data.get("name", "Genesis_Diorama"),
            theme=data.get("theme", "alpine_valley"),
            seed=data.get("seed", 42),
            diorama=diorama,
            peaks=peaks,
            strata=strata,
            tectonic=tectonic,
            coastal=coastal,
            erosion=erosion,
            hydrology=hydrology,
            cavern=cavern,
            biomes=biomes,
            shaders=shaders,
            atmosphere=atmosphere,
            export=export,
        )


# ============================================================================
# Authoritative Built-in Natural Presets (Milestone 4)
# ============================================================================

def build_alpine_peak_glacial_lake_config() -> MapConfig:
    """Preset 1: Alpine Peak & Glacial Lake."""
    return MapConfig(
        name="Alpine_Peak_Glacial_Lake",
        theme="alpine_glacial",
        seed=101,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-18.0, resolution=256),
        peaks=[
            PeakConfig(name="Grand_Horn_North", position=(-22.0, 42.0), height=40.0, radius=26.0, shape="horn"),
            PeakConfig(name="Arete_Ridge_East", position=(28.0, 32.0), height=32.0, radius=22.0, shape="ridge", ridge_angle_deg=60.0),
            PeakConfig(name="Glacial_Cirque_West", position=(-38.0, 12.0), height=24.0, radius=18.0, shape="plateau"),
        ],
        strata=StrataFoldConfig(enabled=True, amplitude=4.5, wavelength=40.0, strike_angle_deg=35.0),
        erosion=ErosionConfig(enabled=True, droplet_count=35000, thermal_iterations=12, talus_angle_deg=35.0, scree_talus_deposition=0.45),
        hydrology=HydrologyConfig(
            lake_center=(-14.0, -8.0),
            lake_radius=24.0,
            lake_level=4.8,
            lake_depth=16.0,
            sea_level=0.0,
            river_enabled=True,
            river_carve_depth=2.2,
            river_meander_amplitude=1.2,
        ),
        cavern=CavernConfig(enabled=False),
        biomes=[
            BiomeScatterConfig(name="Alpine_Conifers", prototype="pine_tall", min_altitude=10.0, max_altitude=28.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.08, seed=101),
            BiomeScatterConfig(name="Krummholz_Shrubs", prototype="broadleaf", min_altitude=6.0, max_altitude=15.0, min_slope_deg=0.0, max_slope_deg=30.0, density=0.06, seed=102),
            BiomeScatterConfig(name="Alpine_Meadow_Moss", prototype="fern", min_altitude=4.5, max_altitude=12.0, min_slope_deg=0.0, max_slope_deg=25.0, density=0.12, seed=103),
            BiomeScatterConfig(name="Moraine_Talus_Boulders", prototype="rock_boulder", min_altitude=5.0, max_altitude=35.0, min_slope_deg=20.0, max_slope_deg=60.0, density=0.06, seed=104),
        ],
        shaders=ShaderConfig(snow_altitude=23.0, water_absorption_color=(0.02, 0.46, 0.65)),
        atmosphere=AtmosphereConfig(sun_time_of_day=8.5, mist_density=0.0018),
        export=ExportConfig(
            glb_path="models/alpine_peak_glacial_lake.glb",
            blend_path="models/alpine_peak_glacial_lake_master.blend",
            heightmap_path="models/alpine_peak_glacial_lake_heightmap.json",
        ),
    )


def build_temperate_forest_river_basin_config() -> MapConfig:
    """Preset 2: Temperate Forest River Basin."""
    return MapConfig(
        name="Temperate_Forest_River_Basin",
        theme="temperate_forest",
        seed=202,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=256),
        peaks=[
            PeakConfig(name="Forested_Ridge_North", position=(-10.0, 48.0), height=26.0, radius=30.0, shape="dome"),
            PeakConfig(name="Valley_Shoulder_East", position=(38.0, 15.0), height=20.0, radius=25.0, shape="dome"),
            PeakConfig(name="Low_Mound_South", position=(-32.0, -28.0), height=16.0, radius=22.0, shape="dome"),
        ],
        strata=StrataFoldConfig(enabled=True, amplitude=3.0, wavelength=55.0, strike_angle_deg=45.0),
        erosion=ErosionConfig(enabled=True, droplet_count=26000, thermal_iterations=8, talus_angle_deg=32.0, scree_talus_deposition=0.35),
        hydrology=HydrologyConfig(
            lake_center=(-14.0, -10.0),
            lake_radius=20.0,
            lake_level=3.5,
            lake_depth=10.0,
            sea_level=0.0,
            river_enabled=True,
            river_meander_amplitude=1.6,
            river_carve_depth=2.4,
        ),
        cavern=CavernConfig(enabled=False),
        biomes=[
            BiomeScatterConfig(name="Canopy_Deciduous_Oaks", prototype="broadleaf", min_altitude=3.0, max_altitude=22.0, min_slope_deg=0.0, max_slope_deg=28.0, density=0.09, seed=201),
            BiomeScatterConfig(name="Understory_Bush_Layer", prototype="fern", min_altitude=3.0, max_altitude=18.0, min_slope_deg=0.0, max_slope_deg=32.0, density=0.14, seed=202),
            BiomeScatterConfig(name="Riparian_River_Reeds", prototype="reeds", min_altitude=3.2, max_altitude=4.5, min_slope_deg=0.0, max_slope_deg=12.0, min_water_dist=0.1, max_water_dist=3.5, density=0.22, seed=203),
            BiomeScatterConfig(name="Forest_Floor_Fallen_Logs", prototype="rock_boulder", min_altitude=3.5, max_altitude=16.0, min_slope_deg=0.0, max_slope_deg=20.0, density=0.04, seed=204),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.08, 0.35, 0.28)),
        atmosphere=AtmosphereConfig(sun_time_of_day=16.5, mist_density=0.0015),
        export=ExportConfig(
            glb_path="models/temperate_forest_river_basin.glb",
            blend_path="models/temperate_forest_river_basin_master.blend",
            heightmap_path="models/temperate_forest_river_basin_heightmap.json",
        ),
    )


def build_tropical_island_coastal_karst_config() -> MapConfig:
    """Preset 3: Tropical Island & Coastal Karst Cliffs."""
    return MapConfig(
        name="Tropical_Island_Coastal_Karst",
        theme="tropical_coastal",
        seed=303,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=256),
        peaks=[
            PeakConfig(name="Karst_Monolith_Alpha", position=(-20.0, 25.0), height=38.0, radius=15.0, shape="horn"),
            PeakConfig(name="Karst_Pinnacle_Beta", position=(22.0, 32.0), height=34.0, radius=14.0, shape="horn"),
            PeakConfig(name="Lagoon_Islet_South", position=(15.0, -25.0), height=18.0, radius=12.0, shape="horn"),
        ],
        tectonic=TectonicScarpConfig(enabled=True, fault_angle_deg=-25.0, displacement_throw=14.0, scarp_sharpness=10.0),
        coastal=CoastalProfileConfig(enabled=True, sea_level=0.0, wave_notch_depth=1.8, beach_slope_coefficient=0.12, beach_width=15.0),
        erosion=ErosionConfig(enabled=True, droplet_count=30000, thermal_iterations=10, talus_angle_deg=45.0, scree_talus_deposition=0.40),
        hydrology=HydrologyConfig(
            sea_level=0.0,
            lake_center=(10.0, -15.0),
            lake_radius=22.0,
            lake_level=0.0,
            lake_depth=8.0,
            river_enabled=False,
        ),
        cavern=CavernConfig(enabled=False),
        biomes=[
            BiomeScatterConfig(name="Tropical_Jungle_Canopy", prototype="broadleaf", min_altitude=1.0, max_altitude=18.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.10, seed=301),
            BiomeScatterConfig(name="Karst_Cliff_Ferns", prototype="fern", min_altitude=3.0, max_altitude=32.0, min_slope_deg=25.0, max_slope_deg=65.0, density=0.12, seed=302),
            BiomeScatterConfig(name="Shoreline_Mangrove_Reeds", prototype="reeds", min_altitude=0.1, max_altitude=1.5, min_slope_deg=0.0, max_slope_deg=15.0, min_water_dist=0.0, max_water_dist=4.0, density=0.18, seed=303),
            BiomeScatterConfig(name="Lagoon_Coral_Boulders", prototype="rock_boulder", min_altitude=-8.0, max_altitude=0.5, min_slope_deg=0.0, max_slope_deg=40.0, density=0.06, seed=304),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.01, 0.48, 0.72)),
        atmosphere=AtmosphereConfig(sun_time_of_day=12.0, mist_density=0.0008),
        export=ExportConfig(
            glb_path="models/tropical_island_coastal_karst.glb",
            blend_path="models/tropical_island_coastal_karst_master.blend",
            heightmap_path="models/tropical_island_coastal_karst_heightmap.json",
        ),
    )


def build_canyon_semiarid_plateau_config() -> MapConfig:
    """Preset 4: Canyon & Semi-arid Plateau."""
    return MapConfig(
        name="Canyon_Semi-arid_Plateau",
        theme="arid_canyon",
        seed=404,
        diorama=DioramaBounds(width=160.0, depth=160.0, z_base=-16.0, resolution=256),
        peaks=[
            PeakConfig(name="Great_Mesa_North", position=(-15.0, 35.0), height=32.0, radius=28.0, shape="plateau"),
            PeakConfig(name="Isolated_Butte_East", position=(32.0, 10.0), height=26.0, radius=14.0, shape="plateau"),
            PeakConfig(name="Canyon_Rim_South", position=(-25.0, -32.0), height=22.0, radius=20.0, shape="plateau"),
        ],
        strata=StrataFoldConfig(enabled=True, terrace_enabled=True, terrace_height=6.0, terrace_sharpness=6.0, terrace_blend=0.85),
        erosion=ErosionConfig(enabled=True, droplet_count=25000, thermal_iterations=12, talus_angle_deg=40.0, scree_talus_deposition=0.55),
        hydrology=HydrologyConfig(
            lake_center=(999.0, 999.0),
            lake_radius=0.0,
            sea_level=0.0,
            river_enabled=True,
            river_carve_depth=3.2,
            river_meander_amplitude=1.4,
        ),
        cavern=CavernConfig(enabled=False),
        biomes=[
            BiomeScatterConfig(name="Arid_Plateau_Juniper", prototype="pine_medium", min_altitude=12.0, max_altitude=32.0, min_slope_deg=0.0, max_slope_deg=25.0, density=0.04, seed=401),
            BiomeScatterConfig(name="Desert_Scrub_Bushes", prototype="fern", min_altitude=2.0, max_altitude=24.0, min_slope_deg=0.0, max_slope_deg=35.0, density=0.07, seed=402),
            BiomeScatterConfig(name="Gully_Wash_Reeds", prototype="reeds", min_altitude=1.0, max_altitude=3.5, min_slope_deg=0.0, max_slope_deg=15.0, min_water_dist=0.1, max_water_dist=3.0, density=0.10, seed=403),
            BiomeScatterConfig(name="Sandstone_Scree_Boulders", prototype="rock_boulder", min_altitude=2.0, max_altitude=30.0, min_slope_deg=25.0, max_slope_deg=65.0, density=0.08, seed=404),
        ],
        shaders=ShaderConfig(snow_altitude=99.0, water_absorption_color=(0.15, 0.30, 0.25)),
        atmosphere=AtmosphereConfig(sun_time_of_day=17.5, mist_density=0.0022, mist_absorption_color=(0.85, 0.70, 0.55)),
        export=ExportConfig(
            glb_path="models/canyon_semiarid_plateau.glb",
            blend_path="models/canyon_semiarid_plateau_master.blend",
            heightmap_path="models/canyon_semiarid_plateau_heightmap.json",
        ),
    )


PRESET_CONFIGS: Dict[str, MapConfig] = {
    "alpine_peak_glacial_lake": build_alpine_peak_glacial_lake_config(),
    "temperate_forest_river_basin": build_temperate_forest_river_basin_config(),
    "tropical_island_coastal_karst": build_tropical_island_coastal_karst_config(),
    "canyon_semiarid_plateau": build_canyon_semiarid_plateau_config(),
}
