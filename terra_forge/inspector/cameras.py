"""
Multi-Angle Inspection Camera Rig & Atmospheric Lighting for TerraForge.
Implements:
1. Solar ephemeris, diurnal tracking & directional vector calculation.
2. Blackbody Correlated Color Temperature (CCT) mapping & Tanner-Helland conversion.
3. Atmospheric air mass extinction (Kasten-Young & Bouguer-Lambert-Beer).
4. World volumetric scattering (ShaderNodeVolumeScatter & ShaderNodeVolumeAbsorption).
5. Standardized multi-angle vision cameras with automated framing math:
   d = max(width, depth) / (sqrt(2) * sin(FOV/2)).
6. Decoupled pure-Python execution via HAS_BPY guard.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional

try:
    import bpy
    from mathutils import Euler, Vector
    HAS_BPY = True
except ImportError:
    bpy = None
    Euler = None
    Vector = None
    HAS_BPY = False


# =============================================================================
# Pure-Python Solar & Atmospheric Mathematical Engines
# =============================================================================

def calculate_solar_position(
    sun_time_of_day: float = 12.0,
    max_elevation_deg: float = 68.0,
) -> Tuple[float, float, Tuple[float, float, float]]:
    """Calculates solar elevation, azimuth, and unit direction vector from time of day.

    Args:
        sun_time_of_day: Solar hour in [0.0, 24.0]. 6.0=dawn, 12.0=noon, 18.0=dusk.
        max_elevation_deg: Maximum elevation at solar noon (default 68.0°).

    Returns:
        (elevation_deg, azimuth_deg, (lx, ly, lz))
    """
    t = float(sun_time_of_day) % 24.0

    # Elevation: sinusoidal diurnal curve
    elev_deg = max_elevation_deg * math.sin((t - 6.0) / 12.0 * math.pi)
    elev_deg = max(-90.0, min(90.0, elev_deg))

    # Azimuth: sweeps 360 degrees clockwise from North (15 deg/hr)
    azim_deg = (t / 24.0 * 360.0) % 360.0

    elev_rad = math.radians(elev_deg)
    azim_rad = math.radians(azim_deg)

    # Unit vector pointing towards the sun (+X East, +Y North, +Z Up)
    lx = math.cos(elev_rad) * math.sin(azim_rad)
    ly = math.cos(elev_rad) * math.cos(azim_rad)
    lz = math.sin(elev_rad)

    return elev_deg, azim_deg, (lx, ly, lz)


def calculate_cct_from_elevation(elevation_deg: float) -> float:
    """Calculates physical Correlated Color Temperature (Kelvin) from solar elevation.

    - Solar elevation < 0° (twilight/night): 7000K - 10000K cool ambient.
    - Solar elevation < 10° (golden hour / dawn / dusk): 2000K - 3200K warm amber/gold.
    - Solar elevation > 45° (midday): 5500K - 6500K crisp white natural daylight.
    """
    elev = float(elevation_deg)

    if elev <= -6.0:
        return 9000.0
    elif elev < 0.0:
        # Civil twilight / blue hour transition
        u = (elev + 6.0) / 6.0
        return 9000.0 - u * (9000.0 - 2200.0)
    elif elev <= 10.0:
        # Golden hour (warm amber/gold: 2200K - 3200K)
        return 2200.0 + (elev / 10.0) * (3200.0 - 2200.0)
    elif elev <= 45.0:
        # Morning/afternoon progression (Hermite smoothstep)
        u = (elev - 10.0) / 35.0
        s = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        return 3200.0 + s * (5800.0 - 3200.0)
    else:
        # High midday (crisp white daylight: 5800K - 6500K)
        u = min(1.0, (elev - 45.0) / 45.0)
        return 5800.0 + u * (6500.0 - 5800.0)


def cct_to_linear_rgb(temp_k: float) -> Tuple[float, float, float]:
    """Converts Correlated Color Temperature (1000K - 40000K) to linear RGB [0.0, 1.0]
    using the Tanner-Helland Planckian approximation with gamma correction."""
    t = max(1000.0, min(40000.0, float(temp_k))) / 100.0

    # Red
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    # Green
    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    g = max(0.0, min(255.0, g))

    # Blue
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    b = max(0.0, min(255.0, b))

    # Convert sRGB to linear photometric RGB
    def to_linear(val: float) -> float:
        c = max(0.0, min(1.0, val / 255.0))
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (to_linear(r), to_linear(g), to_linear(b))


def calculate_optical_air_mass(elevation_deg: float) -> float:
    """Calculates optical air mass m using the Kasten-Young (1989) formula."""
    elev = float(elevation_deg)
    if elev <= 0.0:
        return 38.0
    elev_rad = math.radians(elev)
    denom = math.sin(elev_rad) + 0.50572 * ((elev + 6.07995) ** -1.6364)
    return 1.0 / max(0.02, denom)


def calculate_sun_intensity(
    elevation_deg: float,
    i0: float = 5.5,
    tau_r: float = 0.10,
) -> float:
    """Calculates direct solar intensity using atmospheric extinction:
    I = I0 * exp(-m * tau_R)."""
    elev = float(elevation_deg)
    if elev <= 0.0:
        return 0.0
    m = calculate_optical_air_mass(elev)
    return float(i0 * math.exp(-m * tau_r))


def calculate_fov_from_lens(lens_mm: float, sensor_width_mm: float = 36.0) -> float:
    """Calculates horizontal Field of View (in degrees) from focal length."""
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * lens_mm)))


def calculate_camera_framing_distance(
    width: float,
    depth: float,
    fov_deg: float,
) -> float:
    """Calculates camera distance to guarantee full model framing without clipping:
    d = max(width, depth) / (sqrt(2) * sin(FOV/2))."""
    span = max(float(width), float(depth))
    half_fov_rad = math.radians(fov_deg * 0.5)
    sin_half = math.sin(half_fov_rad)
    return span / (math.sqrt(2.0) * max(0.01, sin_half))


# =============================================================================
# Camera Viewpoint Definitions
# =============================================================================

def build_inspection_camera_configs(
    bounds: Optional[Any] = None,
    hf: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Builds camera configuration dictionaries for standardized viewpoints."""
    width = getattr(bounds, "width", 160.0)
    depth = getattr(bounds, "depth", 160.0)

    # 1. Isometric Diorama: 45° azimuth, 35.264° elevation
    iso_lens = 45.0
    iso_fov = calculate_fov_from_lens(iso_lens)
    iso_dist = calculate_camera_framing_distance(width, depth, iso_fov)

    # Offset along (-1/sqrt(3), -1/sqrt(3), 1/sqrt(3))
    inv_s3 = 1.0 / math.sqrt(3.0)
    iso_x = -iso_dist * inv_s3
    iso_y = -iso_dist * inv_s3
    iso_z = 5.0 + iso_dist * inv_s3

    # 2. Eye-Level First-Person: terrain surface + 2.0m human eye height
    ped_x = -0.18 * width
    ped_y = -0.25 * depth
    if hf is not None and hasattr(hf, "sample_bilinear"):
        ground_z = float(hf.sample_bilinear(ped_x, ped_y))
    else:
        ground_z = 4.5
    cam_eye_z = ground_z + 1.7

    # 3. Closeup Macro: low angle close to water/rock looking across the lake
    macro_x = -2.0
    macro_y = -13.0
    if hf is not None and hasattr(hf, "sample_bilinear"):
        ground_macro = float(hf.sample_bilinear(macro_x, macro_y))
        macro_z = max(ground_macro + 1.8, 6.2)
    else:
        macro_z = 6.5

    configs = {
        "isometric_diorama": {
            "label": "Isometric Diorama (Full Framing)",
            "location": (iso_x, iso_y, iso_z),
            "rotation_euler": (math.radians(60.0), 0.0, math.radians(-45.0)),
            "lens": iso_lens,
            "clip_end": max(500.0, iso_dist * 2.5),
            "dof_enabled": False,
        },
        "eye_level_first_person": {
            "label": "Eye-Level First Person (+1.7m)",
            "location": (ped_x, ped_y, cam_eye_z),
            "rotation_euler": (math.radians(75.0), 0.0, math.radians(-35.0)),
            "target": (0.0, 0.0, 8.0),
            "lens": 24.0,
            "clip_end": 500.0,
            "dof_enabled": False,
        },
        "closeup_macro": {
            "label": "Geological & Shoreline Macro (DoF)",
            "location": (macro_x, macro_y, macro_z),
            "rotation_euler": (math.radians(72.0), 0.0, math.radians(-30.0)),
            "target": (0.0, 6.0, 5.5),
            "lens": 85.0,
            "clip_end": 250.0,
            "dof_enabled": True,
            "focus_distance": 15.0,
            "fstop": 2.8,
        },
        "topdown": {
            "label": "Top-Down (Bird's Eye Overview)",
            "location": (0.0, 0.0, max(width, depth) * 1.35),
            "rotation_euler": (0.0, 0.0, 0.0),
            "lens": 50.0,
            "clip_end": 1000.0,
            "dof_enabled": False,
        },
    }

    # Backward compatibility aliases
    configs["iso_sw"] = configs["isometric_diorama"]
    configs["closeup_water"] = configs["closeup_macro"]
    configs["iso_ne"] = {
        "label": "Northeast Isometric",
        "location": (-iso_x, -iso_y, iso_z),
        "rotation_euler": (math.radians(60.0), 0.0, math.radians(135.0)),
        "lens": iso_lens,
        "clip_end": configs["isometric_diorama"]["clip_end"],
        "dof_enabled": False,
    }

    return configs


# Static default preset table for legacy callers
INSPECTION_PRESETS: Dict[str, Dict[str, Any]] = build_inspection_camera_configs()


# =============================================================================
# Blender Lighting & Atmospheric Setup
# =============================================================================

def setup_sun_lighting(
    config: Optional[Any] = None,
    sun_time_of_day: Optional[float] = None,
    mist_density: float = 0.0015,
    mist_anisotropy: float = 0.70,
) -> Dict[str, Any]:
    """Sets up physical solar vector lighting, blackbody CCT, and volumetric haze."""
    # Resolve parameters from config or defaults
    atmo_cfg = getattr(config, "atmosphere", None)
    if sun_time_of_day is None:
        if atmo_cfg is not None and hasattr(atmo_cfg, "sun_time_of_day"):
            time_hour = atmo_cfg.sun_time_of_day
        else:
            time_hour = 10.5
    else:
        time_hour = sun_time_of_day

    max_elev = getattr(atmo_cfg, "max_elevation_deg", 68.0) if atmo_cfg else 68.0
    i0 = getattr(atmo_cfg, "sun_intensity_max", 5.5) if atmo_cfg else 5.5
    tau_r = getattr(atmo_cfg, "optical_depth", 0.10) if atmo_cfg else 0.10
    m_density = getattr(atmo_cfg, "mist_density", mist_density) if atmo_cfg else mist_density
    m_anisotropy = getattr(atmo_cfg, "mist_anisotropy", mist_anisotropy) if atmo_cfg else mist_anisotropy

    elev_deg, azim_deg, sun_vec = calculate_solar_position(time_hour, max_elevation_deg=max_elev)
    cct_k = calculate_cct_from_elevation(elev_deg)
    sun_rgb = cct_to_linear_rgb(cct_k)
    sun_intensity = calculate_sun_intensity(elev_deg, i0=i0, tau_r=tau_r)

    # Ambient sky color determination
    if elev_deg <= 0.0:
        sky_rgb = (0.03, 0.05, 0.12, 1.0)
        sky_strength = 0.30
    elif elev_deg < 10.0:
        sky_rgb = (0.78, 0.60, 0.45, 1.0)
        sky_strength = 0.75
    else:
        sky_rgb = (0.55, 0.68, 0.85, 1.0)
        sky_strength = 0.95

    lighting_info = {
        "time_of_day": time_hour,
        "elevation_deg": round(elev_deg, 2),
        "azimuth_deg": round(azim_deg, 2),
        "sun_direction": tuple(round(v, 4) for v in sun_vec),
        "cct_kelvin": round(cct_k, 1),
        "sun_color_linear_rgb": tuple(round(c, 4) for c in sun_rgb),
        "sun_intensity": round(sun_intensity, 3),
        "sky_color": sky_rgb,
        "sky_strength": sky_strength,
        "mist_density": m_density,
        "mist_anisotropy": m_anisotropy,
    }

    if not HAS_BPY or bpy is None:
        return lighting_info

    # 1. Setup World Shader with Volumetric Scatter & Absorption
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("TerraForge_World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    out_node = tree.nodes.new("ShaderNodeOutputWorld")
    out_node.location = (400, 0)

    # Surface Background
    bg_node = tree.nodes.new("ShaderNodeBackground")
    bg_node.location = (100, 100)
    bg_node.inputs["Color"].default_value = sky_rgb
    bg_node.inputs["Strength"].default_value = sky_strength
    tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # Volumetric Atmosphere (Optional)
    use_vol = getattr(atmo_cfg, "use_volumetrics", False) if atmo_cfg else False
    if use_vol and m_density > 0:
        vol_scatter = tree.nodes.new("ShaderNodeVolumeScatter")
        vol_scatter.location = (-200, -100)
        vol_scatter.inputs["Color"].default_value = (0.75, 0.85, 0.95, 1.0)
        vol_scatter.inputs["Density"].default_value = m_density
        vol_scatter.inputs["Anisotropy"].default_value = m_anisotropy

        vol_absorb = tree.nodes.new("ShaderNodeVolumeAbsorption")
        vol_absorb.location = (-200, -250)
        vol_absorb.inputs["Color"].default_value = (0.80, 0.88, 0.96, 1.0)
        vol_absorb.inputs["Density"].default_value = m_density * 0.5

        add_shader = tree.nodes.new("ShaderNodeAddShader")
        add_shader.location = (100, -150)
        tree.links.new(vol_scatter.outputs["Volume"], add_shader.inputs[0])
        tree.links.new(vol_absorb.outputs["Volume"], add_shader.inputs[1])
        tree.links.new(add_shader.outputs["Shader"], out_node.inputs["Volume"])

    # Depth Cueing Mist Settings
    if hasattr(world, "mist_settings"):
        world.mist_settings.use_mist = True
        world.mist_settings.start = 8.0
        world.mist_settings.depth = 240.0
        world.mist_settings.falloff = 'QUADRATIC'

    # 2. Setup Sun Key Light
    sun_name = "Sun_Key_Light"
    sun_obj = bpy.data.objects.get(sun_name)
    if not sun_obj:
        sun_data = bpy.data.lights.new(name=sun_name + "_Data", type="SUN")
        sun_obj = bpy.data.objects.new(sun_name, sun_data)
        bpy.context.scene.collection.objects.link(sun_obj)

    # Calculate Euler angles for downward sunlight
    pitch = math.radians(max(0.0, 90.0 - elev_deg))
    yaw = math.radians(90.0 - azim_deg)
    sun_obj.rotation_euler = Euler((pitch, 0.0, yaw), "XYZ")
    sun_obj.data.energy = max(0.25, sun_intensity)
    sun_obj.data.color = sun_rgb

    return lighting_info


# =============================================================================
# Blender Camera Creation Rig
# =============================================================================

def setup_inspection_cameras(
    config: Optional[Any] = None,
    hf: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates or updates the standardized inspection cameras in Blender or returns configs in pure-Python."""
    bounds = getattr(config, "diorama", None)
    presets = build_inspection_camera_configs(bounds=bounds, hf=hf)

    if not HAS_BPY or bpy is None:
        setup_sun_lighting(config=config)
        return presets

    cam_coll_name = "TerraForge_Inspection_Cameras"
    cam_collection = bpy.data.collections.get(cam_coll_name)
    if not cam_collection:
        cam_collection = bpy.data.collections.new(cam_coll_name)
        bpy.context.scene.collection.children.link(cam_collection)

    cameras = {}
    for name, cfg in presets.items():
        cam_obj_name = f"CAM_Inspect_{name}"
        cam_obj = bpy.data.objects.get(cam_obj_name)
        if not cam_obj:
            cam_data = bpy.data.cameras.new(name=cam_obj_name + "_Data")
            cam_obj = bpy.data.objects.new(cam_obj_name, cam_data)
            cam_collection.objects.link(cam_obj)

        cam_obj.location = Vector(cfg["location"])
        if "target" in cfg and hasattr(Vector, "to_track_quat"):
            cam_obj.rotation_euler = (Vector(cfg["target"]) - cam_obj.location).to_track_quat('-Z', 'Y').to_euler()
        else:
            cam_obj.rotation_euler = Euler(cfg["rotation_euler"], "XYZ")
        cam_obj.data.lens = cfg["lens"]
        cam_obj.data.clip_end = cfg["clip_end"]

        # Depth of Field setup
        if cfg.get("dof_enabled", False):
            cam_obj.data.dof.use_dof = True
            cam_obj.data.dof.focus_distance = cfg.get("focus_distance", 10.0)
            cam_obj.data.dof.aperture_fstop = cfg.get("fstop", 2.8)
        else:
            cam_obj.data.dof.use_dof = False

        cameras[name] = cam_obj

    setup_sun_lighting(config=config)
    return cameras
