"""
Milestone 2 Verification Test Suite: Photorealistic Shading & Hydrology.
Tests:
1. Dynamic River Spline Multi-Stage Evaluation (Arbitrary stages, control points, tangents, widths, depths).
2. Scalar Float Indexing & NumPy Deprecation Warning Prevention.
3. Flow Velocity Vectors & 2D Flow Velocity Fields.
4. Triplanar Object-Space PBR Multi-Splatting Shader (5 layers, exponent sharpness p in [4, 8], noise perturbation, scree/moisture/sediment modulation).
5. Beer-Lambert Optical Water Shader (Volume absorption, contact foam, wave bump perturbation).
6. Watertight Water Body Geometry (Dynamic river ribbons, watertight lake cylinder, coastal ocean with skirts).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.hydrology import HydrologyEngine, RiverSplineScalar
from terra_forge.schema.map_config import HydrologyConfig

BLENDER_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"
if not os.path.exists(BLENDER_BIN):
    BLENDER_BIN = shutil.which("blender") or "blender"

HAS_BLENDER = os.path.exists(BLENDER_BIN) or shutil.which(BLENDER_BIN) is not None


class TestDynamicRiverSpline:
    def test_arbitrary_river_stages_evaluation(self):
        """Verify spline evaluation dynamically reads arbitrary river stages with custom control points and widths."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=21)
        custom_stages = [
            {"name": "Upper Creek", "t_start": 0.0, "t_end": 0.4, "start": (-35.0, 35.0, 30.0, 2.5), "end": (-10.0, 15.0, 18.0, 3.5)},
            {"name": "Middle Gorge", "t_start": 0.4, "t_end": 0.7, "start": (-10.0, 15.0, 18.0, 3.5), "end": (10.0, -5.0, 10.0, 5.0)},
            {"name": "Lower Estuary", "t_start": 0.7, "t_end": 1.0, "start": (10.0, -5.0, 10.0, 5.0), "end": (35.0, -35.0, 0.0, 8.0)},
        ]
        cfg = HydrologyConfig(river_stages=custom_stages)
        he = HydrologyEngine(hf, cfg)

        t_samples = np.linspace(0.0, 1.0, 31)
        rx, ry, rz, rw = he.evaluate_river_spline(t_samples)

        assert len(rx) == 31
        assert len(ry) == 31
        assert len(rz) == 31
        assert len(rw) == 31

        # Check endpoints
        assert pytest.approx(rx[0], abs=0.2) == -35.0
        assert pytest.approx(ry[0], abs=0.2) == 35.0
        assert pytest.approx(rz[0], abs=0.2) == 30.0
        assert pytest.approx(rw[0], abs=0.2) == 2.5

        assert pytest.approx(rx[-1], abs=0.2) == 35.0
        assert pytest.approx(ry[-1], abs=0.2) == -35.0
        assert pytest.approx(rz[-1], abs=0.2) == 0.0
        assert pytest.approx(rw[-1], abs=0.2) == 8.0

        # Monotonically descending elevation
        assert np.all(np.diff(rz) <= 0.01)

    def test_scalar_float_indexing_and_deprecation_freedom(self):
        """Verify evaluate_river_spline with scalar float produces RiverSplineScalar without warnings."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=21)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        rx0, ry0, rz0, rw0 = he.evaluate_river_spline(0.0)
        assert isinstance(rx0, RiverSplineScalar)
        assert isinstance(rz0, RiverSplineScalar)

        # Indexing [0] must return float cleanly
        assert rx0[0] == pytest.approx(-8.0, abs=0.1)
        assert ry0[0] == pytest.approx(48.0, abs=0.1)
        assert rz0[0] == pytest.approx(21.5, abs=0.1)
        assert rw0[0] == pytest.approx(3.2, abs=0.1)

        # Direct float conversion must not raise DeprecationWarning
        import warnings
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            flt_z = float(rz0)
            assert flt_z == pytest.approx(21.5, abs=0.1)
        deprecation_warnings = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

        # Scalar arithmetic
        _rx1, _ry1, rz1, _rw1 = he.evaluate_river_spline(1.0)
        dz = rz0 - rz1
        assert float(dz) == pytest.approx(21.5, abs=0.1)
        assert float(dz[0]) == pytest.approx(21.5, abs=0.1)
        assert dz > 10.0

    def test_flow_velocity_vectors(self):
        """Verify flow velocity vectors accelerate over steep cascades and decelerate in flat valleys."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=21)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        # Evaluate velocity at cascade (t=0.15) vs valley meander (t=0.45)
        vx_c, vy_c, vz_c = he.evaluate_flow_velocity(0.15)
        vx_v, vy_v, vz_v = he.evaluate_flow_velocity(0.45)

        speed_c = math.sqrt(float(vx_c)**2 + float(vy_c)**2 + float(vz_c)**2)
        speed_v = math.sqrt(float(vx_v)**2 + float(vy_v)**2 + float(vz_v)**2)

        assert speed_c > speed_v, f"Cascade velocity ({speed_c:.2f}) must exceed valley velocity ({speed_v:.2f})"

    def test_flow_velocity_field_2d(self):
        """Verify compute_flow_velocity_field generates non-zero vectors in the river channel."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=25)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        vx_field, vy_field = he.compute_flow_velocity_field()
        assert vx_field.shape == (25, 25)
        assert vy_field.shape == (25, 25)

        speed_field = np.sqrt(vx_field**2 + vy_field**2)
        assert np.max(speed_field) > 0.5
        assert not np.isnan(speed_field).any()

    def test_river_spline_hermite_tangents_and_meander(self):
        """Verify spline interpolation with explicit Hermite tangents and meander parameters."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=21)
        stages = [
            {
                "name": "Hermite Gorge",
                "t_start": 0.0,
                "t_end": 1.0,
                "start": (0.0, 30.0, 20.0, 3.0),
                "end": (0.0, -30.0, 0.0, 6.0),
                "tangent_in": (10.0, -60.0, -20.0),
                "tangent_out": (-10.0, -60.0, -20.0),
            }
        ]
        cfg = HydrologyConfig(river_stages=stages)
        he = HydrologyEngine(hf, cfg)

        t_samples = np.linspace(0.0, 1.0, 25)
        rx, ry, rz, _rw = he.evaluate_river_spline(t_samples)
        assert len(rx) == 25
        assert pytest.approx(rx[0], abs=0.1) == 0.0
        assert pytest.approx(ry[0], abs=0.1) == 30.0
        assert pytest.approx(rz[0], abs=0.1) == 20.0
        assert pytest.approx(rx[-1], abs=0.1) == 0.0
        assert pytest.approx(ry[-1], abs=0.1) == -30.0
        assert pytest.approx(rz[-1], abs=0.1) == 0.0

    def test_river_spline_out_of_bounds_evaluation(self):
        """Verify spline parameter values outside [0.0, 1.0] return zeros without crashing."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=21)
        cfg = HydrologyConfig()
        he = HydrologyEngine(hf, cfg)

        rx_neg, ry_neg, rz_neg, _rw_neg = he.evaluate_river_spline(-0.5)
        assert float(rx_neg) == 0.0
        assert float(ry_neg) == 0.0
        assert float(rz_neg) == 0.0

        rx_pos, ry_pos, _rz_pos, _rw_pos = he.evaluate_river_spline(1.5)
        assert float(rx_pos) == 0.0
        assert float(ry_pos) == 0.0

    def test_parabolic_channel_carving(self):
        """Verify carve_river_channel cuts a distinct concave depression into the heightfield."""
        hf = Heightfield2D(width=80.0, depth=80.0, resolution=41)
        hf.Z.fill(25.0)
        cfg = HydrologyConfig(river_enabled=True, river_carve_depth=3.0)
        he = HydrologyEngine(hf, cfg)

        he.carve_river_channel(num_samples=100)

        # Elevation along river path must be depressed below initial flat 25.0
        rx, ry, _, _ = he.evaluate_river_spline(0.15)
        gx, gy = hf.world_to_grid(float(rx), float(ry))
        carved_z = hf.Z[int(gy), int(gx)]
        assert carved_z < 25.0, f"River bed elevation ({carved_z}) should be carved below 25.0"
        assert carved_z >= 0.0

    def test_water_distance_field_robustness(self):
        """Verify water distance field accurately identifies water proximity and zero-water boundary."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=31)
        cfg = HydrologyConfig(lake_center=(0.0, 0.0), lake_radius=8.0)
        he = HydrologyEngine(hf, cfg)

        dist = he.compute_water_distance_field()
        gx, gy = hf.world_to_grid(0.0, 0.0)
        assert dist[int(gy), int(gx)] == 0.0

        # Dry diorama (lake_radius=0, river_enabled=False)
        cfg_dry = HydrologyConfig(lake_radius=0.0, river_enabled=False)
        he_dry = HydrologyEngine(hf, cfg_dry)
        dist_dry = he_dry.compute_water_distance_field()
        assert np.all(dist_dry >= 900.0)


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestTerrainTriplanarPBRShader:
    def test_terrain_shader_compilation_and_nodes(self):
        """Verify triplanar terrain shader node tree construction in Blender."""
        blender_code = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_triplanar_terrain_shader

cfg = ShaderConfig(snow_altitude=22.0)
mat = create_triplanar_terrain_shader(cfg)
assert mat is not None
assert mat.use_nodes

nodes = mat.node_tree.nodes
node_types = [n.bl_idname for n in nodes]

# Verify critical node types
assert "ShaderNodeOutputMaterial" in node_types
assert "ShaderNodeBsdfPrincipled" in node_types
assert "ShaderNodeNewGeometry" in node_types
assert "ShaderNodeTexNoise" in node_types
assert "ShaderNodeBump" in node_types
assert "ShaderNodeAttribute" in node_types

# Verify 5 material classes and strata attribute
attr_nodes = [n.attribute_name for n in nodes if n.bl_idname == "ShaderNodeAttribute"]
assert "scree_talus" in attr_nodes
assert "moisture" in attr_nodes
assert "sediment" in attr_nodes
assert "COLOR_0" in attr_nodes

# Verify normal exponent power sharpness (p=6.0)
power_nodes = [n for n in nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "POWER"]
assert len(power_nodes) >= 3
for pn in power_nodes:
    assert pn.inputs[1].default_value == 6.0

print("TERRAIN_SHADER_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender failed: {res.stderr}"
        assert "TERRAIN_SHADER_VERIFIED" in res.stdout


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestBeerLambertWaterShader:
    def test_water_shader_compilation_and_optics(self):
        """Verify Beer-Lambert volume absorption, contact foam, and wave bump in water shader."""
        blender_code = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import ShaderConfig
from terra_forge.blender.shaders import create_water_shader

cfg = ShaderConfig(
    water_absorption_color=(0.04, 0.42, 0.68),
    water_roughness=0.03,
    water_contact_foam=True,
)
mat = create_water_shader(cfg)
assert mat is not None
assert mat.use_nodes

nodes = mat.node_tree.nodes
node_types = [n.bl_idname for n in nodes]

# Verify volume absorption node connected to Output Volume
assert "ShaderNodeVolumeAbsorption" in node_types
vol_node = [n for n in nodes if n.bl_idname == "ShaderNodeVolumeAbsorption"][0]
pytest_col = vol_node.inputs["Color"].default_value
assert pytest_col is not None
assert abs(pytest_col[0] - 0.04) < 1e-3
assert abs(pytest_col[1] - 0.42) < 1e-3
assert abs(pytest_col[2] - 0.68) < 1e-3

# Verify wave bump perturbation connected to BSDF Normal
assert "ShaderNodeBump" in node_types
bump_node = [n for n in nodes if n.bl_idname == "ShaderNodeBump"][0]
assert bump_node.inputs["Strength"].default_value > 0.0

# Verify contact foam Voronoi/Noise texturing
assert "ShaderNodeTexVoronoi" in node_types
assert "ShaderNodeTexNoise" in node_types

# Verify transparency blend method
assert mat.blend_method == "BLEND"

print("WATER_SHADER_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender failed: {res.stderr}"
        assert "WATER_SHADER_VERIFIED" in res.stdout


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender executable not found")
class TestWaterGeometryBuilder:
    def test_dynamic_water_bodies_geometry(self):
        """Verify water builder generates watertight lake, coastal ocean, and dynamic river ribbon."""
        blender_code = """
import sys
sys.path.insert(0, '.')
import bpy
from terra_forge.schema.map_config import HydrologyConfig, ShaderConfig, DioramaBounds
from terra_forge.blender.water_builder import build_water_bodies

cfg = HydrologyConfig(
    lake_center=(10.0, -10.0),
    lake_radius=20.0,
    lake_level=5.0,
    lake_depth=10.0,
    sea_level=0.0,
    river_enabled=True,
    river_stages=[
        {"name": "Stage 1", "t_end": 0.5, "start": (-20.0, 40.0, 20.0, 3.0), "end": (0.0, 20.0, 10.0, 4.0)},
        {"name": "Stage 2", "t_end": 1.0, "start": (0.0, 20.0, 10.0, 4.0), "end": (20.0, -20.0, 0.0, 6.0)},
    ]
)
sh_cfg = ShaderConfig()
bounds = DioramaBounds(width=160.0, depth=160.0, z_base=-16.0)

objs = build_water_bodies(cfg, sh_cfg, diorama_bounds=bounds)
assert len(objs) == 3

# Verify Lake Object
lake_obj = [o for o in objs if "Lake" in o.name][0]
assert len(lake_obj.data.vertices) > 50
assert len(lake_obj.data.polygons) > 20
# Lake must be watertight solid cylinder: all vertices have top and bottom elevation
z_coords = [v.co.z for v in lake_obj.data.vertices]
assert max(z_coords) == 5.0
assert min(z_coords) <= -5.0

# Verify Marine Bay Object
bay_obj = [o for o in objs if "Bay" in o.name][0]
assert len(bay_obj.data.polygons) > 50

# Verify River Ribbon Object
riv_obj = [o for o in objs if "River" in o.name][0]
assert len(riv_obj.data.polygons) > 10

# All water objects must have water material assigned
for obj in objs:
    assert len(obj.data.materials) >= 1
    assert "Water" in obj.data.materials[0].name

print("WATER_GEOMETRY_VERIFIED")
"""
        cmd = [BLENDER_BIN, "-b", "--python-expr", blender_code]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        assert res.returncode == 0, f"Blender failed: {res.stderr}"
        assert "WATER_GEOMETRY_VERIFIED" in res.stdout
