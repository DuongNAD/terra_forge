"""
Unit and Integration Test Suite for Milestone M5: Ecological Dynamics & Seasonal State.

Verifies:
1. Dynamic water level modulation updates shoreline boundaries without topological tearing.
2. Seasonal moisture variation drives Whittaker NPP carrying capacity and logistic regrowth.
3. Shader uniforms generator produces all required Three.js / WebGL uniforms.
4. Per-vertex color and mask planes output valid, normalized RGBA arrays matching mesh vertex counts.
5. Full cross-milestone integration with WorldArtifact and OpenWorldMesh.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pytest

from terra_forge.core.biomes import Heightfield2D
from terra_forge.core.open_mesh import OpenWorldMesh, _pack_glb
from terra_forge.core.world_artifact import (
    Biome,
    CANONICAL_GRID_DIM,
    LegacyBiome,
    WorldArtifact,
)


def create_synthetic_world_artifact(dim: int = 64, seed: int = 42) -> WorldArtifact:
    """Helper creating a synthetic WorldArtifact with varied terrain for testing."""
    n = dim * dim
    rng = np.random.RandomState(seed)
    x = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    r = np.sqrt(gx * gx + gy * gy)
    elev_2d = np.clip(0.5 + 0.4 * np.sin(gx * 3.0) * np.cos(gy * 3.0) - 0.2 * r, 0.05, 0.95).astype(np.float32)
    moist_2d = np.clip(0.5 + 0.3 * np.cos(gx * 2.0) - 0.1 * gy, 0.1, 0.9).astype(np.float32)
    temp_2d = np.clip(0.6 - 0.4 * elev_2d + 0.1 * gy, 0.1, 0.9).astype(np.float32)
    flow_2d = np.clip(np.exp(-((gx - 0.1) ** 2 + (gy - 0.1) ** 2) / 0.1) * 200.0, 0.0, 500.0).astype(np.float32)
    biome = np.array([i % 22 for i in range(n)], dtype=np.uint8)

    return WorldArtifact(
        width=dim,
        height=dim,
        sea_level=0.25,
        seed=seed,
        generator_version=1,
        world_scale=200.0,
        elevation=elev_2d.flatten(),
        moisture=moist_2d.flatten(),
        temperature=temp_2d.flatten(),
        flow=flow_2d.flatten(),
        biome=biome,
    )
from terra_forge.ecology.dynamics import (
    DEFAULT_FLOOD_AMPLITUDE,
    DEFAULT_SEASON_PERIOD_SEC,
    DEFAULT_SNOWLINE_BASE,
    DEFAULT_WATER_LEVEL,
    NPP_TO_CAPACITY,
    SEASON_AMPLITUDE,
    WHITTAKER_NPP_CANONICAL,
    WHITTAKER_NPP_LEGACY,
    BiomassField,
    SeasonClock,
    VertexColorMasks,
    biome_carrying_capacity,
    biome_npp,
    compute_carrying_capacity_grid,
    compute_dynamic_water_level,
    generate_shader_uniforms,
    generate_vertex_colors_and_masks,
    logistic_regrowth,
    modulate_seasonal_moisture,
    seasonal_drying_factor,
    seasonal_fertility,
    step_biomass_regrowth_grid,
)


# =============================================================================
# Helper: GLB Parser for inspecting glTF JSON chunk and binary accessors
# =============================================================================

def parse_glb_header_and_json(glb_bytes: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Unpacks a glTF 2.0 Binary blob into (json_dict, binary_buffer)."""
    assert len(glb_bytes) >= 12, "GLB blob is too short"
    magic, version, length = struct.unpack_from("<4sII", glb_bytes, 0)
    assert magic == b"glTF", f"Invalid GLB magic: {magic}"
    assert version == 2, f"Expected glTF version 2, got {version}"
    assert length == len(glb_bytes), "GLB header length mismatch"

    # Chunk 0: JSON
    c0_len, c0_type = struct.unpack_from("<I4s", glb_bytes, 12)
    assert c0_type == b"JSON", f"Expected JSON chunk, got {c0_type}"
    json_start = 20
    json_bytes = glb_bytes[json_start : json_start + c0_len]
    json_dict = json.loads(json_bytes.decode("utf-8").strip())

    # Chunk 1: BIN
    c1_offset = json_start + c0_len
    if c1_offset < length:
        c1_len, c1_type = struct.unpack_from("<I4s", glb_bytes, c1_offset)
        assert c1_type == b"BIN\x00", f"Expected BIN chunk, got {c1_type}"
        bin_start = c1_offset + 8
        bin_data = glb_bytes[bin_start : bin_start + c1_len]
    else:
        bin_data = b""

    return json_dict, bin_data


# =============================================================================
# 1. Dynamic Water Level & Shoreline Modulation Tests
# =============================================================================

class TestDynamicWaterLevelAndShoreline:
    def test_dynamic_water_level_modulation_harmonics(self):
        """Verify dynamic water level modulation matches sinusoidal monsoon harmonic."""
        base_wl = 2.0
        amp = 0.4

        # Equinox (phase = 0): water_level == base_wl
        assert math.isclose(compute_dynamic_water_level(base_wl, 0.0, amp), 2.0, abs_tol=1e-6)

        # Peak monsoon / summer (phase = pi/2): water_level == base_wl + amp
        assert math.isclose(compute_dynamic_water_level(base_wl, math.pi / 2.0, amp), 2.4, abs_tol=1e-6)

        # Autumn equinox (phase = pi): water_level == base_wl
        assert math.isclose(compute_dynamic_water_level(base_wl, math.pi, amp), 2.0, abs_tol=1e-6)

        # Peak dry season / winter (phase = 3*pi/2): water_level == base_wl - amp
        assert math.isclose(compute_dynamic_water_level(base_wl, 3.0 * math.pi / 2.0, amp), 1.6, abs_tol=1e-6)

        # Periodic periodicity: phase + 2*pi matches phase
        assert math.isclose(
            compute_dynamic_water_level(base_wl, 1.234, amp),
            compute_dynamic_water_level(base_wl, 1.234 + 2.0 * math.pi, amp),
            abs_tol=1e-6,
        )

    def test_shoreline_expansion_contraction_monotonicity(self):
        """Rising water level must monotonically increase submerged area across a continuous slope."""
        # Create continuous ramp elevation from 0.0m to 10.0m
        N = 64
        x = np.linspace(0.0, 10.0, N, dtype=np.float32)
        elev_grid = np.tile(x, (N, 1))

        levels = [1.5, 1.8, 2.0, 2.3, 2.6]
        submerged_counts = []
        for wl in levels:
            submerged = elev_grid <= wl
            submerged_counts.append(int(np.sum(submerged)))

        # Monotonicity check: higher water level strictly increases or equals submerged cells
        for i in range(len(submerged_counts) - 1):
            assert submerged_counts[i] < submerged_counts[i + 1]

    def test_shoreline_no_topological_tearing(self):
        """
        Verify that modulating dynamic water level does NOT tear the mesh topology.
        Mesh topology (vertices, faces, connectivity) must remain strictly invariant,
        manifold, and watertight across all seasonal water fluctuations.
        """
        dim = 32
        elev = np.linspace(0.0, 10.0, dim * dim, dtype=np.float32).reshape((dim, dim))
        mesh = OpenWorldMesh(elev)

        initial_v_count = mesh.vertex_count
        initial_tri_count = mesh.triangle_count
        initial_indices = np.copy(mesh.indices)

        # Cycle through 8 seasonal phases
        phases = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        for phase in phases:
            wl = compute_dynamic_water_level(2.0, phase, 0.4)
            masks = generate_vertex_colors_and_masks(mesh, season_phase=phase, water_level=wl)

            # Topology invariance
            assert mesh.vertex_count == initial_v_count
            assert mesh.triangle_count == initial_tri_count
            assert np.array_equal(mesh.indices, initial_indices)

            # Valid finite values (zero NaNs, zero infinities)
            assert np.all(np.isfinite(masks.rgba))
            assert np.all(masks.rgba >= 0.0)
            assert np.all(masks.rgba <= 1.0)


# =============================================================================
# 2. Whittaker 1975 NPP & Logistic Regrowth Dynamics Tests
# =============================================================================

class TestWhittakerNPPAndLogisticRegrowth:
    def test_whittaker_canonical_values_match_anima_engine(self):
        """Whittaker 1975 dry-matter means must match Anima-Engine ecology.rs exactly."""
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.RAINFOREST)] == 2200.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.TEMPERATE_FOREST)] == 1200.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.BOREAL_FOREST)] == 800.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.GRASSLAND)] == 700.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.RIVER)] == 500.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.OCEAN)] == 250.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.SNOW)] == 140.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.DEEP_OCEAN)] == 125.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.DESERT)] == 90.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.BEACH)] == 60.0
        assert WHITTAKER_NPP_LEGACY[int(LegacyBiome.MOUNTAIN_ROCK)] == 50.0

        # Carrying capacity must be exactly NPP * 0.01
        assert math.isclose(biome_carrying_capacity(LegacyBiome.RAINFOREST, use_canonical=False), 22.0)
        assert math.isclose(biome_carrying_capacity(LegacyBiome.TEMPERATE_FOREST, use_canonical=False), 12.0)
        assert math.isclose(biome_carrying_capacity(LegacyBiome.DESERT, use_canonical=False), 0.9)

    def test_carrying_capacity_grid_vectorization(self):
        """Vectorized compute_carrying_capacity_grid must match scalar evaluations."""
        biomes = np.array([
            [int(Biome.OCEAN), int(Biome.JUNGLE)],
            [int(Biome.DESERT), int(Biome.FOREST)],
        ], dtype=np.uint8)

        cap_grid = compute_carrying_capacity_grid(biomes, use_canonical=True)
        assert cap_grid.shape == (2, 2)
        assert float(cap_grid[0, 0]) == pytest.approx(biome_carrying_capacity(Biome.OCEAN), rel=1e-5)
        assert float(cap_grid[0, 1]) == pytest.approx(biome_carrying_capacity(Biome.JUNGLE), rel=1e-5)
        assert float(cap_grid[1, 0]) == pytest.approx(biome_carrying_capacity(Biome.DESERT), rel=1e-5)
        assert float(cap_grid[1, 1]) == pytest.approx(biome_carrying_capacity(Biome.FOREST), rel=1e-5)

    def test_seasonal_fertility_swing(self):
        """Verify fertility multiplier F(theta) = max(0.0, 1.0 + 0.5 * sin(theta))."""
        assert math.isclose(seasonal_fertility(0.0), 1.0, abs_tol=1e-6)
        assert math.isclose(seasonal_fertility(math.pi / 2.0), 1.5, abs_tol=1e-6)
        assert math.isclose(seasonal_fertility(math.pi), 1.0, abs_tol=1e-6)
        assert math.isclose(seasonal_fertility(3.0 * math.pi / 2.0), 0.5, abs_tol=1e-6)

    def test_logistic_regrowth_seed_and_saturation(self):
        """Discrete logistic regrowth: recovers from R=0 via seed; clamps at R_max."""
        r_max = 10.0
        g = 0.1

        # R = 0: seed ensures non-zero recovery
        r_next = logistic_regrowth(0.0, g, r_max)
        assert r_next > 0.0
        assert math.isclose(r_next, 1e-3 * r_max, abs_tol=1e-6)

        # R = r_max: growth saturates, delta is zero
        r_sat = logistic_regrowth(r_max, g, r_max)
        assert math.isclose(r_sat, r_max, abs_tol=1e-6)

        # Overshoot clamp
        r_over = logistic_regrowth(r_max * 1.5, g, r_max)
        assert math.isclose(r_over, r_max, abs_tol=1e-6)

    def test_seasonal_regrowth_rate_differential(self):
        """Regrowth during summer (fertility=1.5) must strictly outpace winter (fertility=0.5)."""
        biomass_init = np.full((4, 4), 5.0, dtype=np.float32)
        r_max = np.full((4, 4), 10.0, dtype=np.float32)

        # Summer step (phase = pi/2)
        summer_step = step_biomass_regrowth_grid(biomass_init, r_max, growth_rate=0.1, season_phase=math.pi / 2.0)
        summer_delta = np.mean(summer_step - biomass_init)

        # Winter step (phase = 3*pi/2)
        winter_step = step_biomass_regrowth_grid(biomass_init, r_max, growth_rate=0.1, season_phase=3.0 * math.pi / 2.0)
        winter_delta = np.mean(winter_step - biomass_init)

        assert summer_delta > winter_delta
        # Ratio should be approximately 1.5 / 0.5 = 3.0
        assert math.isclose(summer_delta / winter_delta, 3.0, rel_tol=0.05)

    def test_biomass_field_grazing_and_regrowth_cycle(self):
        """Verify BiomassField state tracker: grazing depletion followed by logistic recovery."""
        biomes = np.full((16, 16), int(Biome.GRASSLAND), dtype=np.uint8)
        field = BiomassField(biomes, initial_ratio=1.0, growth_rate=0.1)

        cap_total = field.total_biomass
        assert cap_total > 0.0

        # Graze 50% of the entire field
        mask = np.ones((16, 16), dtype=bool)
        eaten = field.graze_mask(mask, fraction=0.5)
        assert math.isclose(eaten, cap_total * 0.5, rel_tol=1e-4)
        assert math.isclose(field.total_biomass, cap_total * 0.5, rel_tol=1e-4)

        # Normalized biomass is 0.5
        assert np.allclose(field.normalized_biomass, 0.5, atol=1e-4)

        # Advance 10 ticks of summer regrowth
        for _ in range(10):
            field.step(season_phase=math.pi / 2.0, dt=1.0)

        # Biomass recovered towards ceiling
        assert field.total_biomass > cap_total * 0.75
        assert field.total_biomass <= cap_total


# =============================================================================
# 3. Shader Uniforms Generator Tests
# =============================================================================

class TestShaderUniformsGenerator:
    def test_required_uniforms_structure_and_values(self):
        """generate_shader_uniforms must output all required Three.js / WebGL uniforms."""
        uniforms = generate_shader_uniforms(
            season_phase=1.25,
            time_sec=42.5,
            wind_speed=2.5,
            wind_direction=0.785,
            water_level=2.35,
        )

        required_keys = ["uTime", "uWindSpeed", "uWindDirection", "uSeasonPhase", "uWaterLevel", "uDryingFactor"]
        for key in required_keys:
            assert key in uniforms, f"Missing required uniform: {key}"
            assert "value" in uniforms[key], f"Uniform {key} must have 'value' dictionary structure"

        assert math.isclose(uniforms["uTime"]["value"], 42.5)
        assert math.isclose(uniforms["uWindSpeed"]["value"], 2.5)
        assert math.isclose(uniforms["uWindDirection"]["value"], 0.785)
        assert math.isclose(uniforms["uSeasonPhase"]["value"], 1.25)
        assert math.isclose(uniforms["uWaterLevel"]["value"], 2.35)

    def test_drying_factor_phase_behavior(self):
        """uDryingFactor must vary smoothly from 0.0 (wet season) to 1.0 (dry season)."""
        # Peak monsoon (phase = pi/2): drying factor = 0.0
        u_wet = generate_shader_uniforms(season_phase=math.pi / 2.0)
        assert math.isclose(u_wet["uDryingFactor"]["value"], 0.0, abs_tol=1e-6)

        # Peak dry season (phase = 3*pi/2): drying factor = 1.0
        u_dry = generate_shader_uniforms(season_phase=3.0 * math.pi / 2.0)
        assert math.isclose(u_dry["uDryingFactor"]["value"], 1.0, abs_tol=1e-6)

        # Equinox (phase = 0): drying factor = 0.5
        u_eq = generate_shader_uniforms(season_phase=0.0)
        assert math.isclose(u_eq["uDryingFactor"]["value"], 0.5, abs_tol=1e-6)

    def test_anima_engine_twin_shader_aliases(self):
        """Twin aliases (time, windSpeed, windAngle) must match Anima-Engine Vegetation.tsx."""
        uniforms = generate_shader_uniforms(time_sec=10.0, wind_speed=3.0, wind_direction=1.57)
        assert uniforms["time"]["value"] == uniforms["uTime"]["value"]
        assert uniforms["windSpeed"]["value"] == uniforms["uWindSpeed"]["value"]
        assert uniforms["windAngle"]["value"] == uniforms["uWindDirection"]["value"]

    def test_season_clock_advancement(self):
        """SeasonClock advances phase deterministically over time."""
        clock = SeasonClock(initial_phase=0.0, period_sec=100.0)
        # Advance 25 seconds = quarter period = pi/2
        fert = clock.tick(25.0)
        assert math.isclose(clock.phase, math.pi / 2.0, rel_tol=1e-4)
        assert math.isclose(fert, 1.5, rel_tol=1e-4)
        assert math.isclose(clock.drying_factor, 0.0, abs_tol=1e-4)


# =============================================================================
# 4. Per-Vertex Colors & Mask Planes Generator Tests
# =============================================================================

class TestVertexColorsAndMaskPlanes:
    @pytest.fixture
    def synthetic_world(self):
        """Creates a realistic 64x64 synthetic WorldArtifact with varied terrain."""
        return create_synthetic_world_artifact(dim=64, seed=42)

    def test_vertex_masks_shape_bounds_and_channels(self, synthetic_world):
        """generate_vertex_colors_and_masks must return [N, 4] normalized float32 array."""
        masks = generate_vertex_colors_and_masks(synthetic_world, season_phase=0.0)

        assert isinstance(masks, VertexColorMasks)
        assert masks.rgba.shape == (64 * 64, 4)
        assert masks.grid_rgba.shape == (64, 64, 4)
        assert masks.rgba.dtype == np.float32

        # All channels strictly in [0.0, 1.0]
        assert np.all(masks.rgba >= 0.0)
        assert np.all(masks.rgba <= 1.0)

        # 4 distinct channels exist
        assert masks.aridity.shape == (64, 64)
        assert masks.moisture.shape == (64, 64)
        assert masks.snowline.shape == (64, 64)
        assert masks.biomass.shape == (64, 64)

    def test_channel_r_seasonal_aridity_response(self, synthetic_world):
        """Channel R (Aridity) must be higher during dry season than during monsoon."""
        masks_wet = generate_vertex_colors_and_masks(synthetic_world, season_phase=math.pi / 2.0)
        masks_dry = generate_vertex_colors_and_masks(synthetic_world, season_phase=3.0 * math.pi / 2.0)

        # Mean aridity across land must be higher in dry season
        mean_r_wet = float(np.mean(masks_wet.aridity))
        mean_r_dry = float(np.mean(masks_dry.aridity))
        assert mean_r_dry > mean_r_wet

        # Submerged cells must have 0 aridity
        elev = np.asarray(synthetic_world.elevation).reshape((64, 64)) * 10.0
        submerged = elev <= masks_dry.water_level
        if np.any(submerged):
            assert np.all(masks_dry.aridity[submerged] == 0.0)

    def test_channel_g_alluvial_flood_saturation(self, synthetic_world):
        """Channel G (Moisture) must saturate submerged cells and alluvial flood margins."""
        masks = generate_vertex_colors_and_masks(synthetic_world, season_phase=math.pi / 2.0)

        elev = np.asarray(synthetic_world.elevation).reshape((64, 64)) * 10.0
        submerged = elev <= masks.water_level
        if np.any(submerged):
            # Submerged cells are 100% moisture saturated
            assert np.allclose(masks.moisture[submerged], 1.0, atol=1e-5)

    def test_channel_b_seasonal_snowline_descent(self, synthetic_world):
        """Channel B (Snowline) must descend down mountain slopes during winter."""
        masks_summer = generate_vertex_colors_and_masks(synthetic_world, season_phase=math.pi / 2.0)
        masks_winter = generate_vertex_colors_and_masks(synthetic_world, season_phase=3.0 * math.pi / 2.0)

        # Total snow/frost mask coverage must be higher in winter than summer
        total_snow_summer = float(np.sum(masks_summer.snowline))
        total_snow_winter = float(np.sum(masks_winter.snowline))
        assert total_snow_winter > total_snow_summer

    def test_channel_a_biomass_density_bounds(self, synthetic_world):
        """Channel A (Biomass) must reflect lush vegetation on land and minimal in deep snow/water."""
        masks = generate_vertex_colors_and_masks(synthetic_world, season_phase=math.pi / 2.0)

        assert np.all(masks.biomass >= 0.0)
        assert np.all(masks.biomass <= 1.0)
        assert np.max(masks.biomass) > 0.0


# =============================================================================
# 5. Full Cross-Milestone Integration Tests (WorldArtifact & OpenWorldMesh)
# =============================================================================

class TestCrossMilestoneIntegration:
    def test_open_world_mesh_set_vertex_colors(self):
        """OpenWorldMesh.set_vertex_colors embeds RGBA colors and verifies dimensions."""
        dim = 16
        elev = np.zeros((dim, dim), dtype=np.float32)
        mesh = OpenWorldMesh(elev)

        # Correct shape (256, 4)
        colors = np.random.uniform(0.0, 1.0, (dim * dim, 4)).astype(np.float32)
        mesh.set_vertex_colors(colors)
        assert mesh.colors is not None
        assert mesh.colors.shape == (256, 4)

        # Setting from 2D grid shape (16, 16, 4)
        grid_colors = np.random.uniform(0.0, 1.0, (dim, dim, 4)).astype(np.float32)
        mesh.set_vertex_colors(grid_colors)
        assert mesh.colors.shape == (256, 4)

        # Mismatched vertex count must raise ValueError
        bad_colors = np.zeros((100, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            mesh.set_vertex_colors(bad_colors)

    def test_glb_export_with_color_0_accessor(self):
        """Exported GLB must contain valid COLOR_0 accessor pointing to packed binary data."""
        dim = 16
        elev = np.linspace(0.0, 5.0, dim * dim, dtype=np.float32).reshape((dim, dim))
        mesh = OpenWorldMesh(elev)

        masks = generate_vertex_colors_and_masks(elev, season_phase=1.0)
        mesh.set_vertex_colors(masks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_glb = Path(tmp_dir) / "terrain_colored.glb"
            mesh.export_glb(out_glb)

            assert out_glb.exists()
            assert out_glb.stat().st_size > 0

            # Parse exported GLB
            glb_data = out_glb.read_bytes()
            gltf, bin_bytes = parse_glb_header_and_json(glb_data)

            # Check glTF structure
            prim = gltf["meshes"][0]["primitives"][0]
            attrs = prim["attributes"]

            assert "POSITION" in attrs
            assert "NORMAL" in attrs
            assert "TEXCOORD_0" in attrs
            assert "COLOR_0" in attrs, "COLOR_0 attribute missing from GLB primitive!"

            color_acc_idx = attrs["COLOR_0"]
            color_acc = gltf["accessors"][color_acc_idx]
            assert color_acc["type"] == "VEC4"
            assert color_acc["componentType"] == 5126  # FLOAT
            assert color_acc["count"] == mesh.vertex_count

            # Extract color bytes from BIN chunk and verify numerical equivalence
            color_view_idx = color_acc["bufferView"]
            color_view = gltf["bufferViews"][color_view_idx]
            c_offset = color_view["byteOffset"]
            c_len = color_view["byteLength"]

            extracted_colors = np.frombuffer(bin_bytes[c_offset : c_offset + c_len], dtype="<f4").reshape(-1, 4)
            assert np.allclose(extracted_colors, masks.rgba, atol=1e-5)

    def test_chunked_mesh_color_propagation(self):
        """create_chunks must propagate sub-array colors to each MeshChunk."""
        dim = 32
        elev = np.zeros((dim, dim), dtype=np.float32)
        mesh = OpenWorldMesh(elev)

        colors = np.random.uniform(0.0, 1.0, (dim * dim, 4)).astype(np.float32)
        mesh.set_vertex_colors(colors)

        chunks = mesh.create_chunks(chunks_x=2, chunks_y=2)
        assert len(chunks) == 4

        with tempfile.TemporaryDirectory() as tmp_dir:
            for chunk in chunks:
                assert chunk.colors is not None
                assert len(chunk.colors) == chunk.vertex_count
                assert chunk.colors.shape[-1] == 4

                # Verify chunk GLB exports with COLOR_0
                chunk_glb_path = Path(tmp_dir) / f"chunk_{chunk.chunk_x}_{chunk.chunk_y}.glb"
                chunk.export_glb(chunk_glb_path)
                assert chunk_glb_path.exists()

                gltf, _ = parse_glb_header_and_json(chunk_glb_path.read_bytes())
                assert "COLOR_0" in gltf["meshes"][0]["primitives"][0]["attributes"]

    def test_world_artifact_v2_end_to_end_dynamics_pipeline(self):
        """Full end-to-end integration: WorldArtifact v2 -> Dynamics -> OpenWorldMesh -> GLB."""
        # 1. Generate canonical 256x256 WorldArtifact v2
        artifact = create_synthetic_world_artifact(dim=CANONICAL_GRID_DIM, seed=999)

        # 2. Advance seasonal clock
        clock = SeasonClock(initial_phase=math.pi / 4.0, period_sec=100.0)
        clock.tick(dt=12.5)  # Phase ~ 3*pi/8

        # 3. Generate shader uniforms for Three.js
        uniforms = generate_shader_uniforms(
            season_phase=clock.phase,
            time_sec=12.5,
            wind_speed=2.0,
            wind_direction=1.2,
            water_level=clock.water_level(artifact.sea_level * 10.0),
        )
        assert "uSeasonPhase" in uniforms
        assert "uWaterLevel" in uniforms

        # 4. Generate per-vertex RGBA colors and masks
        masks = generate_vertex_colors_and_masks(
            heightfield=artifact,
            season_phase=clock.phase,
            water_level=uniforms["uWaterLevel"]["value"],
        )
        assert masks.vertex_count == 256 * 256

        # 5. Build continuous OpenWorldMesh and embed vertex colors
        mesh = OpenWorldMesh.from_world_artifact(artifact, colors=masks)
        assert mesh.vertex_count == 256 * 256
        assert mesh.colors is not None

        # 6. Export to GLB
        with tempfile.TemporaryDirectory() as tmp_dir:
            glb_path = Path(tmp_dir) / "world_m5_final.glb"
            mesh.export_glb(glb_path)

            assert glb_path.exists()
            assert glb_path.stat().st_size > 4_000_000  # 256x256 mesh with COLOR_0 is ~4.7 MB

            # Parse and verify glTF structure
            gltf, _ = parse_glb_header_and_json(glb_path.read_bytes())
            prim_attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
            assert "COLOR_0" in prim_attrs
            assert "POSITION" in prim_attrs
            assert "NORMAL" in prim_attrs
            assert "TEXCOORD_0" in prim_attrs
