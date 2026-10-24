"""
Milestone M1 Unit Test Suite: Data Contract & Headless Export.

Verifies:
1. WorldArtifact v2 binary format specification, header layout, and 36-byte packing.
2. FNV-1a 32-bit checksum implementation and parity against Anima-Engine test vectors.
3. 22 Canonical Biomes taxonomy, 11 legacy biomes, and bidirectional mappings.
4. Whittaker / physical biome classification from continuous environmental variables.
5. Exact byte-for-byte cross-language compatibility with Anima-Engine fixtures.
6. Error handling: rejection of bad magic, unsupported version, corrupt payload, and trailing bytes.
7. MapManifest generator, schema validation, canonical camera views, and SHA-256 computation.
8. OpenWorldMesh continuous top-surface heightfield generation, chunking, OBJ and GLB export.
"""

import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import pytest

from terra_forge.core.world_artifact import (
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WORLD_ARTIFACT_HEADER_LEN,
    CANONICAL_WORLD_SCALE,
    CANONICAL_GRID_DIM,
    CANONICAL_TOTAL_SIZE_256,
    FNV1A_32_BASIS,
    HEADER_STRUCT,
    Biome,
    LegacyBiome,
    BIOME_COUNT,
    LEGACY_BIOME_COUNT,
    fnv1a_32,
    map_biome_frontend_to_backend,
    map_biome_backend_to_frontend,
    map_biome_array_to_backend,
    map_biome_array_to_frontend,
    classify_biome,
    classify_biomes_vectorized,
    WorldArtifact,
)
from terra_forge.core.manifest import (
    CANONICAL_VIEW_IDS,
    CANONICAL_VIEW_CAMERAS,
    generate_map_manifest,
    validate_map_manifest,
    save_map_manifest,
    load_map_manifest,
)
from terra_forge.core.open_mesh import (
    OpenWorldMesh,
    MeshChunk,
)
from terra_forge.core.heightfield import Heightfield2D

# Paths to Anima-Engine fixtures
ANIMA_ENGINE_DIR = Path("e:/Project/03_Engines_Simulation/Anima-Engine")
FIXTURE_4X4 = ANIMA_ENGINE_DIR / "src-tauri/tests/fixtures/world_4x4.anmw"
FIXTURE_128 = ANIMA_ENGINE_DIR / "src-tauri/tests/fixtures/world_real_128.anmw"
FIXTURE_256 = ANIMA_ENGINE_DIR / "artifacts/world_256.anmw"
MANIFEST_JSON = ANIMA_ENGINE_DIR / "map_manifest.json"


# =============================================================================
# 1. Header Layout & Constants
# =============================================================================
class TestWorldArtifactConstants:
    def test_magic_and_version(self):
        assert WORLD_ARTIFACT_MAGIC == b"ANMW"
        assert WORLD_ARTIFACT_VERSION == 2
        assert WORLD_ARTIFACT_HEADER_LEN == 36
        assert CANONICAL_WORLD_SCALE == 200.0
        assert CANONICAL_GRID_DIM == 256

    def test_header_struct_size(self):
        assert HEADER_STRUCT.size == 36
        # Test packing dummy values
        packed = HEADER_STRUCT.pack(b"ANMW", 2, 256, 256, 0.3, 1337, 1, 200.0, 0)
        assert len(packed) == 36

    def test_canonical_256_file_size(self):
        # 36 header + 256*256*(4*4 + 1) = 36 + 1,114,112 = 1,114,148
        expected_payload = 256 * 256 * 17
        assert expected_payload == 1_114_112
        assert 36 + expected_payload == CANONICAL_TOTAL_SIZE_256
        assert CANONICAL_TOTAL_SIZE_256 == 1_114_148


# =============================================================================
# 2. FNV-1a 32-bit Checksum
# =============================================================================
class TestFNV1aChecksum:
    def test_empty_bytes_basis(self):
        assert fnv1a_32(b"") == FNV1A_32_BASIS
        assert fnv1a_32(b"") == 0x811C9DC5

    def test_known_test_vectors(self):
        # Reference FNV-1a test vectors (XOR then multiply)
        assert fnv1a_32(b"a") == 0xE40C292C
        assert fnv1a_32(b"foobar") == 0xBF9CF968

    @pytest.mark.skipif(not FIXTURE_4X4.exists(), reason="fixture world_4x4 not found")
    def test_fixture_4x4_checksum(self):
        data = FIXTURE_4X4.read_bytes()
        header_csum = struct.unpack_from("<I", data, 32)[0]
        payload = data[36:]
        computed = fnv1a_32(payload)
        assert computed == header_csum
        assert computed == 0x153AB135

    @pytest.mark.skipif(not FIXTURE_256.exists(), reason="fixture world_256 not found")
    def test_fixture_256_checksum(self):
        data = FIXTURE_256.read_bytes()
        header_csum = struct.unpack_from("<I", data, 32)[0]
        payload = data[36:]
        computed = fnv1a_32(payload)
        assert computed == header_csum
        assert computed == 0xBDE9D1F2


# =============================================================================
# 3. Biome Taxonomy & Mappings
# =============================================================================
class TestBiomeTaxonomy:
    def test_enum_counts(self):
        assert BIOME_COUNT == 22
        assert LEGACY_BIOME_COUNT == 11
        assert len(Biome) == 22
        assert len(LegacyBiome) == 11

    def test_frontend_to_backend_projection(self):
        for front in range(22):
            back = map_biome_frontend_to_backend(front)
            assert 0 <= back <= 10

        assert map_biome_frontend_to_backend(Biome.OCEAN) == LegacyBiome.OCEAN
        assert map_biome_frontend_to_backend(Biome.BEACH) == LegacyBiome.BEACH
        assert map_biome_frontend_to_backend(Biome.DESERT) == LegacyBiome.DESERT
        assert map_biome_frontend_to_backend(Biome.RIVER) == LegacyBiome.RIVER
        assert map_biome_frontend_to_backend(Biome.SNOW) == LegacyBiome.SNOW
        assert map_biome_frontend_to_backend(Biome.ROCK) == LegacyBiome.MOUNTAIN_ROCK
        # Out-of-bounds fallback
        assert map_biome_frontend_to_backend(99) == LegacyBiome.GRASSLAND

    def test_backend_to_frontend_lifting(self):
        for back in range(11):
            front = map_biome_backend_to_frontend(back)
            assert 0 <= front < 22

        assert map_biome_backend_to_frontend(LegacyBiome.DEEP_OCEAN) == Biome.OCEAN
        assert map_biome_backend_to_frontend(LegacyBiome.OCEAN) == Biome.OCEAN
        assert map_biome_backend_to_frontend(LegacyBiome.BEACH) == Biome.BEACH
        assert map_biome_backend_to_frontend(LegacyBiome.RIVER) == Biome.RIVER
        assert map_biome_backend_to_frontend(LegacyBiome.SNOW) == Biome.SNOW

    def test_legacy_roundtrip_preservation(self):
        # Roundtrip legacy -> canonical -> legacy is identity for all 1..10
        # (0: DeepOcean folds to 1: Ocean as documented)
        for back in range(11):
            rt = map_biome_frontend_to_backend(map_biome_backend_to_frontend(back))
            if back == 0:
                assert rt == LegacyBiome.OCEAN
            else:
                assert rt == back

    def test_vectorized_mappings(self):
        front_arr = np.array([0, 1, 2, 12, 13, 21], dtype=np.uint8)
        back_arr = map_biome_array_to_backend(front_arr)
        expected = np.array([
            map_biome_frontend_to_backend(x) for x in front_arr
        ], dtype=np.uint8)
        assert np.array_equal(back_arr, expected)

    def test_whittaker_biome_classifier(self):
        # Ocean: below sea level
        assert classify_biome(elev=0.1, temp=0.5, moist=0.5, flow=0.0, slope=0.0, sea_level=0.3) == Biome.OCEAN

        # Glacier / Snow: high & very cold
        assert classify_biome(elev=0.9, temp=0.02, moist=0.5, flow=0.0, slope=0.1, sea_level=0.3) == Biome.GLACIER
        assert classify_biome(elev=0.8, temp=0.10, moist=0.5, flow=0.0, slope=0.1, sea_level=0.3) == Biome.SNOW

        # Desert: hot and arid
        assert classify_biome(elev=0.4, temp=0.55, moist=0.05, flow=0.0, slope=0.05, sea_level=0.3) == Biome.DESERT

        # Jungle: hot and humid
        assert classify_biome(elev=0.4, temp=0.80, moist=0.80, flow=0.0, slope=0.05, sea_level=0.3) == Biome.JUNGLE

        # River: high flow
        assert classify_biome(elev=0.4, temp=0.50, moist=0.50, flow=0.75, slope=0.05, sea_level=0.3) == Biome.RIVER

    def test_vectorized_classification(self):
        n = 16
        elev = np.linspace(0.1, 0.9, n)
        temp = np.linspace(0.8, 0.1, n)
        moist = np.full(n, 0.5)
        flow = np.zeros(n)
        slope = np.zeros(n)

        vec_biomes = classify_biomes_vectorized(elev, temp, moist, flow, slope, sea_level=0.3)
        assert len(vec_biomes) == n
        for i in range(n):
            expected = classify_biome(elev[i], temp[i], moist[i], flow[i], slope[i], sea_level=0.3)
            assert vec_biomes[i] == int(expected)


# =============================================================================
# 4. WorldArtifact Encoding & Decoding Roundtrip
# =============================================================================
class TestWorldArtifactRoundtrip:
    def _create_synthetic(self, w: int = 8, h: int = 8) -> WorldArtifact:
        n = w * h
        elevation = np.linspace(0.0, 1.0, n, dtype=np.float32)
        moisture = np.linspace(0.1, 0.9, n, dtype=np.float32)
        temperature = np.linspace(0.2, 0.8, n, dtype=np.float32)
        flow = np.linspace(0.0, 0.5, n, dtype=np.float32)
        biome = np.array([i % 22 for i in range(n)], dtype=np.uint8)

        return WorldArtifact(
            width=w,
            height=h,
            sea_level=0.35,
            seed=4242,
            generator_version=2,
            world_scale=200.0,
            elevation=elevation,
            moisture=moisture,
            temperature=temperature,
            flow=flow,
            biome=biome,
        )

    def test_synthetic_roundtrip(self):
        orig = self._create_synthetic(8, 8)
        data = orig.to_bytes()

        assert len(data) == 36 + 8 * 8 * 17
        assert data[:4] == b"ANMW"

        decoded = WorldArtifact.from_bytes(data)
        assert decoded.width == orig.width
        assert decoded.height == orig.height
        assert pytest.approx(decoded.sea_level) == orig.sea_level
        assert decoded.seed == orig.seed
        assert decoded.generator_version == orig.generator_version
        assert pytest.approx(decoded.world_scale) == orig.world_scale

        assert np.array_equal(decoded.elevation, orig.elevation)
        assert np.array_equal(decoded.moisture, orig.moisture)
        assert np.array_equal(decoded.temperature, orig.temperature)
        assert np.array_equal(decoded.flow, orig.flow)
        assert np.array_equal(decoded.biome, orig.biome)
        assert decoded.checksum() == orig.checksum()

    def test_peek_seed(self):
        orig = self._create_synthetic(4, 4)
        data = orig.to_bytes()
        seed = WorldArtifact.peek_seed(data)
        assert seed == orig.seed
        assert seed == 4242

    def test_file_io(self):
        orig = self._create_synthetic(16, 16)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_artifact.anmw"
            orig.to_file(path)

            loaded = WorldArtifact.from_file(path)
            assert loaded.width == orig.width
            assert loaded.to_bytes() == orig.to_bytes()


# =============================================================================
# 5. Cross-Language Anima-Engine Fixtures
# =============================================================================
class TestCrossLanguageFixtures:
    @pytest.mark.skipif(not FIXTURE_4X4.exists(), reason="fixture world_4x4 not found")
    def test_world_4x4_fixture(self):
        raw = FIXTURE_4X4.read_bytes()
        artifact = WorldArtifact.from_bytes(raw)

        assert artifact.width == 4
        assert artifact.height == 4
        assert artifact.sea_level == 0.5
        assert artifact.seed == 1337
        assert artifact.generator_version == 20
        assert artifact.world_scale == 200.0
        assert artifact.checksum() == 0x153AB135

        # Strict byte-for-byte identity upon re-serialization
        assert artifact.to_bytes() == raw

    @pytest.mark.skipif(not FIXTURE_256.exists(), reason="fixture world_256 not found")
    def test_world_256_fixture(self):
        raw = FIXTURE_256.read_bytes()
        assert len(raw) == 1_114_148

        artifact = WorldArtifact.from_bytes(raw)
        assert artifact.width == 256
        assert artifact.height == 256
        assert pytest.approx(artifact.sea_level, abs=1e-4) == 0.31722
        assert artifact.seed == 1346747564
        assert artifact.checksum() == 0xBDE9D1F2

        # Strict byte-for-byte identity upon re-serialization
        re_encoded = artifact.to_bytes()
        assert len(re_encoded) == 1_114_148
        assert re_encoded == raw
        assert artifact.sha256() == "sha256:94641ead70b78ba07c62115129fa7aca6934c83464117e13871575d55b39d8d7"

    @pytest.mark.skipif(not FIXTURE_128.exists(), reason="fixture world_real_128 not found")
    def test_world_128_fixture(self):
        raw = FIXTURE_128.read_bytes()
        artifact = WorldArtifact.from_bytes(raw)

        assert artifact.width == 128
        assert artifact.height == 128
        assert artifact.to_bytes() == raw


# =============================================================================
# 6. Integrity & Error Handling
# =============================================================================
class TestIntegrityAndErrorHandling:
    def _get_valid_bytes(self) -> bytearray:
        n = 16
        a = WorldArtifact(
            width=4,
            height=4,
            sea_level=0.5,
            seed=100,
            generator_version=2,
            world_scale=200.0,
            elevation=np.zeros(n, dtype=np.float32),
            moisture=np.zeros(n, dtype=np.float32),
            temperature=np.zeros(n, dtype=np.float32),
            flow=np.zeros(n, dtype=np.float32),
            biome=np.zeros(n, dtype=np.uint8),
        )
        return bytearray(a.to_bytes())

    def test_reject_short_header(self):
        with pytest.raises(ValueError, match="too short"):
            WorldArtifact.from_bytes(b"ANMW")

    def test_reject_bad_magic(self):
        b = self._get_valid_bytes()
        b[0] = ord("X")
        with pytest.raises(ValueError, match="bad artifact magic"):
            WorldArtifact.from_bytes(bytes(b))

    def test_reject_unsupported_version(self):
        b = self._get_valid_bytes()
        # Set version to 1
        struct.pack_into("<I", b, 4, 1)
        with pytest.raises(ValueError, match="unsupported artifact version 1"):
            WorldArtifact.from_bytes(bytes(b))

    def test_reject_corrupt_payload_checksum(self):
        b = self._get_valid_bytes()
        # Corrupt one payload byte at the end
        b[-1] ^= 0xFF
        with pytest.raises(ValueError, match="checksum mismatch.*world data is corrupt"):
            WorldArtifact.from_bytes(bytes(b))

    def test_reject_truncated_payload(self):
        b = self._get_valid_bytes()
        truncated = bytes(b[:-10])
        with pytest.raises(ValueError, match="artifact truncated"):
            WorldArtifact.from_bytes(truncated)

    def test_reject_trailing_bytes(self):
        b = self._get_valid_bytes()
        padded = bytes(b) + b"\x00\x00\x00\x00"
        with pytest.raises(ValueError, match="4 trailing byte"):
            WorldArtifact.from_bytes(padded)


# =============================================================================
# 7. Heightfield Synthesis & Terrain Map Downsampling
# =============================================================================
class TestHeightfieldIntegration:
    def test_from_heightfield(self):
        hf = Heightfield2D(width=160.0, depth=160.0, resolution=64)
        hf.Z[10:30, 10:30] = 15.0
        artifact = WorldArtifact.from_heightfield(hf, seed=777)

        assert artifact.width == 64
        assert artifact.height == 64
        assert artifact.seed == 777
        assert len(artifact.elevation) == 64 * 64
        assert len(artifact.biome) == 64 * 64

        raw = artifact.to_bytes()
        decoded = WorldArtifact.from_bytes(raw)
        assert decoded.checksum() == artifact.checksum()

    def test_to_terrain_map(self):
        orig = TestWorldArtifactRoundtrip()._create_synthetic(8, 8)
        tm = orig.to_terrain_map(4, 4)

        assert tm["width"] == 4
        assert tm["height"] == 4
        assert len(tm["elevations"]) == 16
        assert len(tm["flows"]) == 16
        assert len(tm["biomes"]) == 16
        # All projected biomes must be in legacy 0..10
        assert all(b <= 10 for b in tm["biomes"])


# =============================================================================
# 8. MapManifest Generation & Validation
# =============================================================================
class TestMapManifest:
    @pytest.mark.skipif(not FIXTURE_256.exists(), reason="fixture world_256 not found")
    def test_manifest_generation_from_fixture(self):
        w256 = WorldArtifact.from_file(FIXTURE_256)
        manifest = generate_map_manifest(w256, artifact_rel_path="artifacts/world_256.anmw")

        # Schema and top-level fields
        assert manifest["schemaVersion"] == 1
        assert manifest["$schema"] == "./map_manifest.schema.json"

        # worldArtifact
        wa = manifest["worldArtifact"]
        assert wa["path"] == "artifacts/world_256.anmw"
        assert wa["magic"] == "ANMW"
        assert wa["version"] == 2
        assert wa["width"] == 256
        assert wa["height"] == 256
        assert wa["bytes"] == 1_114_148
        assert wa["checksum"] == "sha256:94641ead70b78ba07c62115129fa7aca6934c83464117e13871575d55b39d8d7"

        # coordinateSystem
        cs = manifest["coordinateSystem"]
        assert cs["worldMinXZ"] == -100
        assert cs["worldMaxXZ"] == 100
        assert cs["worldMinY"] == 0
        assert cs["worldMaxY"] == 10
        assert cs["gridDim"] == 256
        assert cs["worldScale"] == 200

        # biomeTaxonomy
        bt = manifest["biomeTaxonomy"]
        assert bt["canonicalCount"] == 22
        assert bt["legacyCount"] == 11

        # views
        views = manifest["views"]
        assert len(views) == 8
        view_ids = [v["id"] for v in views]
        assert view_ids == CANONICAL_VIEW_IDS

        # Validate against validator
        valid, errors = validate_map_manifest(manifest)
        assert valid, f"manifest validation failed: {errors}"

    def test_validator_rejects_missing_fields(self):
        bad_manifest = {"schemaVersion": 1}
        valid, errors = validate_map_manifest(bad_manifest)
        assert not valid
        assert any("worldArtifact" in e for e in errors)
        assert any("coordinateSystem" in e for e in errors)
        assert any("biomeTaxonomy" in e for e in errors)
        assert any("views" in e for e in errors)

    def test_save_and_load_manifest(self):
        w = TestWorldArtifactRoundtrip()._create_synthetic(8, 8)
        manifest = generate_map_manifest(w, artifact_rel_path="artifacts/world_8.anmw")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            save_map_manifest(manifest, path)
            loaded = load_map_manifest(path)
            assert loaded["schemaVersion"] == 1
            assert loaded["worldArtifact"]["width"] == 8


# =============================================================================
# 9. OpenWorldMesh Continuous Top-Surface Generator
# =============================================================================
class TestOpenWorldMesh:
    def test_mesh_construction(self):
        # 4x4 heightfield grid
        Z = np.zeros((4, 4), dtype=np.float32)
        Z[1:3, 1:3] = 5.0

        mesh = OpenWorldMesh(Z, world_min_xz=-100.0, world_max_xz=100.0, world_min_y=0.0, world_max_y=10.0)

        # Top surface only: exactly 16 vertices, NO diorama walls or bottom baseplate
        assert mesh.vertex_count == 16
        # 3x3 quads * 2 triangles/quad = 18 triangles = 54 index values
        assert mesh.triangle_count == 18
        assert len(mesh.indices) == 54

        # Bounds check
        assert mesh.aabb_min[0] >= -100.001
        assert mesh.aabb_max[0] <= 100.001
        assert mesh.aabb_min[1] >= 0.0
        assert mesh.aabb_max[1] <= 10.0

        # Normals check: normalized unit vectors
        norms = np.linalg.norm(mesh.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)

    def test_mesh_chunking_continuity(self):
        # 16x16 grid partitioned into 2x2 chunks
        Z = np.linspace(0.0, 10.0, 16 * 16).reshape((16, 16))
        mesh = OpenWorldMesh(Z)

        chunks = mesh.create_chunks(chunks_x=2, chunks_y=2)
        assert len(chunks) == 4

        # Check that total chunk triangles equals full mesh triangles
        total_tris = sum(c.triangle_count for c in chunks)
        assert total_tris == mesh.triangle_count

    def test_headless_export_obj_and_bin(self):
        Z = np.zeros((8, 8), dtype=np.float32)
        mesh = OpenWorldMesh(Z)

        with tempfile.TemporaryDirectory() as td:
            obj_path = Path(td) / "open_world.obj"
            bin_path = Path(td) / "elevation.bin"

            mesh.export_obj(obj_path)
            mesh.export_binary_heightfield(bin_path)

            assert obj_path.exists() and obj_path.stat().st_size > 0
            # Bin file must be exactly 8*8*4 = 256 bytes
            assert bin_path.stat().st_size == 8 * 8 * 4

    def test_headless_export_glb(self):
        Z = np.zeros((8, 8), dtype=np.float32)
        mesh = OpenWorldMesh(Z)

        with tempfile.TemporaryDirectory() as td:
            glb_path = Path(td) / "open_world.glb"
            mesh.export_glb(glb_path)

            assert glb_path.exists()
            data = glb_path.read_bytes()
            assert len(data) > 100

            # Inspect glTF binary header
            magic, ver, total_len = struct.unpack_from("<4sII", data, 0)
            assert magic == b"glTF"
            assert ver == 2
            assert total_len == len(data)

            # Inspect JSON chunk
            c0_len, c0_type = struct.unpack_from("<I4s", data, 12)
            assert c0_type == b"JSON"
            json_text = data[20 : 20 + c0_len].decode("utf-8")
            gltf_json = json.loads(json_text)

            assert gltf_json["asset"]["version"] == "2.0"
            assert len(gltf_json["meshes"]) == 1
            assert len(gltf_json["accessors"]) == 4
            assert len(gltf_json["bufferViews"]) == 4

            # Inspect BIN chunk
            c1_off = 20 + c0_len
            c1_len, c1_type = struct.unpack_from("<I4s", data, c1_off)
            assert c1_type == b"BIN\x00"
            assert c1_off + 8 + c1_len == total_len
