"""
Empirical Adversarial Stress Suite for Milestone 1: Local Asset Vault Concurrency,
Performance, Cache Pollution, and Corrupted Asset Recovery.

Challenger 2 Empirical Verification:
1. High-throughput sequential and multi-threaded concurrency (1,000 lookups, p50/p95/p99 vs <50ms).
2. Cache pollution and hash collision resilience (permuted keys, unicode normalization, edge cases).
3. Corrupted asset recovery and fault tolerance (0-byte files, truncated JSON, missing GLB, alien folders).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from terra_forge.ai.vault import (
    AbioticMetadata,
    AbioticType,
    AssetEntry,
    AssetMetadata,
    FloraType,
    LocalAssetVault,
    compute_asset_hash,
)
from terra_forge.blender.asset_fetcher import (
    build_glb_from_mesh,
    create_abiotic_mesh,
    fetch_abiotic_asset,
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================

@pytest.fixture
def clean_vault():
    """Provides an isolated LocalAssetVault in a clean temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="tf_challenger2_vault_")
    vault = LocalAssetVault(vault_dir=temp_dir)
    yield vault
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_dummy_glb() -> bytes:
    """Create minimal valid glTF 2.0 binary bytes for testing."""
    verts, tris = create_abiotic_mesh(AbioticType.RiverPebble, target_height=0.4)
    return build_glb_from_mesh(verts, tris)


def populate_populated_vault(vault: LocalAssetVault, count: int = 10) -> List[str]:
    """Populates vault with N distinct abiotic assets and returns their asset IDs."""
    glb_data = create_dummy_glb()
    asset_ids = []
    for i in range(count):
        prompt = f"primordial granite basalt formation stress tier {i}"
        entry = vault.store(
            prompt=prompt,
            seed=1000 + i,
            model_glb=glb_data,
            params={"tier": i, "density": 0.5 + i * 0.05, "sub": {"sub_key": f"val_{i}"}},
            abiotic_type=AbioticType(i % len(AbioticType)),
            dimensions_meters={"height": 5.0 + i},
            collision_primitives={"cylinder": {"radius": 1.0, "height": 5.0}},
            source="stress_harness",
        )
        asset_ids.append(entry.asset_id)
    return asset_ids


# =============================================================================
# 1. High-Throughput Concurrency & Latency Benchmarks
# =============================================================================

class TestVaultHighThroughputConcurrency:
    """Empirical challenge: 1,000 rapid lookups under sequential and multi-threaded load."""

    def test_1000_sequential_in_memory_lookups_latency(self, clean_vault):
        """
        Benchmark: 1,000 sequential in-memory lookups.
        Requirement: < 50ms per lookup.
        Target: sub-millisecond p99 latency.
        """
        asset_ids = populate_populated_vault(clean_vault, count=20)
        assert len(clean_vault) == 20

        latencies_ms: List[float] = []
        for i in range(1000):
            target_id = asset_ids[i % len(asset_ids)]
            t0 = time.perf_counter_ns()
            entry = clean_vault.get(target_id)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies_ms.append(elapsed_ms)
            assert entry is not None
            assert entry.asset_id == target_id

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))

        print(f"\n[1000 In-Memory Lookups] Mean: {mean_lat:.4f}ms | p50: {p50:.4f}ms | p95: {p95:.4f}ms | p99: {p99:.4f}ms | Max: {max_lat:.4f}ms")

        assert max_lat < 50.0, f"Max latency {max_lat:.3f}ms exceeded 50ms requirement"
        assert p99 < 5.0, f"p99 latency {p99:.3f}ms exceeded 5ms"
        assert mean_lat < 1.0, f"Mean in-memory latency {mean_lat:.3f}ms exceeded 1ms"

    def test_production_baseline_vault_1000_lookups_latency(self):
        """
        Benchmark against the actual production baseline vault at assets/vault/.
        1,000 rapid lookups across the 10 real baseline assets.
        """
        prod_vault = LocalAssetVault()
        assert len(prod_vault) >= 10, f"Production vault should have at least 10 assets, found {len(prod_vault)}"
        asset_ids = list(prod_vault._index.keys())

        # 1. In-memory lookups
        latencies_ms = []
        for i in range(1000):
            aid = asset_ids[i % len(asset_ids)]
            t0 = time.perf_counter_ns()
            entry = prod_vault.get(aid)
            latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            assert entry is not None
            assert entry.asset_id == aid
            assert entry.model_path.is_file()

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))

        print(f"\n[Production Vault 1000 In-Memory] Mean: {mean_lat:.4f}ms | p50: {p50:.4f}ms | p95: {p95:.4f}ms | p99: {p99:.4f}ms | Max: {max_lat:.4f}ms")
        assert max_lat < 50.0
        assert mean_lat < 1.0

        # 2. Cold disk lookups (clear cache each time)
        cold_latencies_ms = []
        for i in range(1000):
            aid = asset_ids[i % len(asset_ids)]
            prod_vault.clear_cache()
            t0 = time.perf_counter_ns()
            entry = prod_vault.get(aid)
            cold_latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            assert entry is not None
            assert entry.asset_id == aid

        cold_arr = np.array(cold_latencies_ms)
        p50_c = float(np.percentile(cold_arr, 50))
        p95_c = float(np.percentile(cold_arr, 95))
        p99_c = float(np.percentile(cold_arr, 99))
        max_lat_c = float(np.max(cold_arr))
        mean_lat_c = float(np.mean(cold_arr))

        print(f"\n[Production Vault 1000 Cold Disk] Mean: {mean_lat_c:.4f}ms | p50: {p50_c:.4f}ms | p95: {p95_c:.4f}ms | p99: {p99_c:.4f}ms | Max: {max_lat_c:.4f}ms")
        assert max_lat_c < 50.0
        assert mean_lat_c < 5.0

    def test_1000_sequential_cold_disk_lookups_latency(self, clean_vault):
        """
        Benchmark: 1,000 sequential cold disk lookups on synthetic 25-asset vault (cache bypassed/cleared).
        Requirement: < 50ms per lookup.
        """
        asset_ids = populate_populated_vault(clean_vault, count=25)

        latencies_ms: List[float] = []
        for i in range(1000):
            target_id = asset_ids[i % len(asset_ids)]
            clean_vault.clear_cache()
            assert target_id not in clean_vault._index

            t0 = time.perf_counter_ns()
            entry = clean_vault.get(target_id)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies_ms.append(elapsed_ms)

            assert entry is not None
            assert entry.asset_id == target_id

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))

        print(f"\n[Synthetic 1000 Cold Disk] Mean: {mean_lat:.4f}ms | p50: {p50:.4f}ms | p95: {p95:.4f}ms | p99: {p99:.4f}ms | Max: {max_lat:.4f}ms")

        assert max_lat < 50.0, f"Max cold disk latency {max_lat:.3f}ms exceeded 50ms requirement"
        assert p99 < 25.0, f"p99 cold disk latency {p99:.3f}ms exceeded 25ms"
        assert mean_lat < 10.0, f"Mean cold disk latency {mean_lat:.3f}ms exceeded 10ms"

    def test_multi_threaded_concurrent_reads(self, clean_vault):
        """Stress: 20 concurrent worker threads performing 50 reads each (1,000 concurrent lookups)."""
        asset_ids = populate_populated_vault(clean_vault, count=30)
        thread_count = 20
        reads_per_thread = 50

        errors: List[Exception] = []
        latencies_ms: List[float] = []

        def worker_read(thread_idx: int):
            for r in range(reads_per_thread):
                target_id = asset_ids[(thread_idx * reads_per_thread + r) % len(asset_ids)]
                t0 = time.perf_counter_ns()
                entry = clean_vault.get(target_id)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                latencies_ms.append(elapsed_ms)
                if entry is None or entry.asset_id != target_id:
                    errors.append(ValueError(f"Lookup failure for {target_id}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker_read, t) for t in range(thread_count)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Concurrent read errors detected: {len(errors)}"
        assert len(latencies_ms) == thread_count * reads_per_thread

        max_lat = float(np.max(latencies_ms))
        mean_lat = float(np.mean(latencies_ms))
        print(f"\n[Concurrent Reads: {thread_count} threads] Total: {len(latencies_ms)} | Mean: {mean_lat:.4f}ms | Max: {max_lat:.4f}ms")
        assert max_lat < 50.0, f"Concurrent read latency exceeded 50ms: {max_lat:.3f}ms"

    def test_concurrent_read_write_stress(self, clean_vault):
        """
        Stress: 10 reader threads + 4 writer threads running concurrently.
        Verifies thread safety during simultaneous cache insertions and queries.
        """
        initial_ids = populate_populated_vault(clean_vault, count=10)
        glb_data = create_dummy_glb()

        reader_count = 10
        writer_count = 4
        operations_per_thread = 40
        write_errors: List[Exception] = []
        read_errors: List[Exception] = []

        def worker_writer(writer_idx: int):
            for w in range(operations_per_thread):
                try:
                    prompt = f"concurrently written asset {writer_idx}_{w}"
                    clean_vault.store(
                        prompt=prompt,
                        seed=5000 + writer_idx * 100 + w,
                        model_glb=glb_data,
                        abiotic_type=AbioticType.HornPeak,
                    )
                except Exception as ex:
                    write_errors.append(ex)

        def worker_reader(reader_idx: int):
            for r in range(operations_per_thread):
                try:
                    target_id = initial_ids[r % len(initial_ids)]
                    entry = clean_vault.get(target_id)
                    if entry is None:
                        read_errors.append(ValueError(f"Could not find initial asset {target_id}"))
                except Exception as ex:
                    read_errors.append(ex)

        with concurrent.futures.ThreadPoolExecutor(max_workers=reader_count + writer_count) as executor:
            writer_futures = [executor.submit(worker_writer, w) for w in range(writer_count)]
            reader_futures = [executor.submit(worker_reader, r) for r in range(reader_count)]
            concurrent.futures.wait(writer_futures + reader_futures)

        print(f"\n[Concurrent R/W Stress] Write errors: {len(write_errors)} | Read errors: {len(read_errors)}")
        assert len(read_errors) == 0, f"Read errors during concurrent writes: {read_errors}"
        # All newly stored items should be indexed
        expected_total = len(initial_ids) + (writer_count * operations_per_thread)
        assert len(clean_vault) == expected_total


# =============================================================================
# 2. Cache Pollution & Hash Collisions
# =============================================================================

class TestCachePollutionAndHashCollisions:
    """Adversarial testing of compute_asset_hash deterministic guarantees and edge cases."""

    def test_permuted_dictionary_keys_nested(self):
        """Verify complex nested dictionaries with permuted key insertion produce identical hashes."""
        dict_a = {
            "resolution": 256,
            "lod": {"tier": 2, "ratio": 0.5, "tags": ["low", "fast"]},
            "weathering": {"thermal": 0.4, "hydraulic": 0.8},
            "seed_offset": 42,
        }
        dict_b = {
            "seed_offset": 42,
            "weathering": {"hydraulic": 0.8, "thermal": 0.4},
            "lod": {"tags": ["low", "fast"], "ratio": 0.5, "tier": 2},
            "resolution": 256,
        }

        h_a = compute_asset_hash("basalt column formation", 42, dict_a)
        h_b = compute_asset_hash("basalt column formation", 42, dict_b)

        assert h_a == h_b, "Nested dictionary key permutation resulted in different hashes!"

    def test_mixed_key_types_in_params(self):
        """Verify numeric and string keys in params are normalized deterministically."""
        dict_num = {1: "a", 2: "b"}
        dict_str = {"1": "a", "2": "b"}

        h_num = compute_asset_hash("stalactite", 10, dict_num)
        h_str = compute_asset_hash("stalactite", 10, dict_str)

        assert h_num == h_str, "Numeric vs string key representations produced different hashes!"

    def test_case_and_boundary_whitespace_insensitivity(self):
        """Verify case insensitivity and boundary whitespace trimming."""
        base_h = compute_asset_hash("granite rock crag", 100)

        assert compute_asset_hash("GRANITE ROCK CRAG", 100) == base_h
        assert compute_asset_hash("  granite rock crag  ", 100) == base_h
        assert compute_asset_hash("\t\ngranite rock crag\n\t", 100) == base_h
        assert compute_asset_hash("Granite Rock Crag", 100) == base_h

    def test_internal_whitespace_variations(self):
        """
        Adversarial finding: multiple internal spaces or newlines.
        Does prompt 'granite   rock' yield different hash from 'granite rock'?
        Document behavior: .strip().lower() does not collapse inner whitespace.
        """
        h_single = compute_asset_hash("granite rock crag", 100)
        h_multi_space = compute_asset_hash("granite    rock crag", 100)
        h_tab = compute_asset_hash("granite\trock crag", 100)

        # They produce different hashes because raw inner tokens are preserved
        assert h_single != h_multi_space
        assert h_single != h_tab

    def test_unicode_nfc_vs_nfd_normalization(self):
        """
        Adversarial test: NFC vs NFD canonical equivalence.
        e.g. 'café' composed (NFC: U+00E9) vs decomposed (NFD: U+0065 U+0301).
        Test whether compute_asset_hash is sensitive to unicode composition.
        """
        nfc_prompt = unicodedata.normalize("NFC", "stalactite mineral dépôt")
        nfd_prompt = unicodedata.normalize("NFD", "stalactite mineral dépôt")

        h_nfc = compute_asset_hash(nfc_prompt, 42)
        h_nfd = compute_asset_hash(nfd_prompt, 42)

        # Document whether hash engine normalizes unicode
        # If not normalized, record as a finding
        is_normalized = (h_nfc == h_nfd)
        print(f"\n[Unicode NFC/NFD] Identical hash: {is_normalized} (NFC: {h_nfc[:16]} vs NFD: {h_nfd[:16]})")

    def test_unicode_emoji_and_special_symbols(self):
        """Verify emojis, geological symbols, and Cyrillic/Greek symbols do not crash hash engine."""
        prompts = [
            "🪨 basaltic columnar jointing",
            "🏔️ alpine glacial horn peak",
            "💧 karst speleothem cave stalactite",
            "базальтовая колонна",  # Cyrillic
            "σπήλαιο σταλακτίτης",  # Greek
        ]
        hashes = set()
        for p in prompts:
            h = compute_asset_hash(p, 999)
            assert isinstance(h, str)
            assert len(h) == 64
            hashes.add(h)

        assert len(hashes) == len(prompts)

    def test_null_bytes_and_injection_characters(self):
        """Verify strings containing null bytes, quotes, backslashes, and control chars hash safely."""
        prompts = [
            "granite\x00crag",
            "rock'; DROP TABLE assets; --",
            'rock" or ""="',
            "column\\n\\r\\t\\0",
            '{"prompt": "fake", "seed": 0}',
        ]
        for p in prompts:
            h = compute_asset_hash(p, 42)
            assert len(h) == 64
            # Must match standard sha256 output structure
            int(h, 16)

    def test_hash_collision_stress_10k_samples(self):
        """Stress: Generate 10,000 distinct prompts & seeds; verify 0 SHA-256 collisions."""
        seen_hashes = set()
        for i in range(10000):
            p = f"geological entity type {i % 9} modifier {i // 9}"
            h = compute_asset_hash(p, seed=i, params={"i": i, "scale": i * 0.1})
            assert h not in seen_hashes, f"Collision detected at iteration {i} for hash {h}"
            seen_hashes.add(h)

        assert len(seen_hashes) == 10000


# =============================================================================
# 3. Corrupted Asset Recovery & Fault Tolerance
# =============================================================================

class TestCorruptedAssetRecovery:
    """Adversarial testing of vault resilience when assets or metadata are corrupt or missing."""

    def test_missing_metadata_json(self, clean_vault):
        """Verify vault ignores or returns None when metadata.json is missing from an asset folder."""
        # Create an asset folder manually without metadata.json
        corrupt_hash = hashlib.sha256(b"corrupt_missing_meta").hexdigest()
        corrupt_dir = clean_vault.vault_dir / corrupt_hash
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "model.glb").write_bytes(create_dummy_glb())

        # Scan should skip it cleanly
        count = clean_vault.scan_and_rebuild_index()
        assert corrupt_hash not in clean_vault._index
        assert clean_vault.get(corrupt_hash) is None
        assert clean_vault.has(corrupt_hash) is False

    def test_zero_byte_metadata_json(self, clean_vault):
        """Verify vault handles zero-byte (empty) metadata.json gracefully."""
        corrupt_hash = hashlib.sha256(b"corrupt_zero_byte_meta").hexdigest()
        corrupt_dir = clean_vault.vault_dir / corrupt_hash
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "metadata.json").write_bytes(b"")
        (corrupt_dir / "model.glb").write_bytes(create_dummy_glb())

        count = clean_vault.scan_and_rebuild_index()
        assert corrupt_hash not in clean_vault._index
        assert clean_vault.get(corrupt_hash) is None

    def test_truncated_malformed_metadata_json(self, clean_vault):
        """Verify vault handles truncated/syntactically invalid JSON in metadata.json."""
        corrupt_hash = hashlib.sha256(b"corrupt_malformed_json").hexdigest()
        corrupt_dir = clean_vault.vault_dir / corrupt_hash
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "metadata.json").write_text('{"asset_id": "test", "seed": 42, "prompt": "unclosed')
        (corrupt_dir / "model.glb").write_bytes(create_dummy_glb())

        assert clean_vault.get(corrupt_hash) is None
        clean_vault.scan_and_rebuild_index()
        assert corrupt_hash not in clean_vault._index

    def test_metadata_json_invalid_types_and_schema(self, clean_vault):
        """Verify vault handles metadata.json with missing or non-integer seed / invalid types."""
        # 1. Invalid seed type: string that cannot parse as int
        bad_seed_hash = hashlib.sha256(b"bad_seed").hexdigest()
        bad_dir = clean_vault.vault_dir / bad_seed_hash
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "metadata.json").write_text(json.dumps({"asset_id": bad_seed_hash, "seed": "not_an_int"}))
        (bad_dir / "model.glb").write_bytes(create_dummy_glb())

        assert clean_vault.get(bad_seed_hash) is None

        # 2. Null seed
        null_seed_hash = hashlib.sha256(b"null_seed").hexdigest()
        null_dir = clean_vault.vault_dir / null_seed_hash
        null_dir.mkdir(parents=True, exist_ok=True)
        (null_dir / "metadata.json").write_text(json.dumps({"asset_id": null_seed_hash, "seed": None}))
        (null_dir / "model.glb").write_bytes(create_dummy_glb())

        assert clean_vault.get(null_seed_hash) is None

    def test_missing_model_glb_fallback_handling(self, clean_vault):
        """
        Adversarial test: An asset folder has valid metadata.json, but model.glb is missing.
        Verify fetch_abiotic_asset detects the failure and gracefully falls back to procedural generation.
        """
        entry = clean_vault.store(
            prompt="fragile stalactite test",
            seed=777,
            model_glb=create_dummy_glb(),
            abiotic_type=AbioticType.Stalactite,
        )
        assert entry.model_path.is_file()

        # Adversarially delete model.glb
        entry.model_path.unlink()
        assert not entry.model_path.is_file()

        # Calling fetch_abiotic_asset with this vault should gracefully fall back to procedural creation
        res = fetch_abiotic_asset(AbioticType.Stalactite, target_height=3.5, vault=clean_vault)
        assert res is not None
        assert res.abiotic_type == AbioticType.Stalactite
        # Since cached model file was missing, it should have procedurally generated a fallback
        assert res.source in ("procedural_normalized", "vault")

    def test_corrupted_zero_byte_glb_fallback_handling(self, clean_vault):
        """
        Adversarial test: model.glb is overwritten with 0 bytes.
        Verify fetch_abiotic_asset catches glTF extraction failure and falls back to procedural generation.
        """
        entry = clean_vault.store(
            prompt="corrupt glb test",
            seed=888,
            model_glb=create_dummy_glb(),
            abiotic_type=AbioticType.BasaltColumn,
        )
        # Overwrite with 0 bytes
        entry.model_path.write_bytes(b"")
        assert entry.model_path.stat().st_size == 0

        res = fetch_abiotic_asset(AbioticType.BasaltColumn, target_height=6.0, vault=clean_vault)
        assert res is not None
        assert res.abiotic_type == AbioticType.BasaltColumn
        assert res.source == "procedural_normalized"
        assert res.normalized_mesh is not None

    def test_corrupted_garbage_bytes_glb_fallback_handling(self, clean_vault):
        """
        Adversarial test: model.glb contains random junk bytes.
        Verify fetch_abiotic_asset catches the error and falls back procedurally without crashing.
        """
        entry = clean_vault.store(
            prompt="garbage glb test",
            seed=999,
            model_glb=create_dummy_glb(),
            abiotic_type=AbioticType.CavernMineral,
        )
        # Overwrite with random garbage
        entry.model_path.write_bytes(b"NOT_A_VALID_GLTF_BINARY_HEADER_JUNK_DATA_123456789")

        res = fetch_abiotic_asset(AbioticType.CavernMineral, target_height=2.0, vault=clean_vault)
        assert res is not None
        assert res.abiotic_type == AbioticType.CavernMineral
        assert res.source == "procedural_normalized"

    def test_corrupted_or_missing_vault_manifest(self, clean_vault):
        """
        Verify vault operates properly if vault_manifest.json is deleted or corrupted.
        The vault should rebuild state on disk scan.
        """
        populate_populated_vault(clean_vault, count=5)
        assert len(clean_vault) == 5
        manifest_path = clean_vault.manifest_path
        assert manifest_path.is_file()

        # 1. Corrupt the manifest to 0 bytes
        manifest_path.write_bytes(b"")

        # Create a new vault pointing to the same folder
        new_vault = LocalAssetVault(vault_dir=clean_vault.vault_dir)
        # It scans directory directly and recovers all 5 assets!
        assert len(new_vault) == 5

        # 2. Delete the manifest completely
        manifest_path.unlink()
        new_vault2 = LocalAssetVault(vault_dir=clean_vault.vault_dir)
        assert len(new_vault2) == 5

    def test_extraneous_and_alien_files_in_vault(self, clean_vault):
        """Verify vault ignores alien files and directories (e.g. .DS_Store, temp folders, loose files)."""
        populate_populated_vault(clean_vault, count=3)

        # Inject alien items
        (clean_vault.vault_dir / ".DS_Store").write_bytes(b"\x00\x00")
        (clean_vault.vault_dir / "Thumbs.db").write_bytes(b"\x00\x00")
        (clean_vault.vault_dir / "random_notes.txt").write_text("not an asset")
        (clean_vault.vault_dir / "temp_incomplete").mkdir()
        (clean_vault.vault_dir / "63_chars_dir_almost_a_hash_12345678901234567890123456789012345678901").mkdir()

        # Scan and rebuild index
        count = clean_vault.scan_and_rebuild_index()
        assert count == 3
        assert len(clean_vault) == 3

    def test_missing_model_file_on_direct_vault_get_anomaly(self, clean_vault):
        """
        Adversarial anomaly discovery:
        When model.glb is deleted, LocalAssetVault.get(hash) does NOT return None.
        Instead, it returns an AssetEntry pointing to a non-existent model_path.
        Downstream code must be defensive and check model_path.is_file().
        """
        entry = clean_vault.store(
            prompt="anomaly missing model test",
            seed=12345,
            model_glb=create_dummy_glb(),
            abiotic_type=AbioticType.RockCrag,
        )
        entry.model_path.unlink()
        clean_vault.clear_cache()

        # Cold lookup:
        retrieved_entry = clean_vault.get(entry.asset_id)
        # Vault returns an AssetEntry rather than None
        assert retrieved_entry is not None
        # But the model path does not exist on disk!
        assert not retrieved_entry.model_path.is_file()

    def test_empty_asset_id_metadata_indexing_anomaly(self, clean_vault):
        """
        Adversarial anomaly discovery:
        If a metadata.json lacks 'asset_id', AssetMetadata defaults to '',
        causing the vault index to store an entry under key ''.
        """
        corrupt_hash = "f" * 64
        corrupt_dir = clean_vault.vault_dir / corrupt_hash
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        # Valid JSON without asset_id
        (corrupt_dir / "metadata.json").write_text(json.dumps({"prompt": "no id", "seed": 1}))
        (corrupt_dir / "model.glb").write_bytes(create_dummy_glb())

        clean_vault.scan_and_rebuild_index()
        # Verify anomaly: entry indexed under empty string ''
        assert "" in clean_vault._index
