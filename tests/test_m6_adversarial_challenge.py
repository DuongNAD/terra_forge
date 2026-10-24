"""
Adversarial Stress Testing & Fuzzing Suite for Terra Forge (Milestone M6).
Authored by EMPIRICAL CHALLENGER (teamwork_preview_challenger_8_1).

Evaluates 4 core verification targets:
1. FNV-1a Checksum & WorldArtifact v2 Binary Layout Fuzzing:
   - 1-bit payload bit flips across 200 randomized locations (assert 100% caught).
   - Strict rejection of truncated buffers, bad magic ('ANM!'), unsupported versions, and trailing bytes.
2. Local Asset Vault Latency Stress Benchmark & Hash Determinism:
   - 5,000 consecutive lookups: asserts p99 latency < 50ms (target < 5ms).
   - SHA-256 hash determinism across 200 randomized key insertion permutations.
3. Primordial Prompt Hard Filter Fuzzing:
   - 65 adversarial human-biased prompts (subtle contractions, mixed cases, embedded forbidden tokens).
   - Tests zero-tolerance filter rejection and identifies any bypass vulnerabilities (e.g. snake_case / underscore boundaries).
4. Geometry Normalizer Pivot & Metric Scaling Verification:
   - 20 random procedural meshes: asserts min(y) == 0.0 strictly (tol 1e-6), horizontal center at (0, 0), and metric height matching target.
"""

from __future__ import annotations

import json
import random
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from terra_forge.ai.normalizer import GeometryNormalizer
from terra_forge.ai.vault import FloraType, LocalAssetVault, compute_asset_hash
from terra_forge.core.world_artifact import (
    CANONICAL_TOTAL_SIZE_256,
    HEADER_STRUCT,
    WORLD_ARTIFACT_HEADER_LEN,
    WORLD_ARTIFACT_MAGIC,
    WORLD_ARTIFACT_VERSION,
    WorldArtifact,
    fnv1a_32,
)
from terra_forge.ecology.primordial import (
    HumanBiasError,
    STRICT_POSITIVE_DESCRIPTORS,
    validate_primordial_prompt,
)


# =============================================================================
# 1. FNV-1a Checksum Fuzzing & Binary Layout Rejection Tests
# =============================================================================

class TestFNV1aChecksumFuzzing:
    """Stress tests binary layout and FNV-1a checksum resilience."""

    @pytest.fixture(scope="class")
    def canonical_artifact(self) -> Tuple[WorldArtifact, bytes]:
        dim = 256
        n = dim * dim
        rng = np.random.default_rng(42)
        elev = rng.uniform(0.0, 1.0, n).astype(np.float32)
        moist = rng.uniform(0.0, 1.0, n).astype(np.float32)
        temp = rng.uniform(0.0, 1.0, n).astype(np.float32)
        flow = rng.uniform(0.0, 1.0, n).astype(np.float32)
        biome = rng.integers(0, 22, n, dtype=np.uint8)

        art = WorldArtifact(dim, dim, 2.0, 12345, 1, 200.0, elev, moist, temp, flow, biome)
        raw = art.to_bytes()
        assert len(raw) == CANONICAL_TOTAL_SIZE_256
        return art, raw

    def test_payload_1bit_flip_fuzzing_100_percent_caught(self, canonical_artifact):
        """Verify that 200 randomized 1-bit flips in payload bytes are 100% caught by FNV-1a."""
        _art, raw = canonical_artifact
        rng = np.random.default_rng(9999)
        total_flips = 200
        caught = 0

        for _ in range(total_flips):
            corrupted = bytearray(raw)
            # Pick a random byte in the payload range [36, len(raw))
            byte_idx = int(rng.integers(WORLD_ARTIFACT_HEADER_LEN, len(raw)))
            bit_idx = int(rng.integers(0, 8))
            corrupted[byte_idx] ^= (1 << bit_idx)

            with pytest.raises(ValueError, match="artifact checksum mismatch"):
                WorldArtifact.from_bytes(bytes(corrupted))
            caught += 1

        assert caught == total_flips, f"Expected 100% caught, got {caught}/{total_flips}"

    @pytest.mark.parametrize(
        "trunc_len",
        [0, 10, 35, 36, 100, 500000, CANONICAL_TOTAL_SIZE_256 - 100, CANONICAL_TOTAL_SIZE_256 - 1],
    )
    def test_truncated_binary_strict_rejection(self, canonical_artifact, trunc_len):
        """Verify that any truncated binary raises ValueError."""
        _art, raw = canonical_artifact
        truncated = raw[:trunc_len]
        with pytest.raises(ValueError, match=r"(artifact too short for header|artifact truncated)"):
            WorldArtifact.from_bytes(truncated)

    @pytest.mark.parametrize(
        "bad_magic",
        [b"ANM!", b"ANMB", b"XNMW", b"ANIM", b"\x00\x00\x00\x00", b"\xff\xff\xff\xff"],
    )
    def test_bad_magic_strict_rejection(self, canonical_artifact, bad_magic):
        """Verify that invalid magic numbers raise ValueError('bad artifact magic')."""
        _art, raw = canonical_artifact
        corrupted = bad_magic + raw[4:]
        with pytest.raises(ValueError, match=r"bad artifact magic \(expected ANMW\)"):
            WorldArtifact.from_bytes(corrupted)

    @pytest.mark.parametrize("bad_version", [0, 1, 3, 4, 100, 4294967295])
    def test_unsupported_version_strict_rejection(self, canonical_artifact, bad_version):
        """Verify that versions != 2 raise ValueError('unsupported artifact version')."""
        _art, raw = canonical_artifact
        corrupted = bytearray(raw)
        struct.pack_into("<I", corrupted, 4, bad_version)
        with pytest.raises(ValueError, match=r"unsupported artifact version"):
            WorldArtifact.from_bytes(bytes(corrupted))

    @pytest.mark.parametrize("trailing_len", [1, 5, 1024])
    def test_trailing_bytes_strict_rejection(self, canonical_artifact, trailing_len):
        """Verify that any trailing bytes after payload raise ValueError('trailing byte\(s\)')."""
        _art, raw = canonical_artifact
        corrupted = raw + (b"\x00" * trailing_len)
        with pytest.raises(ValueError, match=r"trailing byte\(s\) after the payload"):
            WorldArtifact.from_bytes(corrupted)


# =============================================================================
# 2. Local Asset Vault Latency Stress Benchmark & Hash Determinism
# =============================================================================

class TestLocalAssetVaultBenchmark:
    """Stress tests vault retrieval latency and hash invariance."""

    @pytest.fixture
    def benchmark_vault(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="terra_forge_challenger_vault_"))
        vault = LocalAssetVault(vault_dir=tmp_dir)
        # Store 10 synthetic assets
        stored_hashes = []
        for i in range(10):
            entry = vault.store(
                prompt=f"primordial ancient specimen {i}",
                seed=1000 + i,
                model_glb=b"glTF_DUMMY_BENCHMARK_BYTES",
                flora_type=FloraType.Pine,
            )
            stored_hashes.append(entry.asset_id)
        yield vault, stored_hashes
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_consecutive_5000_lookups_latency(self, benchmark_vault):
        """Execute 5,000 consecutive vault lookups, assert p99 < 50ms and record percentiles."""
        vault, hashes = benchmark_vault
        rng = np.random.default_rng(42)
        latencies_ms: List[float] = []

        total_lookups = 5000
        for _ in range(total_lookups):
            target_hash = hashes[int(rng.integers(0, len(hashes)))]
            t0 = time.perf_counter()
            res = vault.get(target_hash)
            t1 = time.perf_counter()
            assert res is not None
            latencies_ms.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        p_max = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))

        print(
            f"\n[BENCHMARK] 5,000 Vault Lookups: p50={p50:.4f}ms, p90={p90:.4f}ms, "
            f"p95={p95:.4f}ms, p99={p99:.4f}ms, max={p_max:.4f}ms, mean={mean_lat:.4f}ms"
        )
        assert p99 < 50.0, f"p99 latency {p99:.4f}ms exceeded 50.0ms limit"

    def test_hash_determinism_across_permutations(self):
        """Verify that SHA-256 hash is invariant under 200 random JSON key permutations."""
        prompt = "ancient ironwood monolith"
        seed = 42
        base_dict: Dict[str, Any] = {
            "alpha": 1,
            "beta": "tree",
            "gamma": [1, 2, 3],
            "delta": {"sub1": 10, "sub2": 20},
            "epsilon": 3.1415,
            "zeta": True,
            "eta": None,
            "theta": "primordial",
            "iota": 100,
            "kappa": "evergreen",
        }

        keys = list(base_dict.keys())
        hashes = set()
        for _ in range(200):
            shuffled_keys = keys.copy()
            random.shuffle(shuffled_keys)
            permuted = {k: base_dict[k] for k in shuffled_keys}
            h = compute_asset_hash(prompt, seed, permuted)
            hashes.add(h)

        assert len(hashes) == 1, f"Expected 1 deterministic hash across permutations, got {len(hashes)}"

    def test_whitespace_and_case_normalization(self):
        """Verify prompt case and surrounding whitespace are normalized."""
        h1 = compute_asset_hash("  Ancient Ironwood Monolith  ", 42)
        h2 = compute_asset_hash("ancient ironwood monolith", 42)
        h3 = compute_asset_hash("ANCIENT IRONWOOD MONOLITH", 42)
        assert h1 == h2 == h3


# =============================================================================
# 3. Primordial Prompt Hard Filter Fuzzing (65 Adversarial Cases)
# =============================================================================

ADVERSARIAL_HUMAN_PROMPTS: List[str] = [
    # Subtle contractions & compound words
    "pre-sawed branches on ancient oak",
    "well-hewn timber in pine grove",
    "hand-carved totem beside stream",
    "stone-wall boundary in meadow",
    "paved-walkway remnants near mossy rocks",
    "saw-cut trunk fallen on slope",
    "man-made stone circle on plateau",
    "wood-plank crossing over creek",
    "iron-nailed joints on fallen snag",
    "torch-lit cave entrance in karst",
    
    # Mixed cases
    "PrE-sAwEd timber on hill",
    "ANCIENT_RUIN of an altar",
    "sToNe_FeNcE along the ridge",
    "wOoD pLaNk washed ashore",
    "HaNd-CaRvEd figure on stone",
    "mAn-MaDe monolith",
    "CoBbLeStOnE walkway near river",
    "bRiCk mortar foundation",
    "LuMbEr stack under canopy",
    "pLyWoOd shelter in valley",

    # Embedded forbidden tokens
    "stone_fence overgrown with wild vines",
    "ancient_ruin half buried in river silt",
    "wood plank stranded on pebble shore",
    "sawmark visible on fallen cedar",
    "a forgotten road leading through the gorge",
    "rusty bolt embedded in granite cliff",
    "weathered pier posts standing in shallow water",
    "abandoned log cabin rotting in swamp",
    "stone wall terrace along hillside",
    "old wooden bench under the willow",
    "bridge spanning the deep gorge",
    "fenced clearing in the taiga",
    "weathered brick scattered on rocky slope",
    "ancient statue head carved from limestone",
    "iron nail trapped in petrified wood",
    "asphalt seepage resembling a paved path",
    "dock remnants near the tidal lagoon",
    "primitive cart wheel half buried in mud",
    "faint trail blazed through dense brush",
    "chipped chisel mark on obsidian cliff",
    "hatchet blade resting on wet moss",
    "broken lantern glass in river gravel",
    "timberland harvested hillside",
    "scaffolding erected around the baobab",
    "cottage chimney crumbling in weeds",
    "wharf platform decaying in mangrove",
    "pedestal supporting a basalt pillar",
    "temple courtyard covered in ferns",
    "concrete slab cracked by giant roots",
    "gate latch dangling from weathered post",
    "domesticated rose bush in wild forest",
    "synthetic fibers caught in brambles",
    "manufactured iron rod in riverbed",
    "anthropogenic clearing in primeval canopy",
    "machined groove on basalt column",
    "rafters protruding from collapsed roof",
    "sidewalk edging along mountain stream",
    "mortar binding river stones",
    "shack roof decaying on forest floor",
    "wagon axle lodged between boulders",
    "candle wax dripped on granite shelf",
    "wire snare tangled in underbrush",
    "cut roots showing clean axe marks",
    "saws and axes abandoned in glen",
    "pre-cut lumber floating in lake",
    "carved stone marker at ridge crest",
    "paved courtyard overgrown with lichen",
    "post-and-rail boundary in meadow",
    "manmade water channel through bog",
    "artificial mound shaped by human hands",
    "hewn beam serving as footbridge",
    "ruined tower engulfed by creeping ivy",
    "subtle sawcut across fallen sequoia",
    "stone wall remnants bordering pasture",
    "weathered plank bridge over stream",
]


class TestPrimordialPromptFilterAdversarial:
    """Adversarial stress testing of Primordial Prompt human bias filter."""

    def test_adversarial_human_bias_prompts_rejection(self):
        """
        Evaluate 55+ adversarial prompts against validate_primordial_prompt.
        Records caught vs bypassed prompts empirically.
        """
        caught_count = 0
        bypassed_prompts: List[Tuple[str, str]] = []

        for p in ADVERSARIAL_HUMAN_PROMPTS:
            res = validate_primordial_prompt(p, strict=False)
            if not res.is_valid:
                caught_count += 1
            else:
                bypassed_prompts.append((p, "Allowed as valid primordial prompt"))

        total = len(ADVERSARIAL_HUMAN_PROMPTS)
        print(f"\n[ADVERSARIAL PROMPT FILTER] Caught: {caught_count}/{total} ({caught_count/total*100:.1f}%)")
        if bypassed_prompts:
            print(f"[VULNERABILITY FOUND] {len(bypassed_prompts)} prompts bypassed filter:")
            for bp, reason in bypassed_prompts:
                print(f"  - {bp!r} -> {reason}")

        # The orchestrator requirement is 100% rejection.
        # Any bypass indicates an empirical bug in tokenization / word boundaries.
        assert caught_count == total, (
            f"Adversarial filter failed! Bypassed {len(bypassed_prompts)}/{total} prompts: {bypassed_prompts}"
        )

    def test_control_group_pristine_prompts_pass(self):
        """Verify that genuine primordial descriptors pass with zero false positives."""
        pristine_samples = [
            desc
            for cat in STRICT_POSITIVE_DESCRIPTORS.values()
            for desc in cat[:3]
        ]
        for prompt in pristine_samples:
            res = validate_primordial_prompt(prompt, strict=True)
            assert res.is_valid is True
            assert len(res.detected_tokens) == 0


# =============================================================================
# 4. Geometry Normalizer Pivot & Invariant Verification
# =============================================================================

class TestGeometryNormalizerPivot:
    """Empirical verification of pivot alignment (min_y == 0.0) on 20 procedural meshes."""

    def test_20_random_procedural_meshes_normalization(self):
        normalizer = GeometryNormalizer()
        rng = np.random.default_rng(54321)

        for i in range(20):
            n_pts = int(rng.integers(30, 1500))
            # Vary bounding dimensions and offsets
            pts = rng.uniform(-40.0, 40.0, size=(n_pts, 3)).astype(np.float32)
            # Apply vertical offset and scaling
            pts[:, 1] += rng.uniform(-150.0, 150.0)
            target_h = float(rng.uniform(1.0, 25.0))

            norm_v, stats = normalizer.normalize_vertices(pts, target_height=target_h)

            min_y = float(np.min(norm_v[:, 1]))
            c_x = float((np.min(norm_v[:, 0]) + np.max(norm_v[:, 0])) / 2.0)
            c_z = float((np.min(norm_v[:, 2]) + np.max(norm_v[:, 2])) / 2.0)
            measured_h = float(np.max(norm_v[:, 1]) - np.min(norm_v[:, 1]))

            # Invariant 1: min(y) == 0.0 strictly (tolerance 10^-6)
            assert abs(min_y) < 1e-6, f"Mesh {i:02d}: min_y={min_y:.2e} violated tolerance 1e-6"

            # Invariant 2: horizontal center is at (0, 0) strictly (tolerance 1e-6)
            assert abs(c_x) < 1e-6, f"Mesh {i:02d}: center_x={c_x:.2e} violated tolerance 1e-6"
            assert abs(c_z) < 1e-6, f"Mesh {i:02d}: center_z={c_z:.2e} violated tolerance 1e-6"

            # Invariant 3: metric height matches target (tolerance 1e-5)
            assert abs(measured_h - target_h) < 1e-5, (
                f"Mesh {i:02d}: measured height {measured_h:.5f} != target {target_h:.5f}"
            )

            # Invariant 4: collision primitives extraction
            colliders = normalizer.extract_collision_primitives(norm_v, target_height=target_h)
            assert colliders.cylinder.radius > 0.0
            assert colliders.cylinder.height > 0.0
            assert abs(colliders.aabb.min_vec[1]) < 1e-4
            assert abs(colliders.aabb.size[1] - target_h) < 1e-4
            assert colliders.convex_hull.volume > 0.0
            assert colliders.convex_hull.area > 0.0
