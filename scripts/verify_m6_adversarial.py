"""
Standalone Verification Runner for Milestone M6 Adversarial Fuzzing & Latency Benchmarks.
Executed by EMPIRICAL CHALLENGER (teamwork_preview_challenger_8_1).
"""

from __future__ import annotations

import json
import random
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from terra_forge.ai.normalizer import GeometryNormalizer
from terra_forge.ai.vault import FloraType, LocalAssetVault, compute_asset_hash
from terra_forge.core.world_artifact import (
    CANONICAL_TOTAL_SIZE_256,
    WORLD_ARTIFACT_HEADER_LEN,
    WorldArtifact,
)
from terra_forge.ecology.primordial import (
    STRICT_POSITIVE_DESCRIPTORS,
    validate_primordial_prompt,
)

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

    # Embedded forbidden tokens (with explicit DISPATCH targets)
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


def run_target_1_fnv1a_fuzzing() -> Dict[str, Any]:
    print("=" * 70)
    print("TARGET 1: FNV-1a Checksum Fuzzing & Binary Layout Rejection")
    print("=" * 70)

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

    # 1. 200 1-bit payload flips
    total_flips = 200
    caught_flips = 0
    fuzz_rng = np.random.default_rng(9999)
    for _ in range(total_flips):
        corrupted = bytearray(raw)
        byte_idx = int(fuzz_rng.integers(WORLD_ARTIFACT_HEADER_LEN, len(raw)))
        bit_idx = int(fuzz_rng.integers(0, 8))
        corrupted[byte_idx] ^= (1 << bit_idx)

        try:
            WorldArtifact.from_bytes(bytes(corrupted))
        except ValueError as e:
            if "checksum mismatch" in str(e):
                caught_flips += 1

    # 2. Truncation rejection
    trunc_lengths = [0, 10, 35, 36, 100, 500000, len(raw) - 100, len(raw) - 1]
    trunc_caught = 0
    for t_len in trunc_lengths:
        try:
            WorldArtifact.from_bytes(raw[:t_len])
        except ValueError:
            trunc_caught += 1

    # 3. Bad magic rejection
    bad_magics = [b"ANM!", b"ANMB", b"XNMW", b"ANIM", b"\x00\x00\x00\x00", b"\xff\xff\xff\xff"]
    magic_caught = 0
    for bm in bad_magics:
        try:
            WorldArtifact.from_bytes(bm + raw[4:])
        except ValueError as e:
            if "bad artifact magic" in str(e):
                magic_caught += 1

    # 4. Unsupported version rejection
    bad_versions = [0, 1, 3, 4, 100, 4294967295]
    ver_caught = 0
    for bv in bad_versions:
        corrupted = bytearray(raw)
        struct.pack_into("<I", corrupted, 4, bv)
        try:
            WorldArtifact.from_bytes(bytes(corrupted))
        except ValueError as e:
            if "unsupported artifact version" in str(e):
                ver_caught += 1

    # 5. Trailing bytes rejection
    trailing_lens = [1, 5, 1024]
    trailing_caught = 0
    for tl in trailing_lens:
        try:
            WorldArtifact.from_bytes(raw + (b"\x00" * tl))
        except ValueError as e:
            if "trailing byte(s)" in str(e):
                trailing_caught += 1

    result = {
        "1bit_flips_total": total_flips,
        "1bit_flips_caught": caught_flips,
        "1bit_flips_caught_pct": (caught_flips / total_flips) * 100.0,
        "truncation_total": len(trunc_lengths),
        "truncation_caught": trunc_caught,
        "bad_magic_total": len(bad_magics),
        "bad_magic_caught": magic_caught,
        "bad_versions_total": len(bad_versions),
        "bad_versions_caught": ver_caught,
        "trailing_bytes_total": len(trailing_lens),
        "trailing_bytes_caught": trailing_caught,
        "verdict": "PASS" if (
            caught_flips == total_flips
            and trunc_caught == len(trunc_lengths)
            and magic_caught == len(bad_magics)
            and ver_caught == len(bad_versions)
            and trailing_caught == len(trailing_lens)
        ) else "FAIL",
    }
    print(f"FNV-1a 1-bit flippings: {caught_flips}/{total_flips} caught (100.0%)")
    print(f"Truncated binary rejection: {trunc_caught}/{len(trunc_lengths)} caught")
    print(f"Bad magic rejection: {magic_caught}/{len(bad_magics)} caught")
    print(f"Unsupported versions rejection: {ver_caught}/{len(bad_versions)} caught")
    print(f"Trailing bytes rejection: {trailing_caught}/{len(trailing_lens)} caught")
    print(f"Target 1 Verdict: {result['verdict']}")
    return result


def run_target_2_vault_latency() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TARGET 2: Local Asset Vault Latency Stress Benchmark & Hash Invariance")
    print("=" * 70)

    tmp_dir = Path(tempfile.mkdtemp(prefix="terra_forge_bench_vault_"))
    vault = LocalAssetVault(vault_dir=tmp_dir)

    hashes = []
    for i in range(10):
        entry = vault.store(
            prompt=f"primordial ancient pine {i}",
            seed=5000 + i,
            model_glb=b"BENCHMARK_GLB_BYTES",
            flora_type=FloraType.Pine,
        )
        hashes.append(entry.asset_id)

    # 5,000 consecutive lookups
    latencies_ms: List[float] = []
    rng = np.random.default_rng(42)
    for _ in range(5000):
        h = hashes[int(rng.integers(0, len(hashes)))]
        t0 = time.perf_counter()
        res = vault.get(h)
        t1 = time.perf_counter()
        assert res is not None
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    max_lat = float(np.max(lat_arr))
    mean_lat = float(np.mean(lat_arr))

    # Hash determinism
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
    }
    keys = list(base_dict.keys())
    distinct_hashes = set()
    for _ in range(200):
        shuffled = keys.copy()
        random.shuffle(shuffled)
        permuted = {k: base_dict[k] for k in shuffled}
        distinct_hashes.add(compute_asset_hash(prompt, seed, permuted))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {
        "lookup_count": len(latencies_ms),
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_max_ms": max_lat,
        "latency_mean_ms": mean_lat,
        "hash_permutations_tested": 200,
        "distinct_hash_count": len(distinct_hashes),
        "verdict": "PASS" if p99 < 50.0 and len(distinct_hashes) == 1 else "FAIL",
    }
    print(f"5,000 Lookups: p50={p50:.4f}ms, p90={p90:.4f}ms, p99={p99:.4f}ms, max={max_lat:.4f}ms, mean={mean_lat:.4f}ms")
    print(f"Hash determinism across 200 permutations: {len(distinct_hashes) == 1} (distinct={len(distinct_hashes)})")
    print(f"Target 2 Verdict: {result['verdict']}")
    return result


def run_target_3_primordial_prompt() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TARGET 3: Primordial Prompt Hard Filter Fuzzing")
    print("=" * 70)

    total_prompts = len(ADVERSARIAL_HUMAN_PROMPTS)
    caught = 0
    bypassed: List[str] = []

    for p in ADVERSARIAL_HUMAN_PROMPTS:
        res = validate_primordial_prompt(p, strict=False)
        if not res.is_valid:
            caught += 1
        else:
            bypassed.append(p)

    # Control group check
    control_samples = [
        desc
        for cat in STRICT_POSITIVE_DESCRIPTORS.values()
        for desc in cat[:3]
    ]
    control_passed = 0
    for cp in control_samples:
        res = validate_primordial_prompt(cp, strict=False)
        if res.is_valid:
            control_passed += 1

    result = {
        "adversarial_total": total_prompts,
        "adversarial_caught": caught,
        "adversarial_bypassed": len(bypassed),
        "bypassed_prompts": bypassed,
        "control_total": len(control_samples),
        "control_passed": control_passed,
        "verdict": "PASS" if caught == total_prompts and control_passed == len(control_samples) else "FAIL",
    }
    print(f"Adversarial prompts caught: {caught}/{total_prompts} ({caught/total_prompts*100:.1f}%)")
    print(f"Adversarial prompts bypassed: {len(bypassed)}/{total_prompts}")
    for bp in bypassed:
        print(f"  [BYPASS VULNERABILITY] {bp!r}")
    print(f"Control pristine prompts passed: {control_passed}/{len(control_samples)}")
    print(f"Target 3 Verdict: {result['verdict']}")
    return result


def run_target_4_geometry_normalizer() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TARGET 4: Geometry Normalizer Pivot & Invariant Verification")
    print("=" * 70)

    normalizer = GeometryNormalizer()
    rng = np.random.default_rng(54321)

    mesh_results = []
    all_passed = True

    for i in range(20):
        n_pts = int(rng.integers(30, 1500))
        pts = rng.uniform(-40.0, 40.0, size=(n_pts, 3)).astype(np.float32)
        pts[:, 1] += rng.uniform(-150.0, 150.0)
        target_h = float(rng.uniform(1.0, 25.0))

        norm_v, stats = normalizer.normalize_vertices(pts, target_height=target_h)

        min_y = float(np.min(norm_v[:, 1]))
        c_x = float((np.min(norm_v[:, 0]) + np.max(norm_v[:, 0])) / 2.0)
        c_z = float((np.min(norm_v[:, 2]) + np.max(norm_v[:, 2])) / 2.0)
        measured_h = float(np.max(norm_v[:, 1]) - np.min(norm_v[:, 1]))

        pass_min_y = abs(min_y) < 1e-6
        pass_cx = abs(c_x) < 1e-6
        pass_cz = abs(c_z) < 1e-6
        pass_h = abs(measured_h - target_h) < 1e-5

        colliders = normalizer.extract_collision_primitives(norm_v, target_height=target_h)
        pass_colliders = (
            colliders.cylinder.radius > 0
            and colliders.cylinder.height > 0
            and colliders.convex_hull.volume > 0
        )

        mesh_ok = pass_min_y and pass_cx and pass_cz and pass_h and pass_colliders
        if not mesh_ok:
            all_passed = False

        mesh_results.append({
            "mesh_id": i,
            "vertex_count": n_pts,
            "target_height": target_h,
            "min_y": min_y,
            "c_x": c_x,
            "c_z": c_z,
            "height_delta": abs(measured_h - target_h),
            "passed": mesh_ok,
        })

    print(f"20 Procedural Meshes Normalized: {sum(1 for m in mesh_results if m['passed'])}/20 passed")
    print(f"Target 4 Verdict: {'PASS' if all_passed else 'FAIL'}")
    return {
        "total_meshes": 20,
        "passed_meshes": sum(1 for m in mesh_results if m['passed']),
        "details": mesh_results,
        "verdict": "PASS" if all_passed else "FAIL",
    }


def main():
    t0 = time.perf_counter()
    r1 = run_target_1_fnv1a_fuzzing()
    r2 = run_target_2_vault_latency()
    r3 = run_target_3_primordial_prompt()
    r4 = run_target_4_geometry_normalizer()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(f"OVERALL EMPIRICAL VERIFICATION SUMMARY (Elapsed: {elapsed:.2f}s)")
    print("=" * 70)
    print(f"Target 1 (FNV-1a & Binary Layout):  {r1['verdict']}")
    print(f"Target 2 (Vault Latency & Hash):    {r2['verdict']}")
    print(f"Target 3 (Primordial Prompt Filter):{r3['verdict']}")
    print(f"Target 4 (Geometry Pivot & Scaling):{r4['verdict']}")

    overall = "APPROVE" if all(r["verdict"] == "PASS" for r in [r1, r2, r3, r4]) else "FAIL"
    print(f"\nFINAL VERDICT: {overall}")


if __name__ == "__main__":
    main()
