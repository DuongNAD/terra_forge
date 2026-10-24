"""
Milestone 1 Remediation Adversarial Re-verification Test Suite.
Agent: teamwork_preview_challenger (Challenger M1-2 Re-verification)
Authoritative invariants:
1. D1: Zero single-cell needle spikes (> 4.0m deviation from 3x3 median) on canonical preset and synthetic slopes.
2. D2: Erosion performance < 5.0s for 60,000 and 100,000 droplets on resolution=64 / 20 steps.
3. D3: Cavern roof clearance >= 1.5m across ALL grid cells of the cavern footprint (q <= 1.0) with 0 breaching cells.
4. Mass conservation: exact |Delta M| < 1e-10 across all paths and scales.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import median_filter

from terra_forge.schema.map_config import ErosionConfig, MapConfig
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator

PRESET_PATH = (
    Path(__file__).resolve().parent.parent
    / "terra_forge"
    / "schema"
    / "presets"
    / "genesis_primordial_abiotic.json"
)


def _load_canonical_config() -> MapConfig:
    with open(PRESET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MapConfig.model_validate(data)


def _build_canonical_heightfield(cfg: MapConfig) -> Heightfield2D:
    hf = Heightfield2D(
        width=cfg.diorama.width,
        depth=cfg.diorama.depth,
        resolution=cfg.diorama.resolution,
    )
    for peak in cfg.peaks:
        hf.add_peak(
            peak.position[0],
            peak.position[1],
            peak.height,
            peak.radius,
            peak.shape,
            peak.ridge_angle_deg,
        )
    hf.apply_geological_features(cfg)
    return hf


class TestD1NeedleSpikesAdversarial:
    """Invariant D1: Zero single-cell needle spikes (> 4.0m deviation from 3x3 median)."""

    @pytest.mark.parametrize("seed", [42, 123, 999, 2026])
    def test_needle_spikes_canonical_preset_varied_seeds(self, seed: int):
        """Verify canonical preset across multiple random seeds produces zero needle spikes (>4.0m)."""
        cfg = _load_canonical_config()
        hf = _build_canonical_heightfield(cfg)
        z_pre = hf.Z.copy()

        sim = ErosionSimulator(hf, cfg.erosion)
        sim.simulate_hydraulic(seed=seed)

        delta_z = hf.Z - z_pre
        med_delta = median_filter(delta_z, size=3)
        spike_delta = np.abs(delta_z - med_delta)
        max_spike = float(np.max(spike_delta))

        assert max_spike <= 4.0, (
            f"Seed {seed}: Needle spike detected in erosion delta: {max_spike:.3f}m (> 4.0m)"
        )

    @pytest.mark.parametrize("droplet_count", [300, 4000, 20000])
    def test_needle_spikes_canonical_preset_all_three_paths(self, droplet_count: int):
        """Verify that Path 1 (<=400), Path 2 (401-10000), and Path 3 (>10000) all produce zero needle spikes on preset."""
        cfg = _load_canonical_config()
        hf = _build_canonical_heightfield(cfg)
        z_pre = hf.Z.copy()

        ero_cfg = dataclasses.replace(cfg.erosion, droplet_count=droplet_count)
        sim = ErosionSimulator(hf, ero_cfg)
        sim.simulate_hydraulic(seed=42)

        delta_z = hf.Z - z_pre
        med_delta = median_filter(delta_z, size=3)
        spike_delta = np.abs(delta_z - med_delta)
        max_spike = float(np.max(spike_delta))

        assert max_spike <= 4.0, (
            f"Preset Count {droplet_count}: Needle spike detected in delta: {max_spike:.3f}m (> 4.0m)"
        )

    @pytest.mark.parametrize("droplet_count", [350, 4000, 20000, 60000])
    def test_needle_spikes_synthetic_horn_peak(self, droplet_count: int):
        """Verify on synthetic Matterhorn peak across droplet count tiers."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=64)
        hf.add_peak(0.0, 0.0, height=28.0, radius=18.0, shape="horn")

        z_pre = hf.Z.copy()
        cfg = ErosionConfig(enabled=True, droplet_count=droplet_count, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        delta_z = hf.Z - z_pre
        med_delta = median_filter(delta_z, size=3)
        spike_delta = np.abs(delta_z - med_delta)
        max_spike = float(np.max(spike_delta))

        assert max_spike <= 4.0, (
            f"Synthetic horn peak (count={droplet_count}): max delta spike={max_spike:.3f}m (> 4.0m)"
        )

    @pytest.mark.parametrize("droplet_count", [350, 4000, 20000, 60000])
    def test_needle_spikes_synthetic_planar_slope(self, droplet_count: int):
        """Verify on synthetic planar slope (gradient=0.5) across droplet count tiers."""
        hf = Heightfield2D(width=60.0, depth=60.0, resolution=64)
        hf.Z = 0.5 * (hf.X + 30.0)

        z_pre = hf.Z.copy()
        cfg = ErosionConfig(enabled=True, droplet_count=droplet_count, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        delta_z = hf.Z - z_pre
        med_delta = median_filter(delta_z, size=3)
        spike_delta = np.abs(delta_z - med_delta)
        max_spike = float(np.max(spike_delta))

        assert max_spike <= 4.0, (
            f"Synthetic planar slope (count={droplet_count}): max delta spike={max_spike:.3f}m (> 4.0m)"
        )


class TestD2PerformanceAdversarial:
    """Invariant D2: Erosion performance < 5.0s for 60,000 and 100,000 droplets on resolution=64 / 20 steps."""

    def test_performance_60k_droplets_multirun(self):
        """Verify 60,000 droplets run in < 5.0s consistently across 3 consecutive runs."""
        times = []
        for run_idx in range(3):
            hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
            hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")
            cfg = ErosionConfig(enabled=True, droplet_count=60000, max_droplet_steps=20)
            sim = ErosionSimulator(hf, cfg)

            t0 = time.perf_counter()
            sim.simulate_hydraulic(seed=42 + run_idx)
            dt = time.perf_counter() - t0
            times.append(dt)
            assert dt < 5.0, f"Run {run_idx+1}: 60k droplets took {dt:.3f}s (required < 5.0s)"

        avg_time = sum(times) / len(times)
        max_time = max(times)
        throughput = 60000 / avg_time
        print(f"\n[D2 Bench 60k] Mean: {avg_time:.3f}s, Max: {max_time:.3f}s, Throughput: {throughput:.0f} droplets/s")

    def test_performance_100k_droplets_multirun(self):
        """Verify 100,000 droplets run in < 5.0s consistently across 3 consecutive runs."""
        times = []
        for run_idx in range(3):
            hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
            hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")
            cfg = ErosionConfig(enabled=True, droplet_count=100000, max_droplet_steps=20)
            sim = ErosionSimulator(hf, cfg)

            t0 = time.perf_counter()
            sim.simulate_hydraulic(seed=42 + run_idx)
            dt = time.perf_counter() - t0
            times.append(dt)
            assert dt < 5.0, f"Run {run_idx+1}: 100k droplets took {dt:.3f}s (required < 5.0s)"

        avg_time = sum(times) / len(times)
        max_time = max(times)
        throughput = 100000 / avg_time
        print(f"\n[D2 Bench 100k] Mean: {avg_time:.3f}s, Max: {max_time:.3f}s, Throughput: {throughput:.0f} droplets/s")


class TestD3CavernClearanceAdversarial:
    """Invariant D3: Cavern roof clearance >= 1.5m across ALL grid cells of the cavern footprint (q <= 1.0) with 0 breaching cells."""

    def test_cavern_clearance_canonical_preset_before_and_after_erosion(self):
        """Verify cavern roof clearance >= 1.5m across all footprint cells before AND after erosion."""
        cfg = _load_canonical_config()
        hf = _build_canonical_heightfield(cfg)

        cx, cy, cz = cfg.cavern.center
        rx, ry, h = cfg.cavern.radius_x, cfg.cavern.radius_y, cfg.cavern.height

        def check_clearance(stage_label: str) -> float:
            min_ob = float("inf")
            breaches = []
            for iy in range(hf.res_y):
                for ix in range(hf.res_x):
                    wx = hf.X[iy, ix]
                    wy = hf.Y[iy, ix]
                    q = ((wx - cx) / rx) ** 2 + ((wy - cy) / ry) ** 2
                    if q <= 1.0:
                        roof_z = cz + h * math.sqrt(1.0 - q)
                        surface_z = hf.Z[iy, ix]
                        ob = surface_z - roof_z
                        if ob < min_ob:
                            min_ob = ob
                        if ob < 1.5:
                            breaches.append(((ix, iy), (wx, wy), surface_z, roof_z, ob))
            assert len(breaches) == 0, (
                f"[{stage_label}] Cavern roof clearance breached in {len(breaches)} cells. Min overburden: {min_ob:.3f}m (< 1.5m)"
            )
            return min_ob

        min_pre = check_clearance("Pre-Erosion")
        assert min_pre >= 1.5, f"Pre-erosion min overburden {min_pre:.3f}m < 1.5m"

        sim = ErosionSimulator(hf, cfg.erosion)
        sim.simulate_hydraulic(seed=cfg.seed)

        min_post = check_clearance("Post-Erosion (20k droplets)")
        assert min_post >= 1.5, f"Post-erosion min overburden {min_post:.3f}m < 1.5m"
        print(f"\n[D3 Cavern] Pre-erosion min: {min_pre:.3f}m, Post-erosion min: {min_post:.3f}m")

    def test_cavern_clearance_flat_ground_synthetic_enclosure(self):
        """Synthetic test: Even on completely flat ground (Z=0), ensure_cavern_clearance lifts the terrain so overburden >= 1.5m everywhere."""
        hf = Heightfield2D(width=100.0, depth=100.0, resolution=128)
        hf.Z[:] = 0.0  # completely flat ground at Z=0

        cfg = _load_canonical_config()
        # Place cavern chamber centered at (0, 0, 5) with height 10m -> ceiling reaches Z = 15m
        cfg.cavern.center = [0.0, 0.0, 5.0]
        cfg.cavern.radius_x = 20.0
        cfg.cavern.radius_y = 15.0
        cfg.cavern.height = 10.0

        hf.ensure_cavern_clearance(cfg, min_clearance=1.5)

        cx, cy, cz = cfg.cavern.center
        rx, ry, h = cfg.cavern.radius_x, cfg.cavern.radius_y, cfg.cavern.height

        min_ob = float("inf")
        breaches = []
        for iy in range(hf.res_y):
            for ix in range(hf.res_x):
                wx = hf.X[iy, ix]
                wy = hf.Y[iy, ix]
                q = ((wx - cx) / rx) ** 2 + ((wy - cy) / ry) ** 2
                if q <= 1.0:
                    roof_z = cz + h * math.sqrt(1.0 - q)
                    surface_z = hf.Z[iy, ix]
                    ob = surface_z - roof_z
                    if ob < min_ob:
                        min_ob = ob
                    if ob < 1.5 - 1e-6:
                        breaches.append(((ix, iy), ob))

        assert len(breaches) == 0, f"Flat terrain enclosure breached in {len(breaches)} cells: min_ob={min_ob:.4f}m"
        assert min_ob >= 1.5 - 1e-6


class TestMassConservationAdversarial:
    """Invariant: Exact physical mass conservation (|Delta M| < 1e-10)."""

    @pytest.mark.parametrize("droplet_count", [50, 200, 400, 1000, 5000, 10000, 20000, 60000, 100000])
    def test_mass_conservation_across_droplet_counts(self, droplet_count: int):
        """Verify |Delta M| < 1e-10 across all droplet count tiers (Path 1, Path 2, Path 3)."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        m0 = float(np.sum(hf.Z))
        cfg = ErosionConfig(enabled=True, droplet_count=droplet_count, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        m1 = float(np.sum(hf.Z))
        delta_m = abs(m1 - m0)
        assert delta_m < 1e-10, (
            f"Mass conservation failed for count={droplet_count}: delta_m={delta_m:.4e} (threshold 1e-10)"
        )

    @pytest.mark.parametrize("res", [32, 64, 128])
    def test_mass_conservation_across_resolutions(self, res: int):
        """Verify |Delta M| < 1e-10 across different grid resolutions."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=res)
        hf.add_peak(0.0, 0.0, height=20.0, radius=12.0, shape="cone")

        m0 = float(np.sum(hf.Z))
        cfg = ErosionConfig(enabled=True, droplet_count=15000, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)
        sim.simulate_hydraulic(seed=42)

        m1 = float(np.sum(hf.Z))
        delta_m = abs(m1 - m0)
        assert delta_m < 1e-10, (
            f"Mass conservation failed for res={res}: delta_m={delta_m:.4e} (threshold 1e-10)"
        )
