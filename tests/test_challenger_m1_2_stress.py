"""
Milestone 1 Adversarial Challenge & Stress Test Suite — Challenger M1-2.
Target: terra_forge (Milestone 1: Organic Topography, Anti-Staircasing & Erosion Physics)

Stress Test Invariants:
1. Exact mass conservation in hydraulic erosion: |Delta M| < 1e-10 across 100,000 droplets.
2. Zero single-cell needle spikes: Droplet terminations must not create isolated spikes (> 4m deviation from 3x3 median).
3. Erosion performance benchmark: 60k - 100k droplets must run in < 5.0 seconds.
4. Cavern roof clearance invariant: Rock overburden thickness above karst cavern must be >= 1.5m across all grid cells.
"""

from __future__ import annotations

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


class TestM1HydraulicMassConservationStress:
    """Invariant 1: Exact mass conservation in hydraulic erosion (|Delta M| < 1e-10 across 100,000 droplets)."""

    def test_mass_conservation_100k_droplets(self):
        """Verify |Delta M| < 1e-10 across 100,000 droplets on a 64x64 horn peak terrain."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        initial_mass = float(np.sum(hf.Z))
        cfg = ErosionConfig(enabled=True, droplet_count=100000, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)

        sim.simulate_hydraulic(seed=42)

        final_mass = float(np.sum(hf.Z))
        delta_m = abs(final_mass - initial_mass)
        assert delta_m < 1e-10, f"Mass conservation violated across 100k droplets: delta_m={delta_m:.4e}"


class TestM1NeedleSpikesStress:
    """Invariant 2: Zero single-cell needle spikes (> 4m deviation from 3x3 median)."""

    def test_zero_single_cell_needle_spikes_on_canonical_preset(self):
        """Verify that droplet terminations on the canonical preset do not produce needle spikes (> 4m deviation from 3x3 median)."""
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = MapConfig.model_validate(data)
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

        z_pre = hf.Z.copy()
        sim = ErosionSimulator(hf, cfg.erosion)
        sim.simulate_hydraulic(seed=cfg.seed)

        delta_z = hf.Z - z_pre
        med_delta = median_filter(delta_z, size=3)
        spike_delta = np.abs(delta_z - med_delta)
        max_spike = float(np.max(spike_delta))

        # Check deviation of final terrain from 3x3 median
        med_z = median_filter(hf.Z, size=3)
        dev_z = np.abs(hf.Z - med_z)
        max_dev_z = float(np.max(dev_z))

        assert max_spike <= 4.0, (
            f"Needle spike detected in erosion delta: max deviation from 3x3 median is {max_spike:.3f}m (> 4.0m threshold)"
        )


class TestM1ErosionPerformanceBenchmark:
    """Invariant 3: Erosion performance benchmark (60k-100k droplets run in < 5 seconds)."""

    def test_performance_60k_droplets_under_5_seconds(self):
        """Verify 60,000 droplets run in < 5.0 seconds."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        cfg = ErosionConfig(enabled=True, droplet_count=60000, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)

        t0 = time.perf_counter()
        sim.simulate_hydraulic(seed=42)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Performance benchmark failed: 60,000 droplets took {elapsed:.3f}s (required < 5.0s)"

    def test_performance_100k_droplets_under_5_seconds(self):
        """Verify 100,000 droplets run in < 5.0 seconds."""
        hf = Heightfield2D(width=50.0, depth=50.0, resolution=64)
        hf.add_peak(0.0, 0.0, height=25.0, radius=15.0, shape="horn")

        cfg = ErosionConfig(enabled=True, droplet_count=100000, max_droplet_steps=20)
        sim = ErosionSimulator(hf, cfg)

        t0 = time.perf_counter()
        sim.simulate_hydraulic(seed=42)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Performance benchmark failed: 100,000 droplets took {elapsed:.3f}s (required < 5.0s)"


class TestM1CavernRoofClearanceInvariant:
    """Invariant 4: Cavern roof clearance invariant (rock overburden thickness >= 1.5m across all grid cells)."""

    def test_cavern_roof_clearance_all_grid_cells_preset(self):
        """Verify that rock overburden thickness above the karst cavern chamber is >= 1.5m across all grid cells."""
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = MapConfig.model_validate(data)
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

        cx, cy, cz = cfg.cavern.center
        rx, ry, h = cfg.cavern.radius_x, cfg.cavern.radius_y, cfg.cavern.height

        violating_cells = []
        min_overburden = float("inf")

        for iy in range(hf.res_y):
            for ix in range(hf.res_x):
                wx = hf.X[iy, ix]
                wy = hf.Y[iy, ix]
                q = ((wx - cx) / rx) ** 2 + ((wy - cy) / ry) ** 2
                if q <= 1.0:
                    local_roof_z = cz + h * math.sqrt(1.0 - q)
                    surface_z = hf.Z[iy, ix]
                    overburden = surface_z - local_roof_z
                    if overburden < min_overburden:
                        min_overburden = overburden
                    if overburden < 1.5:
                        violating_cells.append(
                            {
                                "grid": (ix, iy),
                                "world": (round(float(wx), 2), round(float(wy), 2)),
                                "surface_z": round(float(surface_z), 2),
                                "roof_z": round(float(local_roof_z), 2),
                                "overburden": round(float(overburden), 2),
                            }
                        )

        assert len(violating_cells) == 0, (
            f"Cavern roof clearance breached in {len(violating_cells)} grid cells. "
            f"Minimum overburden: {min_overburden:.3f}m (< 1.5m required). "
            f"Worst violating cell: {violating_cells[0] if violating_cells else None}"
        )
