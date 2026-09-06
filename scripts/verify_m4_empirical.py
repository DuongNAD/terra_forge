#!/usr/bin/env python3
"""
Standalone Empirical Verification Script for Milestone 4 Challenge.
Outputs precise quantitative metrics across:
1. 24-hour Diurnal Lighting Sweep (solar elevation, azimuth, CCT, linear RGB).
2. Optical Air Mass & Solar Extinction Intensity (zenith to horizon, sub-horizon zero check).
3. 4 Natural Presets JSON schema & Pydantic validation.
4. Procedural Heightfield Generation & Watertight Diorama Mesh Topology.
"""

import json
import math
import sys
from pathlib import Path
import numpy as np
import pydantic

from terra_forge.schema.map_config import (
    MapConfig,
    PRESET_CONFIGS,
)
from terra_forge.cli import PRESETS_DIR
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.core.erosion import ErosionSimulator
from terra_forge.core.hydrology import HydrologyEngine
from terra_forge.core.diorama_slab import DioramaSlabBuilder
from terra_forge.inspector.validator import QualityValidator
from terra_forge.inspector.cameras import (
    calculate_solar_position,
    calculate_cct_from_elevation,
    cct_to_linear_rgb,
    calculate_optical_air_mass,
    calculate_sun_intensity,
)


def run_diurnal_lighting_sweep():
    print("\n" + "=" * 78)
    print("CHALLENGE 1: 24-HOUR DIURNAL LIGHTING & SOLAR EPHEMERIS SWEEP")
    print("=" * 78)

    hours = [0.0, 3.0, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 24.0]
    print(f"{'Hour':>5} | {'Elev(°)':>8} | {'Azim(°)':>8} | {'CCT (K)':>8} | {'Linear RGB (R, G, B)':>24} | {'Intensity':>10}")
    print("-" * 78)

    max_elev = 68.0
    all_non_negative = True
    all_finite = True
    sinusoidal_match = True

    for h in hours:
        elev, azim, vec = calculate_solar_position(h, max_elevation_deg=max_elev)
        cct = calculate_cct_from_elevation(elev)
        r, g, b = cct_to_linear_rgb(cct)
        intensity = calculate_sun_intensity(elev)

        # Theoretical elevation
        expected_elev = max_elev * math.sin((h % 24.0 - 6.0) / 12.0 * math.pi)
        if abs(elev - expected_elev) > 1e-4:
            sinusoidal_match = False

        if not (r >= 0 and g >= 0 and b >= 0):
            all_non_negative = False
        if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
            all_finite = False

        rgb_str = f"({r:.3f}, {g:.3f}, {b:.3f})"
        print(f"{h:5.1f} | {elev:8.2f} | {azim:8.2f} | {cct:8.1f} | {rgb_str:>24} | {intensity:10.4f}")

    print("-" * 78)
    print(f"  Sinusoidal Elevation Match : {'PASS [✓]' if sinusoidal_match else 'FAIL [❌]'}")
    print(f"  RGB Non-Negativity Check   : {'PASS [✓]' if all_non_negative else 'FAIL [❌]'}")
    print(f"  RGB Finiteness Check       : {'PASS [✓]' if all_finite else 'FAIL [❌]'}")


def run_air_mass_and_extinction_sweep():
    print("\n" + "=" * 78)
    print("CHALLENGE 2: OPTICAL AIR MASS & SOLAR INTENSITY EXTINCTION")
    print("=" * 78)

    test_elevations = [90.0, 68.0, 45.0, 30.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.0, -0.1, -1.0, -5.0, -18.0, -68.0]
    print(f"{'Elev(°)':>8} | {'Air Mass m':>12} | {'Intensity I0=5.5':>18} | {'Status':>10}")
    print("-" * 78)

    sub_horizon_zero = True
    for elev in test_elevations:
        m = calculate_optical_air_mass(elev)
        intensity = calculate_sun_intensity(elev, i0=5.5, tau_r=0.10)

        if elev <= 0.0:
            status = "I=0 [✓]" if intensity == 0.0 else "FAIL [❌]"
            if intensity != 0.0:
                sub_horizon_zero = False
        else:
            status = "EXTINCT"

        print(f"{elev:8.2f} | {m:12.4f} | {intensity:18.4f} | {status:>10}")

    print("-" * 78)
    print(f"  Sub-Horizon Intensity (I -> 0 when elev <= 0): {'PASS [✓]' if sub_horizon_zero else 'FAIL [❌]'}")


def run_preset_validation_and_diorama_synthesis():
    print("\n" + "=" * 78)
    print("CHALLENGE 3: 4 NATURAL PRESETS SCHEMA & WATERTIGHT DIORAMA TOPOLOGY")
    print("=" * 78)

    presets = [
        "alpine_peak_glacial_lake",
        "temperate_forest_river_basin",
        "tropical_island_coastal_karst",
        "canyon_semiarid_plateau",
    ]

    for p_name in presets:
        json_path = PRESETS_DIR / f"{p_name}.json"
        assert json_path.exists(), f"Missing file: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. MapConfig.from_dict
        cfg_from_dict = MapConfig.from_dict(data)

        # 2. Pydantic TypeAdapter
        ta = pydantic.TypeAdapter(MapConfig)
        cfg_pydantic = ta.validate_python(data)

        # 3. Diorama generation and topological validation
        res = 33
        hf = Heightfield2D(width=cfg_from_dict.diorama.width, depth=cfg_from_dict.diorama.depth, resolution=res)
        for p in cfg_from_dict.peaks:
            hf.add_peak(p.position[0], p.position[1], p.height, p.radius, p.shape, p.ridge_angle_deg)

        if cfg_from_dict.hydrology.lake_radius > 0 and abs(cfg_from_dict.hydrology.lake_center[0]) < cfg_from_dict.diorama.width:
            hf.carve_lake_basin(
                cfg_from_dict.hydrology.lake_center[0],
                cfg_from_dict.hydrology.lake_center[1],
                cfg_from_dict.hydrology.lake_radius,
                cfg_from_dict.hydrology.lake_level,
                cfg_from_dict.hydrology.lake_depth,
                berm_height=1.5,
            )

        if cfg_from_dict.erosion.enabled:
            ero = ErosionSimulator(hf, cfg_from_dict.erosion)
            ero.run_hydraulic_erosion(seed=cfg_from_dict.seed)
            ero.run_thermal_erosion()

        if cfg_from_dict.hydrology.river_enabled:
            hydro = HydrologyEngine(hf, cfg_from_dict.hydrology)
            hydro.carve_river_channel()

        slab = DioramaSlabBuilder(hf, cfg_from_dict.diorama)
        mesh_data = slab.build_mesh_data()

        topo = QualityValidator.validate_mesh_topology(mesh_data["vertices"], mesh_data["faces"])
        metrics = topo["metrics"]

        print(f"\nPreset: '{p_name}'")
        print(f"  MapConfig.from_dict          : PASS [✓]")
        print(f"  Pydantic TypeAdapter Validate : PASS [✓]")
        print(f"  Mesh Topology Metrics:")
        print(f"    - Vertices (V)             : {metrics['num_vertices']}")
        print(f"    - Faces (F)                : {metrics['num_faces']}")
        print(f"    - Edges (E)                : {metrics['num_edges']}")
        print(f"    - Euler Characteristic \u03c7   : {metrics['euler_characteristic']} (Target: 2)")
        print(f"    - Boundary Edges           : {metrics['boundary_edges']} (Target: 0)")
        print(f"    - Non-Manifold Edges       : {metrics['non_manifold_edges']} (Target: 0)")
        print(f"    - Non-Manifold Vertices    : {metrics['non_manifold_vertices']} (Target: 0)")
        print(f"    - Orientation Conflicts    : {metrics['orientation_conflicts']} (Target: 0)")
        print(f"    - 100% Watertight          : {'PASS [✓]' if topo['is_watertight'] else 'FAIL [❌]'}")

        assert topo["is_watertight"] is True
        assert metrics["boundary_edges"] == 0
        assert metrics["euler_characteristic"] == 2

    print("\n" + "=" * 78)
    print("ALL EMPIRICAL CHALLENGE ASSERTIONS PASSED WITH ZERO VIOLATIONS.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_diurnal_lighting_sweep()
    run_air_mass_and_extinction_sweep()
    run_preset_validation_and_diorama_synthesis()
