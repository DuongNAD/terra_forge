# TerraForge Test Infrastructure & Opaque-Box Quality Framework

## 1. Test Philosophy & Principles

TerraForge adopts an **opaque-box, requirement-driven** quality assurance methodology. The test suite is derived directly from user requirements (`ORIGINAL_REQUEST.md`), geological and physical principles, interface contracts (`PROJECT.md`), and the comprehensive architectural specification (`spec_report.md`), rather than probing private internal implementation mechanisms.

### Core Testing Pillars

1. **Opaque-Box Specification Derivation**:
   - Tests assert observable behavior, invariant contracts, physical properties (e.g., mass conservation, angle of repose, solar irradiance dot products), and geometric topology (Euler characteristic $\chi = 2$, zero open boundaries).
   - Tests do not rely on private methods or internal variable names, ensuring that implementation refactoring preserves test validity.

2. **Progressive Testability & Milestone Decoupling**:
   - Features implemented in current or completed milestones execute assertions directly against live code.
   - Features undergoing parallel implementation in upcoming milestones (e.g., in-flight M1 geological folding or M4 preset serialization) are integrated using non-strict xfail markers (`pytest.mark.xfail(strict=False, ...)`), allowing the test runner to execute cleanly with zero failures today while seamlessly converting to green passes as milestones land.

3. **Strict Isolation & Random-Order Resilience**:
   - Each test is fully self-contained, instantiating fresh fixtures, isolated coordinate domains, and deterministic seeds.
   - The test suite is validated with `pytest-randomly` to eliminate inter-test state leakage or ordering dependencies.

4. **Authoritative Expected Output Sources**:
   - Geological equations: Smith-Hovhannisyan sediment capacity, Dean's equilibrium beach profile ($h = A y^{2/3}$), exponential thermal lapse rate ($T = T_0 - \Gamma z$), Beer-Lambert optical attenuation ($I = I_0 e^{-\alpha d}$).
   - Topological invariants: 2-manifold closed surface condition ($V - E + F = 2$, $|\text{Faces}(e)| == 2$).
   - Spatial clustering: Bridson $O(N)$ blue-noise spatial grid criteria ($d \ge r_{\min}$).

---

## 2. Test Architecture (4-Tier Hierarchy)

The E2E test suite is located in `tests/e2e/` and organized into four rigorous tiers:

```
tests/e2e/
├── __init__.py
├── conftest.py                   # Pure-Python topological validator fixture & shared generators
├── test_tier1_features.py        # Tier 1: Requirement-level feature tests (>=5 tests per feature)
├── test_tier2_boundaries.py      # Tier 2: Boundary, extreme stress, and corner case tests
├── test_tier3_combinations.py    # Tier 3: Pairwise cross-feature interactions
└── test_tier4_scenarios.py       # Tier 4: Natural preset end-to-end scenario validations
```

### Tier 1: Requirement-Level Feature Verification (`test_tier1_features.py`)
Provides $\ge 5$ explicit test cases for each requirement feature:
- **R1: Advanced Geological & Terrain Physics**:
  - Hydraulic Droplet Erosion & Sediment Transport (Mass conservation, kinetic velocity, rill incision, alluvial fan dispersion, evaporation)
  - Thermal Talus & Scree Slopes (Angle of repose threshold, sub-critical stability, scree conservation, isotropic 8-neighbor slump, scree map accumulation)
  - Geological Strata & Tectonics (Anticline/syncline folding, strike angle warping, plunge wavelength, tectonic fault scarp displacement, terrace step profiling)
  - Coastal Geomorphology (Wave-cut notch carving, Dean's equilibrium beach profile, cliff steepening, wave berm, bathymetric shelf)
- **R2: Photorealistic Shading & Dynamic Hydrology**:
  - Triplanar PBR Multi-Splatting (Object-space projection without UV stretch, 5 material class blending, snowline altitude and slope shedding, scree texture modulation, moisture-driven moss)
  - Dynamic Hydrology & Water Optics (Beer-Lambert depth absorption, shoreline contact foam, water containment, river velocity vector flow, optical roughness)
- **R3: Multi-Tier Ecosystems & Biomes (including Features 33–36)**:
  - Microclimate Simulation (Solar insolation dot product, elevation temperature lapse rate, timberline boundary, water proximity moisture field, multi-factor niche suitability)
  - Multi-Tier Ecological Stratification (Canopy trees, understory shrubs, ground flora & rock moss, tree decay & windthrow, creature game trails & disturbed soil, underwater aquatic flora & reefs)
  - Poisson-Disk Clustering (Minimum separation enforcement $d \ge r_{\min}$, density modulation, deterministic seed reproducibility, bounding box clipping, candidate acceptance)
- **R4: Atmospheric Lighting & Quality Verification**:
  - Solar ephemeris & Time-of-Day (Sun position vector, correlated color temperature CCT, sky radiance balance)
  - Atmospheric Haze & Depth Cueing (Volumetric mist density, Henyey-Greenstein forward scattering, distance contrast attenuation)
  - Watertight Diorama Block (Euler characteristic $\chi = 2$, 0 boundary edges, 0 non-manifold edges, vertical wall closure, strata vertex color bands)
  - Standardized Multi-Angle Vision (Isometric diorama viewpoint, eye-level first-person viewpoint, close-up geological macro viewpoint, 2x2 grid preview)

### Tier 2: Boundary, Stress & Corner Cases (`test_tier2_boundaries.py`)
Validates engine robustness at extreme mathematical, hydrological, and topological limits:
- Resolution limits: minimal grid ($16\times 16$), intermediate ($64\times 64$, $128\times 128$), production ($256\times 256$), and non-power-of-two odd resolutions ($33\times 33$).
- Flat heightfield: elevation $Z = 0.0$ everywhere; verifies 0-gradient stability, zero phantom erosion, and clean mesh extrusion.
- Sheer vertical cliffs: $90^\circ$ vertical elevation steps; verifies thermal talus relaxation stability and absence of NaN/Inf normals.
- Extreme sea levels: negative sea level, submerged terrain, and diorama overflow conditions.
- Zero water / dry diorama: hydrology completely disabled; water distance field gracefully assigns maximum default distance ($999\text{m}$).
- Solar insolation extremes: vertical zenith sun ($90^\circ$), horizontal grazing sun ($0^\circ$), and night-time negative angles (clamped to zero).
- Microclimate temperature bounds: arctic subzero extremes vs desert heatwaves.
- Cavern overburden boundary: exact $5.0\text{m}$ clearance threshold enforcement.

### Tier 3: Pairwise Cross-Feature Interactions (`test_tier3_combinations.py`)
Validates composite coupling between geological, hydrological, and ecological subsystems:
- **Erosion + Triplanar Texturing**: Alluvial sediment fans and scree talus directly govern gravel/scree material weight over exposed bedrock.
- **Thermal Scree + Vegetation Exclusion**: Unstable scree slopes prevent canopy tree growth while permitting pioneer rock flora and boulder litter.
- **Strata Folding + Watertight Diorama**: Aggressive geological folds and tectonic fault scarps preserve 100% 2-manifold diorama block closure ($\chi = 2$).
- **River Stages + Water Foam**: Steep cascade stages produce high flow velocity and concentrated whitewater foam, whereas gentle valley meanders maintain clear specular flow.
- **Solar Insolation + Moisture Field**: South-facing sun-exposed slopes exhibit lower effective soil moisture compared to sheltered north-facing damp gullies.
- **Biological Game Trails + River Access**: Animal trails naturally forge paths through dense vegetation towards freshwater sources.
- **Underwater Ecosystems + Depth Optics**: Submerged flora and rounded pebbles distribute according to bathymetric water depth and light penetration.

### Tier 4: Real-World Natural Preset Scenarios (`test_tier4_scenarios.py`)
End-to-end validation across the 4 required natural environments:
1. **Alpine Peak & Glacial Lake**: Glacial cirques, knife-edge arêtes, scree talus slopes, high-altitude alpine conifers, and turquoise glacial flour water.
2. **Temperate Forest River Basin**: Sinuous river meander, lush broadleaf canopy, understory ferns, riparian wetlands, and fallen decaying logs.
3. **Tropical Island & Coastal Karst Cliffs**: Sheer limestone karst towers, sandy beaches, wave-cut notches, tropical jungle canopy, and shallow coral lagoon.
4. **Canyon & Semi-arid Plateau**: Layered sandstone mesas, deep river canyon, arid plateau junipers, desert scrub, and dry gravel wash beds.

---

## 3. Test Runner & Execution Commands

### Primary Command
Execute the full E2E test suite:
```bash
pytest tests/e2e
```

### Verbose Output with Test Names
```bash
pytest -v tests/e2e
```

### Targeted Tier Execution
```bash
# Run Tier 1 Feature Coverage
pytest tests/e2e/test_tier1_features.py

# Run Tier 2 Boundaries
pytest tests/e2e/test_tier2_boundaries.py

# Run Tier 3 Combinations
pytest tests/e2e/test_tier3_combinations.py

# Run Tier 4 Presets
pytest tests/e2e/test_tier4_scenarios.py
```

### Run All Tests (Unit + E2E)
```bash
pytest
```

---

## 4. Coverage Matrix

| Category | Requirement | Feature Description | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|---|---|
| **Physics** | R1.1 | Hydraulic Droplet & Momentum Erosion | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Physics** | R1.2 | Thermal Talus Relaxation & Scree | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Physics** | R1.3 | Geological Strata Folding & Tectonics | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Physics** | R1.4 | Coastal Geomorphology & Beach Profiles | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Materials** | R2.1 | 5-Layer Triplanar PBR Multi-Splatting | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Materials** | R2.2 | Dynamic Beer-Lambert Water & Foam | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.1 | Microclimate Simulation (Aspect/Lapse) | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.2 | Multi-Tier Forest Stratification | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.3 | Fast Spatial-Grid Poisson-Disk Scatter | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.4 (F33) | Ground Flora, Wildflowers & Rock Moss | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.5 (F34) | Natural Tree Decay & Windthrow | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.6 (F35) | Biological Creature Impacts & Trails | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Ecology** | R3.7 (F36) | Underwater Ecosystems & Reefs | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **QA / Vision** | R4.1 | Sun Position, CCT & Atmospheric Mist | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **QA / Vision** | R4.2 | Watertight Manifold Diorama Block | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **QA / Vision** | R4.3 | Standard Multi-Angle Vision Cameras | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Presets** | P1 | Alpine Peak & Glacial Lake | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Presets** | P2 | Temperate Forest River Basin | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Presets** | P3 | Tropical Island & Coastal Karst Cliffs | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |
| **Presets** | P4 | Canyon & Semi-arid Plateau | `test_tier1_features.py` | `test_tier2_boundaries.py` | `test_tier3_combinations.py` | `test_tier4_scenarios.py` |

---

## 5. Pass Criteria & Quality Gates

A pull request or milestone handoff is considered **PASS** if and only if:
1. **Zero Test Regressions**: `pytest` exits with code 0 (100% of non-xfail tests pass; 0 unexpected failures, 0 runtime crashes).
2. **Watertight Manifold Invariant**: Every synthesized diorama block satisfies $\chi = V - E + F = 2$, open boundary edges $= 0$, and non-manifold edges $= 0$.
3. **Hydrological Safety**: All water bodies are physically bounded; lake retaining berms maintain a minimum freeboard margin $\ge 0.35\text{m}$.
4. **Ecological Feasibility**: Flora instances are strictly bounded by ecological suitability; 0 canopy trees grow on sheer vertical rock faces ($\theta > 45^\circ$) or submerged beneath deep water.
5. **Headless Execution Efficiency**: Full simulation passes execute in under $1.5\text{s}$ in pure Python / NumPy, enabling instant CI regression verification.
