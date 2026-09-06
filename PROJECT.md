# Project: TerraForge Engine Upgrade

## Architecture
TerraForge is a modular 3D terrain simulation, photorealistic shading, procedural ecology, and diorama generation engine for Blender and AI Coding Agents. The architecture strictly separates pure mathematical/geological algorithms (`terra_forge.core`), data schemas (`terra_forge.schema`), Blender scene/shader/mesh generation (`terra_forge.blender`), quality assurance and inspection (`terra_forge.inspector`), and interface tooling (`terra_forge.cli`, `terra_forge.mcp_server`).

```
                              [ORIGINAL_REQUEST.md]
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
        [Implementation Track]                 [E2E Testing Track]
    ┌──────────────────────────────┐       ┌──────────────────────────────┐
    │ M1: Geological Physics       │       │ Test Infra & Opaque Runner   │
    │ M2: Shading & Hydrology      │       │ Tier 1: Feature Coverage     │
    │ M3: Biomes & Vegetation      │       │ Tier 2: Boundary & Corner    │
    │ M4: Atmosphere, QA & Presets │       │ Tier 3: Pairwise Combination │
    └──────────────┬───────────────┘       │ Tier 4: Real-World Scenarios │
                   ▼                       └──────────────┬───────────────┘
    ┌──────────────────────────────┐                      │
    │ M5: Final E2E Test Suite     │◄─────────────────────┘
    │     Pass & Hardening (T1-T5) │  (publishes TEST_READY.md)
    └──────────────────────────────┘
```

### Module Boundaries
1. `terra_forge.schema`: Pydantic/dataclass configuration models (`map_config.py`, preset JSON files). Completely decoupled from Blender.
2. `terra_forge.core`: Pure Python/NumPy physics simulation engines:
   - `heightfield.py`: Base elevation, strata folding, tectonic fault scarps, coastal wave-cut profiles, lake basins.
   - `erosion.py`: Momentum droplet hydraulic erosion, sediment transport, alluvial fans, 8-neighbor isotropic thermal talus relaxation.
   - `hydrology.py`: River spline networks, water distance fields, velocity vectors, dynamic presets.
   - `biomes.py`: Microclimate simulation (elevation lapse rate, solar insolation aspect, moisture index), multi-tier ecological zonation, Bridson $O(N)$ Poisson-disk clustering.
   - `diorama_slab.py`: Watertight manifold diorama block extrusion, boundary skirts, base floor slab.
3. `terra_forge.blender`: Blender Python API (`bpy`, `bmesh`) scene construction:
   - `shaders.py`: 5-layer Object-Space Triplanar PBR multi-splatting (bedrock, scree, soil, sand, snow) with procedural perturbation; Beer-Lambert water optics with contact shoreline foam.
   - `water_builder.py`: Dynamic water geometry, river ribbons, ocean/bay surfaces with watertight freeboard.
   - `asset_fetcher.py`: Procedural multi-tier vegetation models (canopy trees, understory shrubs, groundcover, detritus logs/boulders).
   - `geonodes.py`: Procedural Geometry Nodes scatter modifiers.
   - `mesh_builder.py`: BMesh diorama mesh generation, vertex coloring, and UV unwrapping.
   - `runner.py`: Headless generation and export pipeline (`-b -P`).
4. `terra_forge.inspector`:
   - `validator.py`: Pure Python + BMesh topological manifold verification ($V - E + F = 2$, 0 non-manifold edges, berm freeboard, cavern rock clearance).
   - `cameras.py`: Three standardized camera viewpoints (isometric diorama, eye-level first-person, close-up geological macro), sun position/blackbody color, volumetric mist.
5. `terra_forge.cli`: Command-line interface (`generate`, `inspect`, `doctor`, `batch`).

---

## Feature Inventory

Every feature identified in the Phase 0 Survey is cataloged below with its assigned milestone. No feature is left unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Hydraulic Droplet Kinematics & Momentum | Correct velocity equation ($- \Delta h \cdot g$), momentum inertia, and directional steering | M1 | Survey 3 & R1 |
| 2 | Vectorized Deposition Mass Conservation | Replace race-condition slice indexing with `np.add.at` for exact sediment conservation | M1 | Survey 3 & R1 |
| 3 | Radial Brush Kernel & Rill Convergence | Precomputed radial erosion/deposition kernels creating dendritic rill and gully networks | M1 | Survey 3 & R1 |
| 4 | Alluvial Sediment Fan Deposition | Terminal deposition spreading creating alluvial fans at mountain breaks in slope | M1 | Survey 3 & R1 |
| 5 | 8-Neighbor Isotropic Thermal Talus | 8-neighbor relaxation with Euclidean distance metrics eliminating cardinal grid artifacts | M1 | Survey 3 & R1 |
| 6 | Scree Layer Tracking | Tracking transported talus debris into a separate `scree_talus` map for texturing & ecology | M1 | Survey 3 & R1 |
| 7 | Geological Strata Folding | Anticline/syncline periodic folding and tectonic domain warping | M1 | Survey 2, 3 & R1 |
| 8 | Tectonic Fault Scarps & Terraces | Fault displacement and continuous canyon step terrace functions | M1 | Survey 2, 3 & R1 |
| 9 | Coastal Geomorphology Profiles | Dean's equilibrium beach profile and coastal cliff wave-cut notch synthesis | M1 | Survey 2, 3 & R1 |
| 10 | Dynamic River Spline Configuration | Generalize river spline evaluation to read custom `river_stages` per preset | M2 | Survey 1, 2 & R2 |
| 11 | 5-Layer Triplanar PBR Multi-Splatting | Object-space triplanar projection blending bedrock, scree, soil, sand, and snow | M2 | Survey 1, 3 & R2 |
| 12 | Procedural Micro-Texture Perturbation | Noise and Voronoi perturbation breaking texture tiling and UV stretching | M2 | Survey 3 & R2 |
| 13 | Sediment & Moisture Shader Modulation | Modulate scree texture with erosion scree map, damp soil/moss with moisture field | M2 | Survey 1, 3 & R2 |
| 14 | Dynamic Beer-Lambert Water Optics | Volumetric optical absorption gradient ($I = I_0 e^{-\alpha d}$) and surface roughness | M2 | Survey 2, 3 & R2 |
| 15 | Shoreline Contact Foam & Wave Bump | Procedural contact foam band and normal bump perturbation along water edges | M2 | Survey 2, 3 & R2 |
| 16 | Microclimate Solar Radiation Aspect | Direct solar insolation calculation ($-\mathbf{n} \cdot \mathbf{L}_{\text{sun}}$) on terrain faces | M3 | Survey 2, 3 & R3 |
| 17 | Elevation Temperature Lapse Rate | Thermal gradient $T(z) = T_0 - \Gamma z$ governing timberline and snowline | M3 | Survey 2, 3 & R3 |
| 18 | Soil Moisture Index Proximity Field | Continuous moisture field from river spline, lake, and marine coast | M3 | Survey 2, 3 & R3 |
| 19 | Multi-Tier Ecological Stratification | Separate canopy trees, understory shrubs, wild grass/moss, and detritus logs/boulders | M3 | Survey 1, 2 & R3 |
| 20 | Bridson $O(N)$ Blue-Noise Poisson-Disk | Spatial grid blue-noise clustering with density map modulation | M3 | Survey 3 & R3 |
| 21 | Procedural Flora & Detritus Prototypes | Procedural canopy trees (pine, broadleaf, palm), shrubs, cacti, and fallen logs | M3 | Survey 1, 2 & R3 |
| 22 | Geometry Nodes Scatter Subsystem | Procedural scatter nodegroups respecting steep cliff and deep water exclusion | M3 | Survey 1, 2 & R3 |
| 23 | Solar Vector & Time-of-Day Lighting | Sun angle calculation with blackbody color temperature and sky radiance | M4 | Survey 2, 3 & R4 |
| 24 | Atmospheric Haze & Volumetric Mist | Depth cueing mist volume shader providing spatial atmospheric depth | M4 | Survey 1, 2 & R4 |
| 25 | Standardized Multi-Angle Vision Cameras | Automated camera positioning: isometric diorama, eye-level, close-up macro | M4 | Survey 1, 2 & R4 |
| 26 | Decoupled Pure Python Mesh Validator | Pure Python topological validator testing 2-manifold invariants without `bpy` | M4 | Survey 1, 2 & R4 |
| 27 | Preset 1: Alpine Peak & Glacial Lake | High alpine horns, cirque lake, glacial moraines, scree slopes, pine forest | M4 | Survey 1, 2 & Presets |
| 28 | Preset 2: Temperate Forest River Basin | Meandering river, broadleaf canopy, understory shrubs, alluvial floodplains | M4 | Survey 1, 2 & Presets |
| 29 | Preset 3: Tropical Island & Karst Cliffs | Wave-cut karst cliffs, sandy beaches, turquoise lagoon, palm vegetation | M4 | Survey 1, 2 & Presets |
| 30 | Preset 4: Canyon & Semi-arid Plateau | Terraced plateau, deep river canyon, talus scree slopes, desert cacti/shrubs | M4 | Survey 1, 2 & Presets |
| 31 | Full E2E Test Suite (Tiers 1-4) | Comprehensive opaque-box test suite verifying all requirements and presets | M5 & E2E | Acceptance Criteria |
| 32 | Adversarial Hardening (Tier 5) | White-box edge case testing and adversarial bug hunting | M5 | Acceptance Criteria |
| 33 | Rich Ground Flora & Rock Moss | Wildflowers, dense grass clumps, turf grass, ferns, mushrooms, and rock-clinging moss | M3 | User Update |
| 34 | Natural Tree Decay & Windthrow | Broken branches, uprooted trees, fallen decaying logs, snapped trunks, rotting wood, and weathered stumps | M3 | User Update |
| 35 | Biological & Creature Impacts | Animal game trails / paths, trampled grass zones, and naturally churned/disturbed soil patches | M3 | User Update |
| 36 | Underwater Ecosystems | Aquatic flora (seaweed, waterweeds, reeds), riverbed/lakebed rounded pebbles, submerged coral/rock reefs | M3 | User Update |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Geological Physics & Terrain Synthesis | Features 1–9 (`erosion.py`, `heightfield.py`, `map_config.py`) | none | DONE |
| M2 | Photorealistic Shading & Hydrology | Features 10–15 (`shaders.py`, `water_builder.py`, `hydrology.py`) | M1 | DONE |
| M3 | Multi-Tier Biomes & Vegetation | Features 16–22, 33–36 (`biomes.py`, `asset_fetcher.py`, `geonodes.py`) | M1 | DONE |
| M4 | Atmosphere, Decoupled QA & 4 Presets | Features 23–30 (`cameras.py`, `validator.py`, `presets/*.json`, `cli.py`) | M2, M3 | DONE |
| M5 | Final E2E Test Pass & Adversarial Hardening | Features 31–32 (100% E2E test pass, Tier 1-5 verification) | M4, E2E Track | DONE |
| E2E | E2E Testing Track | Independent opaque-box test runner and test cases (Tiers 1-4) | none (parallel) | DONE |

---

## Interface Contracts

### M1 (Core Physics) ↔ M2 (Shading & Hydrology)
- `Heightfield2D.Z`: `np.ndarray` of shape `(res, res)`, float32, range $[0, H_{max}]$.
- `Heightfield2D.scree_talus`: `np.ndarray` of shape `(res, res)`, float32, accumulated thermal talus volume.
- `ErosionSimulator.simulate_hydraulic()` returns `(Heightfield2D, sediment_map: np.ndarray, water_depth: np.ndarray)`.
- `HydrologyEngine.evaluate_river_spline(t: float)` returns `(x, y, z, width, depth)` based on `config.hydrology.river_stages`.

### M1 (Core Physics) ↔ M3 (Biomes & Vegetation)
- `Heightfield2D.compute_normals()` returns `(nx, ny, nz)` unit normal vector fields.
- `BiomeEngine.compute_insolation(normals, sun_vector: np.ndarray)` returns `insolation: np.ndarray` in $[0, 1]$.
- `BiomeEngine.compute_temperature(elevation, sea_level, lapse_rate)` returns `temperature: np.ndarray`.
- `BiomeEngine.compute_moisture(water_dist_field)` returns `moisture: np.ndarray` in $[0, 1]$.
- `BiomeEngine.poisson_scatter(suitability_map, min_radius, max_count)` returns `List[Tuple[float, float, float]]` (x, y, z).

### M1/M2/M3 ↔ M4 (QA Validator & Presets)
- `QualityValidator.validate_mesh_topology(vertices, faces)`: Pure Python topology validator returning:
  `{ "euler_characteristic": 2, "is_watertight": True, "boundary_edges": 0, "non_manifold_edges": 0 }`.
- Preset files in `terra_forge/schema/presets/` must adhere to `MapConfig` schema and validate with `pydantic`.

---

## Code Layout

```
terra_forge/
├── __init__.py
├── schema/
│   ├── __init__.py
│   ├── map_config.py        # MapConfig, HeightfieldConfig, ErosionConfig, HydrologyConfig, etc.
│   └── presets/
│       ├── alpine_peak_glacial_lake.json
│       ├── temperate_forest_river_basin.json
│       ├── tropical_island_coastal_karst.json
│       └── canyon_semiarid_plateau.json
├── core/
│   ├── __init__.py
│   ├── heightfield.py       # Heightfield2D, strata folding, tectonic scarps, coastal profiles
│   ├── erosion.py           # Momentum droplet hydraulic erosion, 8-neighbor thermal talus
│   ├── hydrology.py         # River stages, water distance field, flow velocity vectors
│   ├── biomes.py            # Microclimate (insolation, lapse rate), Poisson-disk scatter
│   └── diorama_slab.py      # Manifold slab extrusion, boundary walls, base floor
├── blender/
│   ├── __init__.py
│   ├── shaders.py           # 5-layer Object-Space Triplanar PBR, Beer-Lambert water
│   ├── water_builder.py     # River ribbons, lake/bay meshes with watertight freeboard
│   ├── asset_fetcher.py     # Multi-tier procedural flora & detritus models
│   ├── geonodes.py          # Procedural Geometry Nodes scatter modifiers
│   ├── mesh_builder.py      # Diorama BMesh builder, vertex groups, UVs
│   ├── runner.py            # Headless execution pipeline (-b -P)
│   └── addon.py
├── inspector/
│   ├── __init__.py
│   ├── validator.py         # Pure Python topological validator & BMesh validator
│   ├── cameras.py           # Multi-angle vision rig, sun angle & blackbody, volumetric mist
│   └── vision_rig.py
├── cli.py                   # CLI commands (generate, inspect, doctor, batch)
└── mcp_server.py
tests/
├── test_terra_forge.py      # Unit tests for core components
└── e2e/                     # Independent E2E test suite (Tiers 1-4)
```
