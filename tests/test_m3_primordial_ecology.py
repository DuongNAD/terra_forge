"""
Unit and Integration Tests for Milestone M3: Primordial Nature & Organic Ecology.
Verifies:
1. 100% of primordial prompts pass the filter and contain rich positive descriptors.
2. Filter strictly detects and rejects human-biased prompts with appropriate errors (HumanBiasError).
3. Stratification placement rules correctly evaluate slope, flow, aspect, and moisture fields.
4. Upgraded asset_fetcher.py seamlessly loads normalized biological models integrated with
   LocalAssetVault and GeometryNormalizer (min_y == 0.0 bottom pivot, metric scaling, colliders).
"""

from __future__ import annotations

import math
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from terra_forge.ai.vault import FloraType, LocalAssetVault, DEFAULT_FLORA_PROFILES
from terra_forge.ai.normalizer import GeometryNormalizer, extract_mesh_from_glb
from terra_forge.ecology.primordial import (
    FORBIDDEN_HUMAN_TOKENS,
    HumanBiasError,
    PrimordialPromptEngine,
    ValidationResult,
    validate_primordial_prompt,
    STRICT_POSITIVE_DESCRIPTORS,
)
from terra_forge.ecology.stratification import (
    CANOPY_ALPINE_PINE_RULE,
    CANOPY_BROADLEAF_OAK_RULE,
    CLIFF_SCREE_BASALT_RULE,
    FALLEN_TIMBER_RULE,
    FLUVIAL_PEBBLE_RULE,
    SHADOW_MOISTURE_MOSS_FERN_RULE,
    STANDARD_ECOLOGY_RULES,
    WETLAND_REEDS_RULE,
    EcologicalStratifier,
    OrientationRule,
    StratificationRule,
    StratifiedPlacement,
)
from terra_forge.blender.asset_fetcher import (
    PROTOTYPE_FACTORIES,
    PROTOTYPE_TO_FLORA_TYPE,
    BiologicalAssetResult,
    create_biological_mesh,
    fetch_biological_asset,
    resolve_prototype_object,
)


# =============================================================================
# 1. Primordial Prompt Engine & Positive Descriptors Tests
# =============================================================================

class TestPrimordialPromptEngine:
    @pytest.fixture
    def engine(self):
        return PrimordialPromptEngine()

    def test_all_14_flora_types_generate_valid_primordial_prompts(self, engine):
        """Verify 100% of canonical FloraTypes generate valid prompts passing the human bias filter."""
        assert len(FloraType) == 14

        for ft in FloraType:
            prompt = engine.generate_prompt(ft, seed=int(ft))
            assert isinstance(prompt, str)
            assert len(prompt) > 40

            # 100% must pass validation
            res = engine.validate(prompt, strict=True)
            assert res.is_valid is True
            assert len(res.detected_tokens) == 0

    def test_rich_positive_descriptors_presence(self, engine):
        """Verify prompts contain required rich positive descriptors of primordial nature."""
        required_descriptor_substrings = [
            "ancient", "gnarled", "buttress roots", "bedrock fissures", "bark", "foliage"
        ]

        pine_prompt = engine.generate_prompt(FloraType.Pine, seed=42)
        oak_prompt = engine.generate_prompt(FloraType.Round, seed=42)
        decay_prompt = engine.build_decay_prompt("fallen_log", seed=42)
        rock_prompt = engine.build_geological_prompt("fluvial_boulder", seed=42)
        basalt_prompt = engine.build_geological_prompt("columnar_basalt", seed=42)
        moss_prompt = engine.build_ground_flora_prompt("cliff_moss", seed=42)
        fern_prompt = engine.build_ground_flora_prompt("primordial_fern", seed=42)

        # Assert key descriptors
        assert "ironwood" in pine_prompt or "conifer" in pine_prompt
        assert "buttress roots grasping bedrock fissures" in pine_prompt
        assert "storm-weathered asymmetrical crown" in pine_prompt

        assert "broadleaf oak" in oak_prompt or "colossal" in oak_prompt
        assert "buttress roots grasping bedrock fissures" in oak_prompt

        assert "splintered storm windthrow" in decay_prompt
        assert "decaying heartwood hollow with shelf fungi" in decay_prompt

        assert "fluvial rounded river boulders" in rock_prompt.lower()
        assert "fluvial water-worn erosion facets" in rock_prompt.lower()

        assert "columnar basalt" in basalt_prompt.lower()
        assert "hexagonal" in basalt_prompt.lower()

        assert "cliff-clinging moss" in moss_prompt.lower()
        assert "primordial fern fronds" in fern_prompt.lower()

    def test_seed_determinism(self, engine):
        """Verify prompt generation with identical seed is bit-exact deterministic."""
        prompt1 = engine.generate_prompt(FloraType.Pine, seed=12345)
        prompt2 = engine.generate_prompt(FloraType.Pine, seed=12345)
        assert prompt1 == prompt2

        # Different seed chooses different render qualifiers
        prompt3 = engine.generate_prompt(FloraType.Pine, seed=12346)
        assert prompt1 != prompt3

    def test_specialized_archetype_prompts(self, engine):
        """Verify specialized builder methods produce valid prompts without human bias."""
        tree_prompt = engine.build_tree_prompt("acacia", seed=10)
        assert "umbrella-crown savanna ironwood" in tree_prompt.lower()
        assert engine.validate(tree_prompt, strict=True).is_valid

        stump_prompt = engine.build_decay_prompt("stump", seed=11)
        assert "splintered storm windthrow" in stump_prompt.lower()
        assert engine.validate(stump_prompt, strict=True).is_valid

        fungus_prompt = engine.build_ground_flora_prompt("fungus", seed=12)
        assert "bioluminescent shade fungi" in fungus_prompt.lower()
        assert engine.validate(fungus_prompt, strict=True).is_valid

    def test_custom_style_modifiers_validation(self, engine):
        """Verify valid modifiers are appended, while human-biased modifiers are rejected."""
        valid_mod = ["dew-kissed needles", "lichen-encrusted roots"]
        prompt = engine.generate_prompt(FloraType.Pine, style_modifiers=valid_mod)
        assert "dew-kissed needles" in prompt
        assert "lichen-encrusted roots" in prompt
        assert engine.validate(prompt, strict=True).is_valid

        # Reject biased modifier
        biased_mod = ["sawed base"]
        with pytest.raises(HumanBiasError) as exc_info:
            engine.generate_prompt(FloraType.Pine, style_modifiers=biased_mod)
        assert "saw" in exc_info.value.detected_tokens or "sawed" in exc_info.value.detected_tokens


# =============================================================================
# 2. Human Bias Rejection Filter Tests
# =============================================================================

class TestHumanBiasFilter:
    @pytest.mark.parametrize("biased_term", [
        "saw", "saws", "sawed", "sawing", "cut", "cuts", "cutting",
        "lumber", "plank", "planks", "board", "boards",
        "fence", "fences", "post", "posts",
        "ruin", "ruins", "ruined",
        "road", "roads", "path", "paths", "trail", "paved", "asphalt",
        "bench", "benches", "nail", "nails",
        "brick", "bricks", "concrete", "mortar",
        "artificial", "man-made", "manmade",
        "axe", "tool", "hewn", "trimmed",
        "building", "house", "cabin", "cart", "bridge",
        "column base", "stone wall", "paved walkway",
    ])
    def test_forbidden_tokens_strictly_rejected(self, biased_term: str):
        """Verify each forbidden token raises HumanBiasError in strict mode."""
        test_prompt = f"Ancient primordial forest tree with a {biased_term} in the clearing."
        with pytest.raises(HumanBiasError) as exc_info:
            validate_primordial_prompt(test_prompt, strict=True)

        assert any(biased_term.lower() in tok.lower() for tok in exc_info.value.detected_tokens)
        assert exc_info.value.prompt == test_prompt

    def test_non_strict_validation_result(self):
        """Verify strict=False returns clean ValidationResult without throwing an exception."""
        test_prompt = "Old gnarled tree near an abandoned lumber plank road with rusty nails"
        res = validate_primordial_prompt(test_prompt, strict=False)

        assert res.is_valid is False
        assert bool(res) is False
        assert "lumber" in res.detected_tokens
        assert "plank" in res.detected_tokens
        assert "road" in res.detected_tokens
        assert "nails" in res.detected_tokens or "nail" in res.detected_tokens
        assert "Human bias detected" in res.details

    def test_clean_prompts_pass_validation(self):
        """Verify authentic natural prompts with complex vocabulary pass cleanly."""
        clean_prompts = [
            "Colossal primordial twisted ironwood tree, massive gnarled exposed buttress roots grasping bedrock fissures",
            "Storm-toppled rotting tree trunk, splintered fibrous jagged fracture at natural trunk snap point",
            "Smooth rounded riverbed boulder, fluvial water-worn erosion facets, subtle quartz veins",
            "Jagged columnar basalt outcropping, natural tectonic fracture planes, weathered alpine scree talus",
            "Dense velvety cliff-clinging moss carpet adhering tightly to vertical granite bedrock crags",
            "Primordial fern fronds, delicate arched bipinnate foliage, unfurling spiral fiddlehead crosiers",
        ]
        for p in clean_prompts:
            res = validate_primordial_prompt(p, strict=True)
            assert res.is_valid is True
            assert len(res.detected_tokens) == 0

    def test_case_and_boundary_handling(self):
        """Verify case insensitivity and prevent false positives on substrings."""
        # Case variations
        with pytest.raises(HumanBiasError):
            validate_primordial_prompt("ANCIENT TREE WITH SAW MARKS")
        with pytest.raises(HumanBiasError):
            validate_primordial_prompt("pine with CuT bark")
        with pytest.raises(HumanBiasError):
            validate_primordial_prompt("forest clearing near a RuIn")

        # Hyphenated variants
        with pytest.raises(HumanBiasError):
            validate_primordial_prompt("tree with saw-cut base")
        with pytest.raises(HumanBiasError):
            validate_primordial_prompt("artificial man-made clearing")


# =============================================================================
# 3. Organic Ecological Stratification Tests
# =============================================================================

class TestEcologicalStratification:
    @pytest.fixture
    def stratifier(self):
        return EcologicalStratifier()

    def test_canopy_deep_root_cliff_exclusion(self, stratifier):
        """Verify deep-root canopies strictly avoid sheer cliff faces (slope > 35 deg)."""
        pine_rule = CANOPY_ALPINE_PINE_RULE
        assert pine_rule.max_slope_deg == 35.0

        # Point evaluation on gentle slope vs vertical cliff
        gentle_slope_score = stratifier.evaluate_point(
            rule=pine_rule,
            elevation=15.0,
            slope_deg=10.0,
            flow=5.0,
            insolation=0.8,
            moisture=0.5,
        )
        assert gentle_slope_score > 0.5, "Expected high suitability on gentle 10° slope"

        # Hard slope ceiling: slope > 35° must strictly evaluate to 0.0
        steep_cliff_score = stratifier.evaluate_point(
            rule=pine_rule,
            elevation=15.0,
            slope_deg=38.0,
            flow=5.0,
            insolation=0.8,
            moisture=0.5,
        )
        assert steep_cliff_score == 0.0, "Canopy tree must strictly evaluate to 0.0 on cliff > 35°"

        sheer_wall_score = stratifier.evaluate_point(
            rule=pine_rule,
            elevation=15.0,
            slope_deg=65.0,
            flow=5.0,
            insolation=0.8,
            moisture=0.5,
        )
        assert sheer_wall_score == 0.0

    def test_cliff_moss_thrives_on_steep_slopes(self, stratifier):
        """Verify cliff moss thrives on steep slopes (> 35 deg) where trees cannot grow."""
        moss_rule = SHADOW_MOISTURE_MOSS_FERN_RULE
        assert moss_rule.max_slope_deg >= 70.0

        # On a 50° shadowed, moist cliff face
        cliff_moss_score = stratifier.evaluate_point(
            rule=moss_rule,
            elevation=20.0,
            slope_deg=50.0,
            flow=10.0,
            insolation=0.2,   # Shadowed aspect
            moisture=0.85,    # High humidity
        )
        assert cliff_moss_score > 0.5, "Cliff moss must thrive on steep moist shadowed cliffs"

        # Canopy oak tree on the exact same cliff evaluates to 0.0
        oak_score = stratifier.evaluate_point(
            rule=CANOPY_BROADLEAF_OAK_RULE,
            elevation=20.0,
            slope_deg=50.0,
            flow=10.0,
            insolation=0.2,
            moisture=0.85,
        )
        assert oak_score == 0.0

    def test_fallen_timber_slope_gradient_orientation(self, stratifier):
        """Verify fallen logs on slopes orient their long axis along gradient descent."""
        rule = FALLEN_TIMBER_RULE
        assert rule.orientation_rule == OrientationRule.SLOPE_GRADIENT

        # Sloping downward along +X (dz/dx = -0.5, dz/dy = 0.0) -> downslope vector (-dz/dx, -dz/dy) = (+0.5, 0.0)
        # Downslope yaw should be 0 radians (along +X)
        euler, quat = stratifier.calculate_orientation(
            rule=rule,
            slope_gradient=(-0.5, 0.0),
            flow_vector=None,
        )
        pitch, yaw, roll = euler
        assert abs(yaw - 0.0) < 1e-3, f"Expected yaw ~0 along downslope +X, got {yaw}"

        # Sloping downward along +Y (dz/dx = 0.0, dz/dy = -0.6) -> downslope vector (0.0, +0.6)
        # Downslope yaw should be +pi/2 (90 deg)
        euler_y, _ = stratifier.calculate_orientation(
            rule=rule,
            slope_gradient=(0.0, -0.6),
            flow_vector=None,
        )
        assert abs(euler_y[1] - (math.pi / 2.0)) < 1e-3, f"Expected yaw ~pi/2, got {euler_y[1]}"

    def test_fallen_timber_stream_flow_alignment(self, stratifier):
        """Verify fallen logs caught in stream flow orient along the flow velocity vector."""
        rule = FALLEN_TIMBER_RULE
        # Flow vector pointing diagonal (vx=10.0, vy=10.0) -> angle pi/4 (45 deg)
        euler, quat = stratifier.calculate_orientation(
            rule=rule,
            slope_gradient=(0.0, 0.0),
            flow_vector=(10.0, 10.0),
        )
        yaw = euler[1]
        assert abs(yaw - (math.pi / 4.0)) < 1e-3, f"Expected yaw ~pi/4 along flow vector, got {yaw}"

        # Check quaternion is valid unit quaternion
        qx, qy, qz, qw = quat
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        assert abs(norm_sq - 1.0) < 1e-4

    def test_fluvial_pebble_stream_bed_deposition(self, stratifier):
        """Verify fluvial riverbed pebbles deposit strictly where stream flow velocity is significant."""
        rule = FLUVIAL_PEBBLE_RULE

        # Active stream channel: flow = 60.0, low slope 5°, high moisture 0.95
        stream_bed_score = stratifier.evaluate_point(
            rule=rule,
            elevation=2.0,
            slope_deg=5.0,
            flow=60.0,
            insolation=0.5,
            moisture=0.95,
        )
        assert stream_bed_score > 0.6, "Expected high fluvial pebble suitability in stream channel"

        # Dry upland plateau: flow = 0.0, slope 5°, moisture 0.3
        dry_upland_score = stratifier.evaluate_point(
            rule=rule,
            elevation=2.0,
            slope_deg=5.0,
            flow=0.0,
            insolation=0.5,
            moisture=0.3,
        )
        assert dry_upland_score == 0.0, "Pebble suitability must be 0.0 outside active water channels"

    def test_shadow_and_moisture_microclimate_niche(self, stratifier):
        """Verify moisture-loving shade flora concentrates in shadowed, high-humidity zones."""
        rule = SHADOW_MOISTURE_MOSS_FERN_RULE

        # Damp shadowed northern hollow: moisture = 0.90, insolation = 0.15 (dark shadow)
        shaded_damp_score = stratifier.evaluate_point(
            rule=rule,
            elevation=10.0,
            slope_deg=20.0,
            flow=10.0,
            insolation=0.15,
            moisture=0.90,
        )
        assert shaded_damp_score > 0.6, "Expected high suitability in shaded damp zone"

        # Sun-scorched arid slope: moisture = 0.20, insolation = 0.95 (blazing sun)
        sun_scorched_score = stratifier.evaluate_point(
            rule=rule,
            elevation=10.0,
            slope_deg=20.0,
            flow=10.0,
            insolation=0.95,
            moisture=0.20,
        )
        assert sun_scorched_score == 0.0, "Expected 0.0 suitability on arid sun-scorched slope"

    def test_vectorized_suitability_map_dimensions_and_bounds(self, stratifier):
        """Verify compute_suitability_map operates on 2D arrays and bounds all values in [0.0, 1.0]."""
        dim = 64
        elevation = np.linspace(2.0, 40.0, dim * dim).reshape((dim, dim))
        slope = np.linspace(0.0, 60.0, dim * dim).reshape((dim, dim))
        moisture = np.full((dim, dim), 0.6, dtype=np.float32)
        insolation = np.full((dim, dim), 0.5, dtype=np.float32)
        flow = np.full((dim, dim), 15.0, dtype=np.float32)

        suit_map = stratifier.compute_suitability_map(
            rule=CANOPY_ALPINE_PINE_RULE,
            elevation=elevation,
            slope_deg=slope,
            moisture=moisture,
            insolation=insolation,
            flow=flow,
        )

        assert suit_map.shape == (dim, dim)
        assert suit_map.dtype == np.float32
        assert float(np.min(suit_map)) >= 0.0
        assert float(np.max(suit_map)) <= 1.0

        # Cells with slope > 35° must have exactly 0.0
        cliff_mask = slope > 35.0
        assert np.all(suit_map[cliff_mask] == 0.0)

    def test_generate_stratified_points_generation(self, stratifier):
        """Verify generate_stratified_points produces valid physics-aligned placements."""
        dim = 32
        elev = np.full((dim, dim), 10.0, dtype=np.float32)
        # Gentle slope 10°
        slope = np.full((dim, dim), 10.0, dtype=np.float32)
        moist = np.full((dim, dim), 0.6, dtype=np.float32)
        insol = np.full((dim, dim), 0.5, dtype=np.float32)
        flow = np.full((dim, dim), 10.0, dtype=np.float32)

        placements = stratifier.generate_stratified_points(
            rule=CANOPY_ALPINE_PINE_RULE,
            elevation=elev,
            slope_deg=slope,
            moisture=moist,
            insolation=insol,
            flow=flow,
            world_bounds=(-50.0, -50.0, 50.0, 50.0),
            max_instances=30,
            seed=42,
        )

        assert len(placements) > 5
        for p in placements:
            assert isinstance(p, StratifiedPlacement)
            assert -50.0 <= p.x <= 50.0
            assert -50.0 <= p.z <= 50.0
            assert p.suitability > 0.2
            assert p.scale > 0.5
            # Unit quaternion check
            qx, qy, qz, qw = p.rotation_quaternion
            assert abs(qx * qx + qy * qy + qz * qz + qw * qw - 1.0) < 1e-3


# =============================================================================
# 4. Upgraded asset_fetcher.py & Vault / GeometryNormalizer Integration Tests
# =============================================================================

class TestAssetFetcherIntegration:
    @pytest.fixture
    def temp_vault(self):
        tmp = Path(tempfile.mkdtemp(prefix="terra_forge_test_af_vault_"))
        vault = LocalAssetVault(vault_dir=tmp)
        yield vault
        shutil.rmtree(tmp, ignore_errors=True)

    def test_create_biological_mesh_pure_python(self):
        """Verify authentic biological mesh generators generate clean 3D topology in pure Python."""
        prototypes = [
            "pine", "broadleaf", "palm", "shrub", "wildflower",
            "grass", "fern", "mushroom", "moss", "log", "stump",
            "seaweed", "reeds", "pebble", "coral", "boulder",
        ]
        for proto in prototypes:
            verts, tris = create_biological_mesh(proto, target_height=6.0)
            assert len(verts) >= 8, f"Prototype {proto} has too few vertices: {len(verts)}"
            assert len(tris) >= 4, f"Prototype {proto} has too few triangles: {len(tris)}"
            assert verts.shape[1] == 3
            assert tris.shape[1] == 3

    def test_fetch_biological_asset_normalization_guarantees(self):
        """Verify fetch_biological_asset guarantees min_y == 0.0 bottom pivot, centered X/Z, metric height, and colliders."""
        target_h = 7.5
        result = fetch_biological_asset("pine_tall", target_height=target_h, save_to_vault=False)

        assert isinstance(result, BiologicalAssetResult)
        assert result.name == "pine_tall"
        assert result.flora_type == FloraType.Pine
        assert result.target_height == target_h

        # Invariant 1: Bottom pivot min_y == 0.0
        norm_mesh = result.normalized_mesh
        assert norm_mesh is not None
        min_y = float(np.min(norm_mesh.vertices[:, 1]))
        assert abs(min_y) < 1e-5, f"Expected min_y == 0.0, got {min_y}"

        # Invariant 2: Metric height == target_h
        max_y = float(np.max(norm_mesh.vertices[:, 1]))
        assert abs(max_y - target_h) < 1e-5, f"Expected max_y == {target_h}, got {max_y}"

        # Invariant 3: Centered horizontal bounds
        min_x = float(np.min(norm_mesh.vertices[:, 0]))
        max_x = float(np.max(norm_mesh.vertices[:, 0]))
        min_z = float(np.min(norm_mesh.vertices[:, 2]))
        max_z = float(np.max(norm_mesh.vertices[:, 2]))
        assert abs(min_x + max_x) < 1e-4
        assert abs(min_z + max_z) < 1e-4

        # Invariant 4: Collision primitives extracted
        colliders = result.collision_primitives
        assert "cylinder" in colliders
        assert "aabb" in colliders
        assert "convex_hull" in colliders
        assert colliders["cylinder"]["radius"] > 0.0
        assert colliders["cylinder"]["height"] == target_h
        assert colliders["cylinder"]["center"] == [0.0, target_h / 2.0, 0.0]

    def test_fetch_biological_asset_vault_roundtrip(self, temp_vault):
        """Verify fetch_biological_asset saves to LocalAssetVault and subsequently loads from cache."""
        proto = "alpine_pine"
        target_h = 8.0

        # Pass 1: Cold fetch -> creates procedural normalized asset and stores into vault
        assert len(temp_vault) == 0
        res1 = fetch_biological_asset(proto, vault=temp_vault, target_height=target_h, save_to_vault=True)

        assert res1.source == "procedural_normalized"
        assert res1.model_path is not None
        assert res1.model_path.is_file()
        assert len(temp_vault) == 1

        # Check GLB magic bytes on disk
        with open(res1.model_path, "rb") as f:
            glb_header = f.read(12)
        assert glb_header[:4] == b"glTF"

        # Pass 2: Cache hit -> loads model from vault
        res2 = fetch_biological_asset(proto, vault=temp_vault, target_height=target_h)
        assert res2.source == "vault"
        assert res2.model_path == res1.model_path
        assert res2.is_normalized is True
        assert res2.target_height == target_h

    def test_resolve_prototype_object_pure_python(self):
        """Verify resolve_prototype_object works without Blender, returning valid proxy objects."""
        all_prototypes = [
            "pine_tall", "broadleaf", "palm_tropical", "shrub",
            "wildflower", "grass", "fern", "mushroom", "fungus",
            "rock_moss", "fallen_log", "snapped_stump", "seaweed",
            "reeds", "riverbed_pebble", "coral_reef", "rock_boulder",
        ]

        for p in all_prototypes:
            obj = resolve_prototype_object(p)
            assert obj is not None, f"Failed to generate {p}"
            assert len(obj.data.vertices) > 0, f"No vertices for {p}"
            assert len(obj.data.polygons) > 0, f"No polygons for {p}"
            assert len(obj.data.materials) > 0, f"No material for {p}"
            assert "collision_primitives" in obj or "cylinder" in str(obj.get("collision_primitives", {}))

    def test_all_14_canonical_flora_types_mapped(self):
        """Verify PROTOTYPE_TO_FLORA_TYPE covers key prototypes and maps to valid FloraType members."""
        assert len(PROTOTYPE_TO_FLORA_TYPE) >= 17

        for proto_name, flora_enum in PROTOTYPE_TO_FLORA_TYPE.items():
            assert isinstance(flora_enum, FloraType)
            assert flora_enum in DEFAULT_FLORA_PROFILES
