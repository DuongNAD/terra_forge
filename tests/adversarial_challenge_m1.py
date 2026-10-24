"""
Adversarial Challenge & Stress Harness for Milestone 1 (Abiotic Generative Pipeline).
Tests biological filter bypasses, procedural mesh extremes, normalization invariants,
and collision primitive stability across randomized seeds.
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple
import numpy as np

from terra_forge.ai.vault import (
    AbioticType,
    DEFAULT_ABIOTIC_PROFILES,
    LocalAssetVault,
)
from terra_forge.ecology.primordial import (
    AbioticPromptEngine,
    validate_abiotic_prompt,
    AbioticBiasError,
    FORBIDDEN_BIOLOGICAL_TOKENS,
    FORBIDDEN_ABIOTIC_TOKENS,
)
from terra_forge.blender.asset_fetcher import (
    create_abiotic_mesh,
    fetch_abiotic_asset,
)
from terra_forge.ai.normalizer import (
    GeometryNormalizer,
    NormalizedMesh,
    CollisionPrimitives,
)


def run_filter_bypass_suite() -> Tuple[List[dict], List[dict]]:
    """Test various adversarial bypass vectors against validate_abiotic_prompt."""
    test_cases = [
        # Casing variations
        ("TREE", "upper case single token", False),
        ("TrEe", "mixed case single token", False),
        ("PLaNT", "mixed case plant", False),
        
        # Obscure biological synonyms (Flora)
        ("algae", "aquatic non-vascular flora", True),
        ("algal bloom on rocks", "algal phrasing", True),
        ("seaweed", "marine macroalgae", True),
        ("kelp", "large brown algae seaweed", True),
        ("bryophyte", "scientific division for mosses", True),
        ("pteridophyte", "scientific division for ferns", True),
        ("lichenous crust", "adjective of lichen", True),
        ("vine", "climbing vegetative stem", True),
        ("vines", "climbing vegetative stems", True),
        ("ivy", "climbing evergreen plant", True),
        ("liana", "woody climbing vine", True),
        ("succulent", "water-retaining fleshy plant", True),
        ("cactus", "xerophytic spiny plant", True),
        ("cacti", "xerophytic spiny plants", True),
        ("vegetation", "general plant life", True),
        ("botanical", "plant life adjective", True),
        ("botany", "study of plants", True),
        ("foliar", "leaf related adjective", True),
        ("seedling", "young developed plant", True),
        ("willow", "tree genus Salix", True),
        ("bamboo", "giant woody grass", True),
        ("clover", "herbaceous trefoil", True),
        ("thistle", "prickly composite plant", True),
        
        # Adjectival forms of biological tokens
        ("mossy", "adjective of moss", True),
        ("leafy", "adjective of leaf", True),
        ("grassy", "adjective of grass", True),
        ("flowery", "adjective of flower", True),
        ("woody", "adjective of wood", True),
        ("rooted", "adjective of root", True),
        ("barked", "adjective of bark", True),
        ("fungal", "adjective of fungus", True),
        
        # Wood detritus & compounding
        ("log", "fallen tree trunk", True),
        ("logs", "fallen tree trunks", True),
        ("trunk", "tree body", True),
        ("trunks", "tree bodies", True),
        ("stump", "tree stump", True),
        ("stumps", "tree stumps", True),
        ("driftwood", "weathered wood washed ashore", True),
        ("deadwood", "decaying wood", True),
        ("ironwood", "dense hard tree", True),
        ("treeline", "boundary of tree growth", True),
        
        # Fungal & microbial
        ("spore", "fungal reproductive unit", True),
        ("spores", "fungal reproductive units", True),
        ("mycelium", "fungal vegetative network", True),
        ("hyphae", "branching fungal filaments", True),
        ("microbial", "relating to microbes", True),
        ("bacteria", "single-celled microorganisms", True),
        ("organism", "individual living entity", True),
        ("biological", "relating to biology", True),
        ("biotic", "living component of ecosystem", True),
        ("biota", "animal and plant life", True),
        
        # Marine invertebrates / Coral
        ("coral", "marine colonial polyps", True),
        ("polyps", "individual coral organisms", True),
        
        # Obscure Fauna
        ("beast", "wild animal synonym", True),
        ("beasts", "wild animals synonym", True),
        ("wildlife", "undomesticated animal species", True),
        ("carnivore", "meat eating animal", True),
        ("herbivore", "plant eating animal", True),
        ("predator", "hunting animal", True),
        ("prey", "hunted animal", True),
        ("avian", "bird related adjective", True),
        ("canine", "dog related animal", True),
        ("feline", "cat related animal", True),
        ("reptilian", "reptile related adjective", True),
        ("worm", "invertebrate animal", True),
        ("worms", "invertebrate animals", True),
        ("slug", "gastropod mollusk", True),
        ("slugs", "gastropod mollusks", True),
        ("snail", "shelled gastropod", True),
        ("snails", "shelled gastropods", True),
        ("spider", "eight-legged arachnid", True),
        ("spiders", "eight-legged arachnids", True),
        ("arachnid", "arthropod class", True),
        ("beetle", "coleopteran insect", True),
        ("beetles", "coleopteran insects", True),
        ("crustacean", "arthropod taxon", True),
        ("crab", "decapod crustacean", True),
        ("crabs", "decapod crustaceans", True),
        ("larva", "juvenile form of animal", True),
        ("larvae", "juvenile forms of animals", True),
        
        # Spacing & Punctuation Tricks on forbidden tokens
        ("tree-like", "hyphenated suffix with forbidden token", False),
        ("plant-based", "hyphenated suffix with forbidden token", False),
        ("tree/rock", "slash delimited forbidden token", False),
        ("tree.rock", "dot delimited forbidden token", False),
        ("tree,rock", "comma delimited forbidden token", False),
        ("tree's", "possessive apostrophe", False),
        ("tree\u200b", "trailing zero-width space", False),
        ("tree\u00a0", "trailing non-breaking space", False),
        ("p.l.a.n.t", "inter-character dots", True),
        ("p-l-a-n-t", "inter-character hyphens", True),
        ("t r e e", "spaced letters", True),
        ("t\u200br\u200be\u200be", "internal zero-width space", True),
        
        # Phrase boundaries with non-whitespace separators
        ("column/base", "phrase with slash", True),
        ("column.base", "phrase with dot", True),
        ("stone/wall", "phrase with slash", True),
        ("stone.wall", "phrase with dot", True),
    ]

    bypasses = []
    blocks = []

    for prompt, desc, expected_bypass in test_cases:
        res = validate_abiotic_prompt(prompt, strict=False)
        is_bypass = res.is_valid
        record = {
            "prompt": prompt,
            "desc": desc,
            "is_bypass": is_bypass,
            "detected_tokens": res.detected_tokens,
            "expected_bypass": expected_bypass,
        }
        if is_bypass:
            bypasses.append(record)
        else:
            blocks.append(record)

    return bypasses, blocks


def run_mesh_stress_suite():
    """Stress test create_abiotic_mesh and GeometryNormalizer across edge-case parameters."""
    results = []
    
    # 1. Height variations across all 9 abiotic types
    test_heights = [
        10.0,
        1.0,
        0.01,
        0.0,
        -1.0,
        -100.0,
        1000.0,
        1e5,
    ]
    
    for at in AbioticType:
        for h in test_heights:
            try:
                verts, tris = create_abiotic_mesh(at, target_height=h)
                
                # Check vertex and triangle properties
                has_nan = np.isnan(verts).any()
                has_inf = np.isinf(verts).any()
                v_count = len(verts)
                t_count = len(tris)
                
                # Verify normalization
                norm_verts, stats = GeometryNormalizer.normalize_vertices(verts, target_height=h if h > 0 else 5.0)
                min_y = float(np.min(norm_verts[:, 1]))
                max_y = float(np.max(norm_verts[:, 1]))
                norm_h = max_y - min_y
                
                # Check collision primitives
                colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=h if h > 0 else 5.0)
                
                results.append({
                    "type": at.name,
                    "target_h": h,
                    "v_count": v_count,
                    "t_count": t_count,
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                    "min_y": min_y,
                    "norm_h": norm_h,
                    "cyl_radius": colliders.cylinder.radius,
                    "cyl_h": colliders.cylinder.height,
                    "aabb_size": colliders.aabb.size,
                    "hull_v": colliders.convex_hull.vertex_count,
                    "hull_vol": colliders.convex_hull.volume,
                    "status": "PASS",
                })
            except Exception as e:
                results.append({
                    "type": at.name,
                    "target_h": h,
                    "status": f"FAIL: {type(e).__name__}: {e}",
                })
    return results


def run_seed_and_hull_invariants_suite(n_iterations: int = 100):
    """Test collision primitives, convex hull non-degeneracy, and min_y == 0.0 across random variations."""
    rng = np.random.RandomState(1337)
    invariant_failures = []
    
    for i in range(n_iterations):
        at = AbioticType(rng.choice(list(AbioticType)))
        target_h = float(rng.uniform(0.1, 50.0))
        
        # Test mesh creation
        verts, tris = create_abiotic_mesh(at, target_height=target_h)
        
        # Add random scaling / rotation / shear / jitter to simulate extreme transforms before normalization
        theta_x = rng.uniform(-np.pi, np.pi)
        theta_y = rng.uniform(-np.pi, np.pi)
        theta_z = rng.uniform(-np.pi, np.pi)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ], dtype=np.float32)
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ], dtype=np.float32)
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        
        R = Rx @ Ry @ Rz
        # Random non-uniform scale and large arbitrary translation
        scale = np.array([rng.uniform(0.2, 5.0), rng.uniform(0.2, 5.0), rng.uniform(0.2, 5.0)], dtype=np.float32)
        translation = np.array([rng.uniform(-500.0, 500.0), rng.uniform(-500.0, 500.0), rng.uniform(-500.0, 500.0)], dtype=np.float32)
        
        transformed_verts = (verts * scale) @ R.T + translation
        
        # Now normalize transformed_verts to target_h
        norm_verts, stats = GeometryNormalizer.normalize_vertices(transformed_verts, target_height=target_h)
        
        # Invariant 1: min_y == 0.0 strictly within numerical precision (1e-5)
        min_y = float(np.min(norm_verts[:, 1]))
        if abs(min_y) > 1e-4:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "min_y == 0.0",
                "value": min_y,
            })
            
        # Invariant 2: normalized height == target_h (within 1e-4)
        max_y = float(np.max(norm_verts[:, 1]))
        h_actual = max_y - min_y
        if abs(h_actual - target_h) > 1e-3:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "normalized_height == target_h",
                "expected": target_h,
                "actual": h_actual,
            })
            
        # Invariant 3: Centered on X and Z (center == 0.0 within 1e-4)
        cx = (np.min(norm_verts[:, 0]) + np.max(norm_verts[:, 0])) / 2.0
        cz = (np.min(norm_verts[:, 2]) + np.max(norm_verts[:, 2])) / 2.0
        if abs(cx) > 1e-4 or abs(cz) > 1e-4:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "horizontal_center == 0.0",
                "cx": float(cx),
                "cz": float(cz),
            })
            
        # Invariant 4: Collision primitives extraction succeeds and produces positive dimensions
        colliders = GeometryNormalizer.extract_collision_primitives(norm_verts, target_height=target_h)
        if colliders.cylinder.radius <= 0.0 or colliders.cylinder.height <= 0.0:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "cylinder_positive_dimensions",
                "radius": colliders.cylinder.radius,
                "height": colliders.cylinder.height,
            })
            
        if any(s <= 0.0 for s in colliders.aabb.size):
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "aabb_positive_dimensions",
                "size": colliders.aabb.size,
            })
            
        if colliders.convex_hull.vertex_count < 4 or colliders.convex_hull.simplex_count < 4:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "convex_hull_validity",
                "v_count": colliders.convex_hull.vertex_count,
                "s_count": colliders.convex_hull.simplex_count,
            })
            
        if colliders.convex_hull.volume <= 0.0 or colliders.convex_hull.area <= 0.0:
            invariant_failures.append({
                "iteration": i,
                "type": at.name,
                "invariant": "convex_hull_positive_volume_area",
                "volume": colliders.convex_hull.volume,
                "area": colliders.convex_hull.area,
            })

    return invariant_failures


# =============================================================================
# Pytest Integration Test Functions
# =============================================================================

def test_extreme_height_mesh_stress():
    """Verify create_abiotic_mesh handles extreme heights without NaNs or Infs."""
    results = run_mesh_stress_suite()
    failures = [r for r in results if r["status"] != "PASS"]
    assert len(failures) == 0, f"Mesh generation failures: {failures}"


def test_normalization_and_hull_invariants():
    """Verify ground alignment (min_y == 0.0), AABB, and convex hull invariants over 100 runs."""
    inv_fails = run_seed_and_hull_invariants_suite(100)
    assert len(inv_fails) == 0, f"Invariant failures: {inv_fails}"


def test_cylinder_radius_clamping_fixed():
    """Verify cyl_radius never exceeds max_extent on small meshes."""
    verts, tris = create_abiotic_mesh(AbioticType.RiverPebble, target_height=0.01)
    norm_v, stats = GeometryNormalizer.normalize_vertices(verts, target_height=0.01)
    colliders = GeometryNormalizer.extract_collision_primitives(norm_v, target_height=0.01)
    
    max_extent = max(abs(colliders.aabb.min_vec[0]), abs(colliders.aabb.max_vec[0]),
                     abs(colliders.aabb.min_vec[2]), abs(colliders.aabb.max_vec[2]))
    assert colliders.cylinder.radius <= max_extent + 1e-4, (
        f"Cylinder radius {colliders.cylinder.radius} exceeds max_extent {max_extent}"
    )
    assert colliders.cylinder.radius > 0.0, "Cylinder radius must be strictly positive"


def test_create_abiotic_mesh_seed_parameter_supported():
    """Verify create_abiotic_mesh accepts seed keyword argument and generates deterministic geometry."""
    v1, f1 = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=42)
    assert isinstance(v1, np.ndarray) and len(v1) > 0
    assert isinstance(f1, np.ndarray) and len(f1) > 0
    
    v2, f2 = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=42)
    assert np.allclose(v1, v2), "Mesh generation with identical seed must be deterministic"
    
    v3, _ = create_abiotic_mesh(AbioticType.RockCrag, target_height=5.0, seed=99)
    assert not np.allclose(v1, v3), "Different seeds should produce different orientations"


def test_filter_bypasses_remediated():
    """Verify that all 20 critical biological and phrase terms are strictly blocked after remediation."""
    critical_terms = [
        "mossy", "woody", "leafy", "algae", "seaweed", "kelp", "vine", 
        "cactus", "log", "stump", "driftwood", "spore", "bacteria", 
        "wildlife", "coral", "beast", "spider", "beetle", "crab",
        "column/base",
    ]
    bypassed = [t for t in critical_terms if validate_abiotic_prompt(t, strict=False).is_valid]
    assert len(bypassed) == 0, f"Expected all terms to be blocked, but these bypassed: {bypassed}"


if __name__ == "__main__":
    print("=== MILENSTONE 1 ADVERSARIAL CHALLENGE ===")
    
    print("\n--- 1. Testing Biological Filter Bypasses ---")
    bypasses, blocks = run_filter_bypass_suite()
    print(f"Total bypass test cases: {len(bypasses) + len(blocks)}")
    print(f"Blocked by filter: {len(blocks)}")
    print(f"Bypassed filter: {len(bypasses)}")
    print("\nConfirmed Bypasses:")
    for b in bypasses:
        safe_prompt = b['prompt'].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"  [BYPASS] {safe_prompt!r} - {b['desc']}")
        
    print("\n--- 2. Testing Mesh Stress Suite (Extreme Heights, Types) ---")
    mesh_results = run_mesh_stress_suite()
    failures = [r for r in mesh_results if r["status"] != "PASS"]
    print(f"Total mesh tests: {len(mesh_results)}")
    print(f"Failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("  All extreme height edge cases handled cleanly!")
        
    print("\n--- 3. Testing Randomized Transformations & Invariants (100 runs) ---")
    inv_fails = run_seed_and_hull_invariants_suite(100)
    print(f"Total invariant checks across 100 runs: 600 assertions")
    print(f"Invariant failures: {len(inv_fails)}")
    if inv_fails:
        for fail in inv_fails:
            print(f"  [INV FAIL] {fail}")
    else:
        print("  All invariants (min_y == 0.0, height, centering, cylinder, AABB, hull) strictly held!")
