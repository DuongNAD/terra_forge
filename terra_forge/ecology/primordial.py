"""
Primordial Nature Prompt Engine & Human Bias Rejection Filter for TerraForge.
Generates authentic 3D generative prompts for Meshy AI v2 Text-to-3D and procedural pipelines,
enforcing strict positive descriptors of untouched primordial phenomena and a zero-tolerance
filter against human artifacts (saws, cuts, lumber, ruins, roads, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from terra_forge.ai.vault import AbioticType, FloraType


class HumanBiasError(ValueError):
    """Raised when a prompt contains human-biased artifacts, tools, or man-made elements."""

    def __init__(self, message: str, detected_tokens: List[str], prompt: str):
        super().__init__(message)
        self.detected_tokens = detected_tokens
        self.prompt = prompt


@dataclass
class ValidationResult:
    """Outcome of validating a prompt against primordial nature constraints."""

    is_valid: bool
    detected_tokens: List[str] = field(default_factory=list)
    clean_prompt: str = ""
    details: str = ""

    def __bool__(self) -> bool:
        return self.is_valid


# -----------------------------------------------------------------------------
# Canonical Forbidden Human Bias Tokens & Phrases
# -----------------------------------------------------------------------------

FORBIDDEN_HUMAN_TOKENS: Set[str] = {
    # Cutting & Carpentry tools and marks
    "saw", "saws", "sawed", "sawing", "sawcut", "sawmark",
    "cut", "cuts", "cutting",
    "axe", "axes", "ax", "hatchet", "chisel", "hammer",
    "tool", "tools",
    "hewn", "trimmed", "carving", "machined", "carpentry",

    # Processed wood / Construction lumber
    "lumber", "plank", "planks", "board", "boards",
    "beam", "beams", "rafter", "rafters", "plywood", "timberland",
    "log cabin", "scaffolding", "furniture", "bench", "benches", "table",

    # Structural, Architectural & Civil works
    "ruin", "ruins", "ruined",
    "column base", "pedestal", "monument", "statue", "temple",
    "road", "roads", "path", "paths", "trail", "trails", "walkway", "pathway",
    "paved", "pavement", "asphalt", "cobblestone", "sidewalk",
    "bridge", "bridges", "pier", "dock", "wharf",
    "fence", "fences", "fenced", "fencing", "post", "posts", "gate", "gates",
    "wall", "walls", "stone wall", "brick", "bricks", "mortar", "concrete", "cement",
    "house", "houses", "building", "buildings", "cabin", "cabins", "hut", "shack", "cottage",

    # Fasteners, Hardware & Human items
    "nail", "nails", "nailed", "screw", "screws", "bolt", "bolts", "wire", "cable",
    "cart", "carts", "wheel", "wheels", "wagon",
    "lantern", "lanterns", "lamp", "lamps", "torch", "candle",

    # Abstract Human / Artificial tokens
    "artificial", "man-made", "manmade", "man made", "human", "anthropogenic",
    "manufactured", "synthetic", "domesticated", "cultivated",
}

# Multi-word phrase triggers evaluated with direct substring / phrase boundary checks
FORBIDDEN_PHRASES: List[str] = [
    "column base",
    "stone wall",
    "paved walkway",
    "paved road",
    "paved path",
    "man-made",
    "man made",
    "log cabin",
    "retaining wall",
    "fence post",
    "wooden plank",
    "wood plank",
    "cut wood",
    "saw mark",
    "saw cut",
    "hand-carved",
    "hand carved",
    "wood-carved",
    "wood carved",
    "carved wood",
    "carved stone",
]


def _compile_forbidden_regex() -> Tuple[re.Pattern, List[re.Pattern]]:
    """Compile optimized regexes for forbidden single tokens and phrases."""
    # Single word tokens require word boundaries \b...\b
    word_tokens = [t for t in FORBIDDEN_HUMAN_TOKENS if " " not in t and "-" not in t]
    single_word_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(word_tokens, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    phrase_patterns = []
    for p in FORBIDDEN_PHRASES:
        words = re.split(r"[\s\-]+", p)
        pat = r"\b" + r"[\s/\.\-_]+".join(re.escape(w) for w in words) + r"\b"
        phrase_patterns.append(re.compile(pat, re.IGNORECASE))
    return single_word_pattern, phrase_patterns


_SINGLE_WORD_REGEX, _PHRASE_REGEXES = _compile_forbidden_regex()


def validate_primordial_prompt(
    prompt: str,
    strict: bool = True,
) -> ValidationResult:
    """
    Validates a prompt against the zero-tolerance primordial nature filter.

    Parameters:
        prompt: Text prompt to inspect.
        strict: If True, raises HumanBiasError upon finding any forbidden token.
                If False, returns a ValidationResult with is_valid=False.

    Returns:
        ValidationResult indicating whether prompt is pristine primordial nature.
    """
    cleaned = prompt.strip().replace("_", " ")
    detected: List[str] = []

    # 1. Check multi-word phrases first
    for phrase_re in _PHRASE_REGEXES:
        for match in phrase_re.finditer(cleaned):
            match_str = match.group(0).lower()
            if match_str not in detected:
                detected.append(match_str)

    # 2. Check single-word tokens with word boundary
    for match in _SINGLE_WORD_REGEX.finditer(cleaned):
        token = match.group(0).lower()
        if token not in detected:
            detected.append(token)

    if detected:
        details = (
            f"Human bias detected: found forbidden artifact token(s) {detected!r}. "
            "Primordial prompts must have zero tolerance for human artifacts, tools, or structures."
        )
        if strict:
            raise HumanBiasError(details, detected_tokens=detected, prompt=cleaned)
        return ValidationResult(
            is_valid=False,
            detected_tokens=detected,
            clean_prompt=cleaned,
            details=details,
        )

    return ValidationResult(
        is_valid=True,
        detected_tokens=[],
        clean_prompt=cleaned,
        details="Prompt verified pristine primordial nature (zero human bias detected).",
    )


# -----------------------------------------------------------------------------
# Strict Positive Descriptors Library
# -----------------------------------------------------------------------------

STRICT_POSITIVE_DESCRIPTORS: Dict[str, List[str]] = {
    # Core ancient trees & root systems
    "ancient_trees": [
        "ancient gnarled ironwood",
        "buttress roots grasping bedrock fissures",
        "deeply furrowed ancient rough bark with hanging lichens",
        "storm-weathered asymmetrical crown",
        "monolithic primordial broadleaf canopy",
        "gnarled twisted alpine conifer boughs",
        "thick ancient foliage canopy",
        "flared buttress root flare plunging into rocky substrate",
        "massive sprawling scaffold branches",
        "dense evergreen needle whorls",
    ],
    # Tree decay, windthrow & detritus
    "decay_windthrow": [
        "splintered storm windthrow",
        "storm-toppled rotting tree trunk collapsed under heavy gale",
        "jagged splintered fibrous natural snap point",
        "decaying heartwood hollow with shelf fungi and sprouting conks",
        "moist moss-draped decaying bark partially sunken into humus",
        "weathered fractured stump with flared buttress roots",
        "natural windthrow collapse rotting into fertile forest floor",
        "fibrous split fracture along grain of fallen trunk",
    ],
    # Fluvial hydrology & water erosion
    "fluvial_hydrology": [
        "fluvial rounded river boulders",
        "water-smoothed riverbed pebbles",
        "fluvial water-worn erosion facets",
        "smooth water-worn river stone surface with quartz veins",
        "torrential current-sculpted granite boulders",
        "moist sediment rim at fluvial stone base",
        "ancient river polish along stream channel bedrock",
    ],
    # Geology, bedrock & tectonic formations
    "geology_bedrock": [
        "columnar basalt",
        "hexagonal volcanic rock columns with vertical cooling joints",
        "jagged basalt column outcropping",
        "natural tectonic fracture planes in ancient granite",
        "weathered alpine scree talus",
        "deep bedrock fissure cracks with clinging wild moss",
        "primordial volcanic basalt bedrock",
        "fractured granite cliff face",
    ],
    # Ground flora, mosses, lichens & fungi
    "ground_flora": [
        "cliff-clinging moss",
        "primordial fern fronds with delicate unfurling crosiers",
        "dense undulating moss cushions adhering to damp rock",
        "shade-tolerant forest ferns with arching fronds",
        "bioluminescent subterranean fungi emitting gentle ambient glow",
        "earthy umbrella-cap forest mushrooms clustered in moist humus",
        "hardy alpine cushion lichens encrusting bedrock",
        "dense windblown tussock grass clumps",
    ],
    # Aquatic & wetland flora
    "aquatic_wetland": [
        "riparian wetland reed cluster with dark brown velvet seed heads",
        "slender arching wetland reeds anchored in riparian silt",
        "submerged aquatic kelp ribbon holdfast gripping river stone",
        "undulating translucent waterweed ribbons swaying in current",
        "calcified primordial reef coral head with intricate branching",
    ],
}

# Quality & Render style suffixes guaranteed free of human bias
PRIMORDIAL_RENDER_MODIFIERS: List[str] = [
    "photorealistic 8k pbr textures",
    "highly detailed organic surface",
    "subsurface scattering on vegetative foliage",
    "cinematic lighting, untouched primordial wilderness",
    "octane render botanical realism",
    "isolated natural biological asset",
]


class PrimordialPromptEngine:
    """
    Generates rich, authentic prompts for 3D generative assets (Meshy AI v2) and procedural models.
    All generated prompts are guaranteed to pass the HumanBiasError filter.
    """

    def __init__(self, default_render_qualifiers: bool = True):
        self.default_render_qualifiers = default_render_qualifiers

    def validate(self, prompt: str, strict: bool = True) -> ValidationResult:
        """Validate any prompt against the strict human bias filter."""
        return validate_primordial_prompt(prompt, strict=strict)

    def generate_prompt(
        self,
        flora_type: Union[FloraType, int, str],
        seed: Optional[int] = None,
        style_modifiers: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Generate a fully validated primordial prompt for a given FloraType or archetype.

        Parameters:
            flora_type: Canonical FloraType enum, integer, or name string.
            seed: Deterministic random seed for modifier selection.
            style_modifiers: Additional custom descriptor strings.

        Returns:
            Pristine primordial prompt guaranteed to pass human bias validation.
        """
        ft = FloraType.from_any(flora_type)
        idx = int(seed) if seed is not None else 0

        if ft is not None:
            prompt = self._build_for_flora_type(ft, idx)
        else:
            # Check archetype string
            clean_name = str(flora_type).lower().strip()
            prompt = self._build_for_archetype(clean_name, idx)

        if style_modifiers:
            for mod in style_modifiers:
                # Pre-validate modifier
                validate_primordial_prompt(mod, strict=True)
                prompt += f", {mod.strip()}"

        if self.default_render_qualifiers:
            qualifier = PRIMORDIAL_RENDER_MODIFIERS[idx % len(PRIMORDIAL_RENDER_MODIFIERS)]
            prompt += f", {qualifier}"

        # Final verification: guarantee 100% compliance
        validate_primordial_prompt(prompt, strict=True)
        return prompt

    def build_tree_prompt(
        self,
        species: str = "pine",
        seed: int = 42,
    ) -> str:
        """Build prompt specifically tailored for ancient canopy trees."""
        clean = species.lower()
        if "pine" in clean or "conifer" in clean:
            ft = FloraType.Pine
        elif "palm" in clean:
            ft = FloraType.Palm
        elif "acacia" in clean:
            ft = FloraType.Acacia
        elif "jungle" in clean:
            ft = FloraType.Jungle
        elif "dead" in clean or "snag" in clean:
            ft = FloraType.DeadTree
        else:
            ft = FloraType.Round
        return self.generate_prompt(ft, seed=seed)

    def build_decay_prompt(
        self,
        decay_type: str = "fallen_log",
        seed: int = 42,
    ) -> str:
        """Build prompt for natural fallen windthrow and decaying wood."""
        clean = decay_type.lower()
        if "stump" in clean:
            return self._build_stump_prompt(seed)
        return self._build_windthrow_log_prompt(seed)

    def build_geological_prompt(
        self,
        rock_type: str = "boulder",
        seed: int = 42,
    ) -> str:
        """Build prompt for fluvial boulders or columnar basalt."""
        clean = rock_type.lower()
        if "basalt" in clean or "column" in clean:
            return self._build_basalt_prompt(seed)
        return self._build_fluvial_boulder_prompt(seed)

    def build_ground_flora_prompt(
        self,
        flora_type: str = "fern",
        seed: int = 42,
    ) -> str:
        """Build prompt for cliff moss, primordial ferns, or shade fungi."""
        clean = flora_type.lower()
        if "moss" in clean:
            return self._build_moss_prompt(seed)
        elif "fungus" in clean or "mushroom" in clean:
            return self._build_fungus_prompt(seed)
        return self._build_fern_prompt(seed)

    # -------------------------------------------------------------------------
    # Internal Prompt Builders
    # -------------------------------------------------------------------------

    def _build_for_flora_type(self, ft: FloraType, seed: int) -> str:
        if ft == FloraType.Pine:
            return (
                "Colossal ancient gnarled ironwood alpine pine tree, massive gnarled exposed "
                "buttress roots grasping bedrock fissures, deeply furrowed ancient rough bark "
                "with hanging lichens, storm-weathered asymmetrical crown, dense evergreen needle whorls"
            )
        elif ft == FloraType.Round:
            return (
                "Monolithic ancient broadleaf oak tree, colossal spreading canopy, massive flared "
                "buttress roots grasping bedrock fissures, rugged deeply grooved mossy bark, "
                "dense foliage lobes with natural organic leaf clusters"
            )
        elif ft == FloraType.Jungle:
            return (
                "Towering primordial rainforest strangler tree, massive aerial prop roots grasping "
                "granite boulders, dense hanging epiphytic moss and clinging lichens, multi-tiered "
                "umbrella canopy foliage"
            )
        elif ft == FloraType.Cactus:
            return (
                "Primordial colossal giant saguaro cactus, ancient weathered vertical pleats and "
                "fluted ribs, asymmetrical branching arms scarred by desert winds, natural crystalline spine clusters"
            )
        elif ft == FloraType.Rock:
            return self._build_fluvial_boulder_prompt(seed)
        elif ft == FloraType.Acacia:
            return (
                "Ancient umbrella-crown savanna ironwood tree, wind-bent twisted trunk, sprawling "
                "horizontal thorny canopy, deeply fissured dark weathered bark, flared root base in arid soil"
            )
        elif ft == FloraType.Palm:
            return (
                "Primordial wild coastal palm tree, curved ringed fibrous trunk leaning toward sunlight, "
                "radiating arched feathery fronds, dense exposed fibrous root ball gripping sandy shoreline substrate"
            )
        elif ft == FloraType.DeadTree:
            return (
                "Ancient lightning-shattered tree snag, splintered storm windthrow fracture at apex, "
                "bleached skeletal bare branches stripped by natural gales, hollow weathered trunk with shelf fungi"
            )
        elif ft == FloraType.Bush:
            return (
                "Dense primordial understory shrub, tangled multi-stem woody branches, compact foliage "
                "mounds with wild berry accents, ground-hugging natural growth habit"
            )
        elif ft == FloraType.Reed:
            return (
                "Dense riparian wetland reed cluster, slender arching stalks with dark velvet cattail "
                "seed heads, fibrous mud-anchored root mass, aquatic riverbank vegetation"
            )
        elif ft == FloraType.Tuft:
            return (
                "Hardy windblown alpine tussock grass clump, curved narrow fibrous blades, natural radial "
                "cushion habit, clinging to rocky mountainside scree"
            )
        elif ft == FloraType.Coral:
            return (
                "Submerged primordial living reef coral head, calcified fractal branching formations, "
                "pitted marine mineral textures, bioluminescent pigmentation"
            )
        elif ft == FloraType.Kelp:
            return (
                "Giant cold-water marine kelp stand, undulating translucent ribbon fronds swaying in tidal "
                "current, root-like holdfast securely gripping submerged riverbed rock"
            )
        elif ft == FloraType.Seagrass:
            return (
                "Submerged shallow coastal seabed seagrass meadow, flat flexible undulating blades, "
                "interlocking rhizome root network embedded in marine sediment"
            )
        return (
            "Ancient gnarled ironwood tree, buttress roots grasping bedrock fissures, "
            "storm-weathered asymmetrical crown, deeply furrowed ancient rough bark with hanging lichens"
        )

    def _build_for_archetype(self, name: str, seed: int) -> str:
        if "log" in name or "windthrow" in name or "decay" in name:
            return self._build_windthrow_log_prompt(seed)
        elif "stump" in name:
            return self._build_stump_prompt(seed)
        elif "pebble" in name or "river" in name or "boulder" in name:
            return self._build_fluvial_boulder_prompt(seed)
        elif "basalt" in name or "column" in name:
            return self._build_basalt_prompt(seed)
        elif "moss" in name:
            return self._build_moss_prompt(seed)
        elif "fern" in name:
            return self._build_fern_prompt(seed)
        elif "fungus" in name or "mushroom" in name:
            return self._build_fungus_prompt(seed)
        return (
            "Colossal ancient gnarled ironwood tree, massive buttress roots grasping bedrock fissures, "
            "deeply furrowed ancient rough bark with hanging lichens, storm-weathered asymmetrical crown"
        )

    def _build_windthrow_log_prompt(self, seed: int) -> str:
        return (
            "Storm-toppled rotting tree trunk, splintered storm windthrow jagged fracture at natural "
            "snap point, decaying heartwood hollow with shelf fungi and sprouting conks, moist moss-draped "
            "decaying bark partially sunken into forest soil, natural windthrow collapse"
        )

    def _build_stump_prompt(self, seed: int) -> str:
        return (
            "Weathered ancient tree stump, splintered storm windthrow jagged fibrous top peaks, "
            "massive buttress roots grasping bedrock fissures, decaying hollow center with shelf fungi and clinging moss"
        )

    def _build_fluvial_boulder_prompt(self, seed: int) -> str:
        return (
            "Fluvial rounded river boulders, smooth water-worn surface with subtle quartz veins, "
            "fluvial water-worn erosion facets, moist sediment rim at base, ancient torrent polish"
        )

    def _build_basalt_prompt(self, seed: int) -> str:
        return (
            "Columnar basalt rock formation, vertical hexagonal cooling pillars, natural tectonic "
            "fracture planes, weathered alpine scree talus, deep fissure cracks with cliff-clinging moss"
        )

    def _build_moss_prompt(self, seed: int) -> str:
        return (
            "Cliff-clinging moss cushion, dense velvety emerald bryophyte carpet adhering tightly "
            "to vertical granite bedrock crags, natural micro-topography with subtle moisture sheen"
        )

    def _build_fern_prompt(self, seed: int) -> str:
        return (
            "Primordial fern fronds, delicate arched bipinnate foliage, unfurling spiral fiddlehead "
            "crosiers at crown, radiating from moist fibrous rootstock nestled in forest soil"
        )

    def _build_fungus_prompt(self, seed: int) -> str:
        return (
            "Bioluminescent shade fungi cluster, umbrella-shaped caps emitting gentle ambient cyan "
            "glow, slender curving fibrous stalks sprouting from moist decaying forest humus"
        )


# =============================================================================
# Primordial Abiotic Prompt Engine & Negative Filter (0% Flora, 0% Fauna)
# =============================================================================

ABIOTIC_NEGATIVE_PROMPT: str = (
    "tree, trees, plant, plants, flora, leaf, leaves, flower, flowers, "
    "grass, moss, mossy, lichen, shrub, bush, fungus, mushroom, animal, fauna, "
    "creature, bird, fish, wildlife, coral, algae, seaweed, vine, vines, "
    "forest, jungle, swamp, log, stump, spider, insect, "
    "human, person, manmade, architecture, building, "
    "wall, bridge, road, path, trail, ruins, fence, wood plank, lumber, "
    "saw cuts, low quality, flat shading, cartoon, toy"
)


class AbioticBiasError(ValueError):
    """Raised when an abiotic prompt contains biological (flora/fauna) or human-biased artifacts."""

    def __init__(self, message: str, detected_tokens: List[str], prompt: str):
        super().__init__(message)
        self.detected_tokens = detected_tokens
        self.prompt = prompt


FORBIDDEN_BIOLOGICAL_TOKENS: Set[str] = {
    # Flora & vegetative
    "tree", "trees", "plant", "plants", "flora", "leaf", "leaves",
    "flower", "flowers", "grass", "grasses", "moss", "mosses", "lichen", "lichens",
    "shrub", "shrubs", "bush", "bushes", "bark", "branch", "branches",
    "root", "roots", "bough", "boughs", "canopy", "conifer", "pine", "oak",
    "palm", "fern", "ferns", "reed", "reeds", "foliage", "wood", "wooden",
    "timber", "stem", "stems", "petal", "petals", "sprout", "sprouting",
    "seed", "seeds", "sapling", "saplings", "humus", "loam", "topsoil",

    # Adjectival & morphological variants
    "mossy", "woody", "leafy", "grassy", "flowery", "rooted", "barked",
    "fungal", "algal", "lichenous", "vegetative", "floral", "botanical", "foliar",

    # Flora, marine & botanical taxa
    "algae", "seaweed", "kelp", "vine", "vines", "ivy", "liana", "succulent",
    "cactus", "cacti", "vegetation", "weed", "weeds", "botany", "seedling",
    "willow", "bamboo", "clover", "thistle", "bryophyte", "pteridophyte",

    # Biomes & ecological habitats
    "forest", "jungle", "woodland", "swamp", "bog", "peat",

    # Detritus & wood parts
    "log", "logs", "trunk", "trunks", "stump", "stumps",
    "driftwood", "deadwood", "ironwood", "treeline",

    # Fungi, spores & microbiology
    "fungus", "fungi", "mushroom", "mushrooms", "toadstool",
    "spore", "spores", "mycelium", "hyphae", "microbial",
    "bacteria", "organism", "biological", "biotic", "biota",

    # Fauna, marine life & creatures
    "animal", "animals", "fauna", "creature", "creatures",
    "bird", "birds", "fish", "fishes", "insect", "insects",
    "reptile", "reptiles", "mammal", "mammals", "amphibian", "amphibians",
    "fur", "feather", "feathers", "claw", "claws", "fang", "fangs",
    "wildlife", "coral", "polyps", "beast", "beasts",
    "carnivore", "herbivore", "predator", "prey",
    "avian", "canine", "feline", "reptilian",
    "worm", "worms", "slug", "slugs", "snail", "snails",
    "spider", "spiders", "arachnid", "beetle", "beetles",
    "crustacean", "crab", "crabs", "larva", "larvae",
}

FORBIDDEN_ABIOTIC_TOKENS: Set[str] = FORBIDDEN_HUMAN_TOKENS | FORBIDDEN_BIOLOGICAL_TOKENS

FORBIDDEN_ABIOTIC_PHRASES: List[str] = FORBIDDEN_PHRASES + [
    "tree branch", "tree bark", "fallen log", "rotting wood",
    "green leaf", "flowering plant", "forest humus", "moss cushion",
    "cliff moss", "fern frond", "shelf fungus", "wood grain",
]


def _compile_abiotic_forbidden_regex() -> Tuple[re.Pattern, List[re.Pattern]]:
    """Compile regexes for abiotic single tokens and phrases."""
    word_tokens = [t for t in FORBIDDEN_ABIOTIC_TOKENS if " " not in t and "-" not in t]
    single_word_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(word_tokens, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    phrase_patterns = []
    for p in FORBIDDEN_ABIOTIC_PHRASES:
        words = re.split(r"[\s\-]+", p)
        pat = r"\b" + r"[\s/\.\-_]+".join(re.escape(w) for w in words) + r"\b"
        phrase_patterns.append(re.compile(pat, re.IGNORECASE))
    return single_word_pattern, phrase_patterns


_ABIOTIC_SINGLE_WORD_REGEX, _ABIOTIC_PHRASE_REGEXES = _compile_abiotic_forbidden_regex()


def validate_abiotic_prompt(
    prompt: str,
    strict: bool = True,
) -> ValidationResult:
    """
    Validates a prompt against strict abiotic constraints (0% flora, 0% fauna, 0% human artifacts).

    Parameters:
        prompt: Text prompt to inspect.
        strict: If True, raises AbioticBiasError upon finding any forbidden token.
                If False, returns a ValidationResult with is_valid=False.

    Returns:
        ValidationResult indicating whether prompt is pristine sterile abiotic geology.
    """
    cleaned = prompt.strip().replace("_", " ")
    # Strip zero-width and invisible unicode characters
    cleaned_no_zw = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    # Deobfuscate spaced or dot-separated single letters (e.g. 't r e e' -> 'tree', 'p.l.a.n.t' -> 'plant')
    deob = re.sub(
        r"\b[a-zA-Z](?:[.\- ][a-zA-Z])+\b",
        lambda m: re.sub(r"[.\- ]", "", m.group(0)),
        cleaned_no_zw,
    )

    detected: List[str] = []

    for candidate in [cleaned, cleaned_no_zw, deob]:
        # 1. Check multi-word phrases first
        for phrase_re in _ABIOTIC_PHRASE_REGEXES:
            for match in phrase_re.finditer(candidate):
                match_str = match.group(0).lower()
                if match_str not in detected:
                    detected.append(match_str)

        # 2. Check single-word tokens with word boundary
        for match in _ABIOTIC_SINGLE_WORD_REGEX.finditer(candidate):
            token = match.group(0).lower()
            if token not in detected:
                detected.append(token)

    if detected:
        details = (
            f"Abiotic constraint violation: found forbidden biological or human token(s) {detected!r}. "
            "Abiotic prompts must strictly reject flora, fauna, and human structures."
        )
        if strict:
            raise AbioticBiasError(details, detected_tokens=detected, prompt=cleaned)
        return ValidationResult(
            is_valid=False,
            detected_tokens=detected,
            clean_prompt=cleaned,
            details=details,
        )

    return ValidationResult(
        is_valid=True,
        detected_tokens=[],
        clean_prompt=cleaned,
        details="Prompt verified sterile primordial abiotic geology (zero biological or human artifacts detected).",
    )


# -----------------------------------------------------------------------------
# Strict Abiotic Positive Templates & Render Modifiers
# -----------------------------------------------------------------------------

ABIOTIC_POSITIVE_TEMPLATES: Dict[AbioticType, str] = {
    AbioticType.HornPeak: (
        "primordial jagged horn peak, massive granite crag, glacially sheared arete, "
        "sharp bedrock fault lines, frost-weathered rock face, sterile abiotic geology, "
        "sheer pyramidal alpine summit, high-elevation exposed strata"
    ),
    AbioticType.RockCrag: (
        "massive weathered granite rock crag, sheer vertical bedrock cliff face, "
        "tectonic jointing fissures, rugged alpine bedrock outcrop, sterile abiotic stone, "
        "ancient natural geological fracture planes, rough mineral surface"
    ),
    AbioticType.BasaltColumn: (
        "geometric columnar basalt cluster, volcanic columnar jointing, polygonal prismatic "
        "hexagonal pentagonal rock pillars, dark tholeiitic basalt texture, fractured igneous "
        "cooling joints, stepped terrace escarpment, sterile geology"
    ),
    AbioticType.RiverPebble: (
        "cluster of water-worn riverbed pebbles, smooth oblate quartzite stones, fluvial "
        "rounded gravel deposit, eroded river cobbles, polished wet mineral surface, "
        "natural stream bed sediment deposit"
    ),
    AbioticType.RiparianSandbank: (
        "alluvial riparian sandbank sediment bar, natural river berm, fine alluvial quartz "
        "sand deposit, gentle fluvial current ripple marks, water-sculpted river silt shelf, "
        "sterile mineral shoreline deposit"
    ),
    AbioticType.Stalactite: (
        "subterranean limestone stalactite, conical calcite speleothem, mineral dripstone "
        "with crystalline texture, dripping cavern ceiling rock, natural calcified mineral deposition, "
        "fluted speleothem formation"
    ),
    AbioticType.Stalagmite: (
        "subterranean calcite stalagmite, conical floor-rising mineral mound, convex layered "
        "calcite speleothem, cavern floor mineral deposit, wet crystalline stone, "
        "stepped calcite drip mound"
    ),
    AbioticType.CavernColumn: (
        "massive fused cavern column pillar, grand limestone speleothem spanning floor "
        "to ceiling, fluted calcified flowstone pillar, subterranean geological column, "
        "ancient banded limestone pillar"
    ),
    AbioticType.CavernMineral: (
        "cluster of glowing cavern mineral crystals, crystalline fluorite calcite druse, "
        "phosphorescent geological formation, emissive cyan crystal cluster, raw mineral specimen, "
        "faceted crystal geode, mineral crust"
    ),
    AbioticType.KarstArch: (
        "natural rocky karst arch cavern entrance, rugged eroded limestone rock opening, "
        "subterranean cave portal, ancient weathered geological archway, sterile abiotic karst stone, "
        "rough porous limestone cliff cavity"
    ),
}

ABIOTIC_RENDER_MODIFIERS: List[str] = [
    "photorealistic 8k pbr textures",
    "highly detailed geological surface",
    "cinematic lighting, untouched primordial abiotic wilderness",
    "octane render mineral realism",
    "isolated sterile geological asset",
]


class AbioticPromptEngine:
    """
    Generates authentic geological prompts for Meshy AI v2 Text-to-3D and procedural pipelines,
    strictly enforcing positive geological descriptors and 0% biological/flora/fauna and human artifacts.
    """

    NEGATIVE_PROMPT: str = ABIOTIC_NEGATIVE_PROMPT

    def __init__(self, default_render_qualifiers: bool = True):
        self.default_render_qualifiers = default_render_qualifiers
        self.negative_prompt: str = ABIOTIC_NEGATIVE_PROMPT

    def get_negative_prompt(self) -> str:
        """Returns the canonical abiotic negative prompt."""
        return self.negative_prompt

    def validate(self, prompt: str, strict: bool = True) -> ValidationResult:
        """Validate any prompt against abiotic constraints (no flora, fauna, or human bias)."""
        return validate_abiotic_prompt(prompt, strict=strict)

    def _resolve_abiotic_type(self, val: Union[AbioticType, int, str]) -> Optional[AbioticType]:
        """Resolves AbioticType from enum, integer, or descriptive string."""
        res = AbioticType.from_any(val)
        if res is not None:
            return res
        if isinstance(val, str):
            clean = val.lower().strip()
            if "horn" in clean or "peak" in clean:
                return AbioticType.HornPeak
            if "crag" in clean or "cliff" in clean:
                return AbioticType.RockCrag
            if "basalt" in clean or "columnar" in clean:
                return AbioticType.BasaltColumn
            if "pebble" in clean or "gravel" in clean or "cobble" in clean:
                return AbioticType.RiverPebble
            if "sand" in clean or "berm" in clean or "bank" in clean:
                return AbioticType.RiparianSandbank
            if "stalactite" in clean:
                return AbioticType.Stalactite
            if "stalagmite" in clean:
                return AbioticType.Stalagmite
            if "column" in clean or "pillar" in clean:
                return AbioticType.CavernColumn
            if "mineral" in clean or "crystal" in clean or "druse" in clean:
                return AbioticType.CavernMineral
            if "arch" in clean or "portal" in clean or "karst" in clean:
                return AbioticType.KarstArch
        return None

    def generate_prompt(
        self,
        abiotic_type: Union[AbioticType, int, str],
        seed: Optional[int] = None,
        style_modifiers: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Generate a fully validated abiotic prompt for a given AbioticType.
        Guaranteed to pass validate_abiotic_prompt.
        """
        at = self._resolve_abiotic_type(abiotic_type)
        if at is None:
            raise ValueError(f"Unknown abiotic type: {abiotic_type}")

        base_template = ABIOTIC_POSITIVE_TEMPLATES[at]
        parts = [base_template]

        if style_modifiers:
            for mod in style_modifiers:
                validate_abiotic_prompt(mod, strict=True)
                parts.append(mod.strip())

        if self.default_render_qualifiers:
            idx = int(seed) if seed is not None else 0
            qualifier = ABIOTIC_RENDER_MODIFIERS[idx % len(ABIOTIC_RENDER_MODIFIERS)]
            parts.append(qualifier)

        full_prompt = ", ".join(parts)
        # Verify complete prompt against abiotic filter
        validate_abiotic_prompt(full_prompt, strict=True)
        return full_prompt

