import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    import modules.scripts as scripts
    from modules.processing import StableDiffusionProcessing, process_images
except ImportError:  # pragma: no cover - running outside Automatic1111/Forge
    class _ScriptBase:
        pass

    class _ScriptsStub:
        AlwaysVisible = True
        Script = _ScriptBase

    class StableDiffusionProcessing:  # type: ignore[override]
        def __init__(self):
            self.prompt = ""
            self.all_prompts: List[str] = []
            self.negative_prompt = ""
            self.all_negative_prompts: List[str] = []
            self.extra_generation_params: Dict[str, str] = {}

    def process_images(_p: "StableDiffusionProcessing"):
        return None

    scripts = _ScriptsStub()  # type: ignore[assignment]

try:
    import gradio as gr
except ImportError:  # pragma: no cover - allows unit testing without Gradio installed
    class _StubComponent:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *_, **__):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def click(self, *_, **__):
            return None

        def change(self, *_, **__):
            return None

    class _GrStub:
        def __getattr__(self, _):
            return _StubComponent()

    gr = _GrStub()

SCORE_TAGS: Tuple[str, ...] = (
    "score_9",
    "score_8_up",
    "score_7_up",
)

QUALITY_STYLES: Dict[str, Sequence[str]] = {
    "balanced": ("detailed", "sharp_focus", "studio_lighting"),
    "dreamy": ("soft_lighting", "pastel_colors", "glow"),
    "dramatic": ("dramatic_lighting", "high_contrast", "rim_lighting"),
    "vibrant": ("vivid_colors", "dynamic_lighting", "high_saturation"),
    "minimal": ("clean_lines", "muted_palette"),
}

DEFAULT_NEGATIVE = (
    "bad_anatomy, distorted_face, extra_limbs, low_quality, blurry, jpeg_artifacts, "
    "duplicate, disfigured, mutation"
)

IGNORE_LORA_PATTERN = re.compile(r"<[^>]+>")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s']")

IGNORE_WORDS: Set[str] = {
    "expressiveh",
    "g0thicpxl",
    "ffacedetail",
    "plrd",
    "morimee_style",
    "lhata4564",
    "break1",
    "break",
    "mcnm",
}

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|\n+|[;]+")

FEMALE_TERMS = ("female", "girl", "woman", "lady", "mare", "mares")
MALE_TERMS = ("male", "boy", "man", "guy", "stallion", "stallions")

AGE_RULES = {
    "young": ("young", "youthful", "teen", "teenage"),
    "adult": ("adult", "mature"),
    "elderly": ("elderly", "senior"),
}

BODY_RULES = {
    "petite": ("petite", "tiny"),
    "slim": ("slim", "slender", "lithe"),
    "curvy": ("curvy", "voluptuous"),
    "athletic": ("athletic", "toned"),
    "muscular": ("muscular", "powerful"),
}

SPECIES_RULES = [
    ("species", ("elf", "elven"), ("elf", "pointy_ears", "fantasy")),
    ("species", ("dragon",), ("dragon", "fantasy")),
    ("species", ("pony", "mare", "stallion"), ("pony",)),
    ("species", ("anthro", "furry"), ("anthro",)),
    ("species", ("fairy", "fae"), ("fairy", "wings")),
    ("species", ("angel",), ("angel", "wings")),
    ("species", ("demon", "succubus", "incubus"), ("demon", "horns")),
    ("species", ("mermaid",), ("mermaid", "aquatic")),
    ("species", ("android", "cyborg", "robot"), ("android",)),
    ("species", ("cat", "feline"), ("catgirl", "animal_ears", "tail")),
    ("species", ("fox", "kitsune"), ("foxgirl", "animal_ears", "tail")),
    ("species", ("wolf",), ("wolfgirl", "animal_ears", "tail")),
]

POSE_RULES = [
    ("pose", ("sitting", "seated"), ("sitting",)),
    ("pose", ("standing", "upright"), ("standing",)),
    ("pose", ("lying", "reclining"), ("lying",)),
    ("pose", ("kneeling",), ("kneeling",)),
    ("pose", ("walking", "strolling"), ("walking",)),
    ("pose", ("running",), ("running",)),
    ("pose", ("jumping", "leaping"), ("jumping",)),
    ("pose", ("floating", "hovering"), ("floating",)),
]

SETTING_RULES = [
    ("setting", ("bed", "bedroom"), ("bed", "bedroom")),
    ("setting", ("forest", "woods"), ("forest", "outdoors")),
    ("setting", ("meadow", "field"), ("meadow", "outdoors")),
    ("setting", ("beach", "shore"), ("beach", "coast")),
    ("setting", ("ocean", "sea"), ("ocean",)),
    ("setting", ("mountain", "cliff"), ("mountain",)),
    ("setting", ("city", "urban", "street"), ("cityscape",)),
    ("setting", ("castle",), ("castle",)),
    ("setting", ("library",), ("library",)),
    ("setting", ("cafe",), ("cafe",)),
    ("setting", ("rain", "raining"), ("rain", "night")),
]

WARDROBE_RULES = [
    ("wardrobe", ("pajamas", "sleepwear"), ("pajamas",)),
    ("wardrobe", ("dress",), ("dress",)),
    ("wardrobe", ("armor",), ("armor",)),
    ("wardrobe", ("jacket", "coat"), ("jacket",)),
    ("wardrobe", ("boots", "boot"), ("boots",)),
    ("wardrobe", ("crop top", "cropped top"), ("crop_top",)),
    ("wardrobe", ("shorts",), ("shorts",)),
    ("wardrobe", ("pants", "yoga pants", "leggings"), ("pants",)),
    ("wardrobe", ("panties",), ("panties",)),
    ("wardrobe", ("thighhighs", "thigh highs", "thigh-highs"), ("thighhighs",)),
    ("wardrobe", ("lingerie", "underwear"), ("lingerie",)),
    ("wardrobe", ("maid", "maid outfit"), ("maid_outfit",)),
    ("wardrobe", ("suit",), ("suit",)),
]

LIGHTING_RULES = [
    ("lighting", ("sunset", "sunrise"), ("sunset", "warm_light")),
    ("lighting", ("twilight", "dusk"), ("twilight", "soft_light")),
    ("lighting", ("moonlight", "night"), ("moonlight", "cool_light")),
    ("lighting", ("neon", "cyberpunk"), ("neon_light", "vivid_colors")),
    ("lighting", ("studio", "spotlight"), ("studio_light", "dramatic_light")),
    ("lighting", ("candle", "lantern"), ("candlelight", "warm_light")),
    ("lighting", ("rain",), ("rain_glow", "reflections")),
]

MOOD_RULES = [
    ("mood", ("romantic", "intimate"), ("romantic",)),
    ("mood", ("cozy", "snug"), ("cozy",)),
    ("mood", ("dramatic", "tense"), ("dramatic",)),
    ("mood", ("mysterious", "enigmatic"), ("mysterious",)),
    ("mood", ("energetic", "dynamic"), ("energetic",)),
    ("mood", ("calm", "peaceful"), ("calm",)),
    ("mood", ("seductive", "sexy"), ("seductive",)),
]

ADJECTIVE_SYNONYMS: Dict[str, Sequence[str]] = {
    "young": ("young", "youthful"),
    "adult": ("mature", "grown"),
    "elderly": ("elderly", "aged"),
    "petite": ("petite", "delicate"),
    "slim": ("slender", "lithe"),
    "curvy": ("curvy", "voluptuous"),
    "athletic": ("athletic", "toned"),
    "muscular": ("muscular", "powerful"),
    "romantic": ("romantic", "tender"),
    "cozy": ("cozy", "snug"),
    "dramatic": ("dramatic", "intense"),
    "mysterious": ("mysterious", "enigmatic"),
    "energetic": ("energetic", "spirited"),
    "calm": ("calm", "serene"),
    "seductive": ("seductive", "alluring"),
}

POSE_PHRASES = {
    "sitting": ("sitting gracefully", "sitting casually"),
    "standing": ("standing tall", "standing proud"),
    "lying": ("lying down", "reclining"),
    "kneeling": ("kneeling softly", "kneeling with poise"),
    "walking": ("walking forward", "strolling"),
    "running": ("running swiftly", "running with energy"),
    "jumping": ("mid-jump", "leaping"),
    "floating": ("floating gently", "hovering"),
}

SETTING_PHRASES = {
    "bed": ("on a bed", "on soft bedding"),
    "bedroom": ("in a cozy bedroom", "inside a warm bedroom"),
    "forest": ("in a sunlit forest", "among tall trees"),
    "meadow": ("in a flower meadow", "out in the meadow"),
    "beach": ("on a sandy beach", "by the shoreline"),
    "ocean": ("near the ocean", "beside the sea"),
    "mountain": ("on a mountain ridge", "amid mountains"),
    "cityscape": ("in a glowing cityscape", "on neon streets"),
    "castle": ("inside a grand castle", "within castle halls"),
    "library": ("inside a quiet library", "among bookshelves"),
    "cafe": ("in a cozy cafe", "at a corner cafe"),
    "rain": ("in the rain", "under rainy lights"),
    "night": ("at night", "under the night sky"),
}

WARDROBE_PHRASES = {
    "pajamas": ("wearing cozy pajamas", "dressed in soft pajamas"),
    "dress": ("wearing an elegant dress", "in a flowing dress"),
    "armor": ("wearing sleek armor", "armored for battle"),
    "jacket": ("wearing a stylish jacket", "layered with a jacket"),
    "boots": ("with tall boots", "wearing chunky boots"),
    "crop_top": ("wearing a crop top", "in a cropped top"),
    "shorts": ("wearing shorts", "in fitted shorts"),
    "pants": ("wearing fitted pants", "in casual pants"),
    "yoga_pants": ("wearing yoga pants", "in tight yoga pants"),
    "lingerie": ("wearing delicate lingerie", "in revealing lingerie"),
    "maid_outfit": ("wearing a cute maid outfit", "in a classic maid uniform"),
    "panties": ("wearing panties",),
    "thighhighs": ("with thighhighs", "wearing thigh-high socks"),
    "suit": ("dressed in a sharp suit", "wearing a tailored suit"),
}

ACCESSORY_PHRASES = {
    "gloves": ("with matching gloves",),
    "hat": ("topped with a hat",),
    "earrings": ("wearing earrings",),
    "necklace": ("with a delicate necklace",),
    "choker": ("wearing a choker",),
}

FLAIR_TAG_POOL: Dict[str, Sequence[str]] = {
    "light": ("clean_composition", "soft_background"),
    "balanced": ("cinematic_angle", "dynamic_perspective", "depth_of_field"),
    "heavy": ("cinematic_angle", "dynamic_angle", "hyper_detailed", "volumetric_lighting", "dramatic_shadows"),
}

PRESET_RULES: Dict[str, Dict[str, Dict[str, float]]] = {
    "light": {
        "subject": {"min": 1, "max": 3},
        "species": {"min": 0, "max": 1, "prob": 0.7},
        "traits": {"min": 0, "max": 1, "prob": 0.5},
        "body": {"min": 0, "max": 1, "prob": 0.5},
        "pose": {"min": 1, "max": 1},
        "composition": {"min": 0, "max": 1, "prob": 0.5},
        "gaze": {"min": 0, "max": 1, "prob": 0.6},
        "camera": {"min": 0, "max": 1, "prob": 0.5},
        "setting": {"min": 0, "max": 1, "prob": 0.6},
        "wardrobe": {"min": 1, "max": 2},
        "accessory": {"min": 0, "max": 1, "prob": 0.4},
        "color": {"min": 0, "max": 1, "prob": 0.6},
        "props": {"min": 0, "max": 1, "prob": 0.6},
        "lighting": {"min": 0, "max": 1, "prob": 0.4},
        "mood": {"min": 0, "max": 1, "prob": 0.4},
        "quality": {"min": 1, "max": 2},
    },
    "balanced": {
        "subject": {"min": 2, "max": 4},
        "species": {"min": 1, "max": 2},
        "traits": {"min": 1, "max": 2},
        "body": {"min": 0, "max": 1, "prob": 0.6},
        "pose": {"min": 1, "max": 2},
        "composition": {"min": 0, "max": 2, "prob": 0.7},
        "gaze": {"min": 0, "max": 1, "prob": 0.7},
        "camera": {"min": 0, "max": 1, "prob": 0.6},
        "setting": {"min": 1, "max": 2},
        "wardrobe": {"min": 1, "max": 3},
        "accessory": {"min": 0, "max": 2, "prob": 0.6},
        "color": {"min": 0, "max": 2, "prob": 0.7},
        "props": {"min": 0, "max": 2, "prob": 0.6},
        "lighting": {"min": 0, "max": 2, "prob": 0.7},
        "mood": {"min": 0, "max": 1, "prob": 0.7},
        "quality": {"min": 2, "max": 3},
    },
    "heavy": {
        "subject": {"min": 3, "max": 5},
        "species": {"min": 1, "max": 3},
        "traits": {"min": 1, "max": 3},
        "body": {"min": 1, "max": 2},
        "pose": {"min": 1, "max": 2},
        "composition": {"min": 1, "max": 3},
        "gaze": {"min": 0, "max": 2, "prob": 0.8},
        "camera": {"min": 0, "max": 2, "prob": 0.7},
        "setting": {"min": 1, "max": 3},
        "wardrobe": {"min": 2, "max": 4},
        "accessory": {"min": 1, "max": 2},
        "color": {"min": 1, "max": 3},
        "props": {"min": 0, "max": 3, "prob": 0.7},
        "lighting": {"min": 1, "max": 2},
        "mood": {"min": 1, "max": 2},
        "quality": {"min": 3, "max": 4},
    },
}


COLOR_WORDS = {
    "light blue": "light_blue",
    "dark blue": "dark_blue",
    "sky blue": "sky_blue",
    "baby blue": "baby_blue",
    "midnight blue": "midnight_blue",
    "navy": "navy",
    "blue": "blue",
    "hot pink": "hot_pink",
    "pink": "pink",
    "magenta": "magenta",
    "purple": "purple",
    "lavender": "lavender",
    "red": "red",
    "crimson": "crimson",
    "orange": "orange",
    "gold": "gold",
    "golden": "golden",
    "yellow": "yellow",
    "green": "green",
    "emerald": "emerald",
    "teal": "teal",
    "turquoise": "turquoise",
    "white": "white",
    "cream": "cream",
    "ivory": "ivory",
    "black": "black",
    "charcoal": "charcoal",
    "silver": "silver",
    "gray": "gray",
    "grey": "grey",
    "brown": "brown",
    "beige": "beige",
}

CLOTHING_KEYWORDS = {
    "crop top": "crop_top",
    "cropped top": "crop_top",
    "top": "top",
    "shirt": "shirt",
    "blouse": "blouse",
    "shorts": "shorts",
    "skirt": "skirt",
    "pants": "pants",
    "yoga pants": "yoga_pants",
    "leggings": "leggings",
    "jeans": "jeans",
    "jacket": "jacket",
    "coat": "coat",
    "hoodie": "hoodie",
    "sweater": "sweater",
    "boots": "boots",
    "sneakers": "sneakers",
    "sandals": "sandals",
    "panties": "panties",
    "lingerie": "lingerie",
    "thighhighs": "thighhighs",
}

LIGHTING_PATTERNS = [
    (re.compile(r"\bsoft dawn light\b"), ("dawn_light", "soft_lighting")),
    (re.compile(r"\bdawn\b"), ("dawn_light",)),
    (re.compile(r"\bsunrise\b"), ("sunrise_light", "warm_lighting")),
    (re.compile(r"\bwindow light\b"), ("window_light",)),
    (re.compile(r"\bthrough (?:the|a) window\b"), ("window_light", "soft_lighting")),
    (re.compile(r"\bneon light[s]?\b"), ("neon_light", "vivid_colors")),
    (re.compile(r"\bgodrays\b"), ("godrays", "dramatic_lighting")),
    (re.compile(r"\bsoft (?:morning|dawn) light\b"), ("soft_lighting", "dawn_light")),
]

GAZE_PATTERNS = [
    (re.compile(r"\blooking up at the viewer\b"), ("looking_up_at_viewer", "eye_contact")),
    (re.compile(r"\blooks up at the viewer\b"), ("looking_up_at_viewer", "eye_contact")),
    (re.compile(r"\blooking at the viewer\b"), ("looking_at_viewer", "eye_contact")),
    (re.compile(r"\blooks at the viewer\b"), ("looking_at_viewer", "eye_contact")),
    (re.compile(r"\blooking down\b"), ("looking_down",)),
    (re.compile(r"\bface close up\b"), ("face_close_up", "close_up")),
    (re.compile(r"\bfrontal view\b"), ("frontal_view",)),
    (re.compile(r"\bportrait from the side\b"), ("side_profile",)),
    (re.compile(r"\bfrom below\b"), ("from_below", "dynamic_angle")),
]

CAMERA_PATTERNS = [
    (re.compile(r"\bbutt[- ]shot\b"), ("butt_shot", "low_angle")),
    (re.compile(r"\bdynamic angle\b"), ("dynamic_angle",)),
    (re.compile(r"\bhalf ?body\b"), ("half_body",)),
    (re.compile(r"\bclose up\b"), ("close_up",)),
    (re.compile(r"\bshot from the side\b"), ("side_view",)),
]

COMPOSITION_PATTERNS = [
    (re.compile(r"\bon the edge of (?:the )?bed\b"), ("edge_of_bed",)),
    (re.compile(r"\bsitting on the edge\b"), ("edge_of_bed",)),
    (re.compile(r"\bedge of (?:the |her |his )?bed\b"), ("edge_of_bed",)),
    (re.compile(r"\blift skirt\b"), ("lift_skirt",)),
    (re.compile(r"\bupskirt\b"), ("upskirt",)),
    (re.compile(r"\bstanding up\b"), ("standing_pose",)),
]

TRAIT_PATTERNS = [
    (re.compile(r"\bcute smile\b"), ("cute_smile", "smiling")),
    (re.compile(r"\bsquinted eyes\b"), ("squinting",)),
    (re.compile(r"\bgorgeous girl\b"), ("gorgeous",)),
    (re.compile(r"\bkawaii girl\b"), ("kawaii",)),
]

PROP_PATTERNS = [
    (re.compile(r"\bplants\b"), ("plants",)),
    (re.compile(r"\bwindows?\b"), ("windows",)),
    (re.compile(r"\bwhite walls\b"), ("white_walls",)),
    (re.compile(r"\bpolaroid photo\b"), ("polaroid_photo",)),
    (re.compile(r"\bretro aesthetic\b"), ("retro_aesthetic",)),
    (re.compile(r"\bfilm grain\b"), ("film_grain",)),
    (re.compile(r"\bbed\b"), ("bed",)),
]

ADJECTIVE_SUBJECT_PATTERNS = [
    (re.compile(r"\bcute girl\b"), "cute_girl"),
    (re.compile(r"\badorable girl\b"), "adorable_girl"),
    (re.compile(r"\bpretty girl\b"), "pretty_girl"),
    (re.compile(r"\bbeautiful girl\b"), "beautiful_girl"),
    (re.compile(r"\bcute boy\b"), "cute_boy"),
    (re.compile(r"\bhandsome boy\b"), "handsome_boy"),
]


def _compile_colored_clothing_patterns() -> List[Tuple[re.Pattern, str, str]]:
    patterns: List[Tuple[re.Pattern, str, str]] = []
    color_items = sorted(COLOR_WORDS.items(), key=lambda item: -len(item[0]))
    clothing_items = sorted(CLOTHING_KEYWORDS.items(), key=lambda item: -len(item[0]))
    for color_text, color_tag in color_items:
        for phrase, clothing_tag in clothing_items:
            pattern = re.compile(rf"\b{re.escape(color_text)}\s+{re.escape(phrase)}\b")
            patterns.append((pattern, color_tag, clothing_tag))
    return patterns


COLORED_CLOTHING_PATTERNS = _compile_colored_clothing_patterns()

def _split_into_segments(text: str) -> List[str]:
    segments = [
        segment.strip()
        for segment in SENTENCE_SPLIT_REGEX.split(text)
        if segment and segment.strip()
    ]
    if segments:
        return segments
    cleaned = text.strip()
    return [cleaned] if cleaned else []


def _compile_keywords(values: Iterable[str]) -> Tuple[re.Pattern, ...]:
    return tuple(re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE) for v in values)


def _matches_any(text: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _normalize(text: str) -> str:
    cleaned = IGNORE_LORA_PATTERN.sub(" ", text)
    cleaned = NON_ALNUM_PATTERN.sub(" ", cleaned.lower())
    tokens = [token for token in cleaned.split() if token and token not in IGNORE_WORDS]
    return " ".join(tokens)


def _format_tag(tag: str) -> str:
    return tag if ":" in tag else tag.replace(" ", "_")


def _split_extra(extra: str) -> List[str]:
    return [token.strip() for token in extra.split(",") if token.strip()] if extra else []


def _highest_count(values: Iterable[int]) -> int:
    highest = 0
    for value in values:
        if value > highest:
            highest = value
    return highest


def _detect_subject_count(text: str) -> int:
    number_map = {
        "one": 1,
        "single": 1,
        "solo": 1,
        "two": 2,
        "pair": 2,
        "couple": 2,
        "duo": 2,
        "three": 3,
        "trio": 3,
        "four": 4,
        "quartet": 4,
    }
    pattern = re.compile(
        r"\b(\d+|one|two|three|four)\s+"
        r"(girl|girls|woman|women|boy|boys|man|men|pony|ponies|character|characters)\b"
    )
    counts: List[int] = []
    for match in pattern.finditer(text):
        token = match.group(1)
        counts.append(int(token) if token.isdigit() else number_map.get(token, 1))
    for word, value in number_map.items():
        if re.search(rf"\b{word}\b", text):
            counts.append(value)
    return _highest_count(counts) or 1


def _collect_rules(text: str, rules: Sequence[Tuple[str, Sequence[str], Sequence[str]]]) -> Dict[str, Set[str]]:
    found: Dict[str, Set[str]] = {}
    for category, keywords, tags in rules:
        if _matches_any(text, _compile_keywords(keywords)):
            found.setdefault(category, set()).update(tags)
    return found


def _enrich_attributes_with_freeform(text: str, attrs: Dict[str, Set[str]]) -> None:
    lowered = text.lower()

    for pattern, tags in LIGHTING_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("lighting", set()).update(tags)

    for pattern, tags in GAZE_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("gaze", set()).update(tags)
            attrs.setdefault("traits", set()).update(tags)

    for pattern, tags in CAMERA_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("camera", set()).update(tags)

    for pattern, tags in COMPOSITION_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("composition", set()).update(tags)

    for pattern, tags in TRAIT_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("traits", set()).update(tags)

    for pattern, tags in PROP_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("props", set()).update(tags)
            attrs.setdefault("setting", set()).update(tags)

    for color_text, color_tag in sorted(COLOR_WORDS.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(color_text)}\b", lowered):
            attrs.setdefault("color", set()).add(color_tag)
        if re.search(rf"\b{re.escape(color_text)}\s+background\b", lowered):
            attrs.setdefault("setting", set()).add(f"{color_tag}_background")

    for pattern, color_tag, clothing_tag in COLORED_CLOTHING_PATTERNS:
        if pattern.search(lowered):
            wardrobe_set = attrs.setdefault("wardrobe", set())
            wardrobe_set.add(clothing_tag)
            wardrobe_set.add(f"{color_tag}_{clothing_tag}")
            attrs.setdefault("color", set()).add(color_tag)

    for pattern, combined in ADJECTIVE_SUBJECT_PATTERNS:
        if pattern.search(lowered):
            attrs.setdefault("traits", set()).add(combined)


@dataclass
class ConversionResult:
    prompt: str
    subject_phrase: str
    ordered_tags: List[str]
    subject_count: int
    inferred_gender: str
    applied_style: str
    prompt_weight: str

class PonyPromptConverter:
    def _rng(self, seed: int | None = None) -> random.Random:
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        return rng

    def _match_gender(self, text: str) -> Tuple[bool, bool]:
        female = any(re.search(rf"\b{word}\b", text) for word in FEMALE_TERMS)
        male = any(re.search(rf"\b{word}\b", text) for word in MALE_TERMS)
        return female, male

    def _subject_tags(self, text: str) -> Tuple[Set[str], int, str]:
        count = _detect_subject_count(text)
        female, male = self._match_gender(text)
        tags: Set[str] = set()
        gender = "mixed"
        if count == 1:
            tags.add("solo")
            if female and not male:
                tags.update({"1girl", "female"})
                gender = "female"
            elif male and not female:
                tags.update({"1boy", "male"})
                gender = "male"
            else:
                gender = "unknown"
        elif count == 2:
            tags.add("duo")
            if female and not male:
                tags.add("2girls")
                gender = "female"
            elif male and not female:
                tags.add("2boys")
                gender = "male"
            else:
                tags.update({"1girl", "1boy"})
        else:
            tags.add("group")
            gender = "female" if female and not male else "male" if male and not female else "mixed"
        return tags, count, gender

    def _apply_descriptors(self, text: str, store: Dict[str, Set[str]]):
        for tag, keywords in AGE_RULES.items():
            if any(re.search(rf"\b{word}\b", text) for word in keywords):
                store.setdefault("age", set()).add(tag)
        for tag, keywords in BODY_RULES.items():
            if any(re.search(rf"\b{word}\b", text) for word in keywords):
                store.setdefault("body", set()).add(tag)
        if "smile" in text:
            store.setdefault("traits", set()).add("smiling")
        if "serious" in text or "focused" in text:
            store.setdefault("traits", set()).add("serious")

    def _collect_segment_attributes(
        self,
        normalized: str,
    ) -> Tuple[Dict[str, Set[str]], int, str]:
        attrs: Dict[str, Set[str]] = {}
        subject_tags, count, gender = self._subject_tags(normalized)
        if subject_tags:
            attrs["subject"] = set(subject_tags)
        for rules in (
            SPECIES_RULES,
            POSE_RULES,
            SETTING_RULES,
            WARDROBE_RULES,
            LIGHTING_RULES,
            MOOD_RULES,
        ):
            matches = _collect_rules(normalized, rules)
            for category, tags in matches.items():
                attrs.setdefault(category, set()).update(tags)
        self._apply_descriptors(normalized, attrs)
        return attrs, count, gender

    def _choose_adjectives(self, attrs: Dict[str, Set[str]], rng: random.Random, preset: str) -> List[str]:
        pool: List[str] = []
        for category in ("age", "body", "traits", "mood"):
            pool.extend(attrs.get(category, set()))
        rng.shuffle(pool)
        limit = 3 if preset == "heavy" else 2
        adjectives: List[str] = []
        for tag in pool:
            options = ADJECTIVE_SYNONYMS.get(tag)
            if options:
                pick = rng.choice(options)
                if pick not in adjectives:
                    adjectives.append(pick)
            if len(adjectives) >= limit:
                break
        return adjectives

    def _base_noun(self, attrs: Dict[str, Set[str]], count: int, gender: str) -> str:
        species = attrs.get("species", set())
        for candidate in ("elf", "dragon", "pony", "fairy", "angel", "demon", "mermaid", "android", "catgirl", "foxgirl", "wolfgirl"):
            if candidate in species:
                return candidate if count == 1 else f"{candidate}s"
        if "anthro" in species:
            return "anthro" if count == 1 else "anthros"
        if count > 1:
            return "girls" if gender == "female" else "boys" if gender == "male" else "characters"
        return "girl" if gender == "female" else "boy" if gender == "male" else "character"

    def _phrase(self, mapping: Dict[str, Sequence[str]], tags: Set[str], rng: random.Random) -> str:
        choices = list(tags)
        rng.shuffle(choices)
        for tag in choices:
            options = mapping.get(tag)
            if options:
                return rng.choice(options)
        return ""

    def _wardrobe_phrase(self, attrs: Dict[str, Set[str]], rng: random.Random) -> List[str]:
        result: List[str] = []
        garments = list(attrs.get("wardrobe", set()))
        rng.shuffle(garments)
        for tag in garments:
            phrase = self._phrase(WARDROBE_PHRASES, {tag}, rng)
            if phrase:
                result.append(phrase)
            elif "_" in tag:
                friendly = tag.replace("_", " ")
                result.append(f"wearing {friendly}")
        accessories = list(attrs.get("accessory", set()))
        rng.shuffle(accessories)
        for tag in accessories:
            options = ACCESSORY_PHRASES.get(tag)
            if options and rng.random() < 0.6:
                result.append(rng.choice(options))
        return result[:2]

    def _build_subject_phrase(
        self,
        attrs: Dict[str, Set[str]],
        count: int,
        gender: str,
        rng: random.Random,
        preset: str,
    ) -> str:
        adjectives = self._choose_adjectives(attrs, rng, preset)
        noun = self._base_noun(attrs, count, gender)
        pose_phrase = self._phrase(POSE_PHRASES, attrs.get("pose", set()), rng)
        setting_phrase = self._phrase(
            SETTING_PHRASES,
            attrs.get("setting", set()) | attrs.get("lighting", set()),
            rng,
        )
        wardrobe_phrases = list(dict.fromkeys(self._wardrobe_phrase(attrs, rng)))
        fragments: List[str] = []
        if adjectives:
            fragments.append(" ".join(adjectives))
        fragments.append(noun)
        if pose_phrase:
            fragments.append(pose_phrase)
        if setting_phrase:
            fragments.append(setting_phrase)
        if wardrobe_phrases:
            fragments.append(", ".join(wardrobe_phrases))
        phrase = " ".join(fragment for fragment in fragments if fragment).strip(", ")
        return phrase or "character portrait"

    def _select_categories(
        self,
        attrs: Dict[str, Set[str]],
        preset: str,
        rng: random.Random,
        variation: float,
    ) -> Dict[str, List[str]]:
        config = PRESET_RULES[preset]
        variability = max(0.0, min(1.0, variation))
        selected: Dict[str, List[str]] = {}
        for category, tags in attrs.items():
            if category not in config or not tags:
                continue
            rule = config[category]
            available = list(tags)
            rng.shuffle(available)
            max_allowed = min(int(rule.get("max", len(available))), len(available))
            min_required = min(int(rule.get("min", 0)), max_allowed)
            if max_allowed <= 0:
                continue
            base_prob = rule.get("prob", 1.0 if min_required > 0 else 0.5)
            adjusted_prob = max(0.0, min(1.0, base_prob + (variability - 0.5) * 0.6))
            if min_required == 0 and rng.random() > adjusted_prob:
                continue
            span = max_allowed - min_required
            count = min_required
            if span > 0:
                span_limit = max(0, int(round(span * variability)))
                span_limit = min(span, span_limit)
                if span_limit > 0:
                    count = min_required + rng.randint(0, span_limit)
            if count == 0 and min_required > 0:
                count = min_required
            if count == 0:
                continue
            selected[category] = available[:count]
        return selected
    def _assemble_tags(
        self,
        selected: Dict[str, List[str]],
        preset: str,
        extra: Sequence[str],
        rng: random.Random,
        variation: float,
    ) -> List[str]:
        categories = list(selected.keys())
        if "subject" in categories:
            categories.remove("subject")
            categories.insert(0, "subject")
        rng.shuffle(categories[1:])
        ordered: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for category in categories:
            tags = selected[category]
            rng.shuffle(tags)
            for tag in tags:
                formatted = _format_tag(tag)
                if formatted not in seen:
                    ordered.append((category, formatted))
                    seen.add(formatted)
        flair = list(FLAIR_TAG_POOL.get(preset, ()))
        rng.shuffle(flair)
        variability = max(0.0, min(1.0, variation))
        if flair:
            flair_cap = int(round(len(flair) * max(0.0, variability - 0.2)))
            if preset == "light":
                flair_cap = min(flair_cap, 1)
            elif preset == "balanced":
                flair_cap = min(flair_cap, 2)
            else:
                flair_cap = min(flair_cap, len(flair))
            if flair_cap > 0:
                flair_count = rng.randint(0, flair_cap)
                for tag in flair[:flair_count]:
                    formatted = _format_tag(tag)
                    if formatted not in seen:
                        ordered.append(("flair", formatted))
                        seen.add(formatted)
        for tag in extra:
            formatted = _format_tag(tag)
            if formatted not in seen:
                ordered.append(("extra", formatted))
                seen.add(formatted)
        final_tags: List[str] = []
        final_seen: Set[str] = set()
        for category, tag in ordered:
            if category == "subject":
                final_tags.append(tag)
                final_seen.add(tag)
                continue
            drop_chance = 0.0
            if variability < 0.5:
                drop_chance = (0.5 - variability) * 0.45
            if drop_chance > 0 and rng.random() < drop_chance:
                continue
            if tag not in final_seen:
                final_tags.append(tag)
                final_seen.add(tag)
        if variability > 0.75:
            extras = [tag for _, tag in ordered if tag not in final_seen]
            rng.shuffle(extras)
            if extras:
                add_cap = min(len(extras), 1 + int(round((variability - 0.75) * len(extras))))
                add_count = rng.randint(0, add_cap)
                for tag in extras[:add_count]:
                    if tag not in final_seen:
                        final_tags.append(tag)
                        final_seen.add(tag)
        return final_tags

    def convert(
        self,
        text: str,
        quality_style: str = "balanced",
        extra_tags: Sequence[str] | None = None,
        preset: str = "balanced",
        seed: int | None = None,
        variation: float = 0.5,
    ) -> ConversionResult:
        preset_key = preset if preset in PRESET_RULES else "balanced"
        segments = _split_into_segments(text)
        aggregated_attrs: Dict[str, Set[str]] = {}
        segment_attrs: List[Tuple[Dict[str, Set[str]], int, str]] = []
        subject_counts: List[int] = []
        gender_votes: Counter[str] = Counter()

        for segment in segments:
            normalized_segment = _normalize(segment)
            if not normalized_segment:
                continue
            attrs, count, gender = self._collect_segment_attributes(normalized_segment)
            if not attrs:
                continue
            segment_attrs.append((attrs, count, gender))
            for category, tags in attrs.items():
                aggregated_attrs.setdefault(category, set()).update(tags)
            subject_counts.append(count)
            gender_votes[gender] += 1

        if not segment_attrs:
            normalized = _normalize(text)
            attrs, count, gender = self._collect_segment_attributes(normalized)
            segment_attrs.append((attrs, count, gender))
            for category, tags in attrs.items():
                aggregated_attrs.setdefault(category, set()).update(tags)
            subject_counts.append(count)
            gender_votes[gender] += 1

        _enrich_attributes_with_freeform(text, aggregated_attrs)

        subject_count = max(subject_counts) if subject_counts else 1

        gender = "unknown"
        if gender_votes:
            sorted_votes = sorted(
                gender_votes.items(),
                key=lambda item: (item[0] == "unknown", -item[1]),
            )
            if sorted_votes:
                gender = sorted_votes[0][0]

        if "subject" not in aggregated_attrs:
            aggregated_attrs["subject"] = {"solo"}

        subject_tags_set = aggregated_attrs.setdefault("subject", set())
        if subject_count > 1:
            subject_tags_set.discard("solo")
            if subject_count == 2:
                subject_tags_set.add("duo")
            elif subject_count == 3:
                subject_tags_set.add("trio")
                subject_tags_set.add("group")
            elif subject_count >= 4:
                subject_tags_set.add("group")
        else:
            subject_tags_set.add("solo")
            if gender == "female":
                subject_tags_set.add("1girl")
            elif gender == "male":
                subject_tags_set.add("1boy")

        aggregated_attrs.setdefault("quality", set()).update(
            QUALITY_STYLES.get(quality_style, QUALITY_STYLES["balanced"])
        )

        rng = self._rng(seed)
        subject_phrase = self._build_subject_phrase(
            aggregated_attrs,
            subject_count,
            gender,
            rng,
            preset_key,
        )

        if len(segment_attrs) > 1:
            extra_settings: List[str] = []
            for attrs, _, _ in segment_attrs:
                extra_phrase = self._phrase(SETTING_PHRASES, attrs.get("setting", set()), rng)
                if extra_phrase and extra_phrase not in extra_settings and extra_phrase not in subject_phrase:
                    extra_settings.append(extra_phrase)
            if extra_settings and rng.random() < 0.7:
                subject_phrase = f"{subject_phrase}, {rng.choice(extra_settings)}"

        variation = max(0.0, min(1.0, variation))
        selected = self._select_categories(aggregated_attrs, preset_key, rng, variation)
        extra = list(extra_tags) if extra_tags else []
        ordered_tags = self._assemble_tags(selected, preset_key, extra, rng, variation)

        if subject_count == 2 and "duo" not in ordered_tags:
            ordered_tags.insert(0, "duo")
        elif subject_count == 3:
            if "trio" not in ordered_tags:
                ordered_tags.insert(0, "trio")
            if "group" not in ordered_tags:
                ordered_tags.insert(1 if len(ordered_tags) > 0 else 0, "group")
        elif subject_count >= 4 and "group" not in ordered_tags:
            ordered_tags.insert(0, "group")

        priority_tags: List[str] = []
        priority_tags.extend(tag for tag in aggregated_attrs.get("wardrobe", set()) if "_" in tag)
        priority_tags.extend(aggregated_attrs.get("gaze", set()))
        priority_tags.extend(aggregated_attrs.get("lighting", set()))
        priority_tags.extend(aggregated_attrs.get("props", set()))
        priority_tags.extend(aggregated_attrs.get("color", set()))
        priority_tags.extend(tag for tag in aggregated_attrs.get("traits", set()) if tag.endswith("_girl") or tag.endswith("_boy"))

        for tag in priority_tags:
            formatted = _format_tag(tag)
            if formatted not in ordered_tags:
                ordered_tags.insert(0, formatted)

        parts = list(SCORE_TAGS)
        parts.append(subject_phrase)
        parts.extend(ordered_tags)
        prompt = ", ".join(parts)

        return ConversionResult(
            prompt=prompt,
            subject_phrase=subject_phrase,
            ordered_tags=ordered_tags,
            subject_count=subject_count,
            inferred_gender=gender,
            applied_style=quality_style,
            prompt_weight=preset_key,
        )

converter = PonyPromptConverter()

def _generate_preview(
    prompt_text: str,
    quality_style: str,
    preset_key: str,
    variation_amount: float,
    extra_tags: str,
    include_negative: bool,
    seed_value: int | None = None,
) -> Tuple[str, str, str]:
    prompt_text = prompt_text.strip()
    extra = _split_extra(extra_tags)
    if not prompt_text:
        return "", DEFAULT_NEGATIVE if include_negative else "", ""
    result = converter.convert(
        prompt_text,
        quality_style=quality_style,
        extra_tags=extra,
        preset=preset_key,
        seed=seed_value,
        variation=variation_amount,
    )
    negative = DEFAULT_NEGATIVE if include_negative else ""
    meta = (
        f"Subjects: {result.subject_count} | Gender: {result.inferred_gender} | "
        f"Preset: {result.prompt_weight} | Style: {result.applied_style} | "
        f"Variation: {variation_amount:.2f}"
    )
    return result.prompt, negative, meta


class Script(scripts.Script):
    def title(self) -> str:
        return "Pony Prompt Optimizer"

    def show(self, is_img2img: bool) -> bool:
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion("Pony Prompt Optimizer", open=False):
            enable_checkbox = gr.Checkbox(label="Enable pony prompt conversion", value=False)
            override_prompt = gr.Textbox(
                label="Natural description override (optional)",
                lines=3,
                placeholder="Leave blank to convert the main prompt automatically.",
            )
            quality_style = gr.Dropdown(
                label="Quality style",
                choices=list(QUALITY_STYLES.keys()),
                value="balanced",
            )
            preset_radio = gr.Radio(
                label="Prompt weight",
                choices=list(PRESET_RULES.keys()),
                value="balanced",
            )
            variation_slider = gr.Slider(
                label="Variation",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.5,
            )
            extra_tags_box = gr.Textbox(
                label="Extra tags (comma separated)",
                placeholder="e.g. cinematic_angle, intricate_details",
            )
            negative_checkbox = gr.Checkbox(
                label="Auto-fill recommended negative prompt (if empty)",
                value=True,
            )
            seed_slider = gr.Slider(
                label="Preview seed (optional)",
                minimum=0,
                maximum=999999,
                step=1,
                value=0,
            )
            lock_seed = gr.Checkbox(label="Lock preview seed", value=False)
            preview_box = gr.Textbox(
                label="Optimized pony prompt",
                lines=6,
                interactive=False,
                elem_id="ppo-preview",
            )
            negative_box = gr.Textbox(
                label="Negative prompt preview",
                lines=2,
                interactive=False,
            )
            meta_box = gr.Markdown(value="", elem_id="ppo-meta")
            with gr.Row():
                regen_button = gr.Button("Regenerate preview", variant="primary")
                push_button = gr.Button("Push to prompt", variant="secondary", elem_id="ppo-push-button")

            def _seed(seed: int, lock: bool) -> int | None:
                return seed if lock else None

            inputs = [
                override_prompt,
                quality_style,
                preset_radio,
                variation_slider,
                extra_tags_box,
                negative_checkbox,
                seed_slider,
                lock_seed,
            ]

            regen_button.click(
                lambda text, style, preset, variation, extra, neg, seed, lock: _generate_preview(
                    text,
                    style,
                    preset,
                    variation,
                    extra,
                    neg,
                    _seed(seed, lock),
                ),
                inputs=inputs,
                outputs=[preview_box, negative_box, meta_box],
            )

        return [
            enable_checkbox,
            override_prompt,
            quality_style,
            preset_radio,
            variation_slider,
            extra_tags_box,
            negative_checkbox,
        ]

    def process(
        self,
        p: StableDiffusionProcessing,
        enable_conversion: bool,
        override_prompt: str,
        quality_style: str,
        preset_key: str,
        variation_amount: float,
        extra_tags: str,
        auto_negative: bool,
    ) -> None:
        if not enable_conversion:
            return
        source_prompt = override_prompt.strip() or p.prompt
        extra = _split_extra(extra_tags)
        result = converter.convert(
            source_prompt,
            quality_style=quality_style,
            extra_tags=extra,
            preset=preset_key,
            variation=variation_amount,
        )
        p.prompt = result.prompt
        if hasattr(p, "all_prompts") and p.all_prompts:
            p.all_prompts = [result.prompt for _ in p.all_prompts]
        if auto_negative and not getattr(p, "negative_prompt", ""):
            p.negative_prompt = DEFAULT_NEGATIVE
            if hasattr(p, "all_negative_prompts") and p.all_negative_prompts:
                p.all_negative_prompts = [DEFAULT_NEGATIVE for _ in p.all_negative_prompts]
        if hasattr(p, "extra_generation_params"):
            p.extra_generation_params.update(
                {
                    "pony_prompt_source": source_prompt,
                    "pony_prompt_style": result.applied_style,
                    "pony_prompt_subjects": result.subject_count,
                    "pony_prompt_gender": result.inferred_gender,
                    "pony_prompt_weight": result.prompt_weight,
                    "pony_prompt_variation": round(variation_amount, 2),
                }
            )

    def run(
        self,
        p: StableDiffusionProcessing,
        enable_conversion: bool,
        override_prompt: str,
        quality_style: str,
        preset_key: str,
        variation_amount: float,
        extra_tags: str,
        auto_negative: bool,
    ):
        return process_images(p)
