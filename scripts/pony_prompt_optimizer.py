import random
import re
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
    "score_6_up",
    "score_5_up",
    "score_4_up",
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

FEMALE_TERMS = ("female", "girl", "woman", "lady", "mare")
MALE_TERMS = ("male", "boy", "man", "guy", "stallion")

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
    "lingerie": ("wearing delicate lingerie", "in revealing lingerie"),
    "maid_outfit": ("wearing a cute maid outfit", "in a classic maid uniform"),
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
        "subject": {"min": 2, "max": 3},
        "species": {"min": 0, "max": 1, "prob": 0.7},
        "traits": {"min": 0, "max": 1, "prob": 0.5},
        "body": {"min": 0, "max": 1, "prob": 0.5},
        "pose": {"min": 1, "max": 1},
        "setting": {"min": 0, "max": 1, "prob": 0.6},
        "wardrobe": {"min": 1, "max": 2},
        "accessory": {"min": 0, "max": 1, "prob": 0.4},
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
        "setting": {"min": 1, "max": 2},
        "wardrobe": {"min": 1, "max": 3},
        "accessory": {"min": 0, "max": 2, "prob": 0.6},
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
        "setting": {"min": 1, "max": 3},
        "wardrobe": {"min": 2, "max": 4},
        "accessory": {"min": 1, "max": 2},
        "lighting": {"min": 1, "max": 2},
        "mood": {"min": 1, "max": 2},
        "quality": {"min": 3, "max": 4},
    },
}

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
        accessories = list(attrs.get("accessory", set()))
        rng.shuffle(accessories)
        for tag in accessories:
            options = ACCESSORY_PHRASES.get(tag)
            if options and rng.random() < 0.6:
                result.append(rng.choice(options))
        return result[:2]

    def _select_categories(self, attrs: Dict[str, Set[str]], preset: str, rng: random.Random) -> Dict[str, List[str]]:
        config = PRESET_RULES[preset]
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
            if min_required == 0 and rng.random() > rule.get("prob", 0.5):
                continue
            if max_allowed < min_required:
                max_allowed = min_required
            count = min_required
            if max_allowed > min_required:
                count = rng.randint(min_required, max_allowed)
            if count == 0:
                continue
            selected[category] = available[:count]
        return selected
    def _assemble_tags(self, selected: Dict[str, List[str]], preset: str, extra: Sequence[str], rng: random.Random) -> List[str]:
        categories = list(selected.keys())
        if "subject" in categories:
            categories.remove("subject")
            categories.insert(0, "subject")
        rng.shuffle(categories[1:])
        ordered: List[str] = []
        seen: Set[str] = set()
        for category in categories:
            tags = selected[category]
            rng.shuffle(tags)
            for tag in tags:
                formatted = _format_tag(tag)
                if formatted not in seen:
                    ordered.append(formatted)
                    seen.add(formatted)
        flair = list(FLAIR_TAG_POOL.get(preset, ()))
        rng.shuffle(flair)
        flair_count = 0
        if preset == "heavy":
            flair_count = rng.randint(1, min(3, len(flair))) if flair else 0
        elif preset == "balanced" and rng.random() < 0.7:
            flair_count = 1 if flair else 0
        elif preset == "light" and rng.random() < 0.4:
            flair_count = 1 if flair else 0
        for tag in flair[:flair_count]:
            formatted = _format_tag(tag)
            if formatted not in seen:
                ordered.append(formatted)
                seen.add(formatted)
        for tag in extra:
            formatted = _format_tag(tag)
            if formatted not in seen:
                ordered.append(formatted)
                seen.add(formatted)
        return ordered

    def convert(
        self,
        text: str,
        quality_style: str = "balanced",
        extra_tags: Sequence[str] | None = None,
        preset: str = "balanced",
        seed: int | None = None,
    ) -> ConversionResult:
        normalized = _normalize(text)
        attrs: Dict[str, Set[str]] = {}
        subject_tags, count, gender = self._subject_tags(normalized)
        attrs.setdefault("subject", set()).update(subject_tags)
        for rules in (SPECIES_RULES, POSE_RULES, SETTING_RULES, WARDROBE_RULES, LIGHTING_RULES, MOOD_RULES):
            matches = _collect_rules(normalized, rules)
            for category, tags in matches.items():
                attrs.setdefault(category, set()).update(tags)
        self._apply_descriptors(normalized, attrs)
        attrs.setdefault("quality", set()).update(QUALITY_STYLES.get(quality_style, QUALITY_STYLES["balanced"]))
        rng = self._rng(seed)
        preset_key = preset if preset in PRESET_RULES else "balanced"
        adjectives = self._choose_adjectives(attrs, rng, preset_key)
        noun = self._base_noun(attrs, count, gender)
        pose_phrase = self._phrase(POSE_PHRASES, attrs.get("pose", set()), rng)
        setting_phrase = self._phrase(SETTING_PHRASES, attrs.get("setting", set()) | attrs.get("lighting", set()), rng)
        wardrobe_phrases = self._wardrobe_phrase(attrs, rng)
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
        subject_phrase = " ".join(fragments).strip(", ") or "character portrait"
        selected = self._select_categories(attrs, preset_key, rng)
        extra = list(extra_tags) if extra_tags else []
        ordered_tags = self._assemble_tags(selected, preset_key, extra, rng)
        parts = list(SCORE_TAGS)
        parts.append(subject_phrase)
        parts.extend(ordered_tags)
        prompt = ", ".join(parts)
        return ConversionResult(
            prompt=prompt,
            subject_phrase=subject_phrase,
            ordered_tags=ordered_tags,
            subject_count=count,
            inferred_gender=gender,
            applied_style=quality_style,
            prompt_weight=preset_key,
        )

converter = PonyPromptConverter()

def _generate_preview(
    prompt_text: str,
    quality_style: str,
    preset_key: str,
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
    )
    meta = (
        f"Subjects: {result.subject_count} | Gender: {result.inferred_gender} | "
        f"Preset: {result.prompt_weight} | Style: {result.applied_style}"
    )
    negative = DEFAULT_NEGATIVE if include_negative else ""
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
                extra_tags_box,
                negative_checkbox,
                seed_slider,
                lock_seed,
            ]

            regen_button.click(
                lambda text, style, preset, extra, neg, seed, lock: _generate_preview(
                    text, style, preset, extra, neg, _seed(seed, lock)
                ),
                inputs=inputs,
                outputs=[preview_box, negative_box, meta_box],
            )

        return [
            enable_checkbox,
            override_prompt,
            quality_style,
            preset_radio,
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
                }
            )

    def run(
        self,
        p: StableDiffusionProcessing,
        enable_conversion: bool,
        override_prompt: str,
        quality_style: str,
        preset_key: str,
        extra_tags: str,
        auto_negative: bool,
    ):
        return process_images(p)
