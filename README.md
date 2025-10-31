# Pony Prompt Optimizer

An Automatic1111/Forge extension that rewrites natural language descriptions into pony‑friendly prompts. Feed it a sentence like “A young female elf sitting on a bed with pajamas and rocker boots” and it will output a structured prompt that starts with the standard `score_9, score_8_up, …` prefix followed by clean, tag‑style keywords.

## Features

- Replaces the raw sentence with a curated pony prompt (no original text leakage).
- Three density presets (**light**, **balanced**, **heavy**) that control how many tags and flourishes are injected.
- Synonym and flair pools keep prompts fresh—click **Regenerate** to cycle new variants.
- Multi-sentence descriptions are analyzed sentence-by-sentence so characters, outfits, and settings merge cleanly without duplicate tags.
- Variation slider lets you tighten or loosen the prompt length and optional flair tags.
- Push button copies the preview straight into the active prompt textbox (txt2img/img2img).
- Optional quality styles (balanced, dreamy, dramatic, vibrant, minimal) and auto negative prompt fill.
- Preview seed lock lets you reproduce a draft before running a generation.

## Installation

1. Copy the `pony-prompt-optimizer` folder into your `extensions` directory for Automatic1111 or Forge.
2. Restart the web UI.

## Usage

1. Open the **Pony Prompt Optimizer** accordion in either the `txt2img` or `img2img` tab.
2. Check **Enable pony prompt conversion**.
3. Enter your natural language description in the main prompt box **or** type it into the optional override field inside the accordion.
4. Pick a quality style, choose a prompt weight (light/balanced/heavy), set the variation slider (0 = tighter, 1 = looser), add any extra tags, and decide whether to auto-fill the standard negative prompt.
5. Click **Regenerate preview** until you like the result, then hit **Push to prompt** (or just run a generation—the script will replace the prompt automatically).

The converted prompt is stored in `extra generation params` so you can recover it from image metadata later.

## Notes

- The rule set is heuristic. Ambiguous wording (“two swords”, “four lights”) may be interpreted as character counts. When that happens, clarify the sentence (e.g., “a single elf holding two swords”).  
- Extra tags typed into the “Extra tags” box are appended after conversion. Use them for manual tweaks like `cinematic_angle` or `intricate_details`.
- Light/balanced/heavy adjust how many accessory, lighting, and mood tags are appended; start light if you want more manual control.
- Variation values near 0 keep things concise; values near 1 encourage extra flair, lighting, and style tags.

Feel free to extend `scripts/pony_prompt_optimizer.py` with extra rules or presets to match your workflow.
