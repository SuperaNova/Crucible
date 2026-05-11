from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types

from .schemas import SmithingResult

MODEL = "gemini-3.1-flash-lite-preview"

PIXEL_ART_PREFIX = "pixel art sprite, 32x32 grid, RPG game item icon, "
PIXEL_ART_SUFFIX = (
    "pure white background, single centered item, hard pixel edges, "
    "no anti-aliasing, limited 16-color palette, retro SNES 16-bit style, "
    "flat colors, black pixel outline, no gradients, no shadows, "
    "no photorealism, no painterly style"
)

# ---------------------------------------------------------------------------
# Agent 1 — The Appraiser (BLIP Vision-Encoder-Decoder)
# ---------------------------------------------------------------------------
# NOTE: The Appraiser is no longer an LlmAgent. It is replaced by the
# BLIPAppraiser class in autoencoder_appraiser.py, which runs locally as
# a pre-trained Vision-Encoder-Decoder model.
# The captions it produces are injected into session state under the key
# 'appraisal' before the Master Smith runs.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agent 2 — The Master Smith
# ---------------------------------------------------------------------------

_master_smith = LlmAgent(
    name="master_smith",
    model=MODEL,
    instruction=(
        "You are a master blacksmith and RPG item designer. "
        "Two items are being fused in the Crucible.\n\n"
        "The items have been identified by a BLIP Vision-Encoder-Decoder model "
        "(an image captioning Autoencoder). Its appraisal is:\n"
        "{appraisal}\n\n"
        "Style modifier: {style_modifier}\n\n"
        "Using the item descriptions and their keyword tags, design the fusion:\n"
        "  1. fused_name: a creative thematic name that blends both items.\n"
        "  2. reasoning: one or two sentences of in-world lore explaining the fusion.\n"
        f"  3. image_prompt: must begin with '{PIXEL_ART_PREFIX}', "
        "describe exact colors, shapes, and the blended visual elements of both items, "
        f"and must end with '{PIXEL_ART_SUFFIX}'.\n"
        "No photorealistic, painterly, or 3D style words."
    ),
    output_schema=SmithingResult,
    output_key="smithing",
    include_contents="none",
    generate_content_config=types.GenerateContentConfig(temperature=0.75),
)

_pipeline = SequentialAgent(
    name="crucible_pipeline",
    sub_agents=[_master_smith],
)
