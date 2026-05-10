from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types

from .schemas import AppraisalResult, SmithingResult

MODEL = "gemini-2.5-flash"

PIXEL_ART_PREFIX = "pixel art sprite, 32x32 grid, RPG game item icon, "
PIXEL_ART_SUFFIX = (
    "pure white background, single centered item, hard pixel edges, "
    "no anti-aliasing, limited 16-color palette, retro SNES 16-bit style, "
    "flat colors, black pixel outline, no gradients, no shadows, "
    "no photorealism, no painterly style"
)

_appraiser = LlmAgent(
    name="appraiser",
    model=MODEL,
    instruction=(
        "You are an expert RPG item cataloguer. You are shown two pixel art "
        "item sprites from a retro RPG game. The FIRST image is Item A and "
        "the SECOND image is Item B.\n\n"
        "For each sprite identify:\n"
        "  - item_a / item_b: short common name (e.g. 'iron sword')\n"
        "  - item_a_tags / item_b_tags: 3 to 5 descriptive tags about "
        "material, element, or visual style (e.g. ['metal', 'blue', 'flame'])"
    ),
    output_schema=AppraisalResult,
    output_key="appraisal",
    generate_content_config=types.GenerateContentConfig(temperature=0.3),
)

_master_smith = LlmAgent(
    name="master_smith",
    model=MODEL,
    instruction=(
        "You are a master blacksmith and RPG item designer. "
        "Two items are being fused in the Crucible.\n\n"
        "Appraisal: {appraisal}\n"
        "Style modifier: {style_modifier}\n\n"
        "Design the fusion:\n"
        "  1. fused_name: a creative thematic name.\n"
        "  2. reasoning: one or two sentences of in-world lore.\n"
        f"  3. image_prompt: must begin with '{PIXEL_ART_PREFIX}', "
        "describe exact colors and shapes, "
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
    sub_agents=[_appraiser, _master_smith],
)
