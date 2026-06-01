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
# Agent 2 — The Master Smith  (RPG Stage 2: CoT Material Planning)
# ---------------------------------------------------------------------------
#
# This agent mirrors the "Multimodal Chain-of-Thought Planning" stage of the
# RPG framework (Yang et al., ICML 2024). In the original paper the MLLM acts
# as a *spatial* planner — decomposing a complex prompt into bounding-box
# regions, each given a dense localized sub-prompt. Crucible maps that idea
# from spatial regions onto the *parts of an item*: the Smith decomposes the
# fused item into named components (blade, grip, guard, gem, ...) and gives
# each one a concrete material recipe inherited from the two source sprites —
# e.g. "oak-wood grip, polished-gold blade". This part-level blueprint is the
# Crucible analogue of RPG's region-wise detailed recaption.
#
# The Smith does NOT write the diffusion prompt — it emits the structured
# `parts` list and the Forger assembles the prompt deterministically, so the
# localized material detail is guaranteed to survive into generation.
#
# The agent reads exclusively from ADK session state (include_contents="none")
# so that the appraisal is the sole grounding signal — no free-form chat
# history contaminates the reasoning.

_master_smith = LlmAgent(
    name="master_smith",
    model=MODEL,
    instruction=(
        "You are a master blacksmith and RPG item designer operating inside "
        "the Crucible fusion pipeline.\n\n"
        "Two pixel art items (A and B) are being fused. A Vision-Language Model "
        "(Moondream2) has appraised them:\n\n"
        "{appraisal}\n\n"
        "Design the fused item as a concrete, part-by-part material blueprint. "
        "Think carefully through four steps:\n\n"
        "STEP 1 — ARCHETYPE & SKELETON\n"
        "Decide the base form (archetype) of the fused item — usually inherit the "
        "silhouette of whichever source item has the stronger/clearer shape (e.g. a "
        "sword keeps its blade silhouette). Then enumerate that archetype's canonical "
        "physical parts (e.g. a sword -> blade, crossguard, grip, pommel; a shield -> "
        "face, rim, boss). Set `archetype` to a short noun phrase (e.g. 'longsword').\n\n"
        "STEP 2 — PART-BY-PART MATERIAL ASSIGNMENT\n"
        "For EVERY part, output one entry in `parts` with concrete, physical values:\n"
        "  - `part`     : the component name (e.g. 'blade').\n"
        "  - `material` : a specific material (e.g. 'polished gold', 'oak wood', "
        "'blue crystal') — NOT a vague blend word.\n"
        "  - `color`    : the dominant colour of that part (e.g. 'bright yellow').\n"
        "  - `source`   : 'A' or 'B' if that part's look comes from one source item, "
        "or 'fused' if it genuinely blends both.\n"
        "  - `detail`   : one concrete localized surface detail (e.g. 'glowing runes "
        "etched along the edge', 'leather wrap', 'faceted gem set in the center').\n"
        "Make real choices — e.g. blade=steel from A, grip=wood from B — so the result "
        "reads like 'wooden handle with a gold blade', not a muddy average.\n\n"
        "STEP 3 — STRUCTURE SOURCE\n"
        "Set `structure_source` to 'A' or 'B' — whichever item contributes the dominant "
        "silhouette / shape skeleton of the fused result.\n\n"
        "STEP 4 — NAME & LORE\n"
        "Write `fused_name` (a creative thematic name) and `reasoning` (one or two "
        "sentences of in-world lore explaining how the two items combined)."
    ),
    output_schema=SmithingResult,
    output_key="smithing",
    include_contents="none",
    generate_content_config=types.GenerateContentConfig(temperature=0.7),
)

_pipeline = SequentialAgent(
    name="crucible_pipeline",
    sub_agents=[_master_smith],
)
