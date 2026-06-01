from pydantic import BaseModel
from typing import Literal


class ItemAppraisal(BaseModel):
    """
    Structured per-item description produced by the Moondream2 Appraiser
    (RPG Stage 1 — Recaptioning). One of these is built for each input sprite
    and injected into ADK session state for the Master Smith to ground on.

    Fields:
        description: Identity + material + dominant colours of the item, as a single
                     natural-language sentence (e.g. "An iron sword with a grey steel
                     blade and a brown leather-wrapped grip").
        parts:       Visible physical parts, comma-separated (e.g. "blade, crossguard,
                     grip, pommel"); "whole" for a single solid object.
        tags:        Keyword tokens extracted from the description.
    """
    description: str
    parts: str
    tags: list[str]


class PartSpec(BaseModel):
    """
    A single localized part of the fused item, with an explicit material recipe.

    This is the Crucible analogue of an RPG "region": instead of a spatial
    bounding box, each PartSpec is a named component of the item (blade, grip,
    gem, ...) carrying a dense, concrete material description. The collection of
    PartSpecs is the semantic material blueprint that drives generation.

    Fields:
        part:     Name of the component (e.g. "blade", "grip", "gem socket").
        material: Concrete material (e.g. "polished gold", "oak wood").
        color:    Dominant colour of this part (e.g. "bright yellow", "dark brown").
        source:   Which input sprite this part's look is inherited from — "A", "B",
                  or "fused" when it blends both.
        detail:   Localized surface detail (e.g. "glowing runes etched along the edge").
    """
    part: str
    material: str
    color: str
    source: Literal["A", "B", "fused"]
    detail: str


class SmithingResult(BaseModel):
    """
    Structured output from the Master Smith agent (RPG Stage 2 — CoT Planning).

    The Smith no longer hand-writes the diffusion prompt; it emits a structured
    part-level blueprint and the Forger assembles the prompt deterministically.

    Fields:
        fused_name:       Creative thematic name for the fused item.
        reasoning:        One or two sentences of in-world lore explaining the fusion.
        archetype:        The fused item's base form / silhouette (e.g. "longsword",
                          "round shield"), normally inherited from structure_source.
        structure_source: Which input sprite provides the dominant silhouette / shape
                          anchor. Reserved as the ControlNet structural reference for
                          future image-conditioned generation.
        parts:            Ordered list of PartSpecs — the localized material blueprint
                          the Forger renders into the diffusion prompt.
    """
    fused_name: str
    reasoning: str
    archetype: str
    structure_source: Literal["A", "B"]
    parts: list[PartSpec]
