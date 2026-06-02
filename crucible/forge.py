"""
forge.py
--------
Orchestrates the three-stage Crucible pipeline, modelled on the RPG framework
(Yang et al., ICML 2024 — "Mastering Text-to-Image Diffusion: Recaptioning,
Planning, and Generating with Multimodal LLMs"):

  Stage 1 — Recaptioning  : SpriteAppraiser reads both sprites — SigLIP gives the
                             item identity (type) and Moondream2 describes the
                             appearance (colours/materials) + keyword tags.
  Stage 2 — CoT Planning  : Master Smith (Gemini via Google ADK) analyses the
                             appraisal and produces a structured part-by-part
                             material blueprint (+ archetype, structure_source).
  Stage 3 — Generation    : Forger assembles the blueprint into a Flux prompt,
                             fetches the image from Pollinations/Flux.1, and
                             applies pixel-art post-processing.

VRAM management
---------------
SigLIP SO400M (~1.7 GB) and Moondream2 (~3.5 GB) in float16 each fit a 6 GB
card (e.g. GTX 1660 Super). Stages 2 and 3 are API-only (no GPU). SpriteAppraiser
loads the two models in sequence and unloads each — freeing the CUDA cache —
so peak VRAM stays ~3.5 GB and is fully released before Stage 2 begins.
"""

import asyncio
import concurrent.futures
import io
import json
import os
import random
import uuid
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image

from .agents import PIXEL_ART_PREFIX, PIXEL_ART_SUFFIX, _pipeline
from .autoencoder_appraiser import SpriteAppraiser

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UI_PREVIEW_SIZE = 512
POLLINATIONS_SIZE = 512
PIXEL_GRID = 32
PALETTE_COLORS = 16

_APP_NAME = "crucible"
_USER_ID = "local"


# ---------------------------------------------------------------------------
# Forge
# ---------------------------------------------------------------------------

class Forge:
    """
    Top-level orchestrator for the Crucible three-stage pipeline.

    MoondreamAppraiser is NOT loaded at construction time — it is created
    fresh for each quench() call and unloaded immediately after Stage 1 so
    that Stages 2 and 3 (both API-only) run with full VRAM headroom.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[Forge] GEMINI_API_KEY not found.\n"
                "Add it to your .env file as: GEMINI_API_KEY=your_key_here\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        # Agents 2+ — Google ADK pipeline (Master Smith → …)
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=_APP_NAME,
            agent=_pipeline,
            session_service=self._session_service,
        )

    def quench(self, sprite_a, sprite_b) -> tuple:
        """Run the full fusion pipeline and return (preview_a, preview_b, forged, metadata)."""
        raw_a = self._to_pil(sprite_a)   # native sprite (e.g. 16x16)
        raw_b = self._to_pil(sprite_b)

        # ------------------------------------------------------------------
        # Stage 1 — Recaptioning (SigLIP identity + Moondream appearance, local GPU)
        # ------------------------------------------------------------------
        # SpriteAppraiser loads SigLIP then Moondream2 in sequence and unloads each
        # to free VRAM before Stage 2. Stages 2 and 3 are API-only (no GPU). The
        # appraiser upscales the tiny sprite with LANCZOS internally for perception.
        print("[Stage 1 / Recaptioning] Appraising sprites (SigLIP identity + Moondream appearance) ...")
        appraisal = SpriteAppraiser().appraise(raw_a, raw_b)

        # ------------------------------------------------------------------
        # Stage 2 — CoT Material Planning (Master Smith via Google ADK)
        # ------------------------------------------------------------------
        # The Master Smith performs RPG-style chain-of-thought reasoning:
        # archetype/skeleton → per-part material assignment → structure_source.
        # It emits a structured `parts` blueprint; the prompt is assembled in code.
        session_id = str(uuid.uuid4())
        self._run_async(self._session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
            state={"appraisal": _format_appraisal(appraisal)},
        ))

        message = types.Content(
            role="user",
            parts=[types.Part(text="Design the fusion based on the appraisal in your context.")],
        )

        print("[Stage 2 / CoT Planning] Running Master Smith ...")
        for event in self._runner.run(user_id=_USER_ID, session_id=session_id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                author = getattr(event, "author", "agent")
                print(f"  [{author}] {(event.content.parts[0].text or '')[:120]}")

        session = self._run_async(self._session_service.get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id,
        ))

        smithing = _parse_state(session.state.get("smithing"))

        archetype = smithing.get("archetype", "")
        parts = _normalize_parts(smithing.get("parts"))
        fused_name = smithing.get("fused_name", "")
        structure_source = smithing.get("structure_source", "A")

        print(f"[Stage 2] Fused name    : {fused_name}")
        print(f"[Stage 2] Archetype     : {archetype}")
        print(f"[Stage 2] Structure src : {structure_source}")
        print(f"[Stage 2] Reasoning     : {smithing.get('reasoning')}")
        for p in parts:
            print(f"[Stage 2]   - {p.get('part')}: {p.get('material')} "
                  f"({p.get('color')}) [from {p.get('source')}] — {p.get('detail')}")

        prompt = _assemble_prompt(archetype, parts, fallback=fused_name)
        print(f"[Stage 2] Prompt        : {prompt[:160]}")

        # ------------------------------------------------------------------
        # Stage 3 — Generation (Pollinations / Flux.1 + pixel art post-processing)
        # ------------------------------------------------------------------
        # structure_source is forwarded for future ControlNet support
        # (e.g. Replicate flux-canny-dev). Currently unused by _forge().
        forged = self._forge(prompt, structure_source=structure_source)

        preview_a = raw_a.convert("RGB").resize((UI_PREVIEW_SIZE, UI_PREVIEW_SIZE), Image.NEAREST)
        preview_b = raw_b.convert("RGB").resize((UI_PREVIEW_SIZE, UI_PREVIEW_SIZE), Image.NEAREST)

        return preview_a, preview_b, forged, {
            "item_a": _describe_item(appraisal.get("item_a", {})),
            "item_b": _describe_item(appraisal.get("item_b", {})),
            "fused_name": fused_name,
            "reasoning": smithing.get("reasoning", ""),
            "archetype": archetype,
            "parts": parts,
            "structure_source": structure_source,
            "image_prompt": prompt,
        }

    # ------------------------------------------------------------------
    # Stage 3 — The Forger
    # ------------------------------------------------------------------

    def _forge(self, image_prompt: str, *, structure_source: str = "A") -> Image.Image:
        """Pollinations Flux.1 fetch + pixel art post-processing.

        Pixelate first (crush to PIXEL_GRID, scale back up with NEAREST),
        then quantize — so the palette snaps to hard pixel edges, not gradients.

        Args:
            image_prompt:     Full prompt string from the Master Smith.
            structure_source: Which sprite ('A' or 'B') is the structural anchor.
                              Unused here (Pollinations is a text-only API), but the
                              Colab notebook has an optional SDXL + ControlNet-Canny
                              cell that conditions generation on this sprite's silhouette.
        """
        seed = random.randint(0, 2**31 - 1)
        url = (
            f"https://image.pollinations.ai/prompt/{quote(image_prompt)}"
            f"?width={POLLINATIONS_SIZE}&height={POLLINATIONS_SIZE}"
            f"&model=flux&seed={seed}&nologo=true"
        )
        print(f"[Stage 3 / Generation] Fetching from Pollinations/Flux.1 (seed={seed}, structure_src={structure_source}) ...")
        response = requests.get(url, timeout=90)
        response.raise_for_status()

        result = pixelate_quantize(Image.open(io.BytesIO(response.content)))
        print(f"[Stage 3] Done. Output: {result.size}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_async(coro):
        """Run a coroutine from a sync context, even inside a running event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    @staticmethod
    def _to_pil(sprite) -> Image.Image:
        return Image.fromarray(sprite[..., :3].astype("uint8"))


# ---------------------------------------------------------------------------
# Pixel-art post-processing (the "Pixel-Lattice" layer)
# ---------------------------------------------------------------------------

def pixelate_quantize(
    img: Image.Image,
    *,
    grid: int = PIXEL_GRID,
    colors: int = PALETTE_COLORS,
    size: int = UI_PREVIEW_SIZE,
) -> Image.Image:
    """Crush a diffusion output onto a hard pixel grid, then snap to a limited palette.

    Pixelate first (centre-crop to square, downsample to ``grid`` with NEAREST so
    gradients are destroyed, scale back up to ``size`` with NEAREST), THEN quantize
    to ``colors`` via median-cut — so the palette snaps to the already-hard pixel
    edges rather than muddy intermediate colours.

    Shared by the Pollinations/Flux Forger and the notebook's ControlNet path so
    both produce identically post-processed sprites.
    """
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    return (
        img
        .resize((grid, grid), Image.NEAREST)
        .resize((size, size), Image.NEAREST)
        .quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        .convert("RGB")
    )


# ---------------------------------------------------------------------------
# Prompt assembly & state formatting
# ---------------------------------------------------------------------------

# Soft cap on the assembled body (Pollinations is a GET API — keep the URL sane).
_PROMPT_BODY_MAX = 700


def _assemble_prompt(archetype: str, parts: list, *, fallback: str = "") -> str:
    """Deterministically render the Master Smith's part blueprint into a Flux prompt.

    Each PartSpec becomes one localized clause ("blade made of polished gold
    (bright yellow), glowing runes along the edge"), so the per-part material
    detail is guaranteed to reach the diffusion model rather than relying on the
    LLM to free-write a faithful prompt. The body is wrapped with the shared
    pixel-art prefix/suffix style constraints.

    Falls back to `archetype` (or `fallback`, e.g. the fused name) when the
    blueprint is empty.
    """
    archetype = (archetype or "").strip()
    clauses: list[str] = []
    for p in parts:
        part = str(p.get("part", "")).strip()
        material = str(p.get("material", "")).strip()
        color = str(p.get("color", "")).strip()
        detail = str(p.get("detail", "")).strip()
        if not part:
            continue
        clause = part
        if material:
            clause += f" made of {material}"
        if color:
            clause += f" ({color})"
        if detail:
            clause += f", {detail}"
        clauses.append(clause)

    subject = f"a {archetype}" if archetype else (fallback.strip() or "a fantasy item")
    body = subject + (", " + ", ".join(clauses) if clauses else "")
    if len(body) > _PROMPT_BODY_MAX:
        body = body[:_PROMPT_BODY_MAX].rstrip(", ")

    return f"{PIXEL_ART_PREFIX}{body}, {PIXEL_ART_SUFFIX}"


def _describe_item(item: dict) -> str:
    """One-line 'type — appearance' summary for metadata / UI titles."""
    item = item or {}
    type_ = (item.get("type", "") or "").strip()
    desc = (item.get("description", "") or "").strip()
    return f"{type_} — {desc}" if type_ and desc else (type_ or desc)


def _format_appraisal(appraisal: dict) -> str:
    """Render the structured appraisal dict into a clean text block for the
    Master Smith instruction (ADK string-templates session state into `{appraisal}`)."""
    lines: list[str] = []
    for label, key in (("A", "item_a"), ("B", "item_b")):
        item = appraisal.get(key, {}) or {}
        lines.append(f"ITEM {label}:")
        lines.append(f"  type: {item.get('type', '').strip()}")
        lines.append(f"  appearance: {item.get('description', '').strip()}")
    return "\n".join(lines)


def _normalize_parts(value) -> list:
    """Coerce the Smith's `parts` output into a list of plain dicts."""
    if not value:
        return []
    out: list = []
    for p in value:
        if hasattr(p, "model_dump"):
            out.append(p.model_dump())
        elif isinstance(p, dict):
            out.append(p)
    return out


def _parse_state(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)
