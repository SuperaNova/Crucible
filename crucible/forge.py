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
from .autoencoder_appraiser import MoondreamAppraiser

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_INPUT_SIZE = 128
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
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[Forge] GEMINI_API_KEY not found.\n"
                "Add it to your .env file as: GEMINI_API_KEY=your_key_here\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        # Agent 1 — Moondream2 Vision-Language Model (local, pre-trained)
        self._appraiser = MoondreamAppraiser()

        # Agents 2+ — Google ADK pipeline (Master Smith → …)
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=_APP_NAME,
            agent=_pipeline,
            session_service=self._session_service,
        )

    def quench(self, sprite_a, sprite_b) -> tuple:
        """Run the full fusion pipeline and return (preview_a, preview_b, forged, metadata)."""
        img_a = self._to_pil(sprite_a).resize((GEMINI_INPUT_SIZE, GEMINI_INPUT_SIZE), Image.NEAREST)
        img_b = self._to_pil(sprite_b).resize((GEMINI_INPUT_SIZE, GEMINI_INPUT_SIZE), Image.NEAREST)

        # ------------------------------------------------------------------
        # Stage 1 — Moondream2 Appraisal (Vision-Language Model, runs locally)
        # ------------------------------------------------------------------
        print("[MoondreamAppraiser] Running ...")
        appraisal = self._appraiser.appraise(img_a, img_b)
        print(f"[MoondreamAppraiser] Item A: '{appraisal['item_a']}'  tags={appraisal['item_a_tags']}")
        print(f"[MoondreamAppraiser] Item B: '{appraisal['item_b']}'  tags={appraisal['item_b_tags']}")

        # ------------------------------------------------------------------
        # Stage 2 — Master Smith (Google ADK LLM Agent)
        # ------------------------------------------------------------------
        session_id = str(uuid.uuid4())
        self._run_async(self._session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
            state={
                "appraisal": appraisal,
            },
        ))

        # Master Smith reads from session state only — no image parts needed.
        message = types.Content(
            role="user",
            parts=[types.Part(text="Design the fusion based on the appraisal in your context.")],
        )

        print("[Master Smith] Running ...")
        for event in self._runner.run(user_id=_USER_ID, session_id=session_id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                author = getattr(event, "author", "agent")
                print(f"  [{author}] {(event.content.parts[0].text or '')[:120]}")

        session = self._run_async(self._session_service.get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id,
        ))

        smithing = _parse_state(session.state.get("smithing"))

        print(f"[Master Smith] Fused:  {smithing.get('fused_name')}")
        print(f"[Master Smith] Reason: {smithing.get('reasoning')}")
        print(f"[Master Smith] Prompt: {smithing.get('image_prompt')}")

        prompt = smithing.get("image_prompt", "")
        if not prompt.startswith(PIXEL_ART_PREFIX):
            prompt = PIXEL_ART_PREFIX + prompt
        if PIXEL_ART_SUFFIX not in prompt:
            prompt = prompt.rstrip(", ") + ", " + PIXEL_ART_SUFFIX

        # ------------------------------------------------------------------
        # Stage 3 — Forger (Pollinations Flux + pixel art post-processing)
        # ------------------------------------------------------------------
        forged = self._forge(prompt)

        preview_a = img_a.resize((UI_PREVIEW_SIZE, UI_PREVIEW_SIZE), Image.NEAREST)
        preview_b = img_b.resize((UI_PREVIEW_SIZE, UI_PREVIEW_SIZE), Image.NEAREST)

        return preview_a, preview_b, forged, {
            "item_a": appraisal.get("item_a", ""),
            "item_b": appraisal.get("item_b", ""),
            "fused_name": smithing.get("fused_name", ""),
            "reasoning": smithing.get("reasoning", ""),
            "image_prompt": prompt,
        }

    # ------------------------------------------------------------------
    # Agent 3 — The Forger
    # ------------------------------------------------------------------

    def _forge(self, image_prompt: str) -> Image.Image:
        """Pollinations Flux fetch + pixel art post-processing.

        Pixelate first (crush to PIXEL_GRID, scale back up with NEAREST),
        then quantize — so the palette snaps to hard pixel edges, not gradients.
        """
        seed = 676767
        url = (
            f"https://image.pollinations.ai/prompt/{quote(image_prompt)}"
            f"?width={POLLINATIONS_SIZE}&height={POLLINATIONS_SIZE}"
            f"&model=flux&seed={seed}&nologo=true"
        )
        print(f"[Forger] Fetching from Pollinations (seed={seed}) ...")
        response = requests.get(url, timeout=90)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))

        result = (
            img
            .resize((PIXEL_GRID, PIXEL_GRID), Image.NEAREST)
            .resize((UI_PREVIEW_SIZE, UI_PREVIEW_SIZE), Image.NEAREST)
            .quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
            .convert("RGB")
        )
        print(f"[Forger] Done. Output: {result.size}")
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

    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


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
