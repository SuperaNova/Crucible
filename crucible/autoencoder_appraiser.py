"""
autoencoder_appraiser.py
------------------------
Implements the Appraiser agent using Moondream2
(vikhyatk/moondream2, rev 2025-06-21) — a compact 1.8B parameter
Vision-Language Model designed to run efficiently on consumer GPUs.

Loaded via HuggingFace `transformers` using the latest stable revision.
Uses the unified API introduced in 2025 where caption() and query()
accept PIL Images directly — no separate tokenizer or encode_image() needed.

Architecture (Encoder-Decoder family):
  - Encoder: SigLIP Vision Transformer encodes the input image into a dense
             latent visual embedding h = Encoder(x).
  - Decoder: A Phi-based causal language model decodes h into natural
             language given a text prompt: answer = Decoder(h, prompt).

This is architecturally equivalent to an image-conditioned Autoencoder whose
reconstruction target is natural language rather than pixels. Moondream2 was
trained on a broad dataset that includes 2D digital art, game assets, and
stylized icons, making it robust on pixel art sprites.

Reference:
  Kopuri, V. (2024). Moondream2: A Tiny Vision Language Model.
  HuggingFace: https://huggingface.co/vikhyatk/moondream2
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID = "vikhyatk/moondream2"
_REVISION = "2025-06-21"

# VQA prompt — Moondream2 supports full instruction-following so we can be
# specific about what we want. The 'short caption' mode gives us a concise
# one-liner, which is ideal for feeding into the Master Smith prompt.
_APPRAISAL_PROMPT = (
    "You are an RPG item cataloguer. This is a pixel art icon of a fantasy item. "
    "State what the item is (e.g. 'iron sword', 'wooden shield', 'health potion'), "
    "then briefly describe its material, color, and style. One short sentence only."
)


# ---------------------------------------------------------------------------
# MoondreamAppraiser
# ---------------------------------------------------------------------------

class MoondreamAppraiser:
    """
    Encodes two sprites via Moondream2 and returns natural-language descriptions
    that can be used by the Master Smith agent as an appraisal.

    Uses the 2025-06-21 API: model.query(image, prompt) and model.caption(image).
    The model is loaded once and cached on the instance.

    Fits comfortably within 6 GB VRAM (e.g. GTX 1660 Super) in float16.
    """

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[MoondreamAppraiser] Loading '{_MODEL_ID}' (rev {_REVISION}) "
            f"on {self._device.upper()} ..."
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()

        print("[MoondreamAppraiser] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def appraise(self, img_a: Image.Image, img_b: Image.Image) -> dict:
        """
        Caption both sprites and return an appraisal dict compatible with
        the session-state schema expected by the Master Smith.

        Args:
            img_a: PIL Image for sprite A.
            img_b: PIL Image for sprite B.

        Returns:
            dict with keys: item_a, item_b, item_a_tags, item_b_tags
        """
        caption_a = self._caption(img_a)
        caption_b = self._caption(img_b)

        print(f"[MoondreamAppraiser] Item A: {caption_a}")
        print(f"[MoondreamAppraiser] Item B: {caption_b}")

        return {
            "item_a": caption_a,
            "item_b": caption_b,
            "item_a_tags": _extract_tags(caption_a),
            "item_b_tags": _extract_tags(caption_b),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _caption(self, img: Image.Image) -> str:
        """
        Run a directed VQA query on a single PIL image using Moondream2.

        Moondream2 internally encodes the image into its latent embedding
        (Encoder step) then decodes the latent conditioned on the prompt
        (Decoder step). The 2025 API exposes this as a single query() call.
        """
        img_rgb = img.convert("RGB")

        with torch.no_grad():
            answer: str = self._model.query(img_rgb, _APPRAISAL_PROMPT)["answer"]

        return answer.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "with", "on", "in", "and", "or", "to",
    "is", "it", "its", "at", "by", "for", "from", "that", "this",
    "has", "are", "be", "as", "do", "i", "no", "not", "like",
    "item", "sprite", "icon", "pixel", "art", "fantasy", "rpg",
    "one", "short", "sentence",
}


def _extract_tags(caption: str) -> list[str]:
    """
    Extract meaningful keyword tags from a caption by removing stopwords.
    Returns a list of up to 6 unique tokens.
    """
    tokens = caption.lower().replace(",", "").replace(".", "").split()
    tags = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    seen: set[str] = set()
    unique_tags: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    return unique_tags[:6]
