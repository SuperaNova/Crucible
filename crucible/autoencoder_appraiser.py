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

# Directed VQA prompts. Moondream2 supports full instruction-following, so the
# Appraiser asks two targeted questions per sprite to give the Master Smith dense,
# part-level grounding rather than a single vague caption. Moondream stays in
# natural language (it is unreliable at strict JSON); the structured decomposition
# into a part blueprint is the LLM Smith's job downstream.
_IDENTITY_PROMPT = (
    "You are an RPG item cataloguer. This is a pixel art icon of a fantasy item. "
    "State what the item is (e.g. 'iron sword', 'wooden shield', 'health potion'), "
    "then briefly describe its material and dominant colors. One short sentence only."
)
_PARTS_PROMPT = (
    "This is a pixel art icon of a fantasy item. List its distinct visible physical "
    "parts as a short comma-separated list (e.g. 'blade, crossguard, grip, pommel'). "
    "If it is a single solid object with no separate parts, answer 'whole'."
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

    def unload(self) -> None:
        """
        Release the model from GPU memory and clear the CUDA cache.

        Call this after appraise() returns if VRAM is needed for subsequent
        pipeline stages. After calling unload(), this instance must not be
        used again — create a new MoondreamAppraiser for the next run.
        """
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                import torch as _torch
                _torch.cuda.empty_cache()
            print("[MoondreamAppraiser] Model unloaded, VRAM released.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def appraise(self, img_a: Image.Image, img_b: Image.Image) -> dict:
        """
        Appraise both sprites and return a structured per-item dict for the
        Master Smith to ground its part-level material blueprint on.

        Args:
            img_a: PIL Image for sprite A.
            img_b: PIL Image for sprite B.

        Returns:
            dict of the form::

                {
                    "item_a": {"description": str, "parts": str, "tags": [str, ...]},
                    "item_b": {"description": str, "parts": str, "tags": [str, ...]},
                }
        """
        item_a = self._appraise_one(img_a, "A")
        item_b = self._appraise_one(img_b, "B")
        return {"item_a": item_a, "item_b": item_b}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _appraise_one(self, img: Image.Image, label: str) -> dict:
        """
        Run two directed VQA queries on a single sprite: one for identity +
        material + colours, one for its visible parts. Returns a structured
        per-item appraisal dict.
        """
        description = self._query(img, _IDENTITY_PROMPT)
        parts = self._query(img, _PARTS_PROMPT)

        print(f"[MoondreamAppraiser] Item {label}: {description}")
        print(f"[MoondreamAppraiser] Item {label} parts: {parts}")

        return {
            "description": description,
            "parts": parts,
            "tags": _extract_tags(description),
        }

    def _query(self, img: Image.Image, prompt: str) -> str:
        """
        Run a single directed VQA query on a PIL image using Moondream2.

        Moondream2 internally encodes the image into its latent embedding
        (Encoder step) then decodes the latent conditioned on the prompt
        (Decoder step). The 2025 API exposes this as a single query() call.
        """
        img_rgb = img.convert("RGB")

        with torch.no_grad():
            answer: str = self._model.query(img_rgb, prompt)["answer"]

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
