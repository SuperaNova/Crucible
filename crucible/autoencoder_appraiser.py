"""
autoencoder_appraiser.py
------------------------
Stage 1 (RPG Recaptioning). Two specialised vision models, each used only for
what it is actually good at:

  * SiglipClassifier — zero-shot item *identity*. SigLIP (Zhai et al., 2023) is
    a contrastive image-text model; scoring a sprite against a fixed RPG
    vocabulary ("sword", "gem", "potion", ...) is far more reliable than asking
    a small generative VLM the open-ended "what is this?", which hallucinates on
    16x16 art (it once called a blue gem a "health potion"). Classification over
    a closed set is a much easier task than open generation for a small model.

  * MoondreamDescriber — Moondream2 (1.8B VLM) describes *appearance only*
    (colours, materials, textures). It is good at this and bad at naming, so we
    never ask it to name the item or list parts — the cascading "fake sword"
    failures came from Moondream parroting an example parts list.

The Master Smith downstream is given each item's reliable TYPE plus an
appearance description, and derives the fused item's parts itself.

VRAM
----
Both models run on local GPU. They are loaded and unloaded in sequence (SigLIP
first, then Moondream) so peak VRAM stays ~3.5 GB — within a 6 GB budget.

Perception input
----------------
Sprites are tiny (16x16). For perception we upscale with **LANCZOS** (smooth),
NOT nearest-neighbour — hard pixel blocks are out-of-distribution for these
models. The pixelated look is re-imposed only on the final generated output.

Reference:
  Zhai et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP).
  Kopuri, V. (2024). Moondream2: A Tiny Vision Language Model.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOONDREAM_ID = "vikhyatk/moondream2"
_MOONDREAM_REV = "2025-06-21"
# SO400M is the strong zero-shot SigLIP. Swap to "google/siglip-base-patch16-224"
# for a much smaller/faster download at the cost of accuracy.
_SIGLIP_ID = "google/siglip-so400m-patch14-384"

VLM_INPUT_SIZE = 384  # LANCZOS upscale target fed to both models for perception.

# Closed-set item vocabulary for SigLIP zero-shot identity. Tune freely.
# Ambiguous words are disambiguated ("bow" -> "archery bow") so the text encoder
# does not confuse a weapon with a ribbon/bowtie.
ITEM_VOCAB = [
    "sword", "dagger", "axe", "mace", "war hammer", "spear", "archery bow",
    "crossbow", "magic staff", "wand", "shield", "helmet", "body armor",
    "gauntlet", "boot", "cape", "ring", "amulet", "gemstone", "crystal", "orb",
    "potion bottle", "flask", "scroll", "book", "key", "gold coin",
    "treasure chest", "torch", "lantern", "glass bottle", "barrel", "pickaxe",
    "shovel", "fishing rod", "flower", "leaf", "mushroom", "bone", "skull",
    "egg", "feather", "meat", "bread", "fish", "fruit",
]

# Prompt ensembling: each class is scored under several templates and the scores
# are averaged. RPG/video-game context steers ambiguous words toward the item
# sense. This is the standard CLIP/SigLIP zero-shot accuracy trick.
_PROMPT_TEMPLATES = [
    "a pixel art icon of a {}",
    "a {} sprite in a fantasy RPG video game",
    "a small pixel art {}",
]

# Moondream is asked for appearance ONLY — never identity, never parts.
_DESCRIBE_PROMPT = (
    "Describe only the visual appearance of this small icon: its dominant colours, "
    "the materials or textures it appears to be made of, and any glow, pattern, or "
    "outline. Do not name or guess what the object is. One short sentence."
)


def _upscale(img: Image.Image, size: int = VLM_INPUT_SIZE) -> Image.Image:
    """Smooth (LANCZOS) upscale of a tiny sprite for VLM perception."""
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# SigLIP — zero-shot item identity
# ---------------------------------------------------------------------------

class SiglipClassifier:
    """Zero-shot item-type classifier scoring a sprite against ITEM_VOCAB."""

    def __init__(self, vocab: list[str] = ITEM_VOCAB) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vocab = list(vocab)
        self._n_templates = len(_PROMPT_TEMPLATES)
        # Class-major order: [c0t0, c0t1, ..., c1t0, ...] so a simple reshape to
        # (num_classes, num_templates) lines up for per-class averaging.
        self._prompts = [t.format(c) for c in self._vocab for t in _PROMPT_TEMPLATES]

        print(f"[SiglipClassifier] Loading '{_SIGLIP_ID}' on {self._device.upper()} ...")
        self._processor = AutoProcessor.from_pretrained(_SIGLIP_ID)
        self._model = AutoModel.from_pretrained(
            _SIGLIP_ID,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()
        print("[SiglipClassifier] Model loaded.")

    def classify(self, img: Image.Image, topk: int = 3) -> str:
        """Return the best-matching item type; print the top-k for transparency."""
        inputs = self._processor(
            text=self._prompts,
            images=_upscale(img),
            padding="max_length",   # SigLIP requires fixed-length padding.
            return_tensors="pt",
        ).to(self._device)
        # Match image dtype to the (possibly fp16) model weights.
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)

        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]

        # Average the per-template logits down to one score per class.
        scores = logits.float().view(len(self._vocab), self._n_templates).mean(dim=1)
        order = scores.argsort(descending=True)

        top = ", ".join(
            f"{self._vocab[int(i)]} ({scores[int(i)]:.1f})"
            for i in order[:min(topk, len(self._vocab))]
        )
        print(f"[SiglipClassifier] top-{topk}: {top}")

        return self._vocab[int(order[0])]

    def unload(self) -> None:
        if getattr(self, "_model", None) is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
            print("[SiglipClassifier] Model unloaded, VRAM released.")


# ---------------------------------------------------------------------------
# Moondream2 — appearance description (no naming)
# ---------------------------------------------------------------------------

class MoondreamDescriber:
    """Moondream2 VLM used only to describe appearance (colours/materials)."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[MoondreamDescriber] Loading '{_MOONDREAM_ID}' (rev {_MOONDREAM_REV}) "
            f"on {self._device.upper()} ..."
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            _MOONDREAM_ID,
            revision=_MOONDREAM_REV,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()
        print("[MoondreamDescriber] Model loaded.")

    def describe(self, img: Image.Image) -> str:
        with torch.no_grad():
            answer: str = self._model.query(_upscale(img), _DESCRIBE_PROMPT)["answer"]
        return answer.strip()

    def unload(self) -> None:
        if getattr(self, "_model", None) is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
            print("[MoondreamDescriber] Model unloaded, VRAM released.")


# ---------------------------------------------------------------------------
# SpriteAppraiser — orchestrates the two models for Stage 1
# ---------------------------------------------------------------------------

class SpriteAppraiser:
    """
    Stage 1 appraiser: SigLIP identity + Moondream appearance.

    The two models are loaded and unloaded in sequence so they never sit in VRAM
    at the same time (peak ~3.5 GB). Returns a structured per-item dict::

        {
            "item_a": {"type": str, "description": str, "tags": [str, ...]},
            "item_b": {...},
        }
    """

    def appraise(self, img_a: Image.Image, img_b: Image.Image) -> dict:
        # Phase 1 — identity (SigLIP zero-shot)
        clf = SiglipClassifier()
        type_a = clf.classify(img_a)
        type_b = clf.classify(img_b)
        print(f"[Stage 1] SigLIP identity  — A: '{type_a}'   B: '{type_b}'")
        clf.unload()
        del clf

        # Phase 2 — appearance (Moondream2)
        describer = MoondreamDescriber()
        desc_a = describer.describe(img_a)
        desc_b = describer.describe(img_b)
        print(f"[Stage 1] Moondream appearance — A: {desc_a}")
        print(f"[Stage 1] Moondream appearance — B: {desc_b}")
        describer.unload()
        del describer

        return {
            "item_a": {
                "type": type_a,
                "description": desc_a,
                "tags": _extract_tags(f"{type_a} {desc_a}"),
            },
            "item_b": {
                "type": type_b,
                "description": desc_b,
                "tags": _extract_tags(f"{type_b} {desc_b}"),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "with", "on", "in", "and", "or", "to",
    "is", "it", "its", "at", "by", "for", "from", "that", "this",
    "has", "are", "be", "as", "do", "i", "no", "not", "like",
    "item", "sprite", "icon", "pixel", "art", "fantasy", "rpg",
    "one", "short", "sentence", "appears", "made", "looks",
}


def _extract_tags(text: str) -> list[str]:
    """Extract up to 6 unique keyword tags from text by removing stopwords."""
    tokens = text.lower().replace(",", "").replace(".", "").split()
    seen: set[str] = set()
    unique_tags: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or len(tok) <= 2 or tok in seen:
            continue
        seen.add(tok)
        unique_tags.append(tok)
    return unique_tags[:6]
