# Crucible: Autonomous Sprite Synthesis Pipeline
*(Intelligent Systems / CS346 Mini-Project)*

Crucible is a Multi-Agent System (MAS) that uses advanced generative models to autonomously "fuse" two 2D pixel art RPG items into a completely new, technically coherent artifact. Instead of mathematically averaging pixels (which produces blurry artifacts), Crucible uses a **semantic fusion** approach inspired by the **RPG framework** (Yang et al., ICML 2024 — *Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs*).

The RPG paper proposes using an MLLM as a global planner that decomposes a complex prompt into spatial sub-regions, giving each region a dense localized sub-prompt for complementary diffusion. Crucible maps that idea from *spatial regions* onto **item parts**: the LLM decomposes the fused item into named components (blade, grip, guard, gem, …) and assigns each one a concrete material recipe drawn from the two source sprites — e.g. "oak-wood grip, polished-gold blade" — acting as a *semantic material planner* rather than a spatial layout planner.

## I. Project Objective
* **The Concept:** Crucible is designed as a prototype for an in-game crafting mechanic. In many RPGs, crafting systems are static—combining an Iron Ingot and a Stick always yields a generic Iron Sword. Crucible aims to show how generative AI can be embedded into a game engine to allow players to *actually* forge unique items on the fly, resulting in procedurally generated visual assets and lore that didn't exist in the game's original files.
* **The Solution:** A hybrid Multi-Agent System that combines **Vision-Language Models**, **Large Language Models (LLMs)**, and **Latent Diffusion Models**, orchestrated by the **Google Agent Development Kit (ADK)** and structured around the three stages of the RPG framework.

---

## II. Component Models (The Three Pillars)

Crucible maps directly onto the three stages of the RPG framework, adapted from spatial scene generation to sprite composition:

### Stage 1 — Recaptioning: The Vision Appraiser (SigLIP + Moondream2)
* **RPG Role:** Recaptioning — read both source sprites and produce a reliable, grounded description for each, replacing the original paper's "complex prompt → sub-prompt decomposition" with "sprite → appraisal."
* **Architecture:** Two specialised vision models, each used only for what it is good at:
  * **Identity — SigLIP zero-shot classifier** (Zhai et al., 2023; SO400M). The sprite is scored against a fixed RPG vocabulary (`sword`, `gemstone`, `potion bottle`, `shield`, …) using **prompt ensembling** (each class scored under several RPG-context templates and averaged — the standard CLIP/SigLIP accuracy trick), and the best match is taken as the item `type`. Closed-set classification is dramatically more reliable than open-ended generation for a 16×16 icon — a small generative VLM asked "what is this?" will confidently mislabel a blue gem as a "health potion." The top-3 scores are logged for transparency.
  * **Appearance — Moondream2** (1.8B VLM). Moondream is asked for *appearance only* — dominant colours, materials, and textures — and is **never** asked to name the item or list parts (parroted part lists were the cause of cascading "fake sword" outputs). It is genuinely good at this.
* **Why this split:** It plays to each model's strength and removes the two failure modes we observed (mis-identification and example-parroting). The Master Smith downstream is handed each item's trustworthy `type` plus an appearance description, and derives the fused item's parts itself.
* **Perception detail:** Sprites are upscaled with **LANCZOS** (smooth), not nearest-neighbour, before being fed to the models — hard pixel blocks are out-of-distribution for both. The pixelated look is re-imposed only on the final generated output.
* **VRAM:** SigLIP SO400M (~1.7 GB) and Moondream2 (~3.5 GB) in `float16` are loaded and unloaded **in sequence**, so peak VRAM stays ~3.5 GB (within a 6 GB GTX 1660 Super budget) and is fully released before Stage 2.

### Stage 2 — CoT Planning: The Master Smith (Gemini Flash Lite via Google ADK)
* **RPG Role:** Multimodal Chain-of-Thought Planning — the MLLM acts as a global planner. In the original paper the LLM plans spatial bounding-box regions, each with a dense sub-prompt; here it plans a *part-level material blueprint* across the two sprites.
* **Architecture:** Large Language Model (LLM).
* **How it Works:** Given each item's classified `type` (from SigLIP) and appearance description (from Moondream2), the Master Smith performs explicit four-step CoT reasoning:
  1. **Archetype & skeleton** — picks the fused item's base form (`archetype`) and enumerates its canonical physical parts (e.g. a sword → blade, crossguard, grip, pommel).
  2. **Part-by-part material assignment** — for *every* part emits a concrete `material`, `color`, `source` (A / B / fused), and a localized surface `detail`. This is the core of semantic material compositing — real choices like "blade=gold from B, grip=wood from A" rather than a muddy average.
  3. **Structure source** — designates which sprite provides the primary visual skeleton (preserved for future ControlNet conditioning).
  4. **Name & lore** — writes `fused_name` and `reasoning`.
* **Structured output, not a prompt:** The Smith emits a structured `parts` list (validated by the `SmithingResult` schema) — it does **not** hand-write the diffusion prompt. The Forger assembles the prompt deterministically from the blueprint so every per-part material detail is guaranteed to reach the model.
* **Why ADK:** The multi-step structured output and session-state handoff to the Forger justify ADK's orchestration layer here; the planning output is richer than a simple single-call LLM interaction.

### Stage 3 — Generation: The Forger (Flux.1 via Pollinations)
* **RPG Role:** Generation — the diffusion model manifests the fused sprite from the planner's output prompt. In the original paper this is Complementary Regional Diffusion; here it is a single high-quality Flux.1 call conditioned on the Master Smith's structured prompt.
* **Architecture:** Latent Diffusion Model (Transformer-backed).
* **How it Works:** The Forger first assembles the Master Smith's part blueprint into a single diffusion prompt deterministically — each part becomes a localized clause (`"blade made of polished gold (bright yellow), glowing runes along the edge"`) wrapped in the pixel-art style constraints — then Flux.1 runs reverse diffusion guided by that prompt.
* **Structural conditioning (ControlNet):** The default text-only path ignores `structure_source`. The Colab notebook includes an optional **SDXL + ControlNet-Canny** cell that builds a Canny edge map from the `structure_source` sprite and conditions generation on it, so the fused item follows that silhouette — shown side-by-side with the text-only output. (Kept notebook-side; Flux + ControlNet won't fit the 6 GB local budget.)
* **Why Pollinations/Flux.1:** Flux.1 provides state-of-the-art text adherence and image quality without requiring a heavy local diffusion model — keeping the local GPU footprint to just the lightweight Stage-1 vision models (and they are already unloaded by the time Stage 3 runs).

---

## III. System Architecture: Multi-Agent Orchestration

The handoff between these disparate models is managed by the **Google Agent Development Kit (ADK)**, creating a stateful Sequential Pipeline that mirrors the three stages of the RPG framework:

1. **Stage 1 — Recaptioning (Appraiser):** `Input Sprites → SigLIP Zero-Shot (item type) + Moondream2 (appearance) → Per-Item Appraisal (type + description + tags)`
2. **Stage 2 — CoT Planning (Master Smith):** `Appraisal → ADK Session State → LLM CoT Reasoning → Part-Level Material Blueprint + Archetype + Structure Source`
3. **Stage 3 — Generation (Forger):** `Blueprint → Deterministic Prompt Assembly → Flux.1 Diffusion → Pixel Art Post-Processing → Output Sprite`

The Master Smith's `parts` blueprint and `structure_source` outputs are the Crucible-specific adaptation of RPG's rationale + region-planning stage: the `parts` list maps RPG's per-region detail onto per-part material recipes. `structure_source` designates which sprite's silhouette provides the dominant visual skeleton — surfaced today in the UI, and reserved as the ControlNet structural reference for future image-conditioned generation.

---

## IV. Data Methodology & Representation

### Sprite Representation
Sprites are stored as `16x16` RGB tensors (`uint8`, array shape `(N, 16, 16, 3)`). Because the source art is tiny, the Stage-1 vision models receive a smooth **LANCZOS** upscale, while the *final generated output* is crushed back to a hard `32x32` pixel grid via `NEAREST` resampling and 16-colour quantization — so it snaps to a strict pixel grid rather than a blurry digital painting.

### Dataset Filtering (The Smelter)
We rely on the ["Pixel Art" dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) from Kaggle (~89,400 raw `16x16` sprites). To prevent "garbage in, garbage out" (GIGO) in the generative models, a preprocessing heuristic script called the **Smelter** filters this down to ~1,460 structurally sound sprites.
* **Filtering Criteria:** It rejects near-monochrome sprites (low RGB standard deviation), sprites with too few distinct colour buckets, and washed-out / low-saturation sprites.
* **Deduplication:** Exact duplicates are removed via MD5 hashing of raw pixel bytes.

---

## V. Optimization & Safety

### Prompt Engineering & Domain Adaptation
To force a modern Diffusion model (Flux.1) to render assets that look like retro SNES sprites, strict Domain Adaptation techniques are applied during prompt assembly:
* **Style Prefixing & Suffixing:** Every prompt is forcibly wrapped with strict stylistic constraints (`"pixel art sprite, 32x32 grid, RPG game item icon"` prefix; `"hard pixel edges, no anti-aliasing, limited 16-color palette, retro SNES 16-bit style"` suffix).
* **Deterministic Blueprint Assembly:** Rather than trusting the LLM to free-write a faithful prompt, the Forger renders the Master Smith's structured `parts` list into the diffusion prompt in code — one localized clause per part (material + colour + surface detail) — so every per-part material choice is guaranteed to reach Flux.1.
* **Hallucination Control:** Moondream2 is asked for *appearance only* and explicitly told **not** to name the item or list parts — identity comes from the SigLIP classifier instead. This removes the two failure modes we observed: mislabelled items and parroted part lists driving a wrong archetype.

### Key Parameters & Reproducibility
* **Random Seeds per Forge:** The Pollinations Flux.1 API is called with a freshly sampled random `seed` each run, allowing infinite variation of the same item fusion. Seeds are logged to the console for reproducibility.
* **VRAM Management:** The two local vision models (SigLIP SO400M, then Moondream2) are loaded lazily and unloaded **in sequence** with `torch.cuda.empty_cache()` during Stage 1. This keeps peak VRAM ~3.5 GB — within the 6 GB budget of a GTX 1660 Super — while Stages 2 and 3 use no GPU at all.
* **Output Post-Processing:** The generated image is crushed to a `32x32` grid via `NEAREST` resampling, then scaled back up and quantized to 16 colors using Median-Cut, ensuring hard pixel edges throughout.

---

## Running the Project

### Prerequisites
* Python 3.10+
* A CUDA-capable GPU is strongly recommended (the SigLIP + Moondream2 vision models run on CPU but are significantly slower)
* A [Kaggle API token](https://www.kaggle.com/settings) for dataset download
* A [Google AI Studio API key](https://aistudio.google.com/app/apikey) for the Gemini agent

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

### 3. Prepare the Data (The Smelter)
*(Requires a `kaggle.json` token in `~/.kaggle/` or `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables)*
```bash
python setup.py
```
This will download the Kaggle dataset, run the Smelter heuristic filters, and save a clean sprite array to `data/sprites_clean.npy`.

### 4. Launch the Crucible UI
```bash
python app.py
```
Open the Gradio UI in your browser, select two sprites using the sliders, and click **Forge**.

---

## Project Structure

```
Crucible/
├── app.py                        # Gradio UI entry point
├── setup.py                      # First-run data download & Smelter pipeline
├── requirements.txt
├── .env.example                  # Template for GEMINI_API_KEY
├── crucible/
│   ├── __init__.py
│   ├── agents.py                 # Google ADK agent definitions (Master Smith)
│   ├── autoencoder_appraiser.py  # Stage 1: SigLIP identity + Moondream2 appearance
│   ├── forge.py                  # Forge class — orchestrates the full pipeline
│   ├── schemas.py                # Pydantic schemas for structured agent output
│   └── smelter.py                # Dataset filtering & deduplication utility
├── data/
│   ├── pixel_art/                # Raw Kaggle dataset (downloaded by setup.py)
│   └── sprites_clean.npy         # Filtered sprite array (produced by setup.py)
└── notebooks/
    ├── crucible.ipynb            # Colab T4 end-to-end test harness (clones + runs the package)
    └── rpg_items.ipynb           # Exploratory data analysis notebook
```
