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

### Stage 1 — Recaptioning: The Vision Appraiser (Moondream2)
* **RPG Role:** Recaptioning — the MLLM reads both source sprites and produces rich semantic descriptions, replacing the original paper's "complex prompt → sub-prompt decomposition" with "sprite → appraisal" for each input.
* **Architecture:** Vision-Language Model (Encoder-Decoder).
* **How it Works:**
  * The **Encoder** (SigLIP Vision Transformer) compresses the input sprite into a dense latent visual embedding.
  * The **Decoder** (Phi-based Causal Language Model) decodes that embedding into natural language conditioned on a VQA prompt. The Appraiser runs **two directed queries per sprite** — one for identity + material + dominant colours, one for its visible parts — giving the Master Smith the part-level grounding it needs for material assignment.
* **Why it was chosen:** Moondream2 (1.8B parameters) was chosen because its training distribution included 2D digital art and game assets, making it highly accurate at reading stylized RPG icons. It fits within 6 GB VRAM in `float16`. After Stage 1 completes, the model is **explicitly unloaded** from GPU to free VRAM for subsequent stages.

### Stage 2 — CoT Planning: The Master Smith (Gemini Flash Lite via Google ADK)
* **RPG Role:** Multimodal Chain-of-Thought Planning — the MLLM acts as a global planner. In the original paper the LLM plans spatial bounding-box regions, each with a dense sub-prompt; here it plans a *part-level material blueprint* across the two sprites.
* **Architecture:** Large Language Model (LLM).
* **How it Works:** Given the two Moondream2 appraisals, the Master Smith performs explicit four-step CoT reasoning:
  1. **Archetype & skeleton** — picks the fused item's base form (`archetype`) and enumerates its canonical physical parts (e.g. a sword → blade, crossguard, grip, pommel).
  2. **Part-by-part material assignment** — for *every* part emits a concrete `material`, `color`, `source` (A / B / fused), and a localized surface `detail`. This is the core of semantic material compositing — real choices like "blade=gold from B, grip=wood from A" rather than a muddy average.
  3. **Structure source** — designates which sprite provides the primary visual skeleton (preserved for future ControlNet conditioning).
  4. **Name & lore** — writes `fused_name` and `reasoning`.
* **Structured output, not a prompt:** The Smith emits a structured `parts` list (validated by the `SmithingResult` schema) — it does **not** hand-write the diffusion prompt. The Forger assembles the prompt deterministically from the blueprint so every per-part material detail is guaranteed to reach the model.
* **Why ADK:** The multi-step structured output and session-state handoff to the Forger justify ADK's orchestration layer here; the planning output is richer than a simple single-call LLM interaction.

### Stage 3 — Generation: The Forger (Flux.1 via Pollinations)
* **RPG Role:** Generation — the diffusion model manifests the fused sprite from the planner's output prompt. In the original paper this is Complementary Regional Diffusion; here it is a single high-quality Flux.1 call conditioned on the Master Smith's structured prompt.
* **Architecture:** Latent Diffusion Model (Transformer-backed).
* **How it Works:** The Forger first assembles the Master Smith's part blueprint into a single diffusion prompt deterministically — each part becomes a localized clause (`"blade made of polished gold (bright yellow), glowing runes along the edge"`) wrapped in the pixel-art style constraints — then Flux.1 runs reverse diffusion guided by that prompt. The `structure_source` field is preserved in metadata for future ControlNet-Canny conditioning (e.g. via Replicate `flux-canny-dev`) without requiring code refactoring.
* **Why Pollinations/Flux.1:** Flux.1 provides state-of-the-art text adherence and image quality without requiring local GPU inference, which is critical given the 6 GB VRAM constraint already consumed by Moondream2.

---

## III. System Architecture: Multi-Agent Orchestration

The handoff between these disparate models is managed by the **Google Agent Development Kit (ADK)**, creating a stateful Sequential Pipeline that mirrors the three stages of the RPG framework:

1. **Stage 1 — Recaptioning (Appraiser):** `Input Sprites → SigLIP Encoder → Latent Embedding → Phi Decoder → Per-Item Appraisal (description + parts + tags)`
2. **Stage 2 — CoT Planning (Master Smith):** `Appraisal → ADK Session State → LLM CoT Reasoning → Part-Level Material Blueprint + Archetype + Structure Source`
3. **Stage 3 — Generation (Forger):** `Blueprint → Deterministic Prompt Assembly → Flux.1 Diffusion → Pixel Art Post-Processing → Output Sprite`

The Master Smith's `parts` blueprint and `structure_source` outputs are the Crucible-specific adaptation of RPG's rationale + region-planning stage: the `parts` list maps RPG's per-region detail onto per-part material recipes. `structure_source` designates which sprite's silhouette provides the dominant visual skeleton — surfaced today in the UI, and reserved as the ControlNet structural reference for future image-conditioned generation.

---

## IV. Data Methodology & Representation

### Sprite Representation
Sprites are represented as `32x32` or `64x64` RGBA tensors. Before generation is finalized, the system applies hard-edge quantization and `NEAREST` neighbor downsampling to ensure the output snaps to a strict pixel grid rather than outputting blurry digital paintings.

### Dataset Filtering (The Smelter)
We rely on the ["Pixel Art" dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) from Kaggle (~1,400 sprites). To prevent "garbage in, garbage out" (GIGO) in the generative models, a preprocessing heuristic script called the **Smelter** was built.
* **Filtering Criteria:** It automatically purges sprites that are too small, have messy alpha channel transparency, or lack sufficient color variance. Only structurally sound sprites are allowed into the Appraiser's context window.
* **Deduplication:** Exact duplicates are removed via MD5 hashing of raw pixel bytes.

---

## V. Optimization & Safety

### Prompt Engineering & Domain Adaptation
To force a modern Diffusion model (Flux.1) to render assets that look like retro SNES sprites, strict Domain Adaptation techniques are applied during prompt assembly:
* **Style Prefixing & Suffixing:** Every prompt is forcibly wrapped with strict stylistic constraints (`"pixel art sprite, 32x32 grid, RPG game item icon"` prefix; `"hard pixel edges, no anti-aliasing, limited 16-color palette, retro SNES 16-bit style"` suffix).
* **Deterministic Blueprint Assembly:** Rather than trusting the LLM to free-write a faithful prompt, the Forger renders the Master Smith's structured `parts` list into the diffusion prompt in code — one localized clause per part (material + colour + surface detail) — so every per-part material choice is guaranteed to reach Flux.1.
* **Hallucination Control:** The Moondream2 VLM is prompted with specific negative instructions to ensure the Master Smith receives clean, objective physical descriptions rather than brand names or game-specific lore.

### Key Parameters & Reproducibility
* **Random Seeds per Forge:** The Pollinations Flux.1 API is called with a freshly sampled random `seed` each run, allowing infinite variation of the same item fusion. Seeds are logged to the console for reproducibility.
* **VRAM Management:** Moondream2 is loaded lazily at the start of each run and unloaded (with `torch.cuda.empty_cache()`) immediately after Stage 1. This keeps peak VRAM within the 6 GB budget of a GTX 1660 Super while Stages 2 and 3 use no GPU at all.
* **Output Post-Processing:** The generated image is crushed to a `32x32` grid via `NEAREST` resampling, then scaled back up and quantized to 16 colors using Median-Cut, ensuring hard pixel edges throughout.

---

## Running the Project

### Prerequisites
* Python 3.10+
* A CUDA-capable GPU is strongly recommended (Moondream2 runs on CPU but is significantly slower)
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
│   ├── autoencoder_appraiser.py  # Stage 1: Moondream2 Vision-Language Model
│   ├── forge.py                  # Forge class — orchestrates the full pipeline
│   ├── schemas.py                # Pydantic schemas for structured agent output
│   └── smelter.py                # Dataset filtering & deduplication utility
├── data/
│   ├── pixel_art/                # Raw Kaggle dataset (downloaded by setup.py)
│   └── sprites_clean.npy         # Filtered sprite array (produced by setup.py)
└── notebooks/
    └── rpg_items.ipynb           # Exploratory data analysis notebook
```
