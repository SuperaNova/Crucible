# Crucible: Autonomous Sprite Synthesis Pipeline
*(Intelligent Systems / CS346 Mini-Project)*

Crucible is a Multi-Agent System (MAS) that uses advanced generative models to autonomously "fuse" two 2D pixel art RPG items into a completely new, technically coherent artifact. Instead of mathematically averaging pixels (which produces blurry artifacts), Crucible uses a **semantic fusion** approach: it uses AI to "see" the items, a second AI to "think" of a logical hybrid, and a third AI to "draw" the result.

## I. Project Objective
* **The Concept:** Crucible is designed as a prototype for an in-game crafting mechanic. In many RPGs, crafting systems are static—combining an Iron Ingot and a Stick always yields a generic Iron Sword. Crucible aims to show how generative AI can be embedded into a game engine to allow players to *actually* forge unique items on the fly, resulting in procedurally generated visual assets and lore that didn't exist in the game's original files.
* **The Solution:** A hybrid Multi-Agent System that combines **Vision Autoencoders**, **Large Language Models (LLMs)**, and **Latent Diffusion Models**. This pipeline intelligently understands the *meaning* of base assets, reasons out a conceptual fusion, and generates high-fidelity pixel art.

---

## II. Component Models (The Three Pillars)

To satisfy the requirements of combining modern AI architectures, Crucible is built upon three distinct neural methodologies:

### 1. The Vision Autoencoder (Moondream2)
* **Role:** The **Appraiser** Agent.
* **Architecture:** Vision-Encoder-Decoder (Vision-Language Model).
* **How it Works:**
  * The **Encoder** (SigLIP Vision Transformer) compresses the high-dimensional input image (the sprite) into a low-dimensional, dense latent embedding space.
  * The **Decoder** (Phi-based Causal Language Model) reconstructs that embedding—not back into pixels, but into a natural language representation (a text caption).
* **Why it was chosen:** We utilize this as a Semantic Autoencoder. Moondream2 (1.8B parameters) was chosen because its training distribution included vast amounts of 2D digital art and game UI, making it highly accurate at reading stylized RPG icons. It fits comfortably within 6 GB VRAM in `float16`, making it viable on consumer hardware (e.g., GTX 1660 Super).

### 2. The Reasoning Engine (Gemini 3.1 Flash Lite Preview)
* **Role:** The **Master Smith** Agent.
* **Architecture:** Large Language Model (LLM).
* **How it Works:** LLMs act as probabilistic engines trained to predict token sequences. In our system, Gemini acts as the "brain," performing zero-shot logical reasoning.
* **Function in Pipeline:** It does not "see" the images directly. It ingests the text output from the Appraiser, cross-references its massive internal knowledge of metallurgy and fantasy lore, and outputs structured JSON containing the new item's name, its lore, and a visual synthesis prompt. Orchestrated via the **Google Agent Development Kit (ADK)**.

### 3. The Generative Foundation (Flux.1 via Pollinations)
* **Role:** The **Forger** Agent.
* **Architecture:** Latent Diffusion Model (Transformer-backed).
* **How it Works:** Diffusion models are trained by adding Gaussian noise to an image (forward diffusion) and learning to predict and remove that noise (reverse diffusion/denoising).
* **Function in Pipeline:** It takes the text prompt from the Master Smith and runs the reverse diffusion process on a random noise tensor to "manifest" the fused sprite. Flux was selected due to its superior text-adherence compared to older Convolutional U-Net models.

---

## III. System Architecture: Multi-Agent Orchestration

The handoff between these disparate models is managed by the **Google Agent Development Kit (ADK)**, creating a seamless, stateful Sequential Pipeline:

1. **Stage 1 (Appraiser):** `Input Image -> SigLIP Encoder -> Latent Embedding -> Phi Decoder -> Text Caption`
2. **Stage 2 (Master Smith):** `Text Captions -> ADK Session State -> LLM Context Window -> Fused Prompt & Lore`
3. **Stage 3 (Forger):** `Fused Prompt -> Flux Diffusion Model -> Denoised Tensor -> Output Sprite`

This separation of concerns allows each model to specialize: Moondream2 for Vision, Gemini for Logic, and Flux for Art.

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
To force a modern Diffusion model (Flux) to render assets that look like retro SNES sprites, strict Domain Adaptation techniques are applied via Prompt Engineering:
* **Style Prefixing & Suffixing:** Every prompt generated by the LLM is forcibly wrapped with strict stylistic constraints (e.g., `"pixel art sprite, 32x32 grid, RPG game item icon"`, `"hard pixel edges, no anti-aliasing, limited 16-color palette, retro SNES 16-bit style"`).
* **Hallucination Control:** The Moondream2 VLM is prompted with specific negative instructions to ensure the Master Smith receives clean, objective physical descriptions rather than brand names or game-specific lore.

### Key Parameters & Reproducibility
* **Deterministic Forging:** The Pollinations Flux API is called with an explicit integer `seed`, enabling strict reproducibility when the same seed is reused or infinite variation by changing it.
* **Quantization & Local Execution:** The Moondream2 Appraiser is loaded in `float16` precision to ensure it fits entirely within 6 GB VRAM on consumer-grade hardware without sacrificing accuracy.
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
