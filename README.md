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
* **Architecture:** Vision-Encoder-Decoder.
* **How it Works:** 
  * The **Encoder** (SigLIP Vision Transformer) compresses the high-dimensional input image (the sprite) into a low-dimensional, dense latent embedding space.
  * The **Decoder** (Phi-1.5 Causal Language Model) reconstructs that embedding—not back into pixels, but into a natural language representation (a text caption).
* **Why it was chosen:** We utilize this as a Semantic Autoencoder. Moondream2 (1.8B parameters) was chosen because its training distribution included vast amounts of 2D digital art and game UI, making it highly accurate at reading stylized RPG icons.

### 2. The Reasoning Engine (Gemini 3.1 Flash Lite Preview)
* **Role:** The **Master Smith** Agent.
* **Architecture:** Large Language Model (LLM).
* **How it Works:** LLMs act as probabilistic engines trained to predict token sequences. In our system, Gemini acts as the "brain," performing zero-shot logical reasoning.
* **Function in Pipeline:** It does not "see" the images directly. It ingests the text output from the Autoencoder, cross-references its massive internal knowledge of metallurgy and fantasy lore, and outputs structured JSON containing the new item's name, its lore, and a visual synthesis prompt.

### 3. The Generative Foundation (Flux.1 via Pollinations)
* **Role:** The **Forger** Agent.
* **Architecture:** Latent Diffusion Model (Transformer-backed).
* **How it Works:** Diffusion models are trained by adding Gaussian noise to an image (forward diffusion) and learning to predict and remove that noise (reverse diffusion/denoising).
* **Function in Pipeline:** It takes the text prompt from the Master Smith and runs the reverse diffusion process on a random noise tensor to "manifest" the fused sprite. Flux was selected due to its superior text-adherence compared to older Convolutional U-Net models.

---

## III. System Architecture: Multi-Agent Orchestration

The handoff between these disparate models is managed by the **Google Agent Development Kit (ADK)**, creating a seamless, stateful Sequential Pipeline:

1. **Stage 1 (Appraiser):** `Input Image -> ViT Encoder -> Latent Embedding -> Phi Decoder -> Text Caption`
2. **Stage 2 (Master Smith):** `Text Captions -> ADK Session State -> LLM Context Window -> Fused Prompt & Lore`
3. **Stage 3 (Forger):** `Fused Prompt -> Flux Diffusion Model -> Denoised Tensor -> Output Sprite`

This separation of concerns allows each model to specialize: Moondream for Vision, Gemini for Logic, and Flux for Art.

---

## IV. Data Methodology & Representation

### Sprite Representation
Sprites are represented as `32x32` or `64x64` RGBA tensors. Before generation is finalized, the system applies hard-edge quantization and `NEAREST` neighbor downsampling to ensure the output snaps to a strict pixel grid rather than outputting blurry digital paintings.

### Dataset Filtering (The Smelter)
We rely on the "16-bit RPG Item Collection" dataset from Kaggle (~1,400 sprites). To prevent "garbage in, garbage out" (GIGO) in the generative models, a preprocessing heuristic script called the **Smelter** was built.
* **Filtering Criteria:** It automatically purges sprites that are too small, have messy alpha channel transparency, or lack sufficient color variance. Only structurally sound sprites are allowed into the Appraiser's context window.

---

## V. Optimization & Safety

### Prompt Engineering & Domain Adaptation
To force a modern 2025-era Diffusion model (Flux) to render assets that look like 1990-era SNES sprites, we utilized strict Domain Adaptation techniques via Prompt Engineering:
* **Style Prefixing:** Every prompt generated by the LLM is forcibly wrapped with strict stylistic constraints: `"pixel art, isolated on white background, 8-bit RPG icon"`.
* **Hallucination Control:** The Moondream2 VLM is prompted with specific negative instructions (`"Do not mention Minecraft, video games, or brand names."`) to ensure the Master Smith receives clean, objective physical descriptions rather than copyrighted game lore.

### Key Parameters & Reproducibility
* **Deterministic Forging:** The Pollinations Flux API is called with explicitly generated random integer `seeds`. This allows infinite variations of the same item fusion, or strict reproducibility if a seed is reused.
* **Quantization & Local Execution:** The Moondream2 Autoencoder is loaded in `float16` precision to ensure it fits entirely within the 6GB VRAM constraint of consumer-grade hardware (e.g., GTX 1660 Super) without sacrificing technical accuracy.

---

## Running the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Data (The Smelter):**
   *(Requires a `kaggle.json` token in `~/.kaggle/`)*
   ```bash
   python setup.py
   ```

3. **Launch the Crucible UI:**
   *(Requires `GOOGLE_API_KEY` in `.env`)*
   ```bash
   python app.py
   ```
