# The Crucible — A Hybrid VAE-LLM Multi-Agent Pipeline for RPG Sprite Synthesis

An autonomous intelligence pipeline combining a **Vision-Encoder-Decoder (BLIP)** for local sprite appraisal with a **Google ADK-driven** LLM for creative synthesis. The BLIP model encodes each sprite into a latent visual representation and decodes it into a natural language caption, which is then handed off to the Master Smith LLM agent via shared session-state memory. The resulting high-entropy prompt is decoded into a 16-bit artifact using a CLIP-aligned generative backend and a custom pixel-lattice post-processing layer.

Built with **BLIP** (Salesforce Research, ICML 2022) and the **Google Agent Development Kit (ADK)**.

---

## Architecture

The system uses a Multi-Agent System (MAS) architecture where agents are autonomous units that communicate via a shared session state.

```
[sprite_a]  [sprite_b]
     \            /
      v          v
  Agent 1 — The Appraiser
  (BLIP Vision-Encoder-Decoder, Salesforce/blip-image-captioning-base)
  Encodes both sprites into latent visual features,
  decodes them into natural language captions.
  Writes result to state['appraisal'] (no API call — runs locally).
            |
            v
  Agent 2 — The Master Smith
  (LlmAgent, gemini-3.1-flash-lite-preview, text only)
  Reads {appraisal} captions from state, designs fusion,
  and writes to state['smithing'].
            |
            v
  Agent 3 — The Forger
  (Deterministic pipeline)
  Pollinations Flux -> pixelate -> quantize
  Output: PIL.Image (512x512, 16-color pixel art)
```

### Why BLIP + ADK?
The Appraiser is now a **Vision-Encoder-Decoder** model — the same architectural family as Autoencoders and VAEs — running fully locally without API costs. BLIP encodes each sprite's visual content into a latent representation and decodes it into a descriptive caption. This caption is injected into the ADK session state, allowing the Master Smith LLM to reason about the sprites without ever seeing the raw pixels. The Google ADK `SequentialAgent` orchestrator then manages the remaining flow.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
```

Get a key at: https://aistudio.google.com/app/apikey

### 3. Download and prepare the dataset

You need Kaggle credentials. Either:

- Place `kaggle.json` at `~/.kaggle/kaggle.json`
- Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

Download your token at: https://www.kaggle.com/settings -> API -> Create New Token

Then run:

```bash
python setup.py
```

This downloads the `ebrahimelgazar/pixel-art` dataset, runs the `Smelter` filters, and saves `data/sprites_clean.npy`.

### 4. Launch the app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## File Overview

| Path | Role |
|---|---|
| `app.py` | Gradio Blocks UI (entry point) |
| `setup.py` | One-time dataset download and preparation |
| `crucible/forge.py` | Orchestrator class and Forger pipeline |
| `crucible/agents.py` | ADK Agent definitions (Master Smith) |
| `crucible/autoencoder_appraiser.py` | BLIP Vision-Encoder-Decoder Appraiser |
| `crucible/schemas.py` | Pydantic schemas for agent communication |
| `crucible/smelter.py` | Data cleaning heuristics |
| `notebooks/` | Archive of original research/colab versions |

---

## Mathematical Overview

### Why not latent space interpolation?

A natural approach to sprite fusion is to train a Variational Autoencoder (VAE) on the sprite dataset, encode both sprites into latent vectors **z_a** and **z_b**, and decode an interpolated point:

```
z_fused = (1 - alpha) * z_a + alpha * z_b,  alpha in [0, 1]
```

Because a well-trained VAE enforces a smooth posterior `q(z|x) ~ N(mu, sigma^2 I)` via the KL divergence term in the ELBO loss:

```
L = E[log p(x|z)] - KL( q(z|x) || p(z) )
```

Points between **z_a** and **z_b** on the learned manifold are more likely to decode into coherent images than random interpolations in pixel space.

### Why this project uses BLIP for appraisal instead of a pixel-space VAE

The `sprites.npy` dataset has approximately 7,000 items after cleaning — too few to train a VAE whose latent manifold generalizes reliably across item categories. At this scale, the reconstruction loss dominates and the KL term collapses, producing a posterior that memorizes rather than generalizes.

Instead, we use **BLIP** (*Bootstrapping Language-Image Pre-training*, Li et al., ICML 2022) as the Appraiser. BLIP is a **Vision-Encoder-Decoder** model — architecturally equivalent to an image-conditioned Autoencoder — where:

- The **Encoder** (Vision Transformer, ViT-B/16) maps the input sprite to a dense latent representation `h = Encoder(x)`.
- The **Decoder** (BERT-based language model) reconstructs the image's semantic content as natural language: `caption = Decoder(h)`.

This gives us the core Encoder-Decoder principle but leverages a model pre-trained on 129 million image-text pairs, making it robust even to our small sprite dataset. The output caption is then passed to the Master Smith LLM, which acts as a learned generative prior over RPG item semantics — effectively replacing the VAE decoder with a much larger, text-conditioned generative model (Pollinations Flux).

---

## Post-Processing Pipeline

The Forger applies two operations to convert Pollinations output into pixel art:

1. **Pixelation** — crush to 32x32 with `Image.NEAREST`, then scale back to 512x512 with `Image.NEAREST`. This forces a hard block grid regardless of the source image content.

2. **Palette quantization** — `Image.quantize(colors=16, method=MEDIANCUT)`. Applied *after* pixelation so the palette is snapped to already-hard pixel edges. Quantizing at full resolution before pixelation would preserve gradient blending that the subsequent downscale then averages into muddy intermediate colors.
