# The Crucible — A Multi-Agent ADK Orchestration for RPG Sprite Synthesis

An autonomous intelligence pipeline that replaces conventional sequential processing with a **Google ADK-driven** multi-agent system. A vision-augmented Appraiser performs zero-shot semantic extraction of sprite features, delegating creative reasoning to a Master Smith agent via shared session-state memory. The resulting high-entropy prompt is decoded into a 16-bit artifact using a CLIP-aligned generative backend and a custom pixel-lattice post-processing layer.

Built with the **Google Agent Development Kit (ADK)**.

---

## Architecture

The system uses a Multi-Agent System (MAS) architecture where agents are autonomous units that communicate via a shared session state.

```
[sprite_a]  [sprite_b]
     \            /
      v          v
  Agent 1 — The Appraiser
  (LlmAgent, gemini-2.5-flash, vision)
  Identifies both sprites and writes result to state['appraisal'].
            |
            v
  Agent 2 — The Master Smith
  (LlmAgent, gemini-2.5-flash, text only)
  Reads {appraisal} from state, designs fusion, and writes to state['smithing'].
            |
            v
  Agent 3 — The Forger
  (Deterministic pipeline)
  Pollinations Flux -> pixelate -> quantize
  Output: PIL.Image (512x512, 16-color pixel art)
```

### Why ADK?
By using the Google ADK, each agent is defined with its own instruction set and output schema (via Pydantic). The `SequentialAgent` orchestrator manages the flow, while the `Runner` handles session lifecycle and event streaming. This allows for clean separation between vision-based appraisal and text-based creative smithing.

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
| `crucible/agents.py` | ADK Agent definitions (Appraiser, Master Smith) |
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

### Why this project uses prompt-based fusion instead

The `sprites.npy` dataset has approximately 7,000 items after cleaning — too few to train a VAE whose latent manifold generalizes reliably across item categories. At this scale, the reconstruction loss dominates and the KL term collapses, producing a posterior that memorizes rather than generalizes. The resulting latent space is not smooth enough for meaningful interpolation.

The three-agent pipeline sidesteps this by treating the fusion as a language grounding problem: Gemini identifies semantic properties of each sprite (material, element, visual style) and the Master Smith recombines them in prompt space. Pollinations then acts as the generative decoder, leveraging a much larger prior trained on billions of images.

---

## Post-Processing Pipeline

The Forger applies two operations to convert Pollinations output into pixel art:

1. **Pixelation** — crush to 32x32 with `Image.NEAREST`, then scale back to 512x512 with `Image.NEAREST`. This forces a hard block grid regardless of the source image content.

2. **Palette quantization** — `Image.quantize(colors=16, method=MEDIANCUT)`. Applied *after* pixelation so the palette is snapped to already-hard pixel edges. Quantizing at full resolution before pixelation would preserve gradient blending that the subsequent downscale then averages into muddy intermediate colors.
