import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from crucible import Forge

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CLEAN_PATH = Path("data/sprites_clean.npy")

sprites_clean: np.ndarray = np.array([])
load_error: str = ""

if CLEAN_PATH.exists():
    try:
        sprites_clean = np.load(str(CLEAN_PATH))
        print(f"[App] Loaded {len(sprites_clean)} clean sprites.")
    except Exception as e:
        load_error = f"Failed to load sprite data: {e}"
        print(f"[App] ERROR: {load_error}")
else:
    load_error = (
        f"Sprite data not found at '{CLEAN_PATH}'.\n"
        "Run 'python setup.py' to download and prepare the dataset."
    )
    print(f"[App] ERROR: {load_error}")

N = max(len(sprites_clean) - 1, 1)

# ---------------------------------------------------------------------------
# Forge instance (lazy — only fails if API key is missing at button-click time)
# ---------------------------------------------------------------------------

forge_instance: Forge | None = None
forge_error: str = ""

try:
    forge_instance = Forge()
except EnvironmentError as e:
    forge_error = str(e)
    print(f"[App] {forge_error}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _format_blueprint(meta: dict) -> str:
    """Render the Master Smith's part-level material blueprint as a markdown table."""
    archetype = meta.get("archetype", "")
    structure_src = meta.get("structure_source", "")
    parts = meta.get("parts", []) or []

    header = (
        f"**Archetype:** {archetype} &nbsp;·&nbsp; "
        f"**Structure anchor:** Item {structure_src}\n\n"
    )
    if not parts:
        return header + "_No part blueprint returned._"

    rows = [
        "| Part | Material | Color | From | Detail |",
        "|------|----------|-------|------|--------|",
    ]
    for p in parts:
        rows.append(
            f"| {p.get('part', '')} | {p.get('material', '')} | "
            f"{p.get('color', '')} | {p.get('source', '')} | {p.get('detail', '')} |"
        )
    return header + "\n".join(rows)


def idx_to_pil(idx: int) -> Image.Image | None:
    """Return a 128x128 PIL preview for the sprite at the given index."""
    if len(sprites_clean) == 0:
        return None
    sprite = sprites_clean[int(idx)]
    return Image.fromarray(sprite[..., :3].astype("uint8")).resize(
        (128, 128), Image.NEAREST
    )


def run_forge(idx_a: int, idx_b: int):
    """
    Gradio callback for the Forge button.
    Runs the full three-stage RPG pipeline and returns outputs for all UI components.
    """
    if len(sprites_clean) == 0:
        error_msg = f"Cannot forge: {load_error}"
        return None, None, None, error_msg, ""

    if forge_instance is None:
        error_msg = f"Cannot forge: {forge_error}"
        return None, None, None, error_msg, ""

    sprite_a = sprites_clean[int(idx_a)]
    sprite_b = sprites_clean[int(idx_b)]

    try:
        preview_a, preview_b, forged, meta = forge_instance.quench(
            sprite_a, sprite_b
        )
        label = (
            f"## {meta['fused_name']}\n\n"
            f"{meta['reasoning']}"
        )
        plan_text = _format_blueprint(meta)
        return preview_a, forged, preview_b, label, plan_text

    except Exception as e:
        error_msg = f"Forge failed: {e}"
        print(f"[App] ERROR: {error_msg}")
        return None, None, None, error_msg, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="The Crucible",
        theme=gr.themes.Soft(),
        css="""
        #title { text-align: center; font-family: 'Georgia', serif; font-size: 2.5em; margin-bottom: 0px; }
        #subtitle { text-align: center; opacity: 0.8; font-style: italic; margin-bottom: 24px; }
        #status { font-size: 0.9em; text-align: center; padding: 8px; background: var(--background-fill-secondary); border-radius: 8px; margin: 0 auto 20px; max-width: 400px; border: 1px solid var(--border-color-primary); }
        #forge-btn { font-size: 1.2em; height: 50px; font-weight: bold; background: linear-gradient(135deg, #4a90e2, #50e3c2); border: none; color: white !important; }
        #forge-btn:hover { filter: brightness(1.1); }
        #fusion-label { padding: 20px; background: var(--background-fill-secondary); border-left: 4px solid #4a90e2; border-radius: 0 8px 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .gr-image { border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        """,
    ) as demo:

        # Header
        gr.Markdown("# The Crucible", elem_id="title")
        gr.Markdown(
            "Select two sprites, optionally add style notes, and forge them into something new.",
            elem_id="subtitle",
        )

        # Status banner
        if load_error:
            gr.Markdown(
                f"**Setup required:** {load_error}",
                elem_id="status",
            )
        else:
            gr.Markdown(
                f"{len(sprites_clean)} sprites loaded and ready.",
                elem_id="status",
            )

        # Sprite selectors
        with gr.Row():
            with gr.Column():
                idx_a = gr.Slider(
                    minimum=0,
                    maximum=N,
                    value=0,
                    step=1,
                    label="Item A — Index",
                )
                preview_a = gr.Image(
                    label="Item A Preview",
                    height=128,
                    width=128,
                    interactive=False,
                )

            with gr.Column():
                idx_b = gr.Slider(
                    minimum=0,
                    maximum=N,
                    value=min(10, N),
                    step=1,
                    label="Item B — Index",
                )
                preview_b = gr.Image(
                    label="Item B Preview",
                    height=128,
                    width=128,
                    interactive=False,
                )

        # Live preview updates
        idx_a.change(fn=idx_to_pil, inputs=idx_a, outputs=preview_a)
        idx_b.change(fn=idx_to_pil, inputs=idx_b, outputs=preview_b)

        # Forge button
        forge_btn = gr.Button(
            "Forge",
            variant="primary",
            elem_id="forge-btn",
        )

        gr.Markdown("---")

        # Output row
        with gr.Row():
            out_a = gr.Image(
                label="Item A",
                height=256,
                interactive=False,
            )
            out_forged = gr.Image(
                label="Forged Item",
                height=256,
                interactive=False,
            )
            out_b = gr.Image(
                label="Item B",
                height=256,
                interactive=False,
            )

        # Fusion label
        fusion_label = gr.Markdown(elem_id="fusion-label")

        # Composition plan (Stage 2 CoT output)
        composition_md = gr.Markdown(
            elem_id="composition-plan",
        )

        # Wire up Forge button
        forge_btn.click(
            fn=run_forge,
            inputs=[idx_a, idx_b],
            outputs=[out_a, out_forged, out_b, fusion_label, composition_md],
        )

        # Initialise previews on load
        demo.load(fn=idx_to_pil, inputs=idx_a, outputs=preview_a)
        demo.load(fn=idx_to_pil, inputs=idx_b, outputs=preview_b)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(debug=True)
