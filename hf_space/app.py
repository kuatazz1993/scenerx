"""SceneRx — HuggingFace Space demo.

Wraps the AI_City_View vision pipeline in a Gradio interface.
Pre-deploy checklist (see README.md in this directory):
  1. Copy `AI_City_View/pipeline/` here as `./pipeline/`.
  2. Copy `AI_City_View/Semantic_configuration.json` here.
  3. Push to your HF Space.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import gradio as gr
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from pipeline.stage2_ai_inference import stage2_ai_inference  # noqa: E402
from pipeline.stage3_postprocess import stage3_postprocess  # noqa: E402
from pipeline.stage4_intelligent_fmb import (  # noqa: E402
    stage4_intelligent_fmb,
    stage4_metric_fmb,
)
from pipeline.stage5_openness import stage5_openness  # noqa: E402
from pipeline.stage6_generate_images import stage6_generate_images  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AVAILABLE_DEPTH_MODELS = [
    ("DA3 Metric Large (0.35B, 8GB VRAM)", "depth-anything/DA3METRIC-LARGE"),
    ("DA3 Nested Giant-Large 1.1 (1.4B, 16GB+)", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
]
DEFAULT_DEPTH_MODEL = os.environ.get(
    "VISION_DEPTH_MODEL", "depth-anything/DA3METRIC-LARGE"
)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _load_semantic_config() -> list[dict]:
    path = _HERE / "Semantic_configuration.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if "color" in item and "bgr" not in item:
            item["bgr"] = _hex_to_bgr(item["color"])
    return items


SEMANTIC_ITEMS = _load_semantic_config()


def _build_config(depth_model: str) -> Dict[str, Any]:
    return {
        "split_method": "percentile",
        "semantic_items": SEMANTIC_ITEMS,
        "enable_semantic": True,
        "semantic_backend": "oneformer_ade20k",
        "depth_backend": "v3",
        "depth_model_id_v3": depth_model,
        "depth_focal_length": 300,
        "depth_process_res": 672,
        "depth_invert_v3": False,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def analyze(image, depth_model: str):
    """Run the full pipeline on an uploaded image and return key visualizations."""
    if image is None:
        return None, None, None, None, None, "Please upload an image."

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    config = _build_config(depth_model)

    # Stage 2: AI inference (semantic + depth)
    stage2 = stage2_ai_inference(img_bgr, config)
    semantic_map = stage2["semantic_map"]
    depth_map = stage2["depth_map"]
    depth_metric = stage2.get("depth_metric")
    sky_mask = stage2.get("sky_mask")

    # Stage 3: post-process semantic
    stage3 = stage3_postprocess(semantic_map, config)
    semantic_processed = stage3["semantic_map_processed"]

    # Stage 4: FMB layering (prefer metric depth when available)
    if depth_metric is not None:
        stage4 = stage4_metric_fmb(
            depth_metric, config,
            semantic_map=semantic_map,
            sky_mask=sky_mask,
        )
    else:
        stage4 = stage4_intelligent_fmb(depth_map, config, semantic_map=semantic_map)

    fg = stage4["foreground_mask"]
    mg = stage4["middleground_mask"]
    bg = stage4["background_mask"]

    # Stage 5: openness
    stage5 = stage5_openness(semantic_processed, config)
    openness_map = stage5["openness_map"]

    # Stage 6: render visualizations
    stage6 = stage6_generate_images(
        img_bgr, semantic_processed, depth_map, openness_map,
        fg, mg, bg, config,
        depth_metric=depth_metric,
    )
    images = stage6["images"]

    def _bgr_to_rgb(name: str):
        img = images.get(name)
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    layer_stats = stage4.get("layer_stats", {})
    summary_lines = [
        f"**Depth model**: `{depth_model}`",
        f"**Image size**: {img_bgr.shape[1]}×{img_bgr.shape[0]}",
        "",
        "**FMB layer coverage**:",
    ]
    for layer in ("foreground", "middleground", "background"):
        s = layer_stats.get(layer, {})
        pct = s.get("percentage")
        if pct is not None:
            summary_lines.append(f"- {layer}: {pct:.1f}%")
    if "depth_thresholds" in stage4:
        t = stage4["depth_thresholds"]
        summary_lines.append("")
        summary_lines.append(f"**FMB thresholds**: {t}")

    return (
        _bgr_to_rgb("semantic_map"),
        _bgr_to_rgb("depth_map"),
        _bgr_to_rgb("openness_map"),
        _bgr_to_rgb("fmb_map"),
        "\n".join(summary_lines),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SceneRx Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SceneRx — Scene Analysis Demo

        Upload a street-level or park photo. The pipeline runs:

        1. **OneFormer** semantic segmentation (ADE20K, 150 classes)
        2. **Depth Anything V3** monocular depth estimation
        3. **FMB layering** — foreground / middleground / background masks
        4. **Openness map** — per-pixel openness score from semantic classes

        > Companion demo to the SceneRx paper. For the full multi-stage pipeline
        > (zone analysis, indicator computation, design strategies), see the
        > [main repository](https://github.com/ZhenGtai123/scenerx).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input scene", type="pil")
            depth_model_dd = gr.Dropdown(
                choices=AVAILABLE_DEPTH_MODELS,
                value=DEFAULT_DEPTH_MODEL,
                label="Depth model",
            )
            run_btn = gr.Button("Analyze", variant="primary")
            summary = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Row():
                out_semantic = gr.Image(label="Semantic map (ADE20K palette)")
                out_depth = gr.Image(label="Depth map")
            with gr.Row():
                out_openness = gr.Image(label="Openness map")
                out_fmb = gr.Image(label="FMB layers (R=fore, G=mid, B=back)")

    run_btn.click(
        analyze,
        inputs=[input_image, depth_model_dd],
        outputs=[out_semantic, out_depth, out_openness, out_fmb, summary],
    )

    gr.Markdown(
        """
        ---
        **Citation**: see the [paper / project repo](https://github.com/ZhenGtai123/scenerx).
        Models: [OneFormer (CVPR 2023)](https://arxiv.org/abs/2211.06220),
        [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3).
        """
    )


if __name__ == "__main__":
    demo.launch()
