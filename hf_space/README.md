---
title: SceneRx Demo
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered urban scene analysis (segmentation + depth + FMB)
---

# SceneRx — HuggingFace Space

Lightweight Gradio demo of the SceneRx scene-analysis pipeline:

- **OneFormer** (ADE20K) — 150-class semantic segmentation
- **Depth Anything V3** — monocular depth (metric or canonical)
- **FMB layering** — foreground / middleground / background masks
- **Openness map** — per-pixel openness from semantic classes

Companion to the SceneRx paper. For the full platform (zone diagnostics, indicator
computation, LLM-based design strategy generation), see the
[main repository](https://github.com/ZhenGtai123/scenerx).

---

## Hardware

| Tier | Model | Speed |
|---|---|---|
| **CPU Basic** (free) | DA3METRIC-LARGE | ~30–60 s / image (slow but works) |
| **T4 small / ZeroGPU** | DA3METRIC-LARGE | ~2–5 s / image |
| **A10G / A100** | DA3NESTED-GIANT-LARGE-1.1 | <2 s / image, native metric depth |

Set `VISION_DEPTH_MODEL` in the Space's *Settings → Variables and secrets*
to override the default model.

---

## Deploying this Space

The Gradio app in `app.py` imports the AI_City_View pipeline modules. Two ways
to assemble the Space:

### Option A — copy pipeline files (simplest)

```bash
# 1. Create a new HF Space (Gradio SDK) and clone it locally
git clone https://huggingface.co/spaces/<your-username>/scenerx-demo
cd scenerx-demo

# 2. Copy this directory's files
cp -r /path/to/scenerx/hf_space/. .

# 3. Copy required pipeline files from AI_City_View
cp -r /path/to/AI_City_View/pipeline ./pipeline
cp /path/to/AI_City_View/Semantic_configuration.json .

# 4. Push to HF
git add .
git commit -m "Initial SceneRx demo"
git push
```

### Option B — git submodule

```bash
git submodule add https://github.com/ZhenGtai123/AI_City_View.git ai_city_view
ln -s ai_city_view/pipeline pipeline
ln -s ai_city_view/Semantic_configuration.json Semantic_configuration.json
```

> Symlinks may not survive HF's git infrastructure; Option A is safer.

---

## Local testing

```bash
cd hf_space
pip install -r requirements.txt
# Copy pipeline + semantic config into this dir first (see Option A above)
python app.py
# Opens http://127.0.0.1:7860
```

---

## File layout (after assembly)

```
hf_space/
├── app.py                          # Gradio entrypoint
├── requirements.txt
├── README.md                       # this file (with HF frontmatter)
├── pipeline/                       # ← copied from AI_City_View
│   ├── stage2_ai_inference.py
│   ├── stage3_postprocess.py
│   ├── stage4_intelligent_fmb.py
│   ├── stage5_openness.py
│   ├── stage6_generate_images.py
│   └── ...
└── Semantic_configuration.json     # ← copied from AI_City_View
```

---

## License

MIT — see the [main repository](https://github.com/ZhenGtai123/scenerx) for details.
