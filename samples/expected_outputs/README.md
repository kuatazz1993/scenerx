# Expected outputs — paper reproduction targets

Reference outputs produced by SceneRx version `v__.__.__` on the inputs in
`../inputs/`. A reader who runs `make reproduce` should obtain numerically
matching results within the tolerance below.

## Layout

```
expected_outputs/
├── README.md                       # this file
├── manifest.json                   # commit hash + image versions used to generate
├── stage2_vision/
│   └── <image_id>/
│       ├── semantic_map.png        # OneFormer ADE20K segmentation
│       ├── depth_map.png           # canonical or metric depth
│       └── fmb_map.png             # foreground/middleground/background overlay
├── stage3_metrics/
│   └── <project_id>_metrics.json   # per-image indicator values
├── stage4_zone/
│   └── <project_id>_zone.json      # zone-level z-scores + descriptive stats
└── stage5_design/
    └── <project_id>_design.json    # LLM-generated design strategies (informational)
```

## Tolerance

| Output | Comparison | Tolerance |
|---|---|---|
| `semantic_map.png` | per-pixel class id | ≥ 99 % pixel agreement |
| `depth_map.png` (canonical) | normalised L1 | ≤ 0.02 mean error |
| FMB layer percentages | absolute | ≤ 0.5 percentage points |
| Indicator values | relative | ≤ 1 % per indicator |
| Zone z-scores | absolute | ≤ 0.05 |
| LLM design strategies | **excluded** | non-deterministic — informational only |

Differences exceeding tolerance usually mean a model checkpoint mismatch.
Check `manifest.json` against the `VISION_IMAGE_TAG` and `VISION_DEPTH_MODEL`
in your `.env`.

## Regenerating these files

```bash
# Used by maintainers to refresh the reference set after a release:
make reproduce
python scripts/dump_expected_outputs.py \
    --project <project_id> \
    --out samples/expected_outputs/
```

> The dump script is not yet committed — see issue tracker for status.
