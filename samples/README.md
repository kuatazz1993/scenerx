# Sample inputs and reproduction targets

This directory exists so a paper reader can verify SceneRx end-to-end without
sourcing their own imagery first. Layout:

```
samples/
├── inputs/                  # input photos (commit only with permissive licence)
├── expected_outputs/        # reference outputs the pipeline should match
│   ├── README.md            # what each file represents + tolerance
│   └── ...
└── README.md                # this file
```

## Reproduction recipe

```bash
# 1. Bring up the stack with vision-api enabled
make reproduce

# 2. Open the UI and create a project from the bundled images
#    (or use the /api/projects CLI flow described in README §B)

# 3. Run the full pipeline. Outputs land in packages/backend/outputs/<project>/

# 4. Compare against expected_outputs/
diff -r packages/backend/outputs/<project>/ samples/expected_outputs/
```

Numerical tolerance for indicator values is documented in
[`expected_outputs/README.md`](./expected_outputs/README.md). Z-score
diagnostics and LLM-generated text are non-deterministic and excluded from
the strict diff.

## Input formats accepted by the pipeline

| Mode | Input | Notes |
|---|---|---|
| **Single view** | One PNG or JPG | Street-level / park scene; ≥ 1024×768 recommended |
| **Panorama** | Equirectangular 360° JPG/PNG | Auto-cropped into left/center/right by the vision pipeline |

## What does NOT belong here

- Imagery with restrictive licences. Only commit images you have the right to
  redistribute (CC-BY, CC-BY-SA, CC0, or your own with explicit permission).
- Personally identifiable imagery. Anonymise faces and licence plates first.
- Anything > 5 MB per file — use Git LFS or link to an external archive.

## Suggested CC-licensed sources for benchmarking

- **Mapillary** — https://www.mapillary.com/app *(CC-BY-SA)*
- **KartaView** — https://kartaview.org/map *(CC-BY-SA)*
- Your own field photography with a clear release
