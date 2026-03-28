# SceneRx Platform - TODO

## Current Status

Platform pipeline is operational end-to-end:
```
Raw Photo → Vision API (segmentation) → Mask Images (persisted) → Calculator → Metric Value → Aggregation → Zone Analysis → Design Strategy
```

Frontend: React + TypeScript + Chakra UI + TanStack Query + Zustand
Backend: FastAPI + Pydantic + SQLite persistence
External: Vision API (segmentation), LLM providers (Gemini/OpenAI/Anthropic/DeepSeek)

---

## Open Items

### [ ] 1. Celery Tasks — Verify Runtime
Task files exist (`vision_tasks.py`, `metrics_tasks.py`, `analysis_tasks.py`) but need runtime verification with Redis/Celery.

### [ ] 2. In-Memory → Database Persistence
User/auth data is still in-memory (lost on restart). Project data uses SQLite.
Consider migrating auth to SQLite or PostgreSQL.

### [ ] 3. Four-Layer Pipeline Polish
Calculators and shared_layer support 4 layers (full, foreground, middleground, background).
Pipeline processes all layers. Verify edge cases and UI display.

---

## Done

- [x] Vision → Mask Persistence (`mask_filepaths` on `UploadedImage`, `_save_masks_to_project()`)
- [x] Pipeline uses `semantic_map` mask instead of raw image path
- [x] Vision processing step in project pipeline
- [x] Vision analysis results persisted to project
- [x] Reports page = pipeline summary report (overview, indicators, diagnostics, strategies, MD/JSON export)
- [x] Auth enforcement wired up (all mutating endpoints use `get_current_user`; controlled by `AUTH_ENABLED` setting)
- [x] Knowledge base filenames configurable via env (`KB_EVIDENCE_FILE`, etc.) with warning on missing files
- [x] MetricsManager: AST-based INDICATOR dict parsing with regex fallback
- [x] ZoneAnalyzer: all-NaN column guard before StandardScaler
- [x] Multi-model LLM support (Gemini, OpenAI, Anthropic, DeepSeek) with runtime switching
- [x] Pipeline state persists across navigation (Zustand store)
- [x] Indicators pre-fills from currentProject
- [x] Zone Assignment UI in ProjectDetail.tsx (batch select + assign + unassign)
- [x] Calculator scripts present in `data/metrics_code/` (35 calculators)
- [x] Semantic configuration present in `data/Semantic_configuration.json`
- [x] MetricsCalculator loads semantic_colors and injects into calculator modules
- [x] Indicators → Analysis auto-sync (Analysis.tsx reads from store on mount)
- [x] Reports page uses store indicators (auto-select + recommended sort with star)
- [x] ProjectWizard batch zone assignment (PUT /batch-zone endpoint)
- [x] Route ordering fixes (batch-zone, evidence/dimension)
- [x] Pydantic validation: `List[dict]` → `List[ZoneAssignment]` in batch-zone endpoint
- [x] Various frontend fixes (stale closures, useEffect guards, null zone_id)
- [x] Static file serving: mounted `/api/uploads` in main.py
