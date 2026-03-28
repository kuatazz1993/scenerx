# SceneRx Frontend

React + TypeScript frontend for the SceneRx urban greenspace analysis platform.

## Tech Stack

- **React 19** + TypeScript
- **Vite** — build & dev server
- **Chakra UI v2** — component library
- **TanStack Query v5** — server state management
- **Zustand** — client state management
- **Axios** — HTTP client
- **Lucide React** — icons

## Getting Started

```bash
# Install dependencies
npm install

# Start dev server (default: http://localhost:5173)
npm run dev

# Type check
npx tsc --noEmit

# Build for production
npm run build
```

Requires the backend running at `http://localhost:8080` (configured in `src/api/index.ts`).

## Project Structure

```
src/
├── api/            # Axios API client
├── components/     # Shared components (PageShell, PageHeader, EmptyState, etc.)
├── hooks/          # Custom hooks (useApi, useAppToast)
├── pages/          # Page components
│   ├── Dashboard.tsx
│   ├── Projects.tsx / ProjectDetail.tsx / ProjectWizard.tsx
│   ├── VisionAnalysis.tsx
│   ├── Indicators.tsx      # Stage 1: LLM indicator recommendations
│   ├── Calculators.tsx
│   ├── Analysis.tsx         # Stage 2.5 + 3: Zone analysis & design strategies
│   ├── Reports.tsx          # Pipeline summary report + export
│   └── Settings.tsx
├── store/          # Zustand store (useAppStore)
├── types/          # TypeScript type definitions
└── utils/          # Utilities (generateReport, etc.)
```

## Pipeline Flow

The platform follows a 4-step pipeline per project:

1. **Vision** — Upload images, run semantic segmentation, assign to zones
2. **Indicators** — Get LLM-powered indicator recommendations, select relevant ones
3. **Analysis** — Run metrics calculation, zone diagnostics (Stage 2.5), design strategies (Stage 3)
4. **Reports** — View pipeline summary, download Markdown report or export JSON

Pipeline results (recommendations, zone analysis, design strategies) are persisted in the Zustand store across page navigation and cleared when switching projects.

## State Management

The Zustand store (`src/store/useAppStore.ts`) holds:

- `currentProject` — active project context
- `recommendations` — LLM indicator recommendations (from Indicators page)
- `selectedIndicators` — user-selected indicator subset
- `zoneAnalysisResult` — Stage 2.5 zone diagnostics
- `designStrategyResult` — Stage 3 design strategies
- `pipelineResult` — full pipeline execution result
- `calculators`, `semanticClasses` — reference data
