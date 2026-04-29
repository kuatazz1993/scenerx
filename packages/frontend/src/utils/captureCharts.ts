import type { ChartHostHandle } from '../components/analysisCharts/ChartHost';
import type { ChartDescriptor } from '../components/analysisCharts/registry';
import type { ChartContext } from '../components/analysisCharts/ChartContext';

export interface CapturedChart {
  chart_id: string;
  title: string;
  /** Plain-English caption from the LLM summary cache when available, else
   *  the static descriptor.description, else "". */
  caption: string;
  dataURL: string;
  widthPx: number;
  heightPx: number;
}

interface CaptureArgs {
  /** All registry entries on the analysis tab. */
  charts: ChartDescriptor[];
  /** Already-built chart context (for isAvailable / summaryPayload). */
  ctx: ChartContext;
  /** Map<chart_id, ChartHostHandle> populated via ref callbacks in Reports. */
  refs: Map<string, ChartHostHandle | null>;
  /** Override: which chart_ids to actually capture. Default = exportByDefault. */
  selectedIds?: Set<string>;
  /** Optional caption lookup, keyed by chart_id (e.g. from React Query cache). */
  captionFor?: (chartId: string) => string | null;
}

/**
 * Sequentially capture each requested chart's DOM node as a PNG. Charts that
 * are missing data, hidden, or whose ref isn't mounted yet are silently
 * skipped. Returns the captures in registry order so the report can interleave
 * them with markdown sections.
 */
export async function captureChartsForReport({
  charts,
  ctx,
  refs,
  selectedIds,
  captionFor,
}: CaptureArgs): Promise<CapturedChart[]> {
  const out: CapturedChart[] = [];

  const wantsId = (id: string): boolean => {
    if (selectedIds) return selectedIds.has(id);
    return charts.find((c) => c.id === id)?.exportByDefault === true;
  };

  for (const chart of charts) {
    if (!wantsId(chart.id)) continue;
    if (!chart.isAvailable(ctx)) continue;
    const handle = refs.get(chart.id);
    if (!handle) continue;
    try {
      const result = await handle.capturePNG();
      if (!result) continue;
      const caption =
        captionFor?.(chart.id) ?? chart.description ?? '';
      out.push({
        chart_id: chart.id,
        title: chart.title,
        caption,
        dataURL: result.dataURL,
        widthPx: result.widthPx,
        heightPx: result.heightPx,
      });
    } catch (err) {
      // Per-chart failure shouldn't abort the whole export.
      console.warn(`Chart capture failed for ${chart.id}:`, err);
    }
  }

  return out;
}

/**
 * Wait two animation frames so React has flushed any pending state updates
 * (forceMount, accordion toggle) and the browser has painted the result.
 */
export function waitForPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}
