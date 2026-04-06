import type {
  ZoneAnalysisResult,
  ProjectPipelineResult,
  ClusteringResponse,
  ClusteringResult,
  Project,
  ZoneDiagnostic,
  EnrichedZoneStat,
  UploadedImage,
  ImageRecord,
  GlobalIndicatorStats,
  DataQualityRow,
  IndicatorDefinitionInput,
} from '../../types';

/**
 * Unified data bundle passed to every chart descriptor's `isAvailable` and
 * `render` functions. Centralises all derived state so descriptors stay
 * declarative.
 */
export interface ChartContext {
  // Raw state
  zoneAnalysisResult: ZoneAnalysisResult | null;
  pipelineResult: ProjectPipelineResult | null;
  clusteringResult: ClusteringResponse | null;
  currentProject: Project | null;

  // Layer selector (for Statistics tab)
  selectedLayer: string;

  // Derived
  sortedDiagnostics: ZoneDiagnostic[];
  filteredStats: EnrichedZoneStat[]; // stats restricted to selectedLayer
  correlationData: {
    indicators: string[];
    corr: Record<string, Record<string, number>>;
    pval?: Record<string, Record<string, number>>;
  } | null;
  gpsImages: UploadedImage[];
  gpsIndicatorIds: string[];
  effectiveClustering: ClusteringResult | null;

  // v7.0
  imageRecords: ImageRecord[];
  globalIndicatorStats: GlobalIndicatorStats[];
  dataQuality: DataQualityRow[];
  indicatorDefs: Record<string, IndicatorDefinitionInput>;
  analysisMode: 'multi_zone' | 'single_zone';
}

export const LAYERS = ['full', 'foreground', 'middleground', 'background'];
export const LAYER_LABELS: Record<string, string> = {
  full: 'Full',
  foreground: 'FG',
  middleground: 'MG',
  background: 'BG',
};

interface BuildArgs {
  zoneAnalysisResult: ZoneAnalysisResult | null;
  pipelineResult: ProjectPipelineResult | null;
  clusteringResult: ClusteringResponse | null;
  currentProject: Project | null;
  selectedLayer: string;
}

/**
 * Compute all derived chart-context values once from raw state. Cheap enough
 * to recompute on every relevant state change.
 */
export function buildChartContext(args: BuildArgs): ChartContext {
  const { zoneAnalysisResult, pipelineResult, clusteringResult, currentProject, selectedLayer } = args;

  const sortedDiagnostics = zoneAnalysisResult
    ? [...zoneAnalysisResult.zone_diagnostics].sort((a, b) => b.mean_abs_z - a.mean_abs_z)
    : [];

  const filteredStats = zoneAnalysisResult
    ? zoneAnalysisResult.zone_statistics.filter(s => s.layer === selectedLayer)
    : [];

  const correlationData = (() => {
    if (!zoneAnalysisResult) return null;
    const corr = zoneAnalysisResult.correlation_by_layer?.[selectedLayer];
    const pval = zoneAnalysisResult.pvalue_by_layer?.[selectedLayer];
    if (!corr) return null;
    return { indicators: Object.keys(corr), corr, pval };
  })();

  const gpsImages = currentProject
    ? currentProject.uploaded_images.filter(
        img => img.has_gps && img.latitude != null && img.longitude != null,
      )
    : [];

  const gpsIndicatorIds = Array.from(
    new Set(
      gpsImages.flatMap(img =>
        Object.keys(img.metrics_results)
          .filter(k => img.metrics_results[k] != null)
          .map(k => k.split('__')[0]),
      ),
    ),
  ).sort();

  const effectiveClustering =
    clusteringResult?.clustering ?? zoneAnalysisResult?.clustering ?? null;

  // v7.0 data
  const imageRecords = zoneAnalysisResult?.image_records ?? [];
  const globalIndicatorStats = zoneAnalysisResult?.global_indicator_stats ?? [];
  const dataQuality = zoneAnalysisResult?.data_quality ?? [];
  const indicatorDefs = zoneAnalysisResult?.indicator_definitions ?? {};
  const analysisMode = zoneAnalysisResult?.analysis_mode ?? 'multi_zone';

  return {
    zoneAnalysisResult,
    pipelineResult,
    clusteringResult,
    currentProject,
    selectedLayer,
    sortedDiagnostics,
    filteredStats,
    correlationData,
    gpsImages,
    gpsIndicatorIds,
    effectiveClustering,
    imageRecords,
    globalIndicatorStats,
    dataQuality,
    indicatorDefs,
    analysisMode,
  };
}
