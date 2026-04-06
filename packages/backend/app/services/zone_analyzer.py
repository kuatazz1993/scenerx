"""
Zone Analyzer Service  (Stage 2.5 — v7.0 Descriptive)
Stateless, pure numpy/pandas/scipy — no LLM, no I/O.

Consumes flat zone_statistics records + image-level records and returns:
  - Enriched records with Z-scores, percentiles (no priority/classification)
  - Zone diagnostics (mean_abs_z, indicator_status — no status/problems)
  - Correlation / p-value matrices by layer
  - v7.0: global indicator stats (Table M2), data quality (Table M4),
    mode detection, statistical tests (Shapiro-Wilk, Kruskal-Wallis, CV)
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler

from app.models.analysis import (
    ZoneAnalysisRequest,
    ZoneAnalysisResult,
    EnrichedZoneStat,
    ZoneDiagnostic,
    ComputationMetadata,
    ImageRecord,
    GlobalIndicatorStats,
    DataQualityRow,
)

logger = logging.getLogger(__name__)

LAYERS = ["full", "foreground", "middleground", "background"]


class ZoneAnalyzer:
    """Stateless Stage 2.5 analysis service (v6.0 descriptive)."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, request: ZoneAnalysisRequest) -> ZoneAnalysisResult:
        ind_defs = request.indicator_definitions

        # 1) Pivot flat records into per-layer DataFrames of mean values
        raw_by_layer, meta_by_layer = self._build_dataframes(request.zone_statistics)

        if not raw_by_layer:
            return ZoneAnalysisResult(
                computation_metadata=ComputationMetadata(
                    n_indicators=0, n_zones=0,
                    warnings=["No zone statistics were provided — nothing to analyze."],
                ),
            )

        # Derive zone/indicator lists from pivoted data
        sample_layer = next(iter(raw_by_layer.values()))
        zone_names: list[str] = list(sample_layer.index)
        ind_ids: list[str] = list(sample_layer.columns)

        warnings: list[str] = []
        if len(zone_names) < 2:
            warnings.append(
                f"Only {len(zone_names)} zone available — cross-zone z-scores "
                "are undefined (need ≥ 2 zones). All z-score based charts will "
                "show 0. Add more zones or switch to image-level analysis."
            )
        elif len(zone_names) == 2:
            warnings.append(
                "Only 2 zones — z-scores are always ±1 (mathematical property of "
                "standardising 2 values). Correlations between indicators are "
                "undefined with only 2 data points. Add a 3rd zone for "
                "meaningful cross-zone statistics."
            )

        # 2) Z-scores per layer
        zscore_by_layer: dict[str, pd.DataFrame] = {}
        for layer, df_raw in raw_by_layer.items():
            # Guard: drop columns that are entirely NaN (StandardScaler would crash)
            all_nan_cols = df_raw.columns[df_raw.isna().all()]
            if len(all_nan_cols) > 0:
                logger.warning(
                    "Layer '%s': %d indicators have no data and will be marked as z_score=None: %s",
                    layer, len(all_nan_cols), list(all_nan_cols),
                )
            df_valid = df_raw.drop(columns=all_nan_cols)
            if df_valid.empty or len(df_valid) < 2:
                # Cannot compute z-scores with <2 zones or zero valid columns
                logger.warning(
                    "Layer '%s': z-scores undefined (n_zones=%d, n_valid_indicators=%d) — all z-scores set to None",
                    layer, len(df_valid), df_valid.shape[1],
                )
                zscore_by_layer[layer] = pd.DataFrame(np.nan, index=df_raw.index, columns=df_raw.columns)
                continue
            # Identify zero-variance columns (all zones have identical values)
            # StandardScaler produces z=0 for these, which is mathematically correct
            # ("every zone at the mean") but still worth flagging.
            col_std = df_valid.std(ddof=0)
            zero_var_cols = col_std[col_std == 0].index.tolist()
            if zero_var_cols:
                logger.info(
                    "Layer '%s': %d indicators have zero variance across zones (z=0 for all zones): %s",
                    layer, len(zero_var_cols), zero_var_cols,
                )
                # Surface constant-value indicators to the UI — typically a
                # sign of an upstream calculator bug (e.g. a colour mismatch
                # between Vision API output and the calculator's TARGET_RGB
                # makes every image return the same value).
                if layer == "full":
                    const_vals = {
                        ind: float(df_valid[ind].iloc[0]) for ind in zero_var_cols
                    }
                    stuck_at_zero = [ind for ind, v in const_vals.items() if v == 0]
                    stuck_at_100 = [ind for ind, v in const_vals.items() if v == 100]
                    other_const = [
                        f"{ind}={v:.2f}" for ind, v in const_vals.items()
                        if v != 0 and v != 100
                    ]
                    if stuck_at_zero:
                        warnings.append(
                            f"{len(stuck_at_zero)} indicator(s) returned 0 for every zone "
                            f"on the full layer — likely a calculator/mask colour mismatch: "
                            f"{', '.join(stuck_at_zero[:8])}"
                            f"{'…' if len(stuck_at_zero) > 8 else ''}"
                        )
                    if stuck_at_100:
                        warnings.append(
                            f"{len(stuck_at_100)} indicator(s) returned 100 for every zone "
                            f"on the full layer — check the calculator's target classes and "
                            f"the semantic_map colour mapping: "
                            f"{', '.join(stuck_at_100[:8])}"
                            f"{'…' if len(stuck_at_100) > 8 else ''}"
                        )
                    if other_const:
                        warnings.append(
                            f"{len(other_const)} indicator(s) have zero variance across zones "
                            f"(constant value, z-score undefined): {', '.join(other_const[:8])}"
                            f"{'…' if len(other_const) > 8 else ''}"
                        )
            scaler = StandardScaler()
            filled = df_valid.fillna(df_valid.mean()).infer_objects(copy=False)
            scaled = scaler.fit_transform(filled)
            z_df = pd.DataFrame(scaled, index=df_valid.index, columns=df_valid.columns)
            # Re-add dropped all-NaN columns as NaN (z-score is undefined, not zero)
            for col in all_nan_cols:
                z_df[col] = np.nan
            # Restore original column order
            zscore_by_layer[layer] = z_df.reindex(columns=df_raw.columns)

        # 3) Percentiles per layer
        pct_by_layer: dict[str, pd.DataFrame] = {
            layer: df.rank(pct=True) * 100 for layer, df in raw_by_layer.items()
        }

        # 4) Enriched flat records (descriptive only — no priority/classification)
        enriched: list[EnrichedZoneStat] = []
        for rec in request.zone_statistics:
            layer = rec.layer
            if layer not in zscore_by_layer:
                continue
            z_df = zscore_by_layer[layer]
            p_df = pct_by_layer[layer]
            zone = rec.zone_name
            ind = rec.indicator_id
            if zone not in z_df.index or ind not in z_df.columns:
                continue
            z_raw = z_df.loc[zone, ind]
            z_val = None if pd.isna(z_raw) else round(float(z_raw), 4)
            pct_val = float(p_df.loc[zone, ind]) if not pd.isna(p_df.loc[zone, ind]) else None
            enriched.append(EnrichedZoneStat(
                zone_id=rec.zone_id,
                zone_name=rec.zone_name,
                indicator_id=rec.indicator_id,
                layer=rec.layer,
                unit=rec.unit,
                n_images=rec.n_images,
                mean=rec.mean,
                std=rec.std,
                min=rec.min,
                max=rec.max,
                area_sqm=rec.area_sqm,
                z_score=z_val,
                percentile=round(pct_val, 2) if pct_val is not None else None,
            ))

        # 5) Correlations per layer
        corr_out: dict[str, dict] = {}
        pval_out: dict[str, dict] = {}
        for layer, z_df in zscore_by_layer.items():
            corr_m, pval_m = _calc_corr_pval(z_df)
            corr_out[layer] = {
                c1: {c2: round(float(corr_m.loc[c1, c2]), 4) for c2 in corr_m.columns}
                for c1 in corr_m.index
            }
            pval_out[layer] = {
                c1: {c2: round(float(pval_m.loc[c1, c2]), 6) for c2 in pval_m.columns}
                for c1 in pval_m.index
            }

        # 6) Zone diagnostics (descriptive: mean_abs_z + indicator_status)
        diagnostics = self._build_diagnostics(
            zone_names, ind_ids, ind_defs,
            zscore_by_layer, raw_by_layer, meta_by_layer,
        )

        # 7) Layer-level statistics (mean/std/N across zones, per indicator)
        layer_statistics = self._compute_layer_statistics(raw_by_layer, meta_by_layer)

        # 8) Radar profiles from percentiles
        # `radar_profiles`           — full layer only (backward-compat)
        # `radar_profiles_by_layer`  — all 4 layers (matches notebook Fig 4)
        radar_profiles: dict[str, dict[str, float]] = {}
        if "full" in pct_by_layer:
            pct_full = pct_by_layer["full"]
            for zone in zone_names:
                if zone in pct_full.index:
                    radar_profiles[zone] = {
                        ind_id: round(float(pct_full.loc[zone, ind_id]), 2)
                        for ind_id in ind_ids
                        if not pd.isna(pct_full.loc[zone, ind_id])
                    }

        radar_profiles_by_layer: dict[str, dict[str, dict[str, float]]] = {}
        for layer, pct_df in pct_by_layer.items():
            radar_profiles_by_layer[layer] = {}
            for zone in zone_names:
                if zone in pct_df.index:
                    radar_profiles_by_layer[layer][zone] = {
                        ind_id: round(float(pct_df.loc[zone, ind_id]), 2)
                        for ind_id in ind_ids
                        if not pd.isna(pct_df.loc[zone, ind_id])
                    }

        # 9) v7.0: global indicator stats + data quality + mode detection
        image_records = request.image_records
        global_stats = self._compute_global_indicator_stats(
            image_records, ind_ids, ind_defs,
        )
        data_quality = self._compute_data_quality(
            image_records, ind_ids, global_stats,
        )
        analysis_mode = "multi_zone" if len(zone_names) > 1 else "single_zone"

        return ZoneAnalysisResult(
            zone_statistics=enriched,
            zone_diagnostics=diagnostics,
            correlation_by_layer=corr_out,
            pvalue_by_layer=pval_out,
            indicator_definitions=ind_defs,
            layer_statistics=layer_statistics,
            radar_profiles=radar_profiles,
            radar_profiles_by_layer=radar_profiles_by_layer,
            computation_metadata=ComputationMetadata(
                n_indicators=len(ind_ids),
                n_zones=len(zone_names),
                warnings=warnings,
            ),
            image_records=image_records,
            global_indicator_stats=global_stats,
            data_quality=data_quality,
            analysis_mode=analysis_mode,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataframes(
        records: list,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        """Pivot flat records into {layer: DataFrame(zones x indicators)} of mean values.

        Also returns a meta dict keyed by (zone_name, indicator_id, layer) for
        auxiliary fields (std, min, max, n_images, area_sqm, unit).
        """
        by_layer: dict[str, dict[str, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))
        meta: dict[str, dict] = {}

        for rec in records:
            layer = rec.layer
            if layer not in LAYERS:
                continue
            by_layer[layer][rec.zone_name][rec.indicator_id] = rec.mean
            meta[(rec.zone_name, rec.indicator_id, layer)] = {
                "std": rec.std,
                "min": rec.min,
                "max": rec.max,
                "n_images": rec.n_images,
                "area_sqm": rec.area_sqm,
                "unit": rec.unit,
                "zone_id": rec.zone_id,
            }

        result: dict[str, pd.DataFrame] = {}
        for layer, zone_dict in by_layer.items():
            df = pd.DataFrame(zone_dict).T
            df = df.sort_index()
            df = df.reindex(columns=sorted(df.columns))
            result[layer] = df

        return result, meta

    @staticmethod
    def _build_diagnostics(
        zone_names, ind_ids, ind_defs,
        zscore_by_layer, raw_by_layer, meta_by_layer,
    ) -> list[ZoneDiagnostic]:
        """Build purely descriptive zone diagnostics (v6.0)."""
        diagnostics: list[ZoneDiagnostic] = []

        for zone in zone_names:
            indicator_status: dict[str, dict[str, Any]] = {}

            # Build per-indicator status (descriptive only: value + z_score + target_direction)
            for ind_id in ind_ids:
                layer_data: dict[str, dict] = {}
                for layer in LAYERS:
                    if layer not in zscore_by_layer or zone not in zscore_by_layer[layer].index:
                        continue
                    z_raw = zscore_by_layer[layer].loc[zone, ind_id]
                    z_out = None if pd.isna(z_raw) else round(float(z_raw), 3)
                    val = raw_by_layer[layer].loc[zone, ind_id]
                    defn = ind_defs.get(ind_id)
                    layer_data[layer] = {
                        "value": round(float(val), 4) if not pd.isna(val) else None,
                        "z_score": z_out,
                        "target_direction": defn.target_direction if defn else "INCREASE",
                    }
                indicator_status[ind_id] = layer_data

            # mean_abs_z: mean of abs(z_score) across all indicators in full layer
            mean_abs_z = 0.0
            if "full" in zscore_by_layer and zone in zscore_by_layer["full"].index:
                full_z = zscore_by_layer["full"].loc[zone]
                abs_z = full_z.abs().dropna()
                if len(abs_z) > 0:
                    mean_abs_z = round(float(abs_z.mean()), 4)

            # Get zone_id and area_sqm from meta
            zone_id = zone
            area_sqm = 0.0
            point_count = 0
            for ind_id in ind_ids:
                for lyr in LAYERS:
                    m = meta_by_layer.get((zone, ind_id, lyr), {})
                    if m.get("zone_id"):
                        zone_id = m["zone_id"]
                    if m.get("area_sqm"):
                        area_sqm = m["area_sqm"]
                    if m.get("n_images") and m["n_images"] > point_count:
                        point_count = m["n_images"]
                    if zone_id != zone:
                        break
                if zone_id != zone:
                    break

            diagnostics.append(ZoneDiagnostic(
                zone_id=zone_id,
                zone_name=zone,
                area_sqm=area_sqm,
                mean_abs_z=mean_abs_z,
                point_count=point_count,
                indicator_status=indicator_status,
            ))

        # Sort by mean_abs_z descending (most distinctive first)
        diagnostics.sort(key=lambda d: d.mean_abs_z, reverse=True)
        for rank_idx, diag in enumerate(diagnostics, start=1):
            diag.rank = rank_idx
        return diagnostics

    @staticmethod
    def _compute_layer_statistics(
        raw_by_layer: dict[str, pd.DataFrame],
        meta_by_layer: dict,
    ) -> dict[str, dict]:
        """Compute per-indicator per-layer aggregate statistics."""
        result: dict[str, dict] = {}
        for layer, df in raw_by_layer.items():
            for ind_id in df.columns:
                col = df[ind_id].dropna()
                if ind_id not in result:
                    result[ind_id] = {}
                result[ind_id][layer] = {
                    "N": int(len(col)),
                    "Mean": round(float(col.mean()), 4) if len(col) > 0 else None,
                    "Std": round(float(col.std()), 4) if len(col) > 1 else None,
                    "Min": round(float(col.min()), 4) if len(col) > 0 else None,
                    "Max": round(float(col.max()), 4) if len(col) > 0 else None,
                }
        return result

    # ------------------------------------------------------------------
    # v7.0: global statistics + data quality
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_global_indicator_stats(
        image_records: list[ImageRecord],
        ind_ids: list[str],
        ind_defs: dict,
    ) -> list[GlobalIndicatorStats]:
        """Table M2 / S2: per-indicator global descriptive stats with tests."""
        if not image_records:
            return []

        # Group image values: (indicator_id, layer) → list[float]
        by_ind_layer: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rec in image_records:
            by_ind_layer[(rec.indicator_id, rec.layer)].append(rec.value)

        results: list[GlobalIndicatorStats] = []
        for ind_id in ind_ids:
            defn = ind_defs.get(ind_id)
            stat = GlobalIndicatorStats(
                indicator_id=ind_id,
                indicator_name=defn.name if defn else "",
                unit=defn.unit if defn else "",
                target_direction=defn.target_direction if defn else "INCREASE",
            )

            # Per-layer breakdown
            for layer in LAYERS:
                vals = by_ind_layer.get((ind_id, layer), [])
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                stat.by_layer[layer] = {
                    "N": float(len(arr)),
                    "Mean": round(float(np.mean(arr)), 4),
                    "Std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
                    "Min": round(float(np.min(arr)), 4),
                    "Max": round(float(np.max(arr)), 4),
                }

            # CV (full layer)
            full_vals = by_ind_layer.get((ind_id, "full"), [])
            if len(full_vals) > 1:
                arr_full = np.array(full_vals, dtype=float)
                mean_f = float(np.mean(arr_full))
                std_f = float(np.std(arr_full, ddof=1))
                if mean_f != 0:
                    stat.cv_full = round(std_f / abs(mean_f) * 100, 2)

                # Shapiro-Wilk (max 5000 samples)
                try:
                    sample = arr_full[:5000]
                    if len(sample) >= 3:
                        w, p = scipy_stats.shapiro(sample)
                        stat.shapiro_w = round(float(w), 4)
                        stat.shapiro_p = round(float(p), 6)
                except Exception:
                    pass

            # Kruskal-Wallis across layers
            layer_groups = [
                by_ind_layer.get((ind_id, l), [])
                for l in LAYERS
            ]
            layer_groups = [g for g in layer_groups if len(g) >= 2]
            if len(layer_groups) >= 2:
                try:
                    h, p = scipy_stats.kruskal(*layer_groups)
                    stat.kruskal_h = round(float(h), 4)
                    stat.kruskal_p = round(float(p), 6)
                except Exception:
                    pass

            results.append(stat)

        return results

    @staticmethod
    def _compute_data_quality(
        image_records: list[ImageRecord],
        ind_ids: list[str],
        global_stats: list[GlobalIndicatorStats],
    ) -> list[DataQualityRow]:
        """Table M4 / S4: data quality diagnostics per indicator."""
        if not image_records:
            return []

        stats_map = {s.indicator_id: s for s in global_stats}
        # Count images per (indicator, layer)
        count_map: dict[tuple[str, str], int] = defaultdict(int)
        for rec in image_records:
            count_map[(rec.indicator_id, rec.layer)] += 1

        rows: list[DataQualityRow] = []
        for ind_id in ind_ids:
            gs = stats_map.get(ind_id)
            full_n = count_map.get((ind_id, "full"), 0)
            fg_n = count_map.get((ind_id, "foreground"), 0)
            mg_n = count_map.get((ind_id, "middleground"), 0)
            bg_n = count_map.get((ind_id, "background"), 0)

            is_normal = None
            corr_method = "pearson"
            if gs and gs.shapiro_p is not None:
                is_normal = gs.shapiro_p > 0.05
                corr_method = "pearson" if is_normal else "spearman"

            rows.append(DataQualityRow(
                indicator_id=ind_id,
                total_images=full_n,
                fg_coverage_pct=round(fg_n / full_n * 100, 1) if full_n > 0 else None,
                mg_coverage_pct=round(mg_n / full_n * 100, 1) if full_n > 0 else None,
                bg_coverage_pct=round(bg_n / full_n * 100, 1) if full_n > 0 else None,
                is_normal=is_normal,
                correlation_method=corr_method,
            ))
        return rows


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _calc_corr_pval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation + p-value matrices (min 3 data points)."""
    cols = list(df.columns)
    n = len(cols)
    corr_m = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    pval_m = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                corr_m.loc[c1, c2] = 1.0
                pval_m.loc[c1, c2] = 0.0
            elif j > i:
                mask = df[c1].notna() & df[c2].notna()
                if mask.sum() >= 3:
                    x1 = df.loc[mask, c1]
                    x2 = df.loc[mask, c2]
                    # Guard: pearsonr on constant input returns NaN + warning
                    if x1.std(ddof=0) == 0 or x2.std(ddof=0) == 0:
                        continue
                    r, p = scipy_stats.pearsonr(x1, x2)
                    if np.isnan(r) or np.isnan(p):
                        continue
                    corr_m.loc[c1, c2] = r
                    corr_m.loc[c2, c1] = r
                    pval_m.loc[c1, c2] = p
                    pval_m.loc[c2, c1] = p

    return corr_m, pval_m
