"""
Chart Summary Service — generates short LLM interpretations of analysis charts.

Each chart on the Analysis tab can request a "What this means →" summary. The
service is cache-first: results are persisted in a small SQLite table keyed by
(chart_id, project_id, payload_hash). Cache hits return immediately; misses
hit the configured LLMClient and write back.

Cost note: prompts are kept short (≤ 120 token completion). Use a cheap
provider (Gemini Flash, GPT-4o-mini) — defaults inherit from get_llm_client().
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-chart prompt templates
# ---------------------------------------------------------------------------
# Each template takes a string-formatted payload + project context and asks the
# LLM for a short, plain-language interpretation. The output schema is fixed:
# JSON object with `summary` (≤ 120 tokens) and `highlight_points` (≤ 3 items).
#
# Add a new chart_id key to override the default template; otherwise the
# default is used. Keep prompts tight — long prompts add cost without
# improving the summary quality.

_DEFAULT_TEMPLATE = """\
You are reviewing a chart from an urban-greenspace analysis dashboard.
Your audience is a designer or planner without statistics training.

Chart: {chart_title} (id: {chart_id})
Chart description: {chart_description}

Project context:
{project_context}

Chart data (truncated):
{payload}

Write a 2-3 sentence interpretation in plain English. Avoid jargon. If the
data is thin or single-zone, say so honestly. Then list 1-3 short highlight
bullets (each ≤ 15 words) describing the most striking findings.

Respond ONLY with a single JSON object on one line, no markdown fences:
{{"summary": "<2-3 sentences>", "highlight_points": ["<bullet 1>", "<bullet 2>"]}}
"""


_CHART_TEMPLATES: dict[str, str] = {
    # Override examples — add chart-specific guidance when the default is too
    # generic. The key matches `chart_id` from the frontend registry.
    "correlation-heatmap": _DEFAULT_TEMPLATE
    + "\nFocus on the strongest positive and negative pairs (|r| ≥ 0.5).",
    "radar-profiles": _DEFAULT_TEMPLATE
    + "\nFocus on which zones differ most and on which indicators.",
    "spatial-overview": _DEFAULT_TEMPLATE
    + "\nFocus on whether values cluster spatially or are evenly spread.",
    "indicator-deep-dive": _DEFAULT_TEMPLATE
    + "\nFocus on layer (FG/MG/BG) differences and any indicator-level outliers.",
}


def _get_template(chart_id: str) -> str:
    return _CHART_TEMPLATES.get(chart_id, _DEFAULT_TEMPLATE)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chart_summary_cache (
    chart_id      TEXT NOT NULL,
    project_id    TEXT NOT NULL,
    payload_hash  TEXT NOT NULL,
    summary       TEXT NOT NULL,
    highlight_points_json TEXT NOT NULL,
    model         TEXT,
    created_at    REAL NOT NULL,
    PRIMARY KEY (chart_id, project_id, payload_hash)
);
"""


def _payload_hash(payload: dict[str, Any]) -> str:
    """Stable hash of payload — sort keys so semantically identical payloads
    produce the same hash regardless of dict ordering."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class ChartSummaryService:
    def __init__(self, llm_client: LLMClient, cache_db_path: Path):
        self.llm = llm_client
        self.cache_db_path = cache_db_path
        self._ensure_schema()

    # ── cache helpers ────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _lookup(
        self,
        chart_id: str,
        project_id: str,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT summary, highlight_points_json, model FROM chart_summary_cache
                   WHERE chart_id = ? AND project_id = ? AND payload_hash = ?""",
                (chart_id, project_id, payload_hash),
            ).fetchone()
            if row is None:
                return None
            try:
                highlights = json.loads(row["highlight_points_json"])
            except json.JSONDecodeError:
                highlights = []
            return {
                "summary": row["summary"],
                "highlight_points": highlights,
                "model": row["model"] or "",
            }

    def _store(
        self,
        chart_id: str,
        project_id: str,
        payload_hash: str,
        summary: str,
        highlight_points: list[str],
        model: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO chart_summary_cache
                   (chart_id, project_id, payload_hash, summary,
                    highlight_points_json, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chart_id,
                    project_id,
                    payload_hash,
                    summary,
                    json.dumps(highlight_points),
                    model,
                    time.time(),
                ),
            )
            conn.commit()

    # ── public api ───────────────────────────────────────────────────
    def get_cached(
        self,
        chart_id: str,
        project_id: str,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        return self._lookup(chart_id, project_id, payload_hash)

    async def generate(
        self,
        *,
        chart_id: str,
        chart_title: str,
        chart_description: str | None,
        project_id: str,
        payload: dict[str, Any],
        project_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a {summary, highlight_points, cached, model} dict."""
        payload_hash = _payload_hash(payload)
        cached = self._lookup(chart_id, project_id, payload_hash)
        if cached is not None:
            return {**cached, "cached": True}

        # Cap the payload string so the prompt stays small.
        payload_str = json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False)
        if len(payload_str) > 6000:
            payload_str = payload_str[:6000] + "  ...[truncated]"

        ctx_str = (
            json.dumps(project_context, default=str, ensure_ascii=False)
            if project_context
            else "(none)"
        )
        prompt = _get_template(chart_id).format(
            chart_id=chart_id,
            chart_title=chart_title,
            chart_description=chart_description or "(none)",
            project_context=ctx_str,
            payload=payload_str,
        )

        try:
            raw = await self.llm.generate(prompt)
        except Exception as exc:
            logger.warning("Chart summary LLM call failed for %s: %s", chart_id, exc)
            return {
                "summary": "",
                "highlight_points": [],
                "cached": False,
                "model": getattr(self.llm, "model", ""),
                "error": str(exc),
            }

        summary, highlights = _parse_llm_output(raw)
        self._store(
            chart_id,
            project_id,
            payload_hash,
            summary,
            highlights,
            getattr(self.llm, "model", ""),
        )
        return {
            "summary": summary,
            "highlight_points": highlights,
            "cached": False,
            "model": getattr(self.llm, "model", ""),
        }


def _parse_llm_output(raw: str) -> tuple[str, list[str]]:
    """Tolerant JSON parse: strip markdown fences, locate the first JSON
    object, fall back to free-text on parse failure."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop optional ```json fence, keep the inside until the closing ```.
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            summary = str(obj.get("summary", "")).strip()
            highlights = obj.get("highlight_points") or []
            if not isinstance(highlights, list):
                highlights = [str(highlights)]
            highlights = [str(h).strip() for h in highlights if str(h).strip()][:3]
            return summary, highlights
        except json.JSONDecodeError:
            pass
    # Fallback: treat the whole response as the summary.
    return text[:600].strip(), []
