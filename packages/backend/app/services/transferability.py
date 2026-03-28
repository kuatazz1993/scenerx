"""
Transferability Matcher — deterministic Python computation.

Pre-computes how well each evidence record's study context matches the
target project (climate, LCZ, setting, user group).  The result is attached
to each evidence record *before* it reaches the LLM — the model never has
to perform join / match operations itself.
"""

from collections import Counter
from typing import Any


def _match_climate(ctx_code: str, proj_code: str) -> str:
    if not ctx_code or not proj_code or "NA" in ctx_code or "XX" in ctx_code or "NA" in proj_code:
        return "unknown"
    if ctx_code == proj_code:
        return "exact"
    if len(ctx_code) >= 5 and len(proj_code) >= 5 and ctx_code[:5] == proj_code[:5]:
        return "compatible"
    return "mismatch"


def _match_lcz(ctx_code: str, proj_code: str) -> str:
    if not ctx_code or not proj_code or "NA" in ctx_code or "NA" in proj_code:
        return "unknown"
    if ctx_code == proj_code:
        return "exact"
    if ctx_code == "LCZ_URB" or proj_code == "LCZ_URB":
        return "compatible"
    ctx_num = ctx_code.replace("LCZ_", "")
    proj_num = proj_code.replace("LCZ_", "")
    if ctx_num.isdigit() and proj_num.isdigit() and abs(int(ctx_num) - int(proj_num)) <= 1:
        return "compatible"
    return "mismatch"


def _match_setting(ctx_code: str, proj_code: str) -> str:
    if not ctx_code or not proj_code or "NA" in ctx_code or "NA" in proj_code:
        return "unknown"
    if ctx_code == proj_code:
        return "exact"
    if ctx_code == "SET_URB" or proj_code == "SET_URB":
        return "compatible"
    return "mismatch"


def _match_user(ctx_code: str, proj_code: str) -> str:
    if not ctx_code or not proj_code or "NA" in ctx_code or "UNSPECIFIED" in ctx_code or "NA" in proj_code:
        return "unknown"
    if ctx_code == proj_code:
        return "exact"
    if ctx_code == "AGE_ALL" or proj_code == "AGE_ALL":
        return "compatible"
    return "mismatch"


def compute_transferability(
    evidence_record: dict,
    ctx_record: dict | None,
    project: dict,
) -> dict:
    """Compute transferability for one evidence record against the project context."""
    if not ctx_record:
        return {
            "context_id": None,
            "climate_match": "unknown",
            "lcz_match": "unknown",
            "setting_match": "unknown",
            "user_group_match": "unknown",
            "overall": "unknown",
        }

    climate = _match_climate(
        ctx_record.get("climate", {}).get("koppen_zone_id", ""),
        project.get("koppen_zone_id", ""),
    )
    lcz = _match_lcz(
        ctx_record.get("urban_form", {}).get("lcz_type_id", ""),
        project.get("lcz_type_id", ""),
    )
    setting = _match_setting(
        ctx_record.get("urban_form", {}).get("space_type_id", ""),
        project.get("space_type_id", ""),
    )
    user = _match_user(
        ctx_record.get("user", {}).get("age_group_id", ""),
        project.get("age_group_id", ""),
    )

    scores = [climate, lcz, setting, user]
    n_good = sum(1 for s in scores if s in ("exact", "compatible"))
    n_bad = sum(1 for s in scores if s == "mismatch")

    if n_good >= 3 and n_bad == 0:
        overall = "high"
    elif n_bad >= 2:
        overall = "low"
    else:
        overall = "moderate"

    return {
        "context_id": ctx_record.get("context_id"),
        "climate_match": climate,
        "lcz_match": lcz,
        "setting_match": setting,
        "user_group_match": user,
        "overall": overall,
    }


def enrich_evidence(
    evidence_list: list[dict],
    ctx_by_evidence: dict[str, dict],
    project: dict,
) -> list[dict]:
    """Attach ``_transferability`` to each evidence record (shallow copy)."""
    enriched = []
    for e in evidence_list:
        e_copy = dict(e)
        ctx = ctx_by_evidence.get(e["evidence_id"])
        e_copy["_transferability"] = compute_transferability(e, ctx, project)
        enriched.append(e_copy)
    return enriched
