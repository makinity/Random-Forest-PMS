import json
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException

from app.schemas.uwp import SuggestRequest, SuggestResponse, EmployeeRecommendation
from app.store.model_registry import get_model
from app.train.train_model import _derive_fields


def suggest_employees(req: SuggestRequest, db: Session) -> SuggestResponse:
    pipeline = get_model()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Upload a CSV and retrain first.")

    # ── 1. Load indicator context ─────────────────────────────────────────────
    indicator = db.execute(text("""
        SELECT
            si.id,
            si.indicator_text,
            si.target_quantity,
            si.target_timeline,
            f.function_type,
            m.title AS mfo_title,
            uwp.office_id,
            uwp.performance_period_id
        FROM uwp_success_indicators si
        JOIN uwp_mfos m ON m.id = si.uwp_mfo_id
        JOIN uwp_functions f ON f.id = m.uwp_function_id
        JOIN unit_work_plans uwp ON uwp.id = f.unit_work_plan_id
        WHERE si.id = :ind_id
        LIMIT 1
    """), {"ind_id": req.uwp_success_indicator_id}).mappings().first()

    if not indicator:
        raise HTTPException(status_code=404, detail=f"Indicator {req.uwp_success_indicator_id} not found.")

    # ── 2. Load office employees ──────────────────────────────────────────────
    employees = db.execute(text("""
        SELECT
            e.user_id   AS employee_id,
            e.position,
            o.name      AS office_name,
            o.id        AS office_id
        FROM employees e
        JOIN offices o ON o.id = e.office_id
        WHERE e.office_id = :office_id
          AND e.is_active = 1
          AND e.is_disabled = 0
    """), {"office_id": indicator["office_id"]}).mappings().all()

    if not employees:
        raise HTTPException(status_code=404, detail="No active employees found for this office.")

    # ── 3. Office-level context ───────────────────────────────────────────────
    office_size = len(employees)

    # Parse timeline days from target_timeline string — fall back to snapshot median
    timeline_days = _parse_timeline_days(indicator["target_timeline"])

    # Parse target quantity — strip non-numeric characters
    try:
        target_qty = int(str(indicator["target_quantity"] or "1").strip().replace(",", "").split()[0])
    except (ValueError, IndexError):
        target_qty = 1

    # Count how many indicators are already assigned this period per employee
    workload_map = {}
    workload_rows = db.execute(text("""
        SELECT ia.employee_id, COUNT(*) AS cnt
        FROM uwp_indicator_assignments ia
        JOIN uwp_success_indicators si ON si.id = ia.uwp_success_indicator_id
        JOIN uwp_mfos m ON m.id = si.uwp_mfo_id
        JOIN uwp_functions f ON f.id = m.uwp_function_id
        JOIN unit_work_plans uwp ON uwp.id = f.unit_work_plan_id
        WHERE uwp.performance_period_id = :period_id
        GROUP BY ia.employee_id
    """), {"period_id": indicator["performance_period_id"]}).mappings().all()
    for row in workload_rows:
        workload_map[row["employee_id"]] = row["cnt"]

    # ── 4. Build one feature row per employee ─────────────────────────────────
    rows = []
    for emp in employees:
        rows.append({
            "employee_id":                 emp["employee_id"],
            "performance_period_id":       indicator["performance_period_id"],
            "ipcr_id":                     None,
            "uwp_success_indicator_id":    indicator["id"],
            "position":                    emp["position"],
            "office_name":                 emp["office_name"],
            "indicator_text":              indicator["indicator_text"],
            "function_type":               indicator["function_type"],
            "mfo_title":                   indicator["mfo_title"],
            "target_quantity":             target_qty,
            "target_timeline_days":        timeline_days,
            "office_size":                 office_size,
            "previous_final_score":        None,
            "previous_adjectival_rating":  None,
            "employee_count_assigned":     1,
            "current_workload_count":      workload_map.get(emp["employee_id"], 0),
            "was_flagged_for_calibration": False,
            "final_score":                 None,
            "adjectival_rating":           None,
        })

    df = pd.DataFrame(rows)

    # ── 5. Run inference ──────────────────────────────────────────────────────
    # Only drop cols the model was NOT trained on. The model was trained with
    # employee_id / performance_period_id / uwp_success_indicator_id as numeric
    # features (null-filled via the CSV), so we must keep them here too.
    drop_cols = ["id", "created_at", "updated_at", "feasibility_label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    predictions   = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    feasibility_labels, feasibility_probs, risk_levels, fit_scores, fit_labels, warnings = \
        _derive_fields(predictions, probabilities, pipeline.classes_)

    # ── 6. Build ranked recommendations ──────────────────────────────────────
    results = []
    for i, emp in enumerate(employees):
        results.append({
            "employee_id":           emp["employee_id"],
            "fit_score":             float(fit_scores[i]),
            "fit_label":             fit_labels[i],
            "feasibility_label":     feasibility_labels[i],
            "feasibility_probability": float(feasibility_probs[i]),
            "risk_level":            risk_levels[i],
            "warning":               bool(warnings[i]),
        })

    results.sort(key=lambda r: r["fit_score"], reverse=True)

    top = results[0]

    recommendations = [
        EmployeeRecommendation(
            employee_id=r["employee_id"],
            fit_score=r["fit_score"],
            fit_label=r["fit_label"],
            risk_level=r["risk_level"],
            warning=r["warning"],
        )
        for r in results
    ]

    return SuggestResponse(
        feasibility_label=top["feasibility_label"],
        feasibility_probability=top["feasibility_probability"],
        risk_level=top["risk_level"],
        recommendations=recommendations,
    )


def _parse_timeline_days(timeline_str: str | None) -> int:
    """
    Try to extract a number of days from the target_timeline string.
    e.g. "within 5 working days" → 5, "on the 26th day" → 26
    Falls back to 7 if unparseable.
    """
    if not timeline_str:
        return 7
    import re
    nums = re.findall(r"\d+", str(timeline_str))
    if nums:
        return int(nums[0])
    return 7
