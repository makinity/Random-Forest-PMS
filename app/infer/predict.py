import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException

from app.schemas.uwp import SuggestRequest, SuggestResponse, EmployeeRecommendation


def suggest_employees(req: SuggestRequest, db: Session) -> SuggestResponse:
    row = db.execute(text("""
        SELECT feasibility_label, feasibility_probability, risk_level, recommendations
        FROM ml_kpi_predictions
        WHERE uwp_success_indicator_id = :ind_id
        LIMIT 1
    """), {"ind_id": req.uwp_success_indicator_id}).mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for uwp_success_indicator_id={req.uwp_success_indicator_id}. Run /train first."
        )

    recs_raw = row["recommendations"]
    recs = json.loads(recs_raw) if isinstance(recs_raw, str) else recs_raw

    recommendations = [
        EmployeeRecommendation(
            employee_id=r["employee_id"],
            fit_score=r["fit_score"],
            fit_label=r["fit_label"],
            risk_level=r["risk_level"],
            warning=bool(r["warning"]),
        )
        for r in recs
    ]

    return SuggestResponse(
        feasibility_label=row["feasibility_label"],
        feasibility_probability=float(row["feasibility_probability"]),
        risk_level=row["risk_level"],
        recommendations=recommendations,
    )
