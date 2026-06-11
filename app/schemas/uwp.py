from pydantic import BaseModel
from typing import List


class SuggestRequest(BaseModel):
    uwp_success_indicator_id: int
    performance_period_id: int


class EmployeeRecommendation(BaseModel):
    employee_id: int
    fit_score: float
    fit_label: str
    risk_level: str
    warning: bool


class SuggestResponse(BaseModel):
    feasibility_label: str
    feasibility_probability: float
    risk_level: str
    recommendations: List[EmployeeRecommendation]
