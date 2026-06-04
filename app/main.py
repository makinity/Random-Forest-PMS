from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "model_v1.pkl")
label_path = os.path.join(BASE_DIR, "..", "models", "labels.pkl")

model = joblib.load(model_path)
label_names = joblib.load(label_path)

print("Model path:", model_path, os.path.exists(model_path))
print("Label path:", label_path, os.path.exists(label_path))


@app.get("/")
def root():
    return {"message": "AI Service Running"}


class PredictRequest(BaseModel):
    target_quantity: float
    target_timeline_days: float
    mfo_category: int
    office_size: int
    past_completion_rate: float
    employee_count_assigned: int

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: PredictRequest):

    features = [[
        data.target_quantity,
        data.target_timeline_days,
        data.mfo_category,
        data.office_size,
        data.past_completion_rate,
        data.employee_count_assigned
    ]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result_label = label_names[prediction]
    confidence = float(max(probability)) * 100

    # Risk logic
    if result_label == "achievable":
        risk = "LOW"
        recommendation = "APPROVE"
    elif result_label == "at_risk":
        risk = "MEDIUM"
        recommendation = "APPROVE WITH MONITORING"
    else:
        risk = "HIGH"
        recommendation = "REVISE TARGET"

    return {
        "verdict": result_label,
        "confidence": round(confidence, 2),
        "risk": risk,
        "recommendation": recommendation
    }