import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# =========================
# DATASET
# =========================
data = {
    "target_quantity": [100,120,80,150,200,60,90,130,70,110],
    "target_timeline_days": [5,5,3,7,10,3,4,6,2,5],
    "mfo_category": ["RSP","RSP","HRD","PM","PM","HRD","PBM","RSP","PBM","PM"],
    "office_size": [10,12,8,15,20,7,9,11,6,10],
    "past_completion_rate": [90,92,75,85,88,70,80,91,65,87],
    "employee_count_assigned": [3,4,2,5,6,2,3,4,2,3],
    "outcome": [
        "achievable","achievable","at_risk","achievable","at_risk",
        "unrealistic","at_risk","achievable","unrealistic","at_risk"
    ]
}

df = pd.DataFrame(data)

# =========================
# ENCODING
# =========================
df["mfo_category"] = df["mfo_category"].astype("category").cat.codes
df["outcome"] = df["outcome"].astype("category").cat.codes

# Save label names
label_names = df["outcome"].astype("category").cat.categories.tolist()

# =========================
# TRAIN
# =========================
X = df.drop("outcome", axis=1)
y = df["outcome"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# =========================
# SAVE MODEL + LABELS
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_v1.pkl")
joblib.dump(label_names, "models/labels.pkl")

print("✅ Model and labels saved!")