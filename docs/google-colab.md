import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)






from google.colab import files

uploaded = files.upload()






import pandas as pd

df = pd.read_csv("employee_performance_snapshots_2026_06_11.csv")

print("Shape:", df.shape)
df.head()






print(df.info())

print("\nTarget Distribution:")
print(df["feasibility_label"].value_counts())






TARGET = "feasibility_label"

drop_columns = [
    "id",
    "created_at",
    "updated_at",
    "feasibility_label"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df[TARGET]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)






categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numeric_features)







categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numeric_features)






numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])






X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))






rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)





pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])






pipeline.fit(X_train, y_train)

print("Training Complete!")





predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)





accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))





cm = confusion_matrix(y_test, predictions)

print(cm)





feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    pipeline.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.head(20)






all_predictions = pipeline.predict(X)
all_probabilities = pipeline.predict_proba(X)

# Per-employee x per-indicator � feeds ml_kpi_predictions table + AssignModal
prediction_df = pd.DataFrame({
    "employee_id":               df["employee_id"],
    "uwp_success_indicator_id":  df["uwp_success_indicator_id"],
    "performance_period_id":     df["performance_period_id"],

    # Objective 6.3 � Achievable / At Risk / Unrealistic
    "feasibility_label":         all_predictions,

    # Objective 6.4 � Probability percentage
    "feasibility_probability":   all_probabilities.max(axis=1).round(4),

    # Objective 6.4 � Risk level (maps to AssignModal risk field)
    "risk_level": pd.Series(all_predictions).map({
        "achievable":  "Low",
        "at_risk":     "Medium",
        "unrealistic": "High"
    }).values,

    # Objective 6.5 � Fit score 0-100 (maps to AssignModal fitScore / successProb)
    "fit_score": (all_probabilities.max(axis=1) * 100).round(1),

    # Fit label for UI (maps to AssignModal fitLabel)
    "fit_label": pd.Series(all_probabilities.max(axis=1)).map(
        lambda p: "Strong fit" if p >= 0.75 else
                  "Moderate fit" if p >= 0.50 else
                  "Weak fit"
    ).values,

    # Warning flag for AssignModal high-risk screen (maps to ai_prediction.warning)
    "warning": pd.Series(all_predictions) == "unrealistic",
})

prediction_df.head()




prediction_df.to_csv("ml_kpi_predictions_generated.csv", index=False)
print("Saved:", len(prediction_df), "rows")
print("Columns:", prediction_df.columns.tolist())




import joblib
joblib.dump(pipeline, "random_forest_kpi_model.pkl")
print("Model saved!")
