import json
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.store.model_registry import save_model

# Columns to drop before training (mirrors Colab)
_DROP_COLS = ["id", "created_at", "updated_at", "feasibility_label"]
TARGET = "feasibility_label"
MODEL_VERSION = "1.0.0"


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def _derive_fields(predictions, probabilities, classes):
    """Mirrors exact Colab derivation logic."""
    feasibility_labels = predictions

    # probability of the 'achievable' class — this is the true success probability
    if "achievable" in classes:
        achievable_idx = list(classes).index("achievable")
        achievable_probs = probabilities[:, achievable_idx]
    else:
        achievable_probs = probabilities.max(axis=1)

    feasibility_probs = achievable_probs.round(4)

    risk_level = pd.Series(feasibility_probs).map(
        lambda p: "Low" if p >= 0.75 else "Medium" if p >= 0.30 else "High"
    )

    fit_score = (feasibility_probs * 100).round(1)

    fit_label = pd.Series(feasibility_probs).map(
        lambda p: "Strong fit" if p >= 0.75 else "Moderate fit" if p >= 0.50 else "Weak fit"
    )

    warning = pd.Series(feasibility_labels) == "unrealistic"

    return feasibility_labels, feasibility_probs, risk_level.values, fit_score, fit_label.values, warning.values


def run_training(db: Session) -> dict:
    # 1. Load training data
    rows = db.execute(text("SELECT * FROM employee_performance_snapshots")).mappings().all()
    if not rows:
        raise ValueError("employee_performance_snapshots is empty — seed data first.")

    df = pd.DataFrame(rows)

    # 2. Prepare features / labels
    X = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")
    y = df[TARGET]

    # 3. Train
    pipeline = _build_pipeline(X)
    pipeline.fit(X, y)

    # 4. Generate predictions for every row
    all_predictions = pipeline.predict(X)
    all_probabilities = pipeline.predict_proba(X)

    feasibility_labels, feasibility_probs, risk_levels, fit_scores, fit_labels, warnings = \
        _derive_fields(all_predictions, all_probabilities, pipeline.classes_)

    pred_df = pd.DataFrame({
        "employee_id":              df["employee_id"],
        "uwp_success_indicator_id": df["uwp_success_indicator_id"],
        "performance_period_id":    df["performance_period_id"],
        "feasibility_label":        feasibility_labels,
        "feasibility_probability":  feasibility_probs,
        "risk_level":               risk_levels,
        "fit_score":                fit_scores,
        "fit_label":                fit_labels,
        "warning":                  warnings,
    })

    # 5. Group by indicator → UPSERT ml_kpi_predictions
    upsert_count = 0
    grouped = pred_df.groupby("uwp_success_indicator_id")

    for indicator_id, group in grouped:
        recs = (
            group.sort_values("fit_score", ascending=False)
            .drop_duplicates(subset=["employee_id"], keep="first")[
                ["employee_id", "fit_score", "fit_label",
                 "feasibility_label", "feasibility_probability",
                 "risk_level", "warning"]
            ]
            .assign(warning=lambda d: d["warning"].astype(bool))
            .to_dict("records")
        )

        top = recs[0]
        period_id = int(group["performance_period_id"].iloc[0]) \
            if pd.notna(group["performance_period_id"].iloc[0]) else None

        db.execute(text("""
            INSERT INTO ml_kpi_predictions
              (uwp_success_indicator_id, performance_period_id,
               feasibility_label, feasibility_probability, risk_level,
               recommendations, model_version, generated_at)
            VALUES
              (:ind_id, :per_id,
               :f_label, :f_prob, :risk,
               :recs, :ver, NOW())
            ON DUPLICATE KEY UPDATE
              feasibility_label       = VALUES(feasibility_label),
              feasibility_probability = VALUES(feasibility_probability),
              risk_level              = VALUES(risk_level),
              recommendations         = VALUES(recommendations),
              model_version           = VALUES(model_version),
              generated_at            = NOW()
        """), {
            "ind_id": int(indicator_id),
            "per_id": period_id,
            "f_label": top["feasibility_label"],
            "f_prob": float(top["feasibility_probability"]),
            "risk": top["risk_level"],
            "recs": json.dumps(recs),
            "ver": MODEL_VERSION,
        })
        upsert_count += 1

    db.commit()

    # 6. Persist model
    save_model(pipeline)

    return {
        "rows_trained": len(df),
        "indicators_upserted": upsert_count,
        "model_version": MODEL_VERSION,
    }



def train_from_dataframe(df: pd.DataFrame, target_column: str, log_id: int, source_type: str = "sql") -> None:
    """
    Background training entry point for /ml/train-sql and /ml/train-csv.
    Opens its own DB session so it is safe to run after the request session closes.
    """
    from app.db import SessionLocal  # local import avoids circular import at module load

    db: Session = SessionLocal()
    try:
        drop = [c for c in _DROP_COLS + [target_column] if c in df.columns]
        X = df.drop(columns=drop, errors="ignore")
        y = df[target_column]

        pipeline = _build_pipeline(X)
        pipeline.fit(X, y)

        all_predictions = pipeline.predict(X)
        all_probabilities = pipeline.predict_proba(X)

        feasibility_labels, feasibility_probs, risk_levels, fit_scores, fit_labels, warnings = \
            _derive_fields(all_predictions, all_probabilities, pipeline.classes_)

        pred_df = pd.DataFrame({
            "employee_id":              df["employee_id"],
            "uwp_success_indicator_id": df["uwp_success_indicator_id"],
            "performance_period_id":    df["performance_period_id"],
            "feasibility_label":        feasibility_labels,
            "feasibility_probability":  feasibility_probs,
            "risk_level":               risk_levels,
            "fit_score":                fit_scores,
            "fit_label":                fit_labels,
            "warning":                  warnings,
        })

        # CSV training is purely for building the model — the indicator/period/employee
        # IDs in the CSV are synthetic and won't match live FK constraints.
        # Skip the ml_kpi_predictions upsert entirely for CSV source.
        if source_type != "csv":
            for indicator_id, group in pred_df.groupby("uwp_success_indicator_id"):
                recs = (
                    group.sort_values("fit_score", ascending=False)
                    .drop_duplicates(subset=["employee_id"], keep="first")[
                        ["employee_id", "fit_score", "fit_label",
                         "feasibility_label", "feasibility_probability",
                         "risk_level", "warning"]
                    ]
                    .assign(warning=lambda d: d["warning"].astype(bool))
                    .to_dict("records")
                )
                top = recs[0]
                period_id = int(group["performance_period_id"].iloc[0]) \
                    if pd.notna(group["performance_period_id"].iloc[0]) else None

                db.execute(text("""
                    INSERT INTO ml_kpi_predictions
                      (uwp_success_indicator_id, performance_period_id,
                       feasibility_label, feasibility_probability, risk_level,
                       recommendations, model_version, generated_at)
                    VALUES
                      (:ind_id, :per_id, :f_label, :f_prob, :risk, :recs, :ver, NOW())
                    ON DUPLICATE KEY UPDATE
                      feasibility_label       = VALUES(feasibility_label),
                      feasibility_probability = VALUES(feasibility_probability),
                      risk_level              = VALUES(risk_level),
                      recommendations         = VALUES(recommendations),
                      model_version           = VALUES(model_version),
                      generated_at            = NOW()
                """), {
                    "ind_id": int(indicator_id),
                    "per_id": period_id,
                    "f_label": top["feasibility_label"],
                    "f_prob":  float(top["feasibility_probability"]),
                    "risk":    top["risk_level"],
                    "recs":    json.dumps(recs),
                    "ver":     MODEL_VERSION,
                })

        save_model(pipeline)  # saves to settings.MODEL_PATH (random_forest_kpi_model.pkl)

        if source_type == "csv":
            from app.db import engine as db_engine
            snapshot_cols = [c for c in df.columns if c != "feasibility_label"]
            insert_df = df[snapshot_cols].copy()

            # Null out FK columns — CSV rows are synthetic/historical and
            # won't have valid user/period IDs in the live DB.
            for fk_col in ("employee_id", "performance_period_id", "ipcr_id", "uwp_success_indicator_id"):
                if fk_col in insert_df.columns:
                    insert_df[fk_col] = None

            for col in ("created_at", "updated_at"):
                if col in insert_df.columns:
                    insert_df[col] = pd.to_datetime(insert_df[col], errors="coerce")

            with db_engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE employee_performance_snapshots"))
                insert_df.to_sql(
                    "employee_performance_snapshots",
                    con=conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=500,
                )

        db.execute(text("""
            UPDATE ml_model_logs
            SET status = 'success', row_count = :rows, trained_at = NOW(), updated_at = NOW()
            WHERE id = :log_id
        """), {"rows": len(df), "log_id": log_id})
        db.commit()

    except Exception as exc:
        db.rollback()
        db.execute(text("""
            UPDATE ml_model_logs
            SET status = 'failed', error_message = :err, trained_at = NOW(), updated_at = NOW()
            WHERE id = :log_id
        """), {"err": str(exc)[:1000], "log_id": log_id})
        db.commit()
    finally:
        db.close()
