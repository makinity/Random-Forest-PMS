from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import io

from app.db import get_db, check_connection
from app.store.model_registry import model_exists
from app.schemas.uwp import SuggestRequest, SuggestResponse
from app.train.train_model import run_training, train_from_dataframe
from app.infer.predict import suggest_employees

router = APIRouter()

TARGET_COLUMN = "feasibility_label"


def _insert_log(db: Session, source_type: str) -> int:
    result = db.execute(text("""
        INSERT INTO ml_model_logs (source_type, target_column, status, created_at, updated_at)
        VALUES (:src, :col, 'running', NOW(), NOW())
    """), {"src": source_type, "col": TARGET_COLUMN})
    db.commit()
    return result.lastrowid


@router.post("/ml/train-sql")
def train_sql(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT * FROM employee_performance_snapshots")).mappings().all()
    if not rows:
        raise HTTPException(status_code=422, detail="employee_performance_snapshots is empty.")
    df = pd.DataFrame(rows)
    log_id = _insert_log(db, "sql")
    background_tasks.add_task(train_from_dataframe, df, TARGET_COLUMN, log_id)
    return {"status": "running", "log_id": log_id}


@router.post("/ml/train-csv")
async def train_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    if TARGET_COLUMN not in df.columns:
        raise HTTPException(status_code=422, detail=f"CSV must contain column '{TARGET_COLUMN}'.")
    log_id = _insert_log(db, "csv")
    background_tasks.add_task(train_from_dataframe, df, TARGET_COLUMN, log_id, "csv")
    return {"status": "running", "log_id": log_id}


@router.post("/train")
def train(db: Session = Depends(get_db)):
    result = run_training(db)
    return {"status": "ok", **result}


@router.post("/suggest-employees", response_model=SuggestResponse)
def suggest(req: SuggestRequest, db: Session = Depends(get_db)):
    return suggest_employees(req, db)


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db_ok = check_connection()
    snapshot_count = 0
    if db_ok:
        row = db.execute(text("SELECT COUNT(*) AS cnt FROM employee_performance_snapshots")).mappings().first()
        snapshot_count = row["cnt"]

    return {
        "db_connected": db_ok,
        "model_loaded": model_exists(),
        "snapshot_rows": snapshot_count,
    }
