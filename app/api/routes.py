from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db import get_db, check_connection
from app.store.model_registry import model_exists
from app.schemas.uwp import SuggestRequest, SuggestResponse
from app.train.train_model import run_training
from app.infer.predict import suggest_employees

router = APIRouter()


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
