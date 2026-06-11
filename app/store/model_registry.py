import os
import joblib
from app.config.settings import settings

_model = None


def load_model():
    global _model
    if os.path.exists(settings.MODEL_PATH):
        _model = joblib.load(settings.MODEL_PATH)
    return _model


def get_model():
    return _model


def save_model(pipeline):
    global _model
    joblib.dump(pipeline, settings.MODEL_PATH)
    _model = pipeline


def model_exists() -> bool:
    return os.path.exists(settings.MODEL_PATH)
