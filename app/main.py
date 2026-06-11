from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.store.model_registry import load_model
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Smart PMS ML Service", lifespan=lifespan)
app.include_router(router)
