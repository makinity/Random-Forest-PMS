from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    MODEL_PATH: str = "random_forest_kpi_model.pkl"

    class Config:
        env_file = ".env"


settings = Settings()
