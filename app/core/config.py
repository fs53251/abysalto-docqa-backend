from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "DocQA API"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "./data"
    JWT_SECRET: str = "CHANGE_ME"


settings = Settings()
