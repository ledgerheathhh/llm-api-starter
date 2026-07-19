import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek").lower()

    DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "deepseek/deepseek-chat",
    )
    OPENROUTER_SITE_URL: str = os.getenv(
        "OPENROUTER_SITE_URL",
        "http://localhost:8000",
    )
    OPENROUTER_APP_NAME: str = os.getenv(
        "OPENROUTER_APP_NAME",
        "LLM API Starter",
    )


settings = Settings()
