import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_TITLE: str = "RAG Chatbot API"
    APP_DESCRIPTION: str = "A RAG-based chatbot with memory, powered by LangChain & FastAPI"
    APP_VERSION: str = "0.1.0"

    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")


settings = Settings()
