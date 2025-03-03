from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    """Settings for the Lead Finder Agent."""

    groq_api_key: str
    groq_model_user_proxy: str = "qwen-2.5-32b" # Model for User Proxy Agent
    groq_model_extraction_agent: str = "qwen-2.5-32b" # Model for Extraction Agent
    leads_csv_path: str = "leads.csv" # Path to your leads CSV file

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()

config_list_user_proxy = [{ # Config for User Proxy Agent (deepseek model)
    "model": settings.groq_model_user_proxy,
    "api_key": settings.groq_api_key,
    "api_type": "groq"
}]

config_list_extraction = [{ # Config for Extraction Agent (qwen model)
    "model": settings.groq_model_extraction_agent,
    "api_key": settings.groq_api_key,
    "api_type": "groq"
}]