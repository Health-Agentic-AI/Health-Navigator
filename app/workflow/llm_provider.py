import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


load_dotenv(os.path.join(os.getcwd(), "credentials.env"))

DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"
DEFAULT_ZAI_MODEL = "glm-4.7"
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
SUPPORTED_LLM_PROVIDERS = {"google", "z.ai"}


def normalize_llm_provider(provider: Optional[str]) -> str:
    """Normalize and validate LLM provider value from environment."""
    normalized = (provider or "google").strip().lower()
    if normalized in {"z.ai", "zai"}:
        return "z.ai"
    if normalized == "google":
        return "google"
    raise ValueError("LLM_PROVIDER must be either 'google' or 'z.ai'")


def get_active_llm_provider() -> str:
    """Get active provider from environment."""
    return normalize_llm_provider(os.environ.get("LLM_PROVIDER", "google"))


def create_langchain_chat_model(
    *,
    agent_name: Optional[str] = None,
    google_model: Optional[str] = None,
    zai_model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    Build a LangChain chat model based on LLM_PROVIDER.

    Supported providers:
    - google -> ChatGoogleGenerativeAI
    - z.ai   -> ChatOpenAI (OpenAI-compatible endpoint)
    """
    provider = get_active_llm_provider()

    if provider == "google":
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

        model_name = google_model or os.environ.get("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
        kwargs = {
            "model": model_name,
            "google_api_key": google_api_key,
        }
        if agent_name:
            kwargs["name"] = agent_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(**kwargs)

    z_ai_api_key = os.environ.get("ZAI_API_KEY")
    if not z_ai_api_key:
        raise ValueError("ZAI_API_KEY is required when LLM_PROVIDER=z.ai")

    model_name = zai_model or os.environ.get("ZAI_MODEL", DEFAULT_ZAI_MODEL)
    base_url = os.environ.get("ZAI_BASE_URL", DEFAULT_ZAI_BASE_URL)

    kwargs = {
        "model": model_name,
        "api_key": z_ai_api_key,
        "base_url": base_url,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)
