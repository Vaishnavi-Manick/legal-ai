import os
import logging
from typing import Optional, Dict, Any
import httpx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_env_file():
    """Native, zero-dependency .env file parser."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k:
                            os.environ[k] = v
        except Exception as e:
            pass

load_env_file()

logger = logging.getLogger("LegalLLMClient")



class LLMClient:
    """
    Configurable LLM Client supporting Groq, OpenAI, Ollama, and OpenAI-compatible Chat Completion APIs.
    Reads configuration from environment variables or .env file:
      - LLM_PROVIDER (e.g., 'groq', 'openai', 'ollama', 'none')
      - GROQ_API_KEY / OPENAI_API_KEY
      - GROQ_MODEL / OPENAI_MODEL (default: 'llama-3.3-70b-versatile' for Groq, 'gpt-4o-mini' for OpenAI)
      - GROQ_BASE_URL / OPENAI_BASE_URL (default for Groq: 'https://api.groq.com/openai/v1')
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

        # Read Groq-specific variables with fallback to generic OpenAI variables
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.api_key = groq_key or openai_key

        groq_model = os.getenv("GROQ_MODEL", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "").strip()

        if self.provider == "groq":
            self.model = groq_model or openai_model or "llama-3.3-70b-versatile"
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        elif self.provider == "ollama":
            self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        else:
            self.model = openai_model or groq_model or "gpt-4o-mini"
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

        self.endpoint = f"{base_url}/chat/completions"

    def is_available(self) -> bool:
        """Returns True if LLM provider is configured and API key (or local server) is present."""
        if self.provider == "none":
            return False
        if self.provider in ["ollama", "local"] or "11434" in self.endpoint:
            return True
        return bool(self.api_key)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        timeout: float = 45.0
    ) -> Optional[str]:
        """
        Sends chat completion request to configured LLM API (Groq/OpenAI/Ollama) using httpx.
        Returns generated text or None if request fails or provider is unavailable.
        """
        if not self.is_available():
            logger.info("LLM provider is unconfigured or missing API key. Skipping LLM generation.")
            return None

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            logger.info(f"Sending LLM request to provider '{self.provider}', model '{self.model}' at '{self.endpoint}'...")
            with httpx.Client(timeout=timeout, verify=False) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    return content
                logger.warning("LLM API returned empty choices array.")
                return None
            else:
                logger.error(f"LLM API request failed with status code {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception during LLM API generation: {str(e)}")
            return None
