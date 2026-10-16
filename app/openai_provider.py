import json
import logging
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):

    def __init__(self, model=None, fallback_to_local=True):
        self.fallback_to_local = fallback_to_local
        self._local_provider = None
        self._use_local_only = False

        self.model = model or os.getenv(
            "STORY_PLANNER_MODEL",
            "gpt-5.6",
        )

        api_key = os.getenv("OPENAI_API_KEY")

        if OpenAI is None or not api_key:
            if self.fallback_to_local:
                logger.warning(
                    "[OPENAI] OpenAI package not available or OPENAI_API_KEY is not configured. "
                    "Automatically falling back to LocalLLMProvider..."
                )
                print(
                    "[OPENAI] WARNING: OpenAI key not configured. Automatically falling back to Local LLM..."
                )
                self._use_local_only = True
                self.client = None
            else:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured or openai package is missing."
                )
        else:
            self.client = OpenAI(
                api_key=api_key
            )

    def _get_local_provider(self):
        if self._local_provider is None:
            print("[OPENAI] Initializing LocalLLMProvider fallback...")
            from .local_llm_provider import LocalLLMProvider
            self._local_provider = LocalLLMProvider()
        return self._local_provider

    def generate_json(self, prompt: str, temperature: float = None, **kwargs):
        if self._use_local_only:
            return self._get_local_provider().generate_json(
                prompt, temperature=temperature, **kwargs
            )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )

            text = response.output_text

            if not text:
                raise RuntimeError(
                    "OpenAI returned an empty response."
                )

            return json.loads(text)

        except Exception as exc:
            if self.fallback_to_local:
                logger.warning(
                    f"[OPENAI] Request failed ({exc}). "
                    "Automatically falling back to LocalLLMProvider..."
                )
                print(
                    f"\n[OPENAI] WARNING: OpenAI failed ({exc}). "
                    "Automatically switching to Local LLM...\n"
                )
                self._use_local_only = True
                return self._get_local_provider().generate_json(
                    prompt, temperature=temperature, **kwargs
                )

            if isinstance(exc, json.JSONDecodeError):
                raise RuntimeError(
                    "OpenAI returned invalid JSON.\n\n"
                    f"Response:\n{locals().get('text', '')}"
                ) from exc
            raise
                f"Response:\n{text}"