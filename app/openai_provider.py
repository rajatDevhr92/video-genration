import json
import os

from openai import OpenAI

from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, model=None):
        self.model = model or os.getenv(
            "STORY_PLANNER_MODEL",
            "gpt-5.6",
        )

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key
        )

    def generate_json(self, prompt: str):

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        text = response.output_text

        if not text:
            raise RuntimeError(
                "OpenAI returned an empty response."
            )

        try:
            return json.loads(text)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI returned invalid JSON.\n\n"
                f"Response:\n{text}"
            ) from exc