from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs):
        """
        Generate structured JSON from the supplied prompt.
        """
        raise NotImplementedError