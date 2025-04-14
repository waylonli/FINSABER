"""Base class for LLM model providers."""
import abc
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
)


class LLMProvider(abc.ABC):
    """Interface for LLM models."""

    @abc.abstractmethod
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Create a completion from messages in text (and potentially also encoded images)."""
        pass

    @abc.abstractmethod
    def init_provider(self, provider_cfg) -> None:
        """Initialize a provider via a json config."""
        pass

    @abc.abstractmethod
    def assemble_prompt(self, system_prompts: List[str], user_inputs: List[str], image_filenames: List[str]) -> List[str]:
        """Combine parametes in the appropriate way for the provider to use."""
        pass
