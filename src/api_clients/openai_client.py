"""
This module contains the OpenAIClient class for interacting with the OpenAI API.
"""

from typing import Dict, Any
from openai import AsyncOpenAI
from src.constants import OPENAI_API_KEY


class OpenAIClient:
    """A client for interacting with the OpenAI API."""

    def __init__(self) -> None:
        """Initialize the OpenAIClient with the API key."""
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def get_response(self, prompt: str, model: str) -> str:
        """
        Get a response from the OpenAI API for the given prompt and model.

        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for generating the response.

        Returns:
            str: The generated response content.
        """
        # OpenAI mandates the word "JSON" in the prompt for JSON output.
        if "json" in prompt.lower():
            response: Dict[str, Any] = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=1.4,
                response_format={"type": "json_object"},
            )
        else:
            response: Dict[str, Any] = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=1.4,
            )

        return response.choices[0].message.content
