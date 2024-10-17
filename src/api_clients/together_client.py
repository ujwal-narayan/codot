"""
This module contains the TogetherClient class for interacting with the Together API.
"""

from together import AsyncTogether
from together.types import ChatCompletionResponse
from src.constants import TOGETHER_API_KEY, MODEL_MAPPINGS


class TogetherClient:
    """A client for interacting with the Together API."""

    def __init__(self) -> None:
        """Initialize the TogetherClient with the API key."""
        self.client: AsyncTogether = AsyncTogether(api_key=TOGETHER_API_KEY)

    async def get_response(self, prompt: str, model: str) -> str:
        """
        Get a response from the Together API for the given prompt and model.

        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for generating the response.

        Returns:
            str: The generated response content.
        """
        response: ChatCompletionResponse = await self.client.chat.completions.create(
            model=MODEL_MAPPINGS.get(model, model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=1.0,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
        )
        if not response:
            raise ValueError("No response from Together API")
        return response.choices[0].message.content
