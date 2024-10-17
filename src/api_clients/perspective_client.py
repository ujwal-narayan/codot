"""
This module contains the PerspectiveClient class for interacting with the Perspective API.
"""

from typing import Dict, Any
import aiohttp
from src.constants import PERSPECTIVE_API_KEY


class PerspectiveClient:
    """A client for interacting with the Perspective API."""

    @staticmethod
    async def evaluate(text: str) -> float:
        """
        Evaluate the toxicity of the given text using the Perspective API.

        Args:
            text (str): The text to evaluate.

        Returns:
            float: The toxicity score of the text.

        Raises:
            Exception: If there's an error with the Perspective API request.
        """
        url: str = (
            f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}"
        )
        analyze_request: Dict[str, Any] = {
            "comment": {"text": text},
            "requestedAttributes": {"TOXICITY": {}},
            "languages": ["en"],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=analyze_request) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(
                        f"Perspective API error: {response.status}"
                    )
                result: Dict[str, Any] = await response.json()
                return result["attributeScores"]["TOXICITY"]["spanScores"][0]["score"][
                    "value"
                ]
