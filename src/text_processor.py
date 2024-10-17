"""
This module contains the TextProcessor class for text-related operations.
"""

from typing import List
import re
from src.constants import DEFAULT_TOXIC_PROMPT


class TextProcessor:
    """A class for processing and manipulating text."""

    @staticmethod
    def split_text(text: str) -> List[str]:
        """
        Split the input text into sentences.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of sentences.
        """
        lines: List[str] = text.split("\n")
        result: List[str] = []
        for line in lines:
            sentences: List[str] = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()
            ]
            result.extend(sentences)

        if result and result[-1].endswith(".."):
            result[-1] = result[-1][:-1]

        return result

    @staticmethod
    def make_more_toxic_prompt(text: str, custom_prompt: str = "") -> str:
        """
        Generate a prompt for making text more toxic.

        Args:
            text (str): The input text to make more toxic.
            custom_prompt (str, optional): A custom prompt template. Defaults to None.

        Returns:
            str: The generated prompt for making the text more toxic.
        """
        if custom_prompt:
            return custom_prompt.format(text=text)
        else:
            return DEFAULT_TOXIC_PROMPT.format(text=text)
