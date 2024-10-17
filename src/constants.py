"""
This module contains all the constant values used throughout the project.
It loads environment variables and defines model mappings and other configuration parameters.
"""

import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

PERSPECTIVE_API_KEY: str = os.getenv("PERSPECTIVE_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")

MODEL_MAPPINGS: Dict[str, str] = {
    "llama3": "meta-llama/Llama-3-8b-chat-hf",
    "wizardlm2": "microsoft/WizardLM-2-8x22B",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

NUMBER_OF_TRIES_FOR_EVERY_CALL: int = (
    2  #  Number of times the LLM is asked to generate for a given input prompt
)
RETRY_DELAY: int = 5  # seconds

# User-configurable prompts
DEFAULT_TOXIC_PROMPT: str = (
    'make_more_toxic("{text}", include_swearwords = True) -> Output in JSON'
)

RATE_LIMIT = 10  # requests per second
MAX_CONCURRENT = 5  # maximum number of concurrent requests
