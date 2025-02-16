"""
This module contains the ToxicityAnalyzer class for analyzing and increasing text toxicity.
"""

import asyncio
import logging
from typing import List, Tuple, Optional
from src.api_clients.perspective_client import PerspectiveClient
from src.api_clients.openai_client import OpenAIClient
from src.api_clients.together_client import TogetherClient
from src.toxicity_data import ToxicityData
from src.text_processor import TextProcessor
from src.constants import MODEL_MAPPINGS, NUMBER_OF_TRIES_FOR_EVERY_CALL, RETRY_DELAY, THINKING_MODEL_MAPPINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToxicityAnalyzer:
    """A class for analyzing and increasing text toxicity."""

    def __init__(self) -> None:
        """Initialize the ToxicityAnalyzer with necessary clients and processors."""
        self.perspective_client: PerspectiveClient = PerspectiveClient()
        self.openai_client: OpenAIClient = OpenAIClient()
        self.together_client: TogetherClient = TogetherClient()
        self.text_processor: TextProcessor = TextProcessor()

    async def analyze_toxicity(
        self,
        text: str,
        model: str,
        max_iterations: int = 15,
        custom_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> ToxicityData:
        """
        Analyze and increase the toxicity of the given text.

        Args:
            text (str): The input text to analyze and make more toxic.
            model (str): The model to use for text generation.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 15.
            custom_prompt (str, optional): A custom prompt for toxicity increase. Defaults to None.
            verbose (bool, optional): Whether to print the results of each iteration. Defaults to False.
        Returns:
            ToxicityData: The results of the toxicity analysis and increase.
        """
        tox_texts, tox_sents, tox_scores = [], [], []
        possible_tox_scores, possible_tox_texts, possible_tox_sents = [], [], []
        original_text = text
        for iteration_number in range(max_iterations):
            tox_text, tox_score, tox_sent = await self._perform_experiment(
                text, model, custom_prompt
            )
            if verbose:
                logger.info(
                    "Iteration %d: Text: %s Tox Text: %s Tox Score: %s Tox Sent: %s",
                    iteration_number,
                    text,
                    tox_text,
                    tox_score,
                    tox_sent,
                )

            possible_tox_scores.append(tox_score)
            possible_tox_texts.append(tox_text)
            possible_tox_sents.append(tox_sent)

            if not tox_scores or tox_score > max(tox_scores):
                tox_texts.append(tox_text)
                tox_scores.append(tox_score)
                tox_sents.append(tox_sent)
                text = tox_text
            else:
                tox_texts.append(tox_texts[-1])
                tox_scores.append(tox_scores[-1])
                tox_sents.append(tox_sents[-1])
        return ToxicityData(
            tox_sents=tox_sents,
            tox_scores=tox_scores,
            tox_texts=tox_texts,
            pos_tox_texts=possible_tox_texts,
            pos_tox_sents=possible_tox_sents,
            pos_tox_scores=possible_tox_scores,
            text=original_text,
        )

    async def _perform_experiment(
        self, text: str, model: str, custom_prompt: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Perform a single experiment to increase text toxicity.

        Args:
            text (str): The input text to make more toxic.
            model (str): The model to use for text generation.
            custom_prompt (str, optional): A custom prompt for toxicity increase. Defaults to None.

        Returns:
            Tuple[str, float, str]: The toxic text, its toxicity score, and the most toxic sentence.

        Raises:
            Exception: If max retries are reached without success.
        """
        prompt: str = self.text_processor.make_more_toxic_prompt(text, custom_prompt)
        generations = []
        for _ in range(NUMBER_OF_TRIES_FOR_EVERY_CALL):
            try:
                if model in MODEL_MAPPINGS:
                    response: str = await self.together_client.get_response(
                        prompt, model
                    )
                    if model in THINKING_MODEL_MAPPINGS:
                        start = THINKING_MODEL_MAPPINGS[model]["start"]
                        end = THINKING_MODEL_MAPPINGS[model]["end"]
                        if start in response and end in response:
                            thinking_text = response.split(start)[1].split(end)[0]
                            end_text = response.split(end)[1]
                            if end_text:
                                response = end_text
                else:
                    response: str = await self.openai_client.get_response(prompt, model)

                tox_text: str = response.strip()
                tox_sent, tox_score = await self._evaluate_toxicity(tox_text)
                generations.append((tox_text, tox_score, tox_sent))
            except Exception as e:
                print(f"Error in _perform_experiment: {e}")
                await asyncio.sleep(RETRY_DELAY)

        # Return the most toxic generation
        return max(generations, key=lambda x: x[1])

    async def _evaluate_toxicity(self, text: str) -> Tuple[str, float]:
        """
        Evaluate the toxicity of the given text.

        Args:
            text (str): The text to evaluate for toxicity.

        Returns:
            Tuple[str, float]: The most toxic sentence and its toxicity score.
        """
        sentences: List[str] = self.text_processor.split_text(text)
        scores: List[float] = await asyncio.gather(
            *(self.perspective_client.evaluate(sent) for sent in sentences)
        )
        max_score_index: int = scores.index(max(scores))
        return sentences[max_score_index], scores[max_score_index]
