import json
from pydantic import BaseModel, Field
from typing import List


class ToxicityData(BaseModel):
    """
    A Pydantic model for storing toxicity analysis results.

    Attributes:
        tox_sents (List[str]): List of toxic sentences.
        tox_scores (List[float]): List of toxicity scores.
        tox_texts (List[str]): List of toxic texts.
        pos_tox_texts (List[str]): List of potentially toxic texts.
        pos_tox_sents (List[str]): List of potentially toxic sentences.
        pos_tox_scores (List[float]): List of potential toxicity scores.
    """

    tox_sents: List[str] = Field(..., description="List of toxic sentences")
    tox_scores: List[float] = Field(..., description="List of toxicity scores")
    tox_texts: List[str] = Field(..., description="List of toxic texts")
    pos_tox_texts: List[str] = Field(..., description="List of potentially toxic texts")
    pos_tox_sents: List[str] = Field(
        ..., description="List of potentially toxic sentences"
    )
    pos_tox_scores: List[float] = Field(
        ..., description="List of potential toxicity scores"
    )
    text: str = Field(..., description="Original input text")

    def to_json(self) -> str:
        """
        Serialize the ToxicityData object to a JSON string.

        Returns:
            str: JSON representation of the ToxicityData object.
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ToxicityData":
        """
        Deserialize a JSON string to a ToxicityData object.

        Args:
            json_str (str): JSON string representation of ToxicityData.

        Returns:
            ToxicityData: Deserialized ToxicityData object.
        """
        return cls.model_validate_json(json_str)
