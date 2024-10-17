"""
This module contains utility functions for text processing and file operations.
"""

from typing import Any, Text, List, Union
import json
import argparse


def load_json_file(file_path: str) -> Any:
    """
    Load a JSON file and return its contents

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Any : The contents of the JSON file

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the file contains invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {file_path} was not found.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"The file {file_path} contains invalid JSON.") from exc


def save_json_file(data: Any, file_path: str) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data (Any): The dictionary to be saved as JSON.
        file_path (str): The path where the JSON file will be saved.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        raise IOError(f"Error writing to file {file_path}: {str(e)}") from e


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the Toxicity Analyzer.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze and increase text toxicity.")
    parser.add_argument(
        "--input", type=str, required=True, help="Input text or path to input file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of toxicity amplification iterations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="AI model to use (e.g., gpt-3.5-turbo, llama3)",
    )
    parser.add_argument(
        "--output", type=str, default="output.json", help="Path to output JSON file"
    )
    parser.add_argument(
        "--custom_prompt", type=str, help="Custom prompt for toxicity increase"
    )
    return parser.parse_args()


def get_input_text(input_value: str) -> Union[Text, List[str]]:
    """
    Get the input text from either a file or direct input.

    Args:
        input_value (str): File path or direct text input.

    Returns:
        Union[Text, List[str]]: The input text.
    """
    try:
        return [i.get("text") for i in load_json_file(input_value)]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return input_value
