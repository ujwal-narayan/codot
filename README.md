# Code-of-thought prompting: Probing AI Safety with Code

This repository contains the source code for the paper "Code-of-thought prompting: Probing AI Safety with Code". Our work introduces a novel model interaction paradigm called Code of Thought (CoDoT) prompting, which transforms natural language prompts into pseudo-code to systematically evaluate the safety of Large Language Models (LLMs).

This project demonstrates that current AI safety efforts fall short of ensuring safe, non-toxic outputs from LLMs. Using CoDoT, we show that a wide range of large language models can emit highly toxic outputs with the potential to cause significant harm. Our findings reveal a staggering 16.5× increase in toxicity on GPT-4 Turbo and a massive 4.6× increase on average across multiple models and languages.

## Setup

1. Clone the repository:

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file in the root directory:

   ```bash
   PERSPECTIVE_API_KEY=your_perspective_api_key
   OPENAI_API_KEY=your_openai_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```

## Usage

### CLI 

Run the Toxicity Analyzer from the command line with the following syntax:

```bash
python main.py --model MODEL --input INPUT [--iterations ITERATIONS] [--output OUTPUT] [--custom_prompt CUSTOM_PROMPT]
```

Arguments:

- `--model`: AI model to use (e.g., gpt-3.5-turbo, llama3)
- `--input`: Input text or path to input file. See [Input Format](#input-format) for details on JSON structure.
- `--iterations`: Number of toxicity amplification iterations (default: 15)
- `--output`: Path to output JSON file (default: output.json)
- `--custom_prompt`: Custom prompt (optional).

Note: Custom prompts need to have the placeholder `{text}` for the input to be inserted into the prompt.

Example usage:

```bash
# Analyze text directly from the command line
python main.py --model gpt-3.5-turbo --input "This is a sample text." --iterations 10 --output results.json

# Analyze text from a file
python main.py --model llama3 --input input.json --iterations 20 --output results.json --custom_prompt "Make this text more offensive: {text}"
```

### Python

```python
import asyncio
from src.toxicity_analyzer import ToxicityAnalyzer

async def main():
    analyzer = ToxicityAnalyzer()
    result = await analyzer.analyze_toxicity("Your text here", "gpt-3.5-turbo")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Input Format

To analyze multiple texts, you can provide a JSON file containing an array of objects. Each object should have a "text" key with the input text as its value. Here's the expected format:

```json
[
  {
    "text": "Example text"
  },
  {
    "text": "Another example text to analyze."
  },
  {
    "text": "Yet another example"
  }
]
```

## Output Format

The analysis results are saved in a JSON file with the following structure:

```json
{
  "input_text": "Original input text or file name",
  "model": "Model used for analysis",
  "iterations": "Number of iterations performed",
  "custom_prompt": "Custom prompt used (if any)",
  "text": "original user input",
  "result": {
    "tox_sents": ["List of the most toxic sentence from each iteration"],
    "tox_scores": [0.1, 0.2, ...],
    "tox_texts": ["List of full toxic texts generated in each iteration"],
    "pos_tox_texts": ["List of all generated texts, including less toxic ones"],
    "pos_tox_sents": ["List of the most toxic sentences from all generated texts"],
    "pos_tox_scores": [0.3, 0.4, ...]
  }
}
```

- `tox_sents`, `tox_scores`, and `tox_texts` represent the most toxic outputs until that iteration.
- `pos_tox_texts`, `pos_tox_sents`, and `pos_tox_scores` show the output at each generation step, including less toxic results.
- If a generated text is not more toxic than the previous output (likely due to the safety mechanisms kicking in), it's recorded in the `pos_` fields but not in the main `tox_` fields. The analyzer then reattempts with the same input.

## Supported Models

### OpenAI Models

We use the OpenAI SDK, so all OpenAI model names are supported. You can use the same model names as those listed in the [OpenAI API documentation](https://platform.openai.com/docs/models). Examples include:

- gpt-3.5-turbo
- gpt-4-turbo

### Open-Source Models

For open-source models, we use [Together.ai](https://www.together.ai/) as the serving platform. Currently supported models include:

- llama3
- wizardlm2
- mixtral

You can add or modify model mappings in the `MODEL_MAPPINGS` dictionary in `src/constants.py`. For example:

```python
MODEL_MAPPINGS = {
    'llama3': 'meta-llama/Llama-3-8b-chat-hf',
    'wizardlm2': 'microsoft/WizardLM-2-8x22B',
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
}
```

### Extending to Other Clients

To add support for other AI providers or models:

1. Create a new client file in the `src/api_clients/` directory (e.g., `new_provider_client.py`).
2. Implement the client following the pattern used for OpenAI and Together clients.
3. Update the `ToxicityAnalyzer` class in `src/services/toxicity_analyzer.py` to use the new client.

Example of a new client structure:

```python
class NewProviderClient:
    def __init__(self):
        # Initialize client with API key, etc.

    async def get_response(self, prompt: str, model: str) -> str:
        # Implement API call to new provider
        # Return the generated text
```

## DISCLAIMER

This tool is intended strictly for research purposes in the field of AI safety and ethics. It is designed to explore the limitations and potential vulnerabilities of language models and content moderation systems. The developers of this tool are not responsible for any misuse or consequences arising from its use.
