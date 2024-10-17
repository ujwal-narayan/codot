import asyncio
import time
import logging
from typing import List, Dict, Any
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm

from src.toxicity_analyzer import ToxicityAnalyzer
from src.text_utils import save_json_file, get_input_text, parse_arguments
from src.constants import RATE_LIMIT, MAX_CONCURRENT

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

rate_limiter = AsyncLimiter(RATE_LIMIT, 1)  # Allow RATE_LIMIT requests per second
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def analyze_single_text(
    analyzer: ToxicityAnalyzer, text: str, args
) -> Dict[str, Any]:
    """
    Analyze a single text and return the result, respecting rate limits and concurrency limits.

    Parameters
    ----------
    analyzer : ToxicityAnalyzer
        An instance of the ToxicityAnalyzer class.
    text : str
        The text to be analyzed.
    args : argparse.Namespace
        Command-line arguments containing model, iterations, and custom_prompt.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the analysis result or error information.

    Notes
    -----
    This function uses semaphores and rate limiters to control concurrency and API request rates.
    """
    async with semaphore:  # Limit concurrent executions
        async with rate_limiter:  # Respect rate limits
            try:
                result = await analyzer.analyze_toxicity(
                    text=text,
                    model=args.model,
                    max_iterations=args.iterations,
                    custom_prompt=args.custom_prompt,
                )
                return result.model_dump()
            except Exception as e:
                print(f"Error analyzing text: {e}")
                return {"error": str(e), "text": text}


async def analyze_texts_with_retry(
    analyzer: ToxicityAnalyzer, texts: List[str], args, max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Analyze multiple texts with retry mechanism, respecting concurrency limits.

    Parameters
    ----------
    analyzer : ToxicityAnalyzer
        An instance of the ToxicityAnalyzer class.
    texts : List[str]
        A list of texts to be analyzed.
    args : argparse.Namespace
        Command-line arguments containing model, iterations, and custom_prompt.
    max_retries : int, optional
        Maximum number of retry attempts for failed analyses (default is 3).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing analysis results for each text.

    Notes
    -----
    """
    results = []
    retry_texts = texts

    for attempt in range(max_retries):
        if not retry_texts:
            break

        tasks = [
            asyncio.create_task(analyze_single_text(analyzer, text, args))
            for text in retry_texts
        ]

        pbar = tqdm(total=len(tasks), desc=f"Attempt {attempt + 1}/{max_retries}")

        for task in asyncio.as_completed(tasks):
            result = await task
            if "error" not in result:
                results.append(result)
            else:
                retry_texts.append(result["text"])
            pbar.update(1)

        pbar.close()
        retry_texts = [
            text for text in retry_texts if text not in [r.get("text") for r in results]
        ]
        if retry_texts:
            print(
                f"Retrying {len(retry_texts)} texts. Attempt"
                f" {attempt + 2}/{max_retries}"
            )
            await asyncio.sleep(2**attempt)  # Exponential backoff

    if retry_texts:
        print(
            f"Failed to analyze {len(retry_texts)} texts after {max_retries} attempts."
        )

    return results


async def async_main() -> None:
    """
    Asynchronous main function to run the Toxicity Analyzer from command-line arguments.
    The function handles both single text input and multiple text inputs (as a list).

    Returns
    -------
    None
    """
    args = parse_arguments()
    input_text = get_input_text(args.input)
    analyzer = ToxicityAnalyzer()

    start_time = time.time()

    if isinstance(input_text, list):
        results = await analyze_texts_with_retry(analyzer, input_text, args)
        output_data: Dict[str, Any] = {
            "input_text": args.input,
            "model": args.model,
            "iterations": args.iterations,
            "custom_prompt": args.custom_prompt,
            "result": results,
        }
    else:
        result = await analyze_single_text(analyzer, input_text, args)
        output_data: Dict[str, Any] = {
            "input_text": args.input,
            "model": args.model,
            "iterations": args.iterations,
            "custom_prompt": args.custom_prompt,
            "result": result,
        }

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    save_json_file(output_data, args.output)
    print(f"Analysis complete. Results saved to {args.output}")


def main():
    """
    Main function to set up and run the async event loop.

    Returns
    -------
    None
    """
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
    loop.close()


if __name__ == "__main__":
    main()
