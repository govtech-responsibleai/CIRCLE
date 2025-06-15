"""
experiment.py
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Standard Library
import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import datetime

# Third Party
import dotenv
import pandas as pd
from loguru import logger
from tabulate import tabulate
from tqdm.asyncio import tqdm

dotenv.load_dotenv()

# Local imports
from utils import (
    create_overall_summary,
    get_provider_from_model,
    judge_response,
    model_exists_in_overall,
    run_anthropic_model,
    run_google_model,
    run_mistral_model,
    run_oai_model,
)
from utils.judge import JudgeResponse
from utils.system_prompts import ASSISTANT_SYSTEM_PROMPT


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TIMEOUT: int = 60 * 3  # 3 minutes
CONCURRENCY: int = 50

FUNCTION_MAPPING = {
    "anthropic": run_anthropic_model,
    "google": run_google_model,
    "mistral": run_mistral_model,
    "openai": run_oai_model,
}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


async def process_prompt(prompt: str, model_name: str) -> tuple[JudgeResponse, str]:
    """
    Process a prompt and return the result of the judge.

    Args:
        prompt (str): The prompt to process.
        model_name (str): The name of the model to use.

    Returns:
        JudgeResponse: The result of the judge.
    """
    try:
        logger.debug(f"Processing prompt for model {model_name}: {prompt[:100]}...")
        provider = get_provider_from_model(model_name)
        llm_function = FUNCTION_MAPPING[provider]
        response = await llm_function(
            system_prompt=ASSISTANT_SYSTEM_PROMPT,
            prompt=prompt,
            model=model_name,
            timeout=TIMEOUT,
        )
        if isinstance(response, str):
            logger.debug(f"Got response from model {model_name}: {response}...")
        else:
            logger.debug(f"Got response from model {model_name}: {response.model_dump_json()}")
        
        result: JudgeResponse = await judge_response(prompt, response)
        logger.debug(f"Judge result for model {model_name}: {result}")
        
        return result, response
    except Exception as e:
        logger.error(f"Error processing prompt for model {model_name}: {e}")
        raise


async def main(model_name: str, lite_mode: bool = False, verbose: bool = False, test_mode: bool = False) -> None:
    """
    Main function to process the prompts and save the results.

    Args:
        model_name (str): The name of the model to use.
        lite_mode (bool): Whether to run in lite mode (direct prompts only).
        verbose (bool): Whether to print logs to console.

    Returns:
        None
    """
    # Determine paths based on mode
    mode_suffix = "_lite" if lite_mode else ""
    overall_csv_path = f"results/overall{mode_suffix}.csv"
    
    # Set up logging for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/{model_name}{'_lite' if lite_mode else ''}"
    log_file = f"{log_dir}/{timestamp}.log"
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default stderr handler
    logger.remove()
    
    # Always add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="100 MB"
    )
    
    # Add console handler only if verbose mode
    if verbose:
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO"
        )
    
    mode_description = "LITE MODE (direct prompts only)" if lite_mode else "FULL MODE"
    logger.info(f"Starting benchmark execution for model: {model_name} - {mode_description}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timeout: {TIMEOUT} seconds, Concurrency: {CONCURRENCY}")
    
    # Check if model already exists in overall.csv
    logger.info(f"Checking if model '{model_name}' already exists in {overall_csv_path}")
    if model_exists_in_overall(model_name, overall_csv_path):
        logger.warning(f"Model '{model_name}' already exists in {overall_csv_path}. Skipping execution.")
        print(f"Model '{model_name}' already exists in {overall_csv_path}. Skipping execution.")
        return
    
    # Load dataset
    logger.info("Loading benchmark dataset from full_benchmark_dataset.csv")
    try:
        df: pd.DataFrame = pd.read_csv("full_benchmark_dataset.csv")
        
        # Filter for direct prompts only in lite mode
        if lite_mode:
            original_count = len(df)
            df = df[df["type"] == "direct"]
            logger.info(f"Filtered dataset for direct prompts: {len(df)} out of {original_count} prompts")
            print(f"Running in LITE mode: processing {len(df)} direct prompts only")
        
        prompts: list[str] = df["prompt"].tolist()
        
        # Limit to 5 rows for testing if test_mode is enabled
        if test_mode:
            df = df.sample(5)
            prompts = df["prompt"].tolist()
            logger.info(f"TEST MODE: Limited to {len(prompts)} prompts for testing")
            print(f"TEST MODE: Processing only {len(prompts)} prompts")
        
        logger.info(f"Loaded {len(prompts)} prompts from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    semaphore: asyncio.Semaphore = asyncio.Semaphore(CONCURRENCY)

    async def sem_task(prompt):
        async with semaphore:
            return await process_prompt(prompt, model_name)

    tasks: list[asyncio.Task[tuple[JudgeResponse, str]]] = [sem_task(prompt) for prompt in prompts]

    # Use asyncio.gather to preserve order
    logger.info(f"Starting to process {len(tasks)} prompts with concurrency limit of {CONCURRENCY}")
    print("Processing prompts...")
    
    try:
        results_and_responses: list[tuple[JudgeResponse, str]] = await tqdm.gather(*tasks)
        results: list[JudgeResponse] = [r for r, _ in results_and_responses]
        responses: list[str] = [resp for _, resp in results_and_responses]
        logger.info(f"Successfully processed all {len(results)} prompts")
    except Exception as e:
        logger.error(f"Error during prompt processing: {e}")
        raise

    # Count the answers
    logger.info("Analyzing results and counting answers")
    answer_counts: Counter[str] = Counter(result.answer for result in results)
    logger.info(f"Answer distribution: {dict(answer_counts)}")

    # Prepare table data
    table: list[tuple[str, int]] = [(answer, count) for answer, count in answer_counts.items()]
    print(tabulate(table, headers=["Answer", "Count"], tablefmt="github"))

    # Save the results as a CSV file - now they're in the same order as prompts
    logger.info(f"Preparing to save results to CSV")
    df_results: pd.DataFrame = pd.DataFrame([r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in results])
    df_results["prompt"] = prompts
    df_results["response"] = responses

    # Create results directory if it doesn't exist
    results_dir = f"results/{model_name}{'_lite' if lite_mode else ''}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/results{mode_suffix}.csv"
    
    try:
        df_results.to_csv(results_file, index=False)
        logger.info(f"Successfully saved detailed results to {results_file}")
        logger.info(f"Results CSV contains {len(df_results)} rows and {len(df_results.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to save results CSV: {e}")
        raise
    
    # Create or append to overall summary
    logger.info(f"Creating overall summary in {overall_csv_path}")
    try:
        create_overall_summary(model_name, results, df, overall_csv_path)
        logger.info(f"Successfully updated overall summary")
    except Exception as e:
        logger.error(f"Failed to create overall summary: {e}")
        raise
    
    logger.info(f"Benchmark execution completed successfully for model: {model_name} - {mode_description}")
    logger.info(f"Total prompts processed: {len(results)}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Overall summary updated: {overall_csv_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmark for a language model")
    parser.add_argument(
        "model_name", 
        nargs="?", 
        default="gpt-4.1-nano-2025-04-14",
        help="Name of the model to benchmark (default: gpt-4.1-nano-2025-04-14)"
    )
    parser.add_argument(
        "--lite", 
        action="store_true",
        help="Run in lite mode (process only direct prompts)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print logs to console (logs are always saved to file)"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test mode: process only 5 prompts for testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.model_name, args.lite, args.verbose, args.test))
