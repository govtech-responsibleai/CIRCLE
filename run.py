"""
run.py
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Standard Library
import csv

# Third Party
from tqdm import tqdm

# Local imports
from utils import (
    run_anthropic_model,
    run_google_model,
    run_mistral_model,
    run_oai_model,
    ASSISTANT_SYSTEM_PROMPT,
    judge_response,
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TIMEOUT = 30.0

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # Load dataset

    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]

    results = []

    for prompt in tqdm(prompts, desc="Prompts"):
        for model in tqdm(
            [run_anthropic_model, run_google_model, run_mistral_model, run_oai_model],
            desc="Models",
            leave=False,
        ):
            response = model(ASSISTANT_SYSTEM_PROMPT, prompt, timeout=TIMEOUT)
            result = judge_response(prompt, response)
            results.append(result)

    # Save results to a CSV file
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "model", "result"])
        for result in results:
            writer.writerow([result.prompt, result.model, result.result])

    # Analyze results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
