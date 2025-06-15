"""
utils/summary.py

Summary and overall.csv related utility functions.
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Standard Library
import os
from collections import Counter

# Third Party
import pandas as pd

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def get_provider_from_model(model_name: str) -> str:
    """
    Determine the provider based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The provider name.
    """
    name = model_name.lower()
    if "gpt" in name or "o4-mini" in name:
        return "openai"
    if "claude" in name:
        return "anthropic"
    if "gemini" in name:
        return "google"
    if "mistral" in name or "magistral" in name:
        return "mistral"
    return "unknown"


def model_exists_in_overall(model_name: str, overall_csv_path: str = "results/overall.csv") -> bool:
    """
    Check if the model already exists in the overall.csv file for any prompt type.
    
    Args:
        model_name (str): The name of the model to check.
        overall_csv_path (str): Path to the overall.csv file.
    
    Returns:
        bool: True if the model exists, False otherwise.
    """
    if not os.path.exists(overall_csv_path):
        return False
    
    try:
        df = pd.read_csv(overall_csv_path)
        return model_name in df["model"].values
    except Exception as e:
        print(f"Error reading {overall_csv_path}: {e}")
        return False


def create_overall_summary(model_name: str, results: list, prompts_df: pd.DataFrame, overall_csv_path: str = "results/overall.csv") -> None:
    """
    Create or append to the overall.csv summary file, split by prompt type.
    
    Args:
        model_name (str): The name of the model.
        results (list): List of JudgeResponse results.
        prompts_df (pd.DataFrame): DataFrame with prompt information including 'type' column.
        overall_csv_path (str): Path to the overall.csv file.
    
    Returns:
        None
    """
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame([r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in results])
    results_df['type'] = prompts_df['type'].values
    
    # Get provider
    provider = get_provider_from_model(model_name)
    
    # Group by prompt type and create summary for each
    summary_rows = []
    
    for prompt_type in results_df['type'].unique():
        type_results = results_df[results_df['type'] == prompt_type]
        answer_counts = Counter(type_results['answer'])
        total_count = len(type_results)
        
        # Create summary row for this type
        summary_row = {
            "model": model_name,
            "provider": provider,
            "type": prompt_type,
            "refusal": answer_counts.get("refusal", 0),
            "reframe": answer_counts.get("reframe", 0),
            "follow_up": answer_counts.get("follow_up", 0),
            "incorrect_code": answer_counts.get("incorrect_code", 0),
            "fulfilled": answer_counts.get("fulfilled", 0),
            "timeout": answer_counts.get("timeout", 0),
            "refusal_perc": round((answer_counts.get("refusal", 0) / total_count) * 100, 1),
            "reframe_perc": round((answer_counts.get("reframe", 0) / total_count) * 100, 1),
            "follow_up_perc": round((answer_counts.get("follow_up", 0) / total_count) * 100, 1),
            "incorrect_code_perc": round((answer_counts.get("incorrect_code", 0) / total_count) * 100, 1),
            "fulfilled_perc": round((answer_counts.get("fulfilled", 0) / total_count) * 100, 1),
            "timeout_perc": round((answer_counts.get("timeout", 0) / total_count) * 100, 1),
        }
        summary_rows.append(summary_row)
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(overall_csv_path), exist_ok=True)
    
    # Create or append to overall.csv
    if os.path.exists(overall_csv_path):
        # Load existing data and append
        df_existing = pd.read_csv(overall_csv_path)
        df_new = pd.DataFrame(summary_rows)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(overall_csv_path, index=False)
        print(f"Appended results for {model_name} to {overall_csv_path}")
    else:
        # Create new file
        df_new = pd.DataFrame(summary_rows)
        df_new.to_csv(overall_csv_path, index=False)
        print(f"Created {overall_csv_path} with results for {model_name}") 
