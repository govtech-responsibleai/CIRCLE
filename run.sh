#!/bin/bash

set -e

MODELS=(
    # "gpt-4.1-nano-2025-04-14"
    # "gpt-4.1-mini-2025-04-14"
    # "gpt-4.1-2025-04-14"
    # "o4-mini-2025-04-16"
    # "gemini-2.0-flash-001"
    # "gemini-2.5-flash-preview-05-20"
    # "gemini-2.5-pro-preview-06-05"
    # "mistral-medium-2505"
    # "claude-sonnet-4-20250514"
    # "claude-3-5-haiku-20241022"
    "nova-micro"
)

for MODEL in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Running benchmark for model: $MODEL"
    echo "=============================================="
    python experiment.py "$MODEL"
    echo
done
