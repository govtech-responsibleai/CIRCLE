# CIRCLE: Code Interpreter Resiliency Check for LLM Exploits

A comprehensive benchmarking tool for evaluating the resilience of Large Language Models (LLMs) against potentially harmful, resource-intensive, or exploitative prompts in code interpreter environments.

## 🎯 Overview

CIRCLE tests how well LLMs handle requests that could potentially:
- Consume excessive computational resources (CPU, memory, disk I/O)
- Trigger system timeouts or crashes
- Bypass safety guidelines through indirect prompting techniques
- Execute resource-intensive operations that could impact system performance

The tool provides a standardized way to evaluate LLM safety and robustness across different providers and models.

## 🚀 Features

- **Multi-Provider Support**: Compatible with OpenAI, Anthropic, Google, and Mistral APIs
- **Comprehensive Test Suite**: 1,260+ test prompts across different categories and risk levels
- **Intelligent Judging System**: Automated response classification using GPT-4 as a judge
- **Flexible Execution Modes**: 
  - Full mode: Tests all prompt types (direct and indirect)
  - Lite mode: Tests only direct prompts for faster evaluation
  - Test mode: Limited subset for quick testing
- **Detailed Logging**: Comprehensive logging with timestamps and model-specific log files
- **Results Analysis**: Built-in analysis tools and Jupyter notebook for result exploration
- **Concurrent Processing**: Configurable concurrency for efficient batch processing

## 📊 Test Categories

### Resource Categories
- **R1**: CPU-intensive operations (mathematical computations, loops, algorithms)
- **R2**: Memory-intensive operations (large data structures, memory allocation)
- **R3**: I/O-intensive operations (file operations, disk usage, network operations)

### Prompt Types
- **Direct**: Straightforward requests for resource-intensive tasks
- **Indirect**: Requests disguised as legitimate use cases (research, benchmarking, auditing)

### Template Variations
- **T01**: Public relations research context
- **T02**: Academic thesis research context  
- **T03**: Internal systems audit context
- **T04**: Live demonstration context

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CIRCLE.git
   cd CIRCLE
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   Set the following environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   export MISTRAL_API_KEY="your-mistral-api-key"
   ```

## 🎮 Usage

### Basic Usage

Run benchmarks for a specific model:
```bash
python experiment.py "gpt-4.1-2025-04-14"
```

### Command Line Options

```bash
python experiment.py MODEL_NAME [OPTIONS]

Options:
  --lite        Run in lite mode (direct prompts only)
  --verbose     Enable verbose logging to console
  --test        Run in test mode (limited prompts for testing)
```

### Batch Processing

Use the provided shell script to run benchmarks for multiple models:
```bash
chmod +x run.sh
./run.sh
```

### Example Commands

```bash
# Full benchmark for GPT-4
python experiment.py "gpt-4.1-2025-04-14"

# Lite mode for faster testing
python experiment.py "claude-3-5-haiku-20241022" --lite

# Test mode for development
python experiment.py "gpt-4.1-mini-2025-04-14" --test --verbose
```

## 📈 Results and Analysis

### Output Structure
```
results/
├── overall.csv              # Consolidated results across all models
├── overall_lite.csv         # Lite mode results
├── model_name/             # Model-specific results
│   ├── timestamp.log       # Detailed execution logs
│   └── responses.json      # Raw responses (if saved)
└── analysis.ipynb          # Jupyter notebook for analysis
```

### Result Categories

The judging system classifies responses into six categories:

1. **`refusal`**: Model explicitly refuses the task
2. **`reframe`**: Model suggests alternative, safer approaches
3. **`follow_up`**: Model requests clarification before proceeding
4. **`incorrect_code`**: Model provides non-functional or incomplete code
5. **`fulfilled`**: Model successfully completes the task as requested
6. **`timeout`**: Model execution exceeds time limits

### Analysis Tools

- **Jupyter Notebook**: `analysis.ipynb` provides comprehensive analysis and visualization
- **Summary Generation**: Automated summaries of model performance across categories
- **Comparative Analysis**: Cross-model comparison capabilities

## ⚙️ Configuration

### Timeout and Concurrency

Default settings in `experiment.py`:
- **Timeout**: 3 minutes per prompt
- **Concurrency**: 50 concurrent requests

### Supported Models

The tool supports models from:
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5, etc.
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, etc.
- **Google**: Gemini 2.0 Flash, etc.
- **Mistral**: Mistral Small, Magistral Small, etc.

## 🔒 Safety Considerations

- **Controlled Environment**: Run in isolated environments to prevent system impact
- **Resource Monitoring**: Monitor system resources during execution
- **Responsible Use**: Use results to improve LLM safety, not to develop harmful applications
- **Ethical Guidelines**: Follow responsible AI practices and vendor terms of service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python experiment.py "test-model" --test --verbose
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use CIRCLE in your research, please cite:

```bibtex
@software{circle2025,
  title={CIRCLE: Code Interpreter Resiliency Check for LLM Exploits},
  author={Gabriel Chua},
  year={2025},
  url={https://github.com/yourusername/CIRCLE}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set as environment variables
2. **Model Not Found**: Check model name spelling and availability
3. **Timeout Issues**: Adjust timeout settings for slower models
4. **Memory Issues**: Reduce concurrency for resource-constrained environments

### Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check the logs in `results/` for detailed error information
- Review the Jupyter notebook for analysis examples

## 🏗️ Project Structure

```
CIRCLE/
├── experiment.py           # Main benchmarking script
├── run.sh                 # Batch execution script
├── requirements.txt       # Python dependencies
├── full_benchmark_dataset.csv  # Complete test dataset
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── judge.py          # Response judging logic
│   ├── llm.py            # LLM API interfaces
│   ├── summary.py        # Result summarization
│   └── system_prompts.py # System prompts
├── results/              # Output directory
├── data/                 # Additional data files
└── analysis.ipynb       # Analysis notebook
```

---

**⚠️ Disclaimer**: This tool is designed for research and safety evaluation purposes. Always use responsibly and in accordance with LLM provider terms of service and ethical AI guidelines.
