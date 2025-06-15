# CIRCLE: Code Interpreter Resiliency Check for LLM Exploits

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
   Set the following environment variables based on `.env-template`

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
