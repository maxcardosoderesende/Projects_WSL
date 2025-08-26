# Competition Title: Competition August 2025

## Overview

Competition August 2025 introduces automated validation for return forecasting models. Your submission will be automatically loaded, trained, and evaluated by our runner system.

### Key Features

- **Automated Validation**: Submissions are automatically executed and validated
- **PEP 723 Support**: Automatic dependency installation from notebook metadata
- **Notebook Support**: Submit either `.ipynb` notebooks or `.py` files
- **Dependency Management**: Dependencies specified in notebooks using `!pip install` or `!uv pip install` are automatically handled

## File Structure

```
4/
├── aug2025.ipynb          # Main competition notebook with examples
├── data_loader.py         # Data loading utilities
├── evaluation.py          # Evaluation and scoring functions
├── predictor.py           # Base Predictor class interface
├── utils.py               # Utility functions (notebook conversion, PEP 723 support)
├── runner.py              # Automated validation runner with dependency handling
├── sample_submission.py   # Example submission in Python format
├── pyproject.toml         # Project dependencies
├── data/                  # Competition data files
│   ├── features.train_validate.parquet
│   ├── returns.train_validate.parquet
│   └── target.train_validate.parquet
└── results.csv            # Validation results
```

## Quick Start

1. **Implement Your Predictor**: Edit `aug2025.ipynb` and replace the example predictor class with your implementation.

2. **Test Locally**: Run the automated validation:
   ```bash
   python runner.py aug2025.ipynb
   ```

3. **Submit**: Upload your notebook or Python file to the competition page.

## Installation

How to install dependencies and run the code.

We are using `uv` as a package management tool. Feel free to use any other package manager that understands `pyproject.toml` file.
- Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Create venv
```bash
uv venv
```
- Activate venv
```bash
source .venv/bin/activate
```
- Install dependencies. Works best on Linux.
```bash
uv sync
```
- Run Jupiter Notebook
```bash
jupyter notebook aug2025.ipynb
``` 

Alternatively you can use `pip`:
- Create venv
```bash
python3 -m venv alphanova
```
- Activate venv
```bash
source alphanova/bin/activate
```
- Install dependencies. Works best on Linux.
```bash
pip install -r requirements.txt
```
- Run Jupiter Notebook
```bash
jupyter notebook aug2025.ipynb
```