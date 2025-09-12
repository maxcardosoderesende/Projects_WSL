"""Automated runner for August 2025 competition submissions."""

import csv
import importlib
import os
import shutil
import sys
import tempfile
import time

import pandas as pd

from data_loader import load_data, split_data
from evaluation import sharpe_ratio
from predictor import Predictor
from utils import convert_notebook_to_python, extract_predictor_class


def check_cross_sectional_z(predictions: pd.DataFrame) -> bool:
    cross_sectionally_z = False
    if predictions.sum(axis=1).mean() < 10 ** (-8) and predictions.std(axis=1).mean() == 1:
        cross_sectionally_z = True

        print("cross sectionally z scored test passes!")
    else:
        print("cross sectionally z scored test fails!")
    return cross_sectionally_z


def load_predictor_from_file(file_path: str) -> Predictor:
    """Load a predictor instance from a Python file or notebook.

    Args:
        file_path: Path to the submission file (.py or .ipynb)

    Returns:
        Instance of the predictor class
    """
    import subprocess

    # Convert notebook to Python if needed
    if file_path.endswith(".ipynb"):
        python_file = convert_notebook_to_python(file_path)
    else:
        # Copy Python file to temp directory
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.basename(file_path)
        python_file = os.path.join(temp_dir, base_name)
        shutil.copy2(file_path, python_file)

    # Extract predictor class
    predictor_class_name, predictor_instance_name = extract_predictor_class(python_file)

    if predictor_class_name is None:
        raise ValueError(f"No Predictor class found in {file_path}")

    # Check if the file has PEP 723 dependencies
    with open(python_file) as f:
        content = f.read()

    has_pep723 = "# /// script" in content and "# dependencies" in content

    if has_pep723:
        # Extract dependencies and install them
        print("Installing dependencies from PEP 723 metadata...")

        # Parse PEP 723 metadata to get dependencies
        import re

        deps_match = re.search(r"# /// script.*?# dependencies = \[(.*?)\].*?# ///", content, re.DOTALL)
        if deps_match:
            deps_text = deps_match.group(1)
            deps = re.findall(r'"([^"]+)"', deps_text)
            if deps:
                print(f"Found dependencies: {', '.join(deps)}")
                # Install using uv pip
                install_cmd = ["uv", "pip", "install"] + deps
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Failed to install dependencies: {result.stderr}")
                else:
                    print("Successfully installed dependencies")

    # Import the module and get the predictor
    temp_dir = os.path.dirname(python_file)
    sys.path.insert(0, temp_dir)

    try:
        module_name = os.path.basename(python_file).replace(".py", "")
        module = importlib.import_module(module_name)

        # Create new instance
        predictor_class = getattr(module, predictor_class_name)
        predictor = predictor_class()
        print(f"Created new instance of: {predictor_class_name}")

        return predictor

    finally:
        # Clean up sys.path
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)


def validate_submission(
    file_path: str,
    output_file: str = "results.csv",
    data_dir: str = "data",
    verbose: bool = True,
    max_train_time: float = 900.0,
    max_predict_time: float = 60.0,
) -> tuple[float, float, float]:
    """Validate a submission file.

    Args:
        file_path: Path to the submission file
        output_file: Path to output CSV file
        data_dir: Directory containing data files
        verbose: Whether to print progress
        max_train_time: Maximum training time in seconds
        max_predict_time: Maximum prediction time per call in seconds

    Returns:
        Tuple of (train_mse, validate_mse, test_mse)
    """
    if verbose:
        print(f"Processing submission: {file_path}")
        print("=" * 60)

    # Load predictor
    try:
        predictor = load_predictor_from_file(file_path)
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        return None, None, None

    # Load data
    if verbose:
        print("Loading data...")
    returns, features, target_returns = load_data(data_dir)

    # Split data
    train_data, validate_data = split_data(returns, features, target_returns, test_size=0.25)

    # Train predictor
    if verbose:
        print("Training predictor...")
    start_time = time.time()

    try:
        predictor.train(train_data["features"], train_data["target"])
        train_time = time.time() - start_time

        if train_time > max_train_time:
            print(f"WARNING: Training took {train_time:.2f}s (limit: {max_train_time}s)")
    except Exception as e:
        print(f"Training failed: {e}")
        return None, None, None

    if verbose:
        print(f"Training completed in {train_time:.2f} seconds")

    # Make predictions and calculate Sharpe
    results = {}

    # Training predictions
    if verbose:
        print("Making training predictions...")
    start_time = time.time()

    try:
        train_predictions = predictor.predict(train_data["features"])
        predict_time = time.time() - start_time

        if predict_time > max_predict_time:
            print(f"WARNING: Training prediction took {predict_time:.2f}s (limit: {max_predict_time}s)")

        train_sharpe = sharpe_ratio(train_predictions, train_data["returns"])
        results["train"] = train_sharpe

        print("checking that predictions are cross sectionally z scored on training")
        cross_sectionally_z = check_cross_sectional_z(train_predictions)
        results["cross sectional z training"] = cross_sectionally_z

        if verbose:
            print(f"Training Sharpe: {train_sharpe:.20f}")
    except Exception as e:
        print(f"Training prediction failed: {e}")
        return None, None, None

    # Validation predictions
    if verbose:
        print("Making validation predictions...")
    start_time = time.time()

    try:
        validate_predictions = predictor.predict(validate_data["features"])
        predict_time = time.time() - start_time

        if predict_time > max_predict_time:
            print(f"WARNING: Validation prediction took {predict_time:.2f}s (limit: {max_predict_time}s)")

        validate_sharpe = sharpe_ratio(validate_predictions, validate_data["returns"])
        results["validate"] = validate_sharpe

        print("checking that predictions are cross sectionally z scored on validate")
        cross_sectionally_z = check_cross_sectional_z(validate_predictions)
        results["cross sectional z validate"] = cross_sectionally_z

        if verbose:
            print(f"Validation Sharpe: {validate_sharpe:.20f}")
    except Exception as e:
        print(f"Validation prediction failed: {e}")
        return None, None, None

    """ no test period in this case.
    # Test predictions
    if verbose:
        print("Making test predictions...")
    start_time = time.time()

    try:
        test_predictions = predictor.predict(test_data)
        predict_time = time.time() - start_time

        if predict_time > max_predict_time:
            print(f"WARNING: Test prediction took {predict_time:.2f}s (limit: {max_predict_time}s)")

        test_mse = loss_mse(test_predictions, test_target)
        results["test"] = test_mse

        if verbose:
            print(f"Test MSE: {test_mse:.20f}")
    except Exception as e:
        print(f"Test prediction failed: {e}")
        return None, None, None
    """

    # Write results to CSV
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["file_name", "train_sharpe", "validate_sharpe", "timestamp"])
        writer.writerow([file_path, f"{train_sharpe:.20f}", f"{validate_sharpe:.20f}", pd.Timestamp.now().isoformat()])

    if verbose:
        print("=" * 60)
        print("Validation complete!")

    return train_sharpe, validate_sharpe


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate August 2025 competition submissions")
    parser.add_argument("submission", help="Path to submission file (.py or .ipynb)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--max-train-time", type=float, default=900.0, help="Maximum training time in seconds")
    parser.add_argument(
        "--max-predict-time", type=float, default=60.0, help="Maximum prediction time per call in seconds"
    )

    args = parser.parse_args()

    validate_submission(
        args.submission,
        output_file=args.output,
        data_dir=args.data_dir,
        verbose=not args.quiet,
        max_train_time=args.max_train_time,
        max_predict_time=args.max_predict_time,
    )


if __name__ == "__main__":
    main()
