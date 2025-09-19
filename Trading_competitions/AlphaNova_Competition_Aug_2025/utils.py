"""Utility functions for August 2025 competition."""

import ast
import json
import os
import tempfile
from typing import Any

import nbformat
import pandas as pd
from nbconvert import PythonExporter


def convert_notebook_to_python(notebook_path: str) -> str:
    """Convert a Jupyter notebook to a Python script.

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        Path to the generated Python file
    """
    import re

    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)

    # Convert to Python - exclude IPython magics
    exporter = PythonExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    python_code, _ = exporter.from_notebook_node(notebook)

    # Collect dependencies from pip install commands
    dependencies = set()

    # Parse lines to find pip install commands and clean IPython code
    lines = python_code.split("\n")
    cleaned_lines = []

    for line in lines:
        # Check for get_ipython() calls with pip/uv pip install
        if "get_ipython()" in line:
            # Match get_ipython().system('pip install ...') or get_ipython().run_line_magic()
            system_match = re.search(r"get_ipython\(\)\.(?:system|run_line_magic)\(['\"](.+?)['\"]\)", line)
            if system_match:
                cmd = system_match.group(1)
                # Check if it's a pip or uv pip install command
                pip_match = re.match(r"(?:uv\s+)?pip\s+install\s+(.+)", cmd)
                if pip_match:
                    packages_str = pip_match.group(1)
                    # Clean up redirects and pipes
                    packages_str = packages_str.split("2>")[0].split("||")[0].strip()
                    # Split and add each package
                    for pkg in packages_str.split():
                        if pkg and not pkg.startswith("-"):
                            dependencies.add(pkg)
            # Skip the get_ipython line
            continue

        # Check for shell magic commands (!pip install or !uv pip install)
        if line.strip().startswith("!"):
            cmd = line.strip()[1:]  # Remove the !
            pip_match = re.match(r"(?:uv\s+)?pip\s+install\s+(.+)", cmd)
            if pip_match:
                packages_str = pip_match.group(1)
                # Clean up redirects and pipes
                packages_str = packages_str.split("2>")[0].split("||")[0].strip()
                # Split and add each package
                for pkg in packages_str.split():
                    if pkg and not pkg.startswith("-"):
                        dependencies.add(pkg)
            # Skip magic commands
            continue

        # Skip other magic commands
        if line.strip().startswith("%"):
            continue

        cleaned_lines.append(line)

    # Find where to insert PEP 723 metadata
    # Look for the first non-comment, non-docstring line
    insert_index = 0
    in_docstring = False
    docstring_quotes = None

    for i, line in enumerate(cleaned_lines):
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_quotes = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(docstring_quotes) >= 2:  # Single line docstring
                    in_docstring = False
                continue
        else:
            if docstring_quotes in stripped:
                in_docstring = False
            continue

        # Skip empty lines and comments at the beginning
        if not stripped or stripped.startswith("#"):
            continue

        # Found first code line
        insert_index = i
        break

    # Create PEP 723 metadata block if we have dependencies
    if dependencies:
        metadata_lines = ["# /// script", "# dependencies = ["]
        for dep in sorted(dependencies):
            metadata_lines.append(f'#   "{dep}",')
        metadata_lines.append("# ]")
        metadata_lines.append("# ///")
        metadata_lines.append("")  # Empty line after metadata

        # Insert metadata at the appropriate position
        cleaned_lines = cleaned_lines[:insert_index] + metadata_lines + cleaned_lines[insert_index:]

    python_code = "\n".join(cleaned_lines)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.basename(notebook_path).replace(".ipynb", ".py")
    python_file = os.path.join(temp_dir, base_name)
    print(f"Created temp Python file from notebook in {python_file}")

    with open(python_file, "w") as f:
        f.write(python_code)

    return python_file


def extract_predictor_class(python_file: str) -> tuple[str | None, str | None]:
    """Extract the predictor class name and instance from a Python file.

    Args:
        python_file: Path to the Python file

    Returns:
        Tuple of (class_name, instance_name) or (None, None) if not found
    """
    with open(python_file) as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in file: {e}")
        return None, None

    # Find classes that inherit from Predictor
    predictor_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from Predictor
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Predictor":
                    predictor_classes.append(node.name)
                    break

    if not predictor_classes:
        # Look for any class with train and predict methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_train = False
                has_predict = False

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "train":
                            has_train = True
                        elif item.name == "predict":
                            has_predict = True

                if has_train and has_predict:
                    predictor_classes.append(node.name)

    if not predictor_classes:
        return None, None

    # Use the last predictor class found
    predictor_class = predictor_classes[-1]

    # Look for instance creation
    instance_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id == predictor_class:
                        if node.targets and isinstance(node.targets[0], ast.Name):
                            instance_name = node.targets[0].id
                            break

    return predictor_class, instance_name


def extract_imports_and_classes(python_file: str) -> tuple[list[str], list[str]]:
    """Extract import statements and class definitions from a Python file.

    Args:
        python_file: Path to the Python file

    Returns:
        Tuple of (imports, classes)
    """
    with open(python_file) as f:
        content = f.read()

    tree = ast.parse(content)

    imports = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"from {module} import {alias.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    return imports, classes


def validate_predictor_interface(predictor: Any) -> bool:
    """Validate that a predictor implements the required interface.

    Args:
        predictor: Predictor instance to validate

    Returns:
        True if valid, False otherwise
    """
    required_methods = ["set_parms", "train", "predict"]

    for method in required_methods:
        if not hasattr(predictor, method):
            print(f"Missing required method: {method}")
            return False

        if not callable(getattr(predictor, method)):
            print(f"'{method}' is not callable")
            return False

    return True


def create_submission_info(
    file_path: str, train_mse: float, validate_mse: float, test_mse: float, metadata: dict = None
) -> dict:
    """Create submission information dictionary.

    Args:
        file_path: Path to submission file
        train_mse: Training MSE
        validate_mse: Validation MSE
        test_mse: Test MSE
        metadata: Additional metadata

    Returns:
        Dictionary with submission information
    """
    info = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "train_mse": train_mse,
        "validate_mse": validate_mse,
        "test_mse": test_mse,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if metadata:
        info.update(metadata)

    return info


def save_submission_info(info: dict, output_file: str = "submission_info.json") -> None:
    """Save submission information to JSON file.

    Args:
        info: Submission information dictionary
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)


def format_results_table(results: dict) -> str:
    """Format results dictionary as a readable table.

    Args:
        results: Dictionary with train/validate/test results

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 50)
    lines.append("RESULTS SUMMARY")
    lines.append("=" * 50)

    for dataset, metrics in results.items():
        lines.append(f"\n{dataset.upper()} SET:")
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                lines.append(f"  {metric}: {value:.6f}")
        else:
            lines.append(f"  MSE: {metrics:.6f}")

    lines.append("=" * 50)
    return "\n".join(lines)
