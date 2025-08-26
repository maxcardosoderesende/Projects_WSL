"""Data loading utilities for August 2025 competition."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load competition data from parquet files.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Tuple of (returns, features, target_returns)
    """
    returns = pd.read_parquet(f"{data_dir}/returns.train_validate.parquet")
    features = pd.read_parquet(f"{data_dir}/features.train_validate.parquet")
    target_returns = pd.read_parquet(f"{data_dir}/target.train_validate.parquet")

    return returns, features, target_returns


def split_data(
    returns: pd.DataFrame, features: pd.DataFrame, target: pd.DataFrame, test_size: float = 0.25
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split data into train and validation sets.

    Args:
        returns: Returns DataFrame
        features: Features DataFrame
        target_returns: Target returns DataFrame
        test_size: Fraction of data to use for validation

    Returns:
        Tuple of (train_target, validate_target, train_data, validate_data)
        where train_data and validate_data are dictionaries with 'returns' and 'features' keys
    """
    # Split returns
    train_returns, validate_returns = train_test_split(returns, test_size=test_size, shuffle=False)

    # Split features
    train_features, validate_features = train_test_split(features, test_size=test_size, shuffle=False)

    train_target, validate_target = train_test_split(target, test_size=test_size, shuffle=False)

    # Create data dictionaries
    train_data = {"returns": train_returns, "features": train_features, "target": train_target}
    validate_data = {"returns": validate_returns, "features": validate_features, "target": validate_target}

    return train_data, validate_data


def prepare_test_data(test_returns: pd.DataFrame, test_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare test data dictionary.

    Args:
        test_returns: Test returns DataFrame
        test_features: Test features DataFrame

    Returns:
        Dictionary with 'returns' and 'features' keys
    """
    return {"returns": test_returns, "features": test_features}
