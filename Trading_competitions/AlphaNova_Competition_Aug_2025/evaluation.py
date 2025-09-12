"""Evaluation and scoring utilities for August 2025 competition."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def loss_mse(predictions: pd.DataFrame, target: pd.DataFrame) -> float:
    """Calculate Mean Squared Error between predictions and targets.

    Args:
        predictions: Predicted values
        target: True target values

    Returns:
        MSE value
    """
    return mean_squared_error(target.values, predictions.values)


def backtest(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Perform backtest of predictions against actual returns.

    Note: This function is for one-step forward returns, not target returns.

    Args:
        predictions: Signal predictions
        returns: Actual returns

    Returns:
        Portfolio returns series
    """
    predictions = predictions.copy()
    predictions.ffill(inplace=True)

    # Calculate portfolio returns based on predictions
    pf_returns = (predictions.shift(1)).mul(returns.values).sum(axis=1)
    pf_returns.iloc[0] = 0  # First day return is 0

    return pf_returns


def backtest_with_turnover(predictions: pd.DataFrame, returns: pd.DataFrame, cost: float = 0.0003) -> pd.Series:
    """Backtest returns with turnover cost penalty.

    Args:
        predictions: Signal dataframe (z-scored)
        returns: Actual returns
        cost: Turnover penalty per unit

    Returns:
        Penalized return series
    """
    predictions = predictions.copy()
    predictions.ffill(inplace=True)

    pnl = (predictions.shift(1) * returns).sum(axis=1)
    turnover = predictions.diff().abs().sum(axis=1).fillna(0)
    return pnl - cost * turnover


def turnover_penalized_sharpe(predictions: pd.DataFrame, returns: pd.DataFrame, cost: float = 0.0003) -> float:
    """Compute Sharpe ratio penalized by turnover cost."""
    pnl_series = backtest_with_turnover(predictions, returns, cost)
    return pnl_series.mean() / pnl_series.std()


def sharpe_ratio(predictions: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Compute Sharpe ratio penalized by turnover cost."""
    pnl_series = backtest(predictions, returns)
    return pnl_series.mean() / pnl_series.std()


def returns_to_equity(returns: pd.Series) -> pd.Series:
    """Convert returns series to equity curve.

    Args:
        returns: Returns series

    Returns:
        Cumulative equity series
    """
    return returns.cumsum()


def calculate_metrics(predictions: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    """Calculate various performance metrics.

    Args:
        predictions: Predicted values
        target: True target values

    Returns:
        Dictionary of metrics
    """
    mse = loss_mse(predictions, target)

    # Calculate additional metrics
    mae = np.mean(np.abs(predictions.values - target.values))
    rmse = np.sqrt(mse)

    # Calculate correlation
    pred_flat = predictions.values.flatten()
    target_flat = target.values.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    return {"mse": mse, "mae": mae, "rmse": rmse, "correlation": correlation}


def plot_backtest_results(
    train_returns: pd.Series, validate_returns: pd.Series, test_returns: pd.Series = None
) -> None:
    """Plot backtest equity curves.

    Args:
        train_returns: Training period returns
        validate_returns: Validation period returns
        test_returns: Test period returns (optional)
    """
    fig, axes = plt.subplots(1, 3 if test_returns is not None else 2, figsize=(15, 5))

    # Training equity curve
    returns_to_equity(train_returns).plot(ax=axes[0], title="Training Equity Curve")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Cumulative Returns")

    # Validation equity curve
    returns_to_equity(validate_returns).plot(ax=axes[1], title="Validation Equity Curve")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cumulative Returns")

    # Test equity curve if provided
    if test_returns is not None:
        returns_to_equity(test_returns).plot(ax=axes[2], title="Test Equity Curve")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Cumulative Returns")

    plt.tight_layout()
    plt.show()


def evaluate_predictor(
    predictor: Any,
    train_data: dict[str, pd.DataFrame],
    train_target: pd.DataFrame,
    validate_data: dict[str, pd.DataFrame],
    validate_target: pd.DataFrame,
    test_data: dict[str, pd.DataFrame] = None,
    test_target: pd.DataFrame = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate a predictor on train, validation, and optionally test data.

    Args:
        predictor: Trained predictor instance
        train_data: Training data dictionary
        train_target: Training targets
        validate_data: Validation data dictionary
        validate_target: Validation targets
        test_data: Test data dictionary (optional)
        test_target: Test targets (optional)
        verbose: Whether to print results

    Returns:
        Dictionary with metrics for each dataset
    """
    results = {}

    # Training metrics
    train_predictions = predictor.predict(train_data["features"])
    train_metrics = calculate_metrics(train_predictions, train_target)
    results["train"] = train_metrics

    if verbose:
        print("Training Results:")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  RMSE: {train_metrics['rmse']:.6f}")
        print(f"  Correlation: {train_metrics['correlation']:.4f}")

    # Validation metrics
    validate_predictions = predictor.predict(validate_data["features"])
    validate_metrics = calculate_metrics(validate_predictions, validate_target)
    results["validate"] = validate_metrics

    if verbose:
        print("\nValidation Results:")
        print(f"  MSE: {validate_metrics['mse']:.6f}")
        print(f"  RMSE: {validate_metrics['rmse']:.6f}")
        print(f"  Correlation: {validate_metrics['correlation']:.4f}")

    # Test metrics if provided
    if test_data is not None and test_target is not None:
        test_predictions = predictor.predict(test_data["features"])
        test_metrics = calculate_metrics(test_predictions, test_target)
        results["test"] = test_metrics

        if verbose:
            print("\nTest Results:")
            print(f"  MSE: {test_metrics['mse']:.6f}")
            print(f"  RMSE: {test_metrics['rmse']:.6f}")
            print(f"  Correlation: {test_metrics['correlation']:.4f}")

    return results
