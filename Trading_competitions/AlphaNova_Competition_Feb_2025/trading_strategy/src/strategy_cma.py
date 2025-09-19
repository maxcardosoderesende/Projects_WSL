import numpy as np
import pandas as pd

from loguru import logger as log
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def cma_strategy_1(df, train_ratio=0.75, c_mom=0.1, c_rev=-0.25):
    """
    Apply different models based on risk levels to predict future returns,
    using features and risk_distance with the same suffix.

    Parameters:
    - df (pd.DataFrame): DataFrame containing returns, risk distances, and risk levels.
    - train_ratio (float): Percentage of data used for training (default: 0.75).
    - c_mom (float): Momentum model constant.
    - c_rev (float): Mean-reversion speed constant.
    """

    log.info("Computing CMA Strategy 1")
    predicted_returns_dict = {}  # Store predictions for all return.X columns
    actual_returns_dict = {}
    risk_levels_dict = {}

    # Compute global standard deviation for all returns
    std_returns = df[[c for c in df.columns if "return" in c]].std()

    for col in [c for c in df.columns if "return" in c]:  # Iterate over return columns
        log.info(f"Training and predicting CMA 1 for {col}...")
        suffix = col.split(".")[-1]  # Extract suffix (0,1,2,...)
        risk_col = f"risk_level.{suffix}"  # Corresponding risk level column
        risk_dist_col = f"risk_dist.{suffix}"  # Corresponding risk distance column

        # Select feature columns dynamically
        feature_cols = [
            c for c in df.columns if c.endswith(f".{suffix}") and ("feature" in c)
        ]

        # 🚀 **Step 1: Train-Test Split (75% Train, 25% Test)**
        train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

        # **Step 2: Train Regression Model Only Once**
        train_X = train_df[feature_cols]
        train_y = train_df[col]
        model = LinearRegression()
        model.fit(train_X, train_y)  # Train OLS model on 75% of data

        # print(f"\n--- Model Summary for {col} (Risk Level = 1) ---")
        # print("Intercept (α):", model.intercept_)
        # print("Coefficients (β):", dict(zip(feature_cols, model.coef_)))
        # print("R² Score on Training Data:", model.score(train_X, train_y))

        # Store predictions for this return.X column
        predicted_values = pd.Series(index=test_df.index, dtype="float64")

        for idx in test_df.index:  # Only predict on the test dataset
            risk_level = df.loc[idx, risk_col]  # Get risk level

            if risk_level == "1":  # Use pre-trained regression model
                X = pd.DataFrame(
                    df.loc[idx, feature_cols].values.reshape(1, -1),
                    columns=feature_cols,
                )
                predicted_values[idx] = model.predict(X)[0]  # Store predicted value

            elif risk_level == "2":  # Momentum model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_mom
                )

            elif risk_level == "3":  # Reversal model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_rev
                )

            elif risk_level == "4":  # Moving average model
                if (
                    idx >= 2
                ):  # Ensure we have at least 3 data points for rolling average
                    predicted_values[idx] = (
                        df.loc[idx - 2, col] + df.loc[idx - 1, col] + df.loc[idx, col]
                    ) / 3
                else:
                    predicted_values[idx] = df.loc[
                        idx, col
                    ]  # Default to current return if not enough data

        # Store results in dictionaries for later concatenation
        predicted_returns_dict[f"predicted_{col}"] = predicted_values
        actual_returns_dict[f"actual_{col}"] = df.loc[test_df.index, col]
        risk_levels_dict[f"risk_level_{col}"] = df.loc[test_df.index, risk_col]

    # 🚀 **Step 3: Store & Display Predictions & Risk Levels**
    results_df = pd.concat(
        [
            pd.DataFrame(predicted_returns_dict),
            pd.DataFrame(actual_returns_dict),
            pd.DataFrame(risk_levels_dict),
        ],
        axis=1,
    )
    log.success("CMA Strategy 1 computation completed for all return columns!")
    return results_df


def cma_strategy_2(df, train_ratio=0.75, c_mom=0.1, c_rev=-0.25):
    """
    Apply different models based on risk levels to predict future returns,
    using ALL available features instead of just the ones matching the suffix.

    Parameters:
    - df (pd.DataFrame): DataFrame containing returns, risk distances, and risk levels.
    - train_ratio (float): Percentage of data used for training (default: 0.75).
    - c_mom (float): Momentum model constant.
    - c_rev (float): Mean-reversion speed constant.
    """
    log.info("Computing CMA Strategy 2")
    predicted_returns_dict = {}  # Store predictions for all return.X columns
    actual_returns_dict = {}
    risk_levels_dict = {}

    # Compute global standard deviation for all returns
    std_returns = df[[c for c in df.columns if "return" in c]].std()

    # 🚀 Select ALL features (instead of just feature.suffix)
    all_feature_cols = [c for c in df.columns if "feature" in c]

    for col in [c for c in df.columns if "return" in c]:  # Iterate over return columns
        log.info(f"Training and predicting CMA 2 for {col}...")
        suffix = col.split(".")[-1]  # Extract suffix (0,1,2,...)
        risk_col = f"risk_level.{suffix}"  # Corresponding risk level column
        risk_dist_col = f"risk_dist.{suffix}"  # Corresponding risk distance column

        # 🚀 **Step 1: Train-Test Split (75% Train, 25% Test)**
        train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

        # **Step 2: Train Regression Model Only Once**
        train_X = train_df[
            all_feature_cols
        ]  # Use ALL features, not just matching suffix
        train_y = train_df[col]
        model = LinearRegression()
        model.fit(train_X, train_y)  # Train OLS model on 75% of data

        # print(f"\n--- Model Summary for {col} (Risk Level = 1) ---")
        # print("Intercept (α):", model.intercept_)
        # print("Coefficients (β):", dict(zip(all_feature_cols, model.coef_)))
        # print("R² Score on Training Data:", model.score(train_X, train_y))

        # Store predictions for this return.X column
        predicted_values = pd.Series(index=test_df.index, dtype="float64")

        for idx in test_df.index:  # Only predict on the test dataset
            risk_level = df.loc[idx, risk_col]  # Get risk level

            if risk_level == "1":  # Use pre-trained regression model
                X = pd.DataFrame(
                    df.loc[idx, all_feature_cols].values.reshape(1, -1),
                    columns=all_feature_cols,
                )
                predicted_values[idx] = model.predict(X)[0]  # Store predicted value

            elif risk_level == "2":  # Momentum model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_mom
                )

            elif risk_level == "3":  # Reversal model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_rev
                )

            elif risk_level == "4":  # Moving average model
                if (
                    idx >= 2
                ):  # Ensure we have at least 3 data points for rolling average
                    predicted_values[idx] = (
                        df.loc[idx - 2, col] + df.loc[idx - 1, col] + df.loc[idx, col]
                    ) / 3
                else:
                    predicted_values[idx] = df.loc[
                        idx, col
                    ]  # Default to current return if not enough data

        # Store results in dictionaries for later concatenation
        predicted_returns_dict[f"predicted_{col}"] = predicted_values
        actual_returns_dict[f"actual_{col}"] = df.loc[test_df.index, col]
        risk_levels_dict[f"risk_level_{col}"] = df.loc[test_df.index, risk_col]

    # 🚀 **Step 3: Store & Display Predictions & Risk Levels**
    results_df = pd.concat(
        [
            pd.DataFrame(predicted_returns_dict),
            pd.DataFrame(actual_returns_dict),
            pd.DataFrame(risk_levels_dict),
        ],
        axis=1,
    )
    log.success("CMA Strategy 2 computation completed for all return columns!")
    return results_df


def add_lags(df, num_lags=2):
    """
    Generate lagged versions of all feature columns.
    """
    lagged_df = df.copy()
    feature_cols = [c for c in df.columns if "feature" in c]

    for lag in range(1, num_lags + 1):  # Create multiple lags if needed
        for col in feature_cols:
            lagged_df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return lagged_df


def cma_strategy_3(df, train_ratio=0.75, c_mom=0.1, c_rev=-0.25, num_lags=2):
    """
    Apply different models based on risk levels to predict future returns,
    using ALL available features AND their lags.

    Parameters:
    - df (pd.DataFrame): DataFrame containing returns, risk distances, and risk levels.
    - train_ratio (float): Percentage of data used for training (default: 0.75).
    - c_mom (float): Momentum model constant.
    - c_rev (float): Mean-reversion speed constant.
    - num_lags (int): Number of lagged features to include.
    """
    log.info("Computing CMA Strategy 3")
    # Step 1: Create Lagged Features
    df = add_lags(df, num_lags=num_lags)

    predicted_returns_dict = {}  # Store predictions for all return.X columns
    actual_returns_dict = {}
    risk_levels_dict = {}

    # Compute global standard deviation for all returns
    std_returns = df[[c for c in df.columns if "return" in c]].std()

    #  Select ALL features (including lagged versions)
    all_feature_cols = [c for c in df.columns if "feature" in c or "_lag" in c]

    for col in [c for c in df.columns if "return" in c]:  # Iterate over return columns
        log.info(f"Training and predicting CMA 3 for {col}...")
        suffix = col.split(".")[-1]  # Extract suffix (0,1,2,...)
        risk_col = f"risk_level.{suffix}"  # Corresponding risk level column
        risk_dist_col = f"risk_dist.{suffix}"  # Corresponding risk distance column

        feature_cols = all_feature_cols  # Now using all feature columns

        # **Step 2: Train-Test Split (75% Train, 25% Test)**
        train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

        # Remove NaNs introduced by lagging
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # **Step 3: Train Regression Model Only Once**
        train_X = train_df[feature_cols]
        train_y = train_df[col]
        model = LinearRegression()
        model.fit(train_X, train_y)  # Train OLS model on 75% of data

        # print(f"\n--- Model Summary for {col} (Risk Level = 1) ---")
        # print("Intercept (α):", model.intercept_)
        # print("Coefficients (β):", dict(zip(feature_cols, model.coef_)))
        # print("R² Score on Training Data:", model.score(train_X, train_y))

        # Store predictions for this return.X column
        predicted_values = pd.Series(index=test_df.index, dtype="float64")

        for idx in test_df.index:  # Only predict on the test dataset
            risk_level = df.loc[idx, risk_col]  # Get risk level

            if risk_level == "1":  # Use pre-trained regression model
                X = pd.DataFrame(
                    df.loc[idx, feature_cols].values.reshape(1, -1),
                    columns=feature_cols,
                )
                predicted_values[idx] = model.predict(X)[0]  # Store predicted value

            elif risk_level == "2":  # Momentum model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_mom
                )

            elif risk_level == "3":  # Reversal model
                risk_distance = df.loc[idx, risk_dist_col]  # d_it
                sigma_t = std_returns[col]  # sigma_t (global std dev)
                predicted_values[idx] = (
                    df.loc[idx, col] * (risk_distance / sigma_t) * c_rev
                )

            elif risk_level == "4":  # Moving average model
                if (
                    idx >= 2
                ):  # Ensure we have at least 3 data points for rolling average
                    predicted_values[idx] = (
                        df.loc[idx - 2, col] + df.loc[idx - 1, col] + df.loc[idx, col]
                    ) / 3
                else:
                    predicted_values[idx] = df.loc[
                        idx, col
                    ]  # Default to current return if not enough data

        # Store results in dictionaries for later concatenation
        predicted_returns_dict[f"predicted_{col}"] = predicted_values
        actual_returns_dict[f"actual_{col}"] = df.loc[test_df.index, col]
        risk_levels_dict[f"risk_level_{col}"] = df.loc[test_df.index, risk_col]

    # **Step 4: Store & Display Predictions & Risk Levels**
    results_df = pd.concat(
        [
            pd.DataFrame(predicted_returns_dict),
            pd.DataFrame(actual_returns_dict),
            pd.DataFrame(risk_levels_dict),
        ],
        axis=1,
    )
    log.success("CMA Strategy 2 computation completed for all return columns!")
    return results_df


def compute_mse(df):
    """
    Compute Mean Squared Error as accuracy measure
    """

    # Select only columns that start with 'predicted_return.' or 'actual_return.'
    cols_to_divide = [
        col
        for col in df.columns
        if col.startswith(("predicted_return.", "actual_return."))
    ]

    # Divide these columns by 100
    df[cols_to_divide] = df[cols_to_divide] / 100

    # Compute mse
    mse_results = {}
    for i in range(20):  # 20 return columns
        actual_col = f"actual_return.{i}"
        predicted_col = f"predicted_return.{i}"

        mse = mean_squared_error(df[actual_col], df[predicted_col])
        mse_results[f"Return {i}"] = mse

    # Convert to DataFrame for better visualization

    mse_df = pd.DataFrame(list(mse_results.items()), columns=["Return Series", "MSE"])

    return mse_df
