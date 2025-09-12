import pandas as pd
from loguru import logger as log
from sklearn.model_selection import train_test_split

from src.strategy_cma import add_lags
import xgboost as xgb


def xgboost_strategy_1(df, train_ratio=0.75, num_lags=2):
    """
    Trains Independet XGBoost models to predict all return columns using only feature columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - train_ratio (float): Proportion of data used as training set.
    - num_lags (int): Number of lags applied to return columns.

    """
    log.info("Computing XGBoost Strategy 1")
    # Step 1: Identify Returns & Features**
    df = add_lags(df, num_lags=num_lags)  # Add lagged features
    return_cols = [c for c in df.columns if c.startswith("return.")]

    # Dictionaries to store results
    predicted_returns_dict = {}
    actual_returns_dict = {}

    for return_col in return_cols:
        log.info(f"Training and predicting CMA 1 for {return_col}...")
        # Step 2: Select Features Including Lags for Each Return**
        selected_features = [
            c for c in df.columns if "feature" in c or f"{return_col}_lag" in c
        ]

        df.dropna(inplace=True)  # Drop NaN rows created by shifting

        # Step 3: Train-Test Split**
        train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

        # Step 4: Train XGBoost Model**
        model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=100, random_state=42
        )
        model.fit(train_df[selected_features], train_df[return_col])

        # Step 5: Make Predictions**
        predicted_values = model.predict(test_df[selected_features])

        # Step 6: Store Results**
        predicted_returns_dict[f"predicted_{return_col}"] = predicted_values
        actual_returns_dict[f"actual_{return_col}"] = test_df[return_col].values

    # Step 7: Store & Return Results as DataFrame**
    results_df = pd.concat(
        [
            pd.DataFrame(predicted_returns_dict, index=test_df.index),
            pd.DataFrame(actual_returns_dict, index=test_df.index),
        ],
        axis=1,
    )

    log.success("XGBoost Strategy 1 computation completed for all return columns!")
    return results_df


def xgboost_strategy_2(df, train_ratio=0.75, num_lags=2):
    """
    Trains a single XGBoost model to predict all return columns simultaneously.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - train_ratio (float): Proportion of data used as training set.
    - num_lags (int): Number of lags applied to return columns.

    Returns:
    - pd.DataFrame: DataFrame containing actual vs. predicted values for all returns.
    """
    log.info("Computing XGBoost Strategy 2")
    # Step 1: Identify Returns & Features**
    df = add_lags(df, num_lags=num_lags)
    return_cols = [c for c in df.columns if c.startswith("return.")]
    feature_cols = [c for c in df.columns if c.startswith("feature.")]

    df.dropna(inplace=True)  # Drop NaN rows created by shifting

    # Step 2: Train-Test Split**
    train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train, y_test = train_df[return_cols], test_df[return_cols]  # Multi-output

    # Step 3: Train Single XGBoost Model for All Returns**
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    # Step 4: Make Predictions**
    y_pred = model.predict(X_test)

    # Step 5: Store Actual vs. Predicted Values**
    predictions_df = pd.DataFrame(
        y_pred, columns=[f"predicted_{c}" for c in return_cols], index=test_df.index
    )
    actual_df = test_df[return_cols].rename(columns=lambda c: f"actual_{c}")

    log.success("XGBoost Strategy 2 computation completed for all return columns!")
    return pd.concat([actual_df, predictions_df], axis=1)


def xgboost_strategy_3(df, train_ratio=0.75, c_mom=0.1, c_rev=-0.25, num_lags=2):
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
    log.info("Computing XGBoost Strategy 3")
    # tep 1: Create Lagged Features
    df = add_lags(df, num_lags=num_lags)

    predicted_returns_dict = {}  # Store predictions for all return.X columns
    actual_returns_dict = {}
    risk_levels_dict = {}

    # Compute global standard deviation for all returns
    std_returns = df[[c for c in df.columns if "return" in c]].std()

    # Select ALL features (including lagged versions)
    feature_cols = [c for c in df.columns if "feature" in c or "_lag" in c]

    for col in [c for c in df.columns if "return" in c]:
        log.info(f"Training and predicting CMA 3 for {col}...")
        suffix = col.split(".")[-1]  # Extract suffix (0,1,2,...)
        risk_col = f"risk_level.{suffix}"  # Corresponding risk level column
        risk_dist_col = f"risk_dist.{suffix}"  # Corresponding risk distance column

        # **Step 2: Train-Test Split (75% Train, 25% Test)**
        train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=False)

        # Remove NaNs introduced by lagging
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # **Step 3: Train XGBoost Model**
        train_X, train_y = train_df[feature_cols], train_df[col]
        model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=100, random_state=42
        )
        model.fit(train_X, train_y)  # Train XGBoost model

        # print(f"\n--- XGBoost Model Summary for {col} (Risk Level = 1) ---")
        # print("Feature Importance:", model.feature_importances_)

        # **Step 4: Predict Entire Test Set (Batch Mode)**
        predicted_values = pd.Series(index=test_df.index, dtype="float64")

        # Predict ALL rows at once where risk_level == "1"
        mask_risk_1 = test_df[risk_col] == "1"  # Find rows with risk_level = 1
        if mask_risk_1.sum() > 0:  # Ensure there are rows to predict
            X_risk_1 = test_df.loc[mask_risk_1, feature_cols]
            predicted_values.loc[mask_risk_1] = model.predict(X_risk_1)

        # Loop through remaining rows for other risk models
        for idx in test_df.index:
            if not mask_risk_1.loc[idx]:  # Only process if NOT risk_level == 1
                risk_level = df.loc[idx, risk_col]  # Get risk level

                if risk_level == "2":  # Momentum model
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
                            df.loc[idx - 2, col]
                            + df.loc[idx - 1, col]
                            + df.loc[idx, col]
                        ) / 3
                    else:
                        predicted_values[idx] = df.loc[
                            idx, col
                        ]  # Default to current return

        # Store results in dictionaries for later concatenation
        predicted_returns_dict[f"predicted_{col}"] = predicted_values
        actual_returns_dict[f"actual_{col}"] = df.loc[test_df.index, col]
        risk_levels_dict[f"risk_level_{col}"] = df.loc[test_df.index, risk_col]

    # **Step 5: Store & Display Predictions & Risk Levels**
    results_df = pd.concat(
        [
            pd.DataFrame(predicted_returns_dict),
            pd.DataFrame(actual_returns_dict),
            pd.DataFrame(risk_levels_dict),
        ],
        axis=1,
    )
    log.success("XGBoost Strategy 3 computation completed for all return columns!")
    return results_df
