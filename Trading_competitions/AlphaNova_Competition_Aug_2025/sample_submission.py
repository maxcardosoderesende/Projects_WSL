# /// script
# dependencies = [
#   "xgboost",
# ]
# ///

# You only need to keep dependencies (see above), imports and the implementation of the Predictor class in the submission.
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor

warnings.filterwarnings("ignore")


class XGBoostPredictor(Predictor):
    def __init__(self):
        self.model = None
        self.asset_list = []
        self.feature_names = []
        self.trained = False
        self.params = {
            "max_depth": 5,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "verbosity": 0,
        }

    # ----------------------------------------------------------------------
    def cross_sectional_corr_objective(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        """
        Custom XGBoost objective: maximize cross-sectional correlation between predictions
        and targets (no turnover penalty).
        """
        y_true = dtrain.get_label()

        n_assets = len(self.asset_list)

        # Reshape to (T, N)
        y_true = y_true.reshape(-1, n_assets)
        y_pred = y_pred.reshape(-1, n_assets)

        # Cross-sectional centering
        r_centered = y_true - y_true.mean(axis=1, keepdims=True)

        # Gradient = negative centered returns (maximize correlation)
        grad = -r_centered.ravel()

        # Hessian = constant
        hess = np.ones_like(grad)

        return grad, hess

    # ----------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train XGBoost model with custom cross-sectional correlation objective.
        """
        # Transform features
        self.asset_list = features.columns.levels[1].tolist()
        self.feature_names = features.columns.levels[0].tolist()

        # Prepare stacked data
        X = features.stack(level=1)
        y = target.stack()
        mask = ~y.isna()

        X = X.loc[mask]
        y = y.loc[mask]

        # Convert to XGBoost DMatrix
        dtrain = xgb.DMatrix(X.values, label=y.values)

        # Train with custom objective
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 50),
            obj=self.cross_sectional_corr_objective,
        )

        self.trained = True

    # ----------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-sectionally z-scored predictions for each timestamp.
        """
        assert self.trained, "Model must be trained before predicting."

        # Transform features
        index = features.index
        X = features.stack(level=1)

        # Predict
        preds = self.model.predict(xgb.DMatrix(X.values))

        # Reshape to (time, asset)
        pred_matrix = pd.DataFrame(preds.reshape(-1, len(self.asset_list)), index=index, columns=self.asset_list)

        # Cross-sectional z-score at each timestamp

        mean_vals = pred_matrix.mean(axis=1)
        std_vals = pred_matrix.std(axis=1).replace(0, 1)  # avoid division by zero
        pred_matrix = pred_matrix.sub(mean_vals, axis=0).div(std_vals, axis=0).fillna(0.0)
        pred_matrix = pred_matrix.replace([np.inf, -np.inf], 0.0)

        return pred_matrix


# You don't really need the code below in the submission.
# ## Training Example
#
# Below is an example of training and evaluating a predictor.
from data_loader import load_data, split_data
from evaluation import backtest, returns_to_equity, sharpe_ratio

# Load competition data
returns, features, target_returns = load_data("data")


# ## Train/Validation Split
#
# The split is performed automatically by the data loader to ensure consistency across all submissions.

# Split data into train and validation sets
# IMPORTANT: Always use test_size=0.25 as specified
train_data, validate_data = split_data(returns, features, target_returns, test_size=0.25)

# Create and train the example predictor
predictor = XGBoostPredictor()
predictor.train(train_data["features"], train_data["target"])


# Make predictions on training data
train_predictions = predictor.predict(train_data["features"])


# Calculate training Sharpe
sharpe = sharpe_ratio(train_predictions, train_data["returns"])
print(sharpe)
print(f"Training Sharpe: {sharpe:.20f}")


# ### Backtest Visualization (Optional)

# Visualize backtest results (optional, not used for scoring)
pf_returns = backtest(train_predictions, train_data["returns"])
# pf_returns=backtest(train_predictions, train_data['returns'])
returns_to_equity(pf_returns).plot(title="Training Backtest")


# ## Validation Results

# Make predictions on validation data
validate_predictions = predictor.predict(validate_data["features"])


# Calculate validation Sharpe
sharpe = sharpe_ratio(validate_predictions, validate_data["returns"])
print(sharpe)
print(f"Validation Sharpe: {sharpe:.20f}")


# ### Validation Backtest (Optional)

# Visualize validation backtest (optional)
validate_pf_returns = backtest(validate_predictions, validate_data["returns"])
returns_to_equity(validate_pf_returns).plot(title="Validation Backtest")
