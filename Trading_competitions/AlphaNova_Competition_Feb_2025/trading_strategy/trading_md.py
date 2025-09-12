# MD Trading strategy - Compute all strategies

import pandas as pd
from loguru import logger as log
from src.laod_data import load_trading_data
from src.risk_measure import compute_risk_measurement
from src.strategy_cma import cma_strategy_1, cma_strategy_2, cma_strategy_3, compute_mse
from src.strategy_xgboost import (
    xgboost_strategy_1,
    xgboost_strategy_2,
    xgboost_strategy_3,
)


# Patch to rename "__main__" -> "Freight_Prediction_Module"
def rename_main(record):
    if record["name"] == "__main__":
        record["name"] = "AlphaNova_Trading_Strategies"

log = log.patch(rename_main)


log.info("Starting MD trading algorithm - All strategies")

log.info("Load data")
returns, features, target_returns = load_trading_data()

# Scale returns by 100 to avoid loss functions early stops
returns_2 = returns * 100

# Compute the risk interval and classification
df, risk_distance, risk_levels = compute_risk_measurement(returns_2)

# Add risk merasures to features dataframe
features = features.join(risk_distance)
features = features.join(risk_levels)

log.info("Strategy - Conditional Modelling Aprroaxch (CMA)")
df_cma = returns_2.join(features)

df_cma_predicted_1 = cma_strategy_1(df_cma)
mse_df = compute_mse(df_cma_predicted_1)

df_cma_predicted_2 = cma_strategy_2(df_cma)
mse_df_2 = compute_mse(df_cma_predicted_2)

df_cma_predicted_3 = cma_strategy_3(df_cma, num_lags=2)
mse_df_3 = compute_mse(df_cma_predicted_3)

log.info("Strategy - XGBoost")
df_xgboost_1 = xgboost_strategy_1(df_cma)
mse_df_xgboost_1 = compute_mse(df_xgboost_1)

df_xgboost_2 = xgboost_strategy_2(df_cma)
mse_df_xgboost_2 = compute_mse(df_xgboost_2)

df_xgboost_3 = xgboost_strategy_3(df_cma, num_lags=2)
mse_df_xgboost_3 = compute_mse(df_xgboost_3)

log.info("Concatenate all MSE accuracy measures")

# Merge MSE dataframes (first three)
merged_mse_df = mse_df.merge(mse_df_2, on="Return Series", suffixes=("_1", "_2")).merge(
    mse_df_3, on="Return Series"
)

# Rename the final MSE column from mse_df_3
merged_mse_df.rename(columns={"MSE": "MSE_3"}, inplace=True)

# Merge XGBoost results
merged_mse_df = merged_mse_df.merge(
    mse_df_xgboost_1, on="Return Series", suffixes=("", "_xg_1")
).merge(mse_df_xgboost_2, on="Return Series", suffixes=("", "_xg_multi"))

# Ensure correct column renaming
merged_mse_df.rename(
    columns={"MSE": "MSE_xg_1", "MSE_xg_multi": "MSE_xg_multi"}, inplace=True
)

# Merge XGBoost with Risk level results
merged_mse_df = merged_mse_df.merge(mse_df_xgboost_3, on="Return Series")
merged_mse_df.rename(columns={"MSE": "MSE_xgb_risk"}, inplace=True)

# Calculate the average for each column (excluding "Return Series")
avg_row = merged_mse_df.drop(columns="Return Series").mean()
avg_row["Return Series"] = "Average"
merged_mse_df = pd.concat([merged_mse_df, avg_row.to_frame().T], ignore_index=True)
breakpoint()
log.info("All done!")
