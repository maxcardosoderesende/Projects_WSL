import pandas as pd
from loguru import logger as log


def load_trading_data(data_path="../data"):
    returns = pd.read_parquet(f"{data_path}/returns.contest.24.02.25.parquet")
    features = pd.read_parquet(f"{data_path}/features.contest.24.02.25.parquet")
    target_returns = pd.read_parquet(
        f"{data_path}/target_returns.contest.24.02.25.parquet"
    )

    log.success("All datasets loaded successfully.")
    return returns, features, target_returns
