import pandas as pd
import numpy as np
from loguru import logger as log


def compute_risk_measurement(df):
    """
    Function that extracts risk measure:
    Risk_distance = Current Return - Average Return
    """
    log.info("Extracting risk measures per Asset")
    # Compute historical mean and standard deviation for each asset
    mean_returns = df.mean()
    std_returns = df.std()

    # Compute risk interval (distance from mean)
    risk_distance = df - mean_returns

    # Rename columns from return.x to risk_dist.x
    risk_distance.columns = [
        col.replace("return", "risk_dist") for col in risk_distance.columns
    ]

    # Concatenate the risk metrics with the original dataframe
    df = pd.concat([df, risk_distance], axis=1)

    # Define risk classification (low, medium, high, very high)
    risk_levels = pd.DataFrame(index=df.index)

    for col in risk_distance.columns:
        risk_interval_1 = 0.5 * std_returns[col.replace("risk_dist", "return")]
        risk_interval_2 = 1.5 * std_returns[col.replace("risk_dist", "return")]
        risk_interval_3 = 2.5 * std_returns[col.replace("risk_dist", "return")]

        risk_levels[col.replace("risk_dist", "risk_level")] = np.select(
            [
                abs(risk_distance[col]) < risk_interval_1,  # Too comom
                (abs(risk_distance[col]) >= risk_interval_1)
                & (abs(risk_distance[col]) < risk_interval_2),  # Mommentun effect
                (abs(risk_distance[col]) >= risk_interval_2)
                & (abs(risk_distance[col]) < risk_interval_3),  # High Risk
                abs(risk_distance[col])
                >= risk_interval_3,  # Amost zero probability of occurence
            ],
            ["1", "2", "3", "4"],  # Correctly aligned with conditions
            default="Unknown",
        )

    # Concatenate risk levels with the original dataframe
    df = pd.concat([df, risk_levels], axis=1)

    return df, risk_distance, risk_levels
