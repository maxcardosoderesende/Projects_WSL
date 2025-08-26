import warnings
from abc import ABC, abstractmethod

import pandas as pd

warnings.filterwarnings("ignore")


class Predictor(ABC):
    """Base class for all predictors in the competition.

    All submissions must inherit from this class and implement the required methods.
    """

    @abstractmethod
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the predictor on the given data.

        Args:
            features: Feature values
            target: Target returns
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the given data.

        Args:
            features: Feature values

        Returns:
            DataFrame with predictions, same shape as target returns
        """
        pass
