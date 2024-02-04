from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from src.experiment.model.custom_metrics import f1_macro  # type: ignore


class OptWeightedEnsemble:
    """
    Example of a custom predictor that uses a weighted sum of features to make predictions.
    -------------------------------------------------------
    # 目的関数例（MSE）
    def mse(y_true, y_pred):  #  type: ignore
        return np.mean((y_true - y_pred) ** 2)

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([2, 5, 8])

    model = CustomWeightedPredictor(objective_function=mse, optimize_direction="min")
    model.fit(X, y)
    predictions = model.predict(X)
    print("Optimal weights:", model.weights)
    print("Predictions:", predictions)
    """

    def __init__(
        self,
        objective_function: Callable | str = "mse",
        optimize_direction: str = "min",
        optimize_method: str = "Nelder-Mead",
    ) -> None:
        self.weights = None

        self.optimize_direction = optimize_direction
        self.optimize_method = optimize_method
        self.objective_function = objective_function

        self.options = {"disp": True}

    def set_objective_function(self, objective_function: Callable | str) -> None:
        if objective_function == "mse":
            self.objective_function = mean_squared_error
        elif objective_function == "f1_macro":
            self.objective_function = f1_macro

    def fit(self, X, y) -> np.ndarray:  # type: ignore
        self.set_objective_function(self.objective_function)

        def objective(weights: np.ndarray) -> float:
            predictions = np.dot(X, weights)
            score = self.objective_function(y, predictions)  # type: ignore
            return score if self.optimize_direction == "min" else -score

        initial_weights = np.random.rand(X.shape[1])
        result = minimize(
            objective,
            initial_weights,
            method=self.optimize_method,
            options=self.options,
        )
        self.weights = result.x

    def predict(self, X):  # type: ignore
        if self.weights is not None:
            return np.dot(X, self.weights)  # type: ignore
        else:
            raise Exception("Model has not been fitted yet.")

    def set_params(self, **params) -> None:  # type: ignore
        """Set the parameters of this predictor.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self  # type: ignore


class SimpleAverageEnsemble:
    """
    Example of a custom predictor that uses the average of features to make predictions.
    -----------------------------------------------------------------

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])  # この例では y は使用しませんが、fit メソッドの引数として必要です

    model = SimpleAveragePredictor()
    model.fit(X, y)  # 実際には何もしません
    predictions = model.predict(X)
    print("Predictions:", predictions)

    """

    def fit(self, X, y):  # type: ignore
        pass

    def predict(self, X):  # type: ignore
        return np.mean(X, axis=1)

    def set_params(self, **params) -> None:  # type: ignore
        """Set the parameters of this predictor.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")

        return self  # type: ignore
