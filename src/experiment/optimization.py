# type: ignore
from typing import Callable

import joblib
import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Callable,
    n_jobs: int = -1,
    maximize: bool = True,
) -> float:
    search_vals = list(np.arange(0, 1, 0.001))

    def _calc(th: float) -> float:
        score = metrics(y_true, y_pred >= th)
        if not maximize:
            return score * -1
        return score

    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_calc)(th) for th in search_vals)
    max_idx = np.argmax(results)
    return search_vals[max_idx]


def opt_macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    macro_f1 = lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")  # noqa: E731
    th = optimize_threshold(y_true, y_pred, metrics=macro_f1)
    return {"th": th, "score": macro_f1(y_true, y_pred >= th)}
