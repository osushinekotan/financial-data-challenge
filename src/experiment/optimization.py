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


def decode_v2(
    y_pred: np.ndarray,
    pos_ratio: float,
) -> np.ndarray:
    num_positive = int(len(y_pred) * pos_ratio)
    pred_label_idx = np.argsort(y_pred)[::-1][:num_positive]
    y_pred_label = np.zeros_like(y_pred)
    y_pred_label[pred_label_idx] = 1
    return y_pred_label


def opt_macro_f1_score_v2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    n = len(y_pred)
    num_positive = np.sum(y_true)
    pos_ratio = num_positive / n

    y_pred_label = decode_v2(y_pred, pos_ratio)
    score = f1_score(y_true, y_pred_label, average="macro")

    return {"num_all": int(n), "num_positive": int(num_positive), "pos_ratio": pos_ratio, "score": score}
