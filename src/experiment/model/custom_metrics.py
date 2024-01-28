# type: ignore
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def lgb_macro_auc(y_true, y_pred):
    auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class="ovr")
    return "macro_auc", auc, True


def lgb_macro_f1(y_true, y_pred):
    f1 = f1_score(y_true=y_true, y_pred=y_pred >= 0.5, average="macro")
    return "macro_f1", f1, True


def xgb_macro_f1(y_true, y_pred):
    f1 = f1_score(y_true=y_true, y_pred=y_pred >= 0.5, average="macro")
    return f1
