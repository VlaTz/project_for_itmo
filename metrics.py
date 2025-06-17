import numpy as np
from sklearn.metrics import make_scorer
from statsmodels.stats.stattools import durbin_watson


def calculate_quality(y_true, y_pred, metric='WAPE', reset=True, time_weight_coef=None,
                      calculate_durbin_watson=False):
    quality = np.inf
    if reset:
        y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)

    if time_weight_coef is None:
        time_weight_coef = 1
    ramp_size = len(y_true)
    time_weightings = np.logspace(ramp_size, 0, base=time_weight_coef, num=ramp_size)

    if metric == 'MAPE':
        quality = np.mean(((y_pred - y_true) / (1 if np.abs(y_true).sum() == 0 else np.abs(y_true))) * time_weightings)
    elif metric == 'WAPE':
        quality = (abs(y_true - y_pred) * time_weightings).sum() / (time_weightings.sum() if y_true.sum() == 0
                                                                    else (y_true * time_weightings).sum())

    if calculate_durbin_watson:
        try:
            dw = durbin_watson(y_pred - y_true)
            return round(quality, 7), round(dw, 7)
        except FloatingPointError:
            return round(quality, 7), -1

    return round(quality, 7)
