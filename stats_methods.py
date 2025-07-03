# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, Lasso, Ridge,
                                  ElasticNet)
from sklearn.preprocessing import PolynomialFeatures
import logging
from sklearn.exceptions import ConvergenceWarning
from time import time
from metrics import calculate_quality
from service_functions import Mode, get_forecast_info, tren_coeff_find


def seasonal_coefficients(ts, timestamps, time_step, timestamps_res):
    mean_sales = np.mean(ts.values)
    marks_res = {}
    marks = {}
    if time_step == 'YS':
        marks = timestamps.dt.year
        marks_res = timestamps_res.year
    elif time_step == 'QS':
        marks = timestamps.dt.quarter
        marks_res = timestamps_res.quarter
    elif time_step == 'MS':
        marks = timestamps.dt.month
        marks_res = timestamps_res.month
    elif time_step == 'W' or time_step == 'W-MON':
        marks = (timestamps.dt.day - 1 + (
                timestamps - pd.to_timedelta(timestamps.dt.day - 1, unit='d')).dt.dayofweek) // 7 + 1
        marks_res = (timestamps_res.day - 1 + (
                timestamps_res - pd.to_timedelta(timestamps_res.day - 1, unit='d')).dayofweek) // 7 + 1
    elif time_step == 'D':
        marks = timestamps.dt.dayofweek
        marks_res = timestamps_res.dayofweek
    dict_seasonality = {}
    df = pd.DataFrame({'ds': marks, 'y': ts})
    for timestamp in marks.unique():
        values = df[df.ds == timestamp].y
        dict_seasonality[timestamp] = np.mean(values / mean_sales)
    for timestamp in set(marks_res.unique()) - set(marks.unique()):
        dict_seasonality[timestamp] = dict_seasonality.get(timestamp, 1)
    return dict_seasonality, marks_res

def create_x_train_pred(train_len, len_pred):
    """Create arrays for training and prediction."""
    x_train = np.arange(train_len).reshape(-1, 1)
    x_pred = np.arange(train_len + len_pred).reshape(-1, 1)
    ts_res = pd.Series(index=range(train_len + len_pred), dtype='float64')
    return ts_res, x_train, x_pred


def _apply_seasonality(ts_res, ts, timestamps, time_step, n_predict):
    """Helper function to apply seasonal coefficients."""
    timestamps_res = pd.date_range(start=timestamps[0], freq=time_step, periods=len(ts_res))
    dict_seasonality, marks_res = seasonal_coefficients(ts, timestamps, time_step, timestamps_res)
    df = pd.DataFrame({'y_pred': ts_res, 'indexes': marks_res})
    df['y_pred'] = df['y_pred'] * df['indexes'].map(dict_seasonality)
    return df['y_pred'].reset_index(drop=True)


def rol_mean(ts, n_predict, ws):
    """Rolling mean forecast."""
    n = len(ts)
    ts_res = pd.Series(index=range(n + n_predict), dtype='float64')
    ts_copy = ts_res.copy()
    ts_copy.loc[ts.index] = ts.values
    rol = ts_copy.fillna(0).rolling(ws)
    ts_res.loc[ts.index[:ws]] = ts.loc[ts.index[:ws]].values
    return ts_res.where(pd.notna, other=rol.mean())


def exp_smooth(ts, n_predict, alpha):
    """Exponential smoothing forecast."""
    n = len(ts)
    ts_res = pd.Series(index=range(n + n_predict), dtype='float64')
    fit = SimpleExpSmoothing(ts.fillna(0), initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
    ts_res.loc[:] = fit.forecast(len(ts_res)).tolist()
    return ts_res


def holt(timestamps, ts, alpha, beta, n_predict, time_step):
    """Holt's linear trend method with seasonality."""
    n = len(ts)
    ts_res = pd.Series(index=range(n + n_predict), dtype='float64')
    fit = Holt(ts.fillna(0)).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    ts_res.loc[:] = fit.forecast(len(ts_res)).tolist()
    result = _apply_seasonality(ts_res, ts, timestamps, time_step, n_predict)
    return result, {'alpha': alpha, 'beta': beta}


def holt_winters(ts, period, trend, seasonal, n_predict):
    """Holt-Winters seasonal method."""
    n = len(ts)
    try:
        ts_res = pd.Series(index=range(n + n_predict), dtype='float64')
        ts_copy = ts.copy()
        ts_copy = ts_copy.fillna(0).clip(lower=0.0001)
        fit = ExponentialSmoothing(ts_copy, seasonal_periods=period, trend=trend, seasonal=seasonal).fit()
        ts_res.loc[:] = fit.forecast(len(ts_res))[1:].tolist()
        return ts_res, {'period': period, 'trend': trend, 'seasonal': seasonal}
    except FloatingPointError as err:
        logging.debug(f'Holt-Winters error - {err}, params: seasonal={seasonal}, trend={trend}, period={period}')
        return pd.Series(np.inf, index=range(n + n_predict)), {'period': period, 'trend': trend, 'seasonal': seasonal}


def _regression_model(timestamps, ts, n_predict, time_step, model, degree=None):
    """Generic regression model helper."""
    ts_res, x_train, x_pred = create_x_train_pred(len(ts), n_predict)

    if degree:
        poly = PolynomialFeatures(degree, include_bias=False)
        x_train = poly.fit_transform(x_train)
        x_pred = poly.transform(x_pred)

    model.fit(x_train, ts.fillna(0))
    ts_res.loc[:] = model.predict(x_pred)
    return _apply_seasonality(ts_res, ts, timestamps, time_step, n_predict)


def linear_regression(timestamps, ts, n_predict, time_step):
    """Linear regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, LinearRegression())


def polynomial_regression(timestamps, ts, degree, n_predict, time_step):
    """Polynomial regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, LinearRegression(), degree)


def theil_sen_regression(timestamps, ts, degree, n_predict, time_step):
    """Theil-Sen regression forecast."""
    try:
        return _regression_model(timestamps, ts, n_predict, time_step, TheilSenRegressor(random_state=42), degree)
    except ConvergenceWarning:
        return pd.Series(-np.inf, index=range(len(ts) + n_predict))


def ransac_regression(timestamps, ts, degree, n_predict, time_step):
    """RANSAC regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, RANSACRegressor(random_state=42), degree)


def huber_regression(timestamps, ts, degree, n_predict, time_step):
    """Huber regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, HuberRegressor(), degree)


def lasso_regression(timestamps, ts, alpha, n_predict, time_step):
    """Lasso regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, Lasso(alpha=alpha) )


def ridge_regression(timestamps, ts, alpha, n_predict, time_step):
    """Ridge regression forecast."""
    return _regression_model(timestamps, ts, n_predict, time_step, Ridge(alpha=alpha))


def elastic_net(timestamps, ts, alpha, l1, n_predict, time_step):
    """ElasticNet regression forecast."""
    result = _regression_model(timestamps, ts, n_predict, time_step, ElasticNet(alpha=alpha, l1_ratio=l1))
    return result, {'alpha': alpha, 'l1_ratio': l1}


def croston_tsb(timestamps, ts, alpha, beta, n_predict, time_step):
    """Croston TSB method for intermittent demand forecasting."""
    d = ts.values
    cols = len(d)
    d = np.append(d, [np.nan] * n_predict)

    a = np.full(cols + n_predict + 1, np.nan)
    p = np.full_like(a, np.nan)
    f = np.full_like(a, np.nan)

    first_occurrence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurrence]
    p[0] = 1 / (1 + first_occurrence)
    f[0] = p[0] * a[0]

    for t in range(cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * 1 + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    a[cols + 1:] = a[cols]
    p[cols + 1:] = p[cols]
    f[cols + 1:] = f[cols]

    ts_res = pd.Series(f[1:], index=range(cols + n_predict), dtype='float64')
    timestamps_res = pd.date_range(start=timestamps[0], freq=time_step, periods=len(ts_res))
    dict_seasonality, marks_res = seasonal_coefficients(ts, timestamps, time_step, timestamps_res)
    df = pd.DataFrame({'y_pred': ts_res, 'indexes': marks_res})
    df.loc[cols + 1:, 'y_pred'] *= df.loc[cols + 1:, 'indexes'].map(dict_seasonality)

    return df['y_pred'].reset_index(drop=True), {'alpha': alpha, 'beta': beta}


def hyperbolic_regression(timestamps, ts, n_predict, time_step):
    """Hyperbolic regression forecast."""
    n = len(ts)
    x_train = np.arange(1, n + 1)
    x_pred = np.arange(1, n + n_predict + 1)

    sum_1_x = np.sum(1 / x_train)
    sum_1_x2 = np.sum(1 / x_train ** 2)
    sum_y = np.sum(ts.fillna(0))
    sum_y_x = np.sum(ts.fillna(0) / x_train)

    denominator = n * sum_1_x2 - sum_1_x ** 2
    a1 = (n * sum_y_x - sum_1_x * sum_y) / denominator
    a0 = (sum_y - a1 * sum_1_x) / n

    ts_res = pd.Series(a0 + a1 / x_pred, index=range(n + n_predict))
    return _apply_seasonality(ts_res, ts, timestamps, time_step, n_predict)


def const(ts, n_predict):
    """Constant value forecast."""
    return pd.Series(ts.mode().iloc[0], index=range(len(ts) + n_predict), dtype='float64')



def rol_mean_predict(info, key, samples, mode, dates=None):
    """
    Модель на основе скользящего среднего

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели скользящего среднего...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_window_size, max_window_size = dp_df['rol_mean_min_window_size'], dp_df['rol_mean_max_window_size']
    step = 1
    metric = dp_df['eval_metric'].get('rol_mean', 'WAPE')

    sample_index = sample.index.values.reshape(-1, 1)
    num_components = 0
    trend_coef = None

    if mode == Mode.warm:
        trend_coef = info.forecast['specified_parameters'][key]['rol_mean']['trend_coeff']
        num_components = info.forecast['specified_parameters'][key]['rol_mean']['window_size']

        # Линейное уравнение вида y = ax + b
        trend = trend_coef['a'] * sample_index + trend_coef['b']
        sample.loc[:, 'y'] = sample['y'] - trend.squeeze()

    if mode == Mode.cold:
        trend_coef = tren_coeff_find(sample.y)
        trend = trend_coef['a'] * sample_index + trend_coef['b']
        sample.loc[:, 'y'] = sample['y'] - trend.squeeze()

        quality = np.inf

        for window_size in np.arange(start=min_window_size, stop=max_window_size + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = rol_mean(train['y'], dates.n_periods, window_size)

                temp_quality = calculate_quality(
                    (valid.y[test_index] + trend.squeeze()[test_index]).reset_index(drop=True),
                    (prediction[test_index] + trend.squeeze()[test_index]).tolist(),
                    metric,
                    False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = window_size

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = rol_mean(sample['y'], dates.n_periods, num_components)
    result = prepare_result(sample, future_ds,  prediction)

    all_trend = trend_coef['a'] * result.index.values
    level = trend_coef['b']
    result.reset_index(drop=True, inplace=True)

    # Добавим тренд к прогнозу
    result['y_pred'] = np.array(result['y_pred'] + all_trend) + level

    logging.debug(f'Обучение модели скользящего среднего завершено')
    logging.debug(f'Rol_mean_time: {time() - start}')

    return result, {'rol_mean': {'window_size': num_components, 'trend_coeff': trend_coef}}, None


def exp_smoothing_predict(info, key, samples, mode, dates=None):
    """
    Модель на основе экспоненциального сглаживания

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели экспоненциального сглаживания с сезонными коэффициентами...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['exp_smoothing_min_alpha'], dp_df['exp_smoothing_max_alpha']
    step = dp_df['exp_smoothing_step']
    metric = dp_df['eval_metric'].get('exp_smoothing', 'WAPE')

    sample_index = sample.index.values.reshape(-1, 1)
    num_components = 0

    if mode == Mode.warm:
        trend_coef = info.forecast['specified_parameters'][key]['exp_smoothing']['trend_coeff']
        num_components = info.forecast['specified_parameters'][key]['exp_smoothing']['alpha']

        # Линейное уравнение вида y = ax + b
        trend = trend_coef['a'] * sample_index + trend_coef['b']
        sample.loc[:, 'y'] = sample['y'] - trend.squeeze()

    if mode == Mode.cold:
        trend_coef = tren_coeff_find(sample.y)
        trend = trend_coef['a'] * sample_index + trend_coef['b']
        sample.loc[:, 'y'] = sample['y'] - trend.squeeze()
        quality = np.inf

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = exp_smooth(train['y'], dates.n_periods, alpha)

                temp_quality = calculate_quality(
                    (valid.y[test_index] + trend.squeeze()[test_index]).reset_index(drop=True),
                    (prediction[test_index] + trend.squeeze()[test_index]).tolist(),
                    metric,
                    False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = alpha

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = exp_smooth(sample['y'], dates.n_periods, num_components)
    result = prepare_result(sample, future_ds,  prediction)

    all_trend = trend_coef['a'] * result.index.values
    level = trend_coef['b']
    result.reset_index(drop=True, inplace=True)

    # Добавим тренд к прогнозу
    result['y_pred'] = np.array(result['y_pred'] + all_trend) + level

    logging.debug(f'Обучение модели экспоненциального сглаживания с сезонными коэффициентами завершено')
    logging.debug(f'Exp_smooth_time: {time() - start}')

    return result, {'exp_smoothing': {'alpha': num_components, 'trend_coeff': trend_coef}}, None


def holt_predict(info, key, samples, mode, dates=None):
    """
    Модель Хольта

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Хольта с сезонными коэффициентами...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['holt_min_alpha'], dp_df['holt_max_alpha']
    min_beta, max_beta = dp_df['holt_min_beta'], dp_df['holt_max_beta']
    step = dp_df['holt_step']
    metric = dp_df['eval_metric'].get('holt', 'WAPE')
    if mode == Mode.warm:
        parameters = info.forecast['specified_parameters'][key]['holt']

    if mode == Mode.cold:
        quality = np.inf
        parameters = {}

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            for beta in np.arange(start=min_beta, stop=max_beta + step, step=step):
                errors = []
                for train_index, valid_index in samples.split_mode.split(sample):
                    train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                    test_index = (valid.y[-dp_df['test_part']:]).index

                    prediction, temp_parameters = holt(train['ds'], train['y'], alpha, beta, dates.n_periods,
                                                       dates.time_step)

                    temp_quality = calculate_quality(
                        (valid.y[test_index]).reset_index(drop=True),
                        (prediction[test_index]).tolist(),
                        metric,
                        False)

                    errors.append(abs(temp_quality))

                mean_error = sum(errors) / len(errors)
                if mean_error < quality:
                    quality = mean_error
                    parameters = temp_parameters

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction, parameters = holt(sample['ds'], sample['y'], parameters['alpha'], parameters['beta'],
                                  dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели Хольта с сезонными коэффициентами завершено')
    logging.debug(f'Holt_time: {time() - start}')

    return result, {'holt': parameters}, None


def holt_winters_predict(info, key, samples, mode, dates=None):
    """
    Модель Хольта-Винтерса

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Хольта-Винтерса...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_seasonality, max_seasonality = dp_df['holt_winters_min_seasonality'], dp_df['holt_winters_max_seasonality']
    trend_types, seasonal_types = dp_df['holt_winters_trend_types'], dp_df['holt_winters_seasonal_types']
    step = 1
    metric = dp_df['eval_metric'].get('holt_winters', 'WAPE')

    if mode == Mode.warm:
        parameters = info.forecast['specified_parameters'][key]['holt_winters']

    if mode == Mode.cold:
        quality = np.inf
        parameters = {}

        for period in np.arange(start=min_seasonality, stop=max_seasonality + step, step=step):
            for trend in trend_types:
                for seasonal in seasonal_types:
                    errors = []
                    for train_index, valid_index in samples.split_mode.split(sample):
                        train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                        test_index = (valid.y[-dp_df['test_part']:]).index

                        prediction, temp_parameters = holt_winters(train['y'], period, trend, seasonal, dates.n_periods)

                        temp_quality = calculate_quality(
                            (valid.y[test_index]).reset_index(drop=True),
                            (prediction[test_index]).tolist(),
                            metric,
                            False)

                        errors.append(abs(temp_quality))

                    mean_error = sum(errors) / len(errors)
                    if mean_error < quality:
                        quality = mean_error
                        parameters = temp_parameters

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction, parameters = holt_winters(sample['y'], parameters['period'], parameters['trend'],
                                          parameters['seasonal'], dates.n_periods)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели Хольта-Винтерса завершено')
    logging.debug(f'Holt_Winter_time: {time() - start}')

    return result, {'holt_winters': parameters}, None


def linear_regression_predict(info, key, samples, mode, dates=None):
    """
    Модель линейной регрессии

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели линейной регрессии...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = linear_regression(sample['ds'], sample['y'], dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели линейной регрессии завершено')
    logging.debug(f'Linear_Regression_time: {time() - start}')

    return result, {}, None


def polynomial_predict(info, key, samples, mode, dates=None):
    """
    Модель полиномиальной регрессии

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели полиномиальной регрессии...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_degrees, max_degrees = dp_df['polynomial_min_degrees'], dp_df['polynomial_max_degrees']
    step = 1
    metric = dp_df['eval_metric'].get('polynomial', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['polynomial']['degree']

    if mode == Mode.cold:
        quality = np.inf

        for degree in np.arange(start=min_degrees, stop=max_degrees + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = polynomial_regression(train['ds'], train['y'], degree, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = degree

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = polynomial_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели полиномиальной регрессии завершено')
    logging.debug(f'Polynomial_time: {time() - start}')

    return result, {'polynomial': {'degree': num_components}}, None


def theil_sen_predict(info, key, samples, mode, dates=None):
    """
    Модель регрессии Тейла – Сена

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели регрессии Тейла – Сена...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_degrees, max_degrees = dp_df['theil_sen_min_degrees'], dp_df['theil_sen_max_degrees']
    step = 1
    metric = dp_df['eval_metric'].get('theil_sen', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['theil_sen']['degree']

    if mode == Mode.cold:
        quality = np.inf

        for degree in np.arange(start=min_degrees, stop=max_degrees + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = theil_sen_regression(train['ds'], train['y'], degree, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = degree

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = theil_sen_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели регрессии Тейла – Сена завершено')
    logging.debug(f'Theil_Sen_time: {time() - start}')

    return result, {'theil_sen': {'degree': num_components}}, None


def ransac_predict(info, key, samples, mode, dates=None):
    """
    Модель регрессии консенсуса по случайной выборке (RANSAC)

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели регрессии консенсуса по случайной выборке (RANSAC)...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_degrees, max_degrees = dp_df['ransac_min_degrees'], dp_df['ransac_max_degrees']
    step = 1
    metric = dp_df['eval_metric'].get('ransac', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['ransac']['degree']

    if mode == Mode.cold:
        quality = np.inf

        for degree in np.arange(start=min_degrees, stop=max_degrees + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = ransac_regression(train['ds'], train['y'], degree, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = degree

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = ransac_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели регрессии консенсуса по случайной выборке (RANSAC) завершено')
    logging.debug(f'ransac_time: {time() - start}')

    return result, {'ransac': {'degree': num_components}}, None


def huber_predict(info, key, samples, mode, dates=None):
    """
    Модель регрессии Хубера

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели регрессии Хубера...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_degrees, max_degrees = dp_df['huber_min_degrees'], dp_df['huber_max_degrees']
    step = 1
    metric = dp_df['eval_metric'].get('huber', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['huber']['degree']

    if mode == Mode.cold:
        quality = np.inf

        for degree in np.arange(start=min_degrees, stop=max_degrees + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = huber_regression(train['ds'], train['y'], degree, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = degree

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = huber_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели регрессии Хубера завершено')
    logging.debug(f'Huber_time: {time() - start}')

    return result, {'huber': {'degree': num_components}}, None


def lasso_predict(info, key, samples, mode, dates=None):
    """
    Модель Лассо регрессии

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Лассо регрессии...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['lasso_min_alpha'], dp_df['lasso_max_alpha']
    step = dp_df['lasso_step']
    metric = dp_df['eval_metric'].get('lasso', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['lasso']['alpha']

    if mode == Mode.cold:
        quality = np.inf

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = lasso_regression(train['ds'], train['y'], alpha, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = alpha

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = lasso_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели Лассо регрессии завершено')
    logging.debug(f'lasso_time: {time() - start}')

    return result, {'lasso': {'alpha': num_components}}, None


def ridge_predict(info, key, samples, mode, dates=None):
    """
    Модель Риджа регрессии

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Риджа регрессии...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()

    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['ridge_min_alpha'], dp_df['ridge_max_alpha']
    step = dp_df['ridge_step']
    metric = dp_df['eval_metric'].get('ridge', 'WAPE')
    num_components = 0
    if mode == Mode.warm:
        num_components = info.forecast['specified_parameters'][key]['ridge']['alpha']

    if mode == Mode.cold:
        quality = np.inf

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            errors = []
            for train_index, valid_index in samples.split_mode.split(sample):
                train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                test_index = (valid.y[-dp_df['test_part']:]).index

                prediction = ridge_regression(train['ds'], train['y'], alpha, dates.n_periods, dates.time_step)

                temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                 (prediction[test_index]).tolist(), metric, False)

                errors.append(abs(temp_quality))

            mean_error = sum(errors) / len(errors)
            if mean_error < quality:
                quality = mean_error
                num_components = alpha

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = ridge_regression(sample['ds'], sample['y'], num_components, dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели Риджа регрессии завершено')
    logging.debug(f'ridge_time: {time() - start}')

    return result, {'ridge': {'alpha': num_components}}, None


def elastic_net_predict(info, key, samples, mode, dates=None):
    """
    Модель Elastic Net

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Elastic Net...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['elastic_net_min_alpha'], dp_df['elastic_net_max_alpha']
    min_l1, max_l1 = dp_df['elastic_net_min_l1'], dp_df['elastic_net_max_l1']
    step = dp_df['elastic_net_step']
    metric = dp_df['eval_metric'].get('elastic_net', 'WAPE')

    if mode == Mode.warm:
        parameters = info.forecast['specified_parameters'][key]['elastic_net']

    if mode == Mode.cold:
        quality = np.inf
        parameters = {}

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            for l1 in np.arange(start=min_l1, stop=max_l1 + step, step=step):
                errors = []
                for train_index, valid_index in samples.split_mode.split(sample):
                    train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                    test_index = (valid.y[-dp_df['test_part']:]).index

                    prediction, temp_parameters = elastic_net(train['ds'], train['y'], alpha, l1, dates.n_periods,
                                                              dates.time_step)

                    temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                     (prediction[test_index]).tolist(), metric, False)

                    errors.append(abs(temp_quality))

                mean_error = sum(errors) / len(errors)
                if mean_error < quality:
                    quality = mean_error
                    parameters = temp_parameters

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction, parameters = elastic_net(sample['ds'], sample['y'], parameters['alpha'],
                                         parameters['l1_ratio'], dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds, prediction)

    logging.debug(f'Обучение модели Elastic Net завершено')
    logging.debug(f'elastic_net_time: {time() - start}')

    return result, {'elastic_net': parameters}, None


def croston_tsb_predict(info, key, samples, mode, dates=None):
    """
    Модель Кростона TSB

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели Кростона TSB...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()
    dp_df = get_forecast_info(info, key)

    min_alpha, max_alpha = dp_df['croston_tsb_min_alpha'], dp_df['croston_tsb_max_alpha']
    min_beta, max_beta = dp_df['croston_tsb_min_beta'], dp_df['croston_tsb_max_beta']
    step = dp_df['croston_tsb_step']
    metric = dp_df['eval_metric'].get('croston_tsb', 'WAPE')
    temp_parameters = {}

    if mode == Mode.warm:
        parameters = info.forecast['specified_parameters'][key]['croston_tsb']

    if mode == Mode.cold:
        quality = np.inf
        parameters = {}

        for alpha in np.arange(start=min_alpha, stop=max_alpha + step, step=step):
            for beta in np.arange(start=min_beta, stop=max_beta + step, step=step):
                errors = []
                for train_index, valid_index in samples.split_mode.split(sample):
                    train, valid = sample.iloc[train_index], sample.iloc[valid_index]

                    test_index = (valid.y[-dp_df['test_part']:]).index

                    prediction, temp_parameters = croston_tsb(train['ds'], train['y'], alpha, beta, dates.n_periods,
                                                              dates.time_step)

                    temp_quality = calculate_quality((valid.y[test_index]).reset_index(drop=True),
                                                     (prediction[test_index]).tolist(), metric, False)

                    errors.append(abs(temp_quality))

                mean_error = sum(errors) / len(errors)
                if mean_error < quality:
                    quality = mean_error
                    parameters = temp_parameters

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction, fourier_parameters = croston_tsb(sample['ds'], sample['y'], parameters['alpha'],
                                                 parameters['beta'], dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели Кростона TSB завершено')
    logging.debug(f'croston_tsb_time: {time() - start}')

    return result, {'croston_tsb': parameters}, None


def hyperbolic_predict(info, key, samples, mode, dates=None):
    """
    Модель гиперболической регрессии

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param key: Ключ пересечения;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели гиперболической регрессии...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = hyperbolic_regression(sample['ds'], sample['y'], dates.n_periods, dates.time_step)
    result = prepare_result(sample, future_ds,  prediction)

    logging.debug(f'Обучение модели гиперболической регрессии завершено')
    logging.debug(f'hyperbolic_time: {time() - start}')

    return result, {}, None


def const_predict(info, key, samples, mode, dates=None):
    """
    Модель константного прогноза (мода)

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param samples: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param mode: Режим прогнозирования;
    :return: Прогнозные значения и параметры модели
    """
    logging.debug(f'Обучение модели константного прогноза (мода)...')
    start = time()

    sample = samples.train[['ds', 'y']].copy()

    future_ds = pd.date_range(start=max(sample['ds']),
                              freq=info.source['fact']['time_column']['step'],
                              periods=dates.n_periods + 1)[1:]

    prediction = const(sample['y'], dates.n_periods)
    result = prepare_result(sample, future_ds,  prediction)


    logging.debug(f'Обучение модели константного прогноза (мода) завершено')
    logging.debug(f'const_time: {time() - start}')

    return result, {}, None

def prepare_result(sample, future_ds,  prediction):
    result = pd.DataFrame({'ds': np.append(sample['ds'], future_ds), 'y_pred': prediction})
    result = result[result['ds'].isin(np.append(sample['ds'], future_ds))]
    return result