#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging


def remove_outliers(info, df, key):
    """
    Удаление выбросов. Допустимый интервал контролируется параметром "outlier_threshold" - задается пользователем.
    Выбросы заменяются на пороговые значения.

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param df: Фрейм с прогнозом;
    :return: Скорректированный прогноз.
    """
    dp_df = info.source['detailed_preferences']['df']
    dp_index = dp_df[dp_df['key'] == key].index[0]
    outlier_threshold = dp_df.at[dp_index, 'outlier_threshold']

    horizon = dp_df.at[dp_index, 'periods_to_forecast']
    col_index = df.columns.get_loc('y_pred')

    # calculated summary statistics for history
    data_mean = dp_df.at[dp_index, 'mean']
    data_std = dp_df.at[dp_index, 'std']

    # identify outliers borders
    outlier_band = outlier_threshold * data_std
    lower, upper = data_mean - outlier_band, data_mean + outlier_band

    # identify outliers in the forecast
    outliers = [x for x in df.iloc[-horizon:, col_index] if x < lower or x > upper]
    logging.debug('Identified outliers: %d' % len(outliers))

    upper_condition, low_condition = (df.iloc[-horizon:, col_index] > upper), (df.iloc[-horizon:, col_index] < lower)
    df.iloc[-horizon:, col_index].loc[upper_condition] = upper
    df.iloc[-horizon:, col_index].loc[low_condition] = lower

    return df
