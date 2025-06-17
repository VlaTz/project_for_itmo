import json
from collections import namedtuple
from enum import Enum

import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

QualityFile = namedtuple('QualityFile', ['quality_file'])
Info = namedtuple('Info', ['source', 'forecast', 'output'])


class Mode(str, Enum):
    # forecasting_mode
    warm = 'warm'
    cold = 'cold'
    fitted_model = 'fitted_model'


def ts_split(sample_length, n_periods):
    n_splits = (sample_length - n_periods) // n_periods
    if n_splits > 5:
        n_splits = 5
        return TimeSeriesSplit(n_splits=n_splits, test_size=n_periods)
    elif n_splits >= 2:
        n_splits = 2
        return TimeSeriesSplit(n_splits=n_splits, test_size=n_periods)
    else:
        return TimeSeriesSplit(n_splits=2)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def tren_coeff_find(ts):
    poly = np.polynomial.Polynomial.fit(np.arange(len(ts)), ts.values, 1)
    poly = poly.convert()
    if len(poly.coef) < 2:
        trend_coeff = {'a': float(0), 'b': float(poly.coef[0])}
    else:
        trend_coeff = {'a': float(poly.coef[1]), 'b': float(poly.coef[0])}
    return trend_coeff


def get_forecast_info(info, key, get_index=False):
    dp_df = info.source['detailed_preferences']['df']
    dp_index = dp_df[dp_df['key'] == key].index[0]
    if get_index:
        return dp_df.loc[dp_index, :], dp_index
    return dp_df.loc[dp_index, :]


def calculate_feature_t_value(y, df):
    # Fit the linear regression model
    X = sm.add_constant(df)
    model = sm.OLS(y, X).fit()

    return model.tvalues


def get_pca(pca_features, n_components=None):
    principal = PCA(n_components=n_components, random_state=1)
    principal.fit(pca_features)

    return principal


def detect_time_type(user_step):
    if user_step == 'Год':
        step = 'YS'
        type = 'om_year'
    elif user_step == 'Квартал':
        step = 'QS'
        type = 'om_quarter'
    elif user_step == 'Месяц':
        step = 'MS'
        type = 'om_month'
    if user_step == 'Неделя':
        step = 'W-MON'
        type = 'om_weeks'
    elif user_step == 'День':
        step = 'D'
        type = 'om_days'

    return step, type


def find_delimiter_encoding(filename):
    """
    Автоматическое определение кодировки и разделителя

    :filename: Путь к файлу
    :returns: Строковое обозначение кодировки и разделителя

    """
    encoding = 'Windows-1251'
    delimiter = ';'

    return encoding, delimiter


def create_key(df, group_column, out_column_name='key'):
    if df.empty:
        df[out_column_name] = ''
        return df

    if len(group_column) > 1:
        df[out_column_name] = df[group_column].agg(''.join, axis=1)
        df = df.drop(group_column, axis=1)

    elif len(group_column) == 1:
        df = df.rename(columns={group_column[0]: out_column_name})

    return df
