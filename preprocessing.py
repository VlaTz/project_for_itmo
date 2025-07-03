#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import logging
from collections import namedtuple
from time import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from service_functions import get_forecast_info, ts_split, Mode, calculate_feature_t_value, get_pca


def generate_features(info,
                      fact,
                      dates,
                      key,
                      mode='cold'):
    """
    Формирует обучающие и тестовые выборки

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param fact: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param key: Ключ, определяющий пересечение;
    :param mode: Режим прогнозирования;
    :return: Именованный кортеж. Содержит обучающие и тестовые выборки, а также даты
    """
    start = time()
    sample = fact.copy()
    sample_end = max(sample['ds'])
    dp_df = get_forecast_info(info, key)

    # Увеличим выборку на прогнозируемые даты
    forecast_range = pd.date_range(start=sample_end,
                                   freq=info.source['fact']['time_column']['step'],
                                   periods=dates.n_periods + 1)

    forecast_df = pd.DataFrame({'ds': forecast_range,
                                'y': 0})

    sample = pd.concat([sample, forecast_df[1:]]).reset_index(drop=True)

    train = sample[:-dates.n_periods]  # Обучающая выборка
    # Тестовая выборка включает обучающую часть, для получения прогноза по фактическому периоду
    test = sample
    features = [col for col in train.columns if col not in ['key', 'ds', 'y']]  # Названия колонок с признаками

    if mode == Mode.warm:
        Samples = namedtuple('Samples', ['fact', 'train', 'test', 'features'])
        samples = Samples(fact, train, test, features)

    elif mode == Mode.cold:
        # Подготовим split_mode
        split_mode = ts_split(len(train), dp_df['test_part'])
        # Проверка ряда на постоянство значений. Если ряд постоянен, то вносится небольшое изменение.
        # Необходимо для метода CatBoost
        for train_index, test_index in split_mode.split(train):
            if np.var(train.iloc[train_index]['y']) == 0:
                train.loc[min(train_index), 'y'] = train.iloc[min(train_index)]['y'] + 0.001
            for name in features:
                if np.var(train.iloc[train_index][name]) == 0:
                    train.loc[min(train_index), name] = train.iloc[min(train_index)][name] + 0.001
        Samples = namedtuple('Samples', ['fact', 'train', 'test', 'features', 'split_mode'])
        samples = Samples(fact, train, test, features, split_mode)

    logging.debug(f'generate_features_time: {time() - start}')

    return samples


def generate_features_date(info,
                           sample_data,
                           key,
                           date_column='ds'):
    """
    Генерирует дополнительные признаки временного ряда из даты

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample_data: Данные для прогнозирования;
    :param key: Ключ пересечения
    :param date_column: Название колонки с датой
    :return: Датафрейм с дополнительными признаками
    """
    sample = sample_data.copy()
    dp_df = get_forecast_info(info, key)

    # Генерация признаков месяц года и день недели в dummy-режиме
    if dp_df['use_month_of_year']:
        sample = sample.join(pd.get_dummies(sample[date_column].dt.month_name(), dtype=int, drop_first=True))
    if dp_df['use_day_of_week']:
        sample = sample.join(pd.get_dummies(sample[date_column].dt.day_name(), dtype=int, drop_first=True))

    return sample


def generate_features_target_default(sample_data,
                                     dates,
                                     amount_of_pruning):
    """
    Генерирует дополнительные признаки временного ряда из целевого признака

    :param sample_data: Данные для прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param amount_of_pruning: Количество наблюдений, которые можно удалить из выборки;
    :return: Датафрейм с дополнительными признаками
    """
    sample = {}
    # Выполним дифференцирование ряда, добавим сдвиги и скользящее среднее
    for shift in range(dates.n_periods, amount_of_pruning + 1):
        sample[f'shift_{shift}'] = sample_data['y'].shift(shift)

        for period in range(1, amount_of_pruning - shift + 1):
            sample[f'shift_{shift}_diff_{period}'] = sample_data['y'].shift(shift).diff(periods=period)

            if period != 1:
                sample[f'shift_{shift}_mean_{period}'] = sample_data['y'].shift(shift).rolling(period).mean()

    sample = pd.concat([sample_data, pd.DataFrame(sample)], axis=1)

    sample = sample.dropna().reset_index(drop=True)

    return sample


def generate_defined_features(info,
                              sample_data,
                              key,
                              dates):
    """
    Генерация статистически значимых признаков

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample_data: Данные для прогнозирования;
    :param key: Ключ пересечения
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :return:
    """
    start = time()

    gc = info.source['global_correlation']
    dp_df = get_forecast_info(info, key)
    sample = sample_data.copy()
    sample_columns = sample.columns
    amount_of_pruning = dp_df['amount_of_pruning']

    corr_df = gc['corr_df'].copy()
    corr_df = corr_df.loc[corr_df['key'] == key]
    for i, feature_line in corr_df.iterrows():
        top = []
        if feature_line['cf'] not in ['week', 'month', 'year', 'relative_time']:
            top = [x.split('_') for x in json.loads(feature_line['top_features']).keys()]
            if feature_line['cf'] == 'shift':
                top = [[int(temp_shift), 0] for _, temp_shift in top if
                       dates.n_periods <= int(temp_shift) < amount_of_pruning]
                name = "f'shift_{shift}'"
                method = 'sample["y"].shift(shift)'

            elif feature_line['cf'] == 'shift_diff' or feature_line['cf'] == 'shift_mean':
                top = [[int(shift), int(value)] for _, shift, _, value in top
                       if int(shift) >= dates.n_periods
                       and int(shift) + int(value) < amount_of_pruning]

                if feature_line['cf'] == 'shift_diff':
                    name = "f'shift_{shift}_diff_{value}'"
                    method = 'sample["y"].shift(shift).diff(periods=value)'

                elif feature_line['cf'] == 'shift_mean':
                    name = "f'shift_{shift}_mean_{value}'"
                    method = 'sample["y"].shift(shift).rolling(value).mean()'

        elif feature_line['cf'] in ['week', 'month', 'year', 'relative_time']:
            top = [[0, 0]]
            if feature_line['cf'] == 'year':
                name = "f'ds_year'"
                method = 'sample.loc[:, "ds"].dt.year'
            elif feature_line['cf'] == 'month':
                name = "f'ds_month'"
                method = 'sample.loc[:, "ds"].dt.month'
            elif feature_line['cf'] == 'week':
                name = "f'ds_week'"
                method = 'sample.loc[:, "ds"].dt.isocalendar().week.astype("int64")'
            elif feature_line['cf'] == 'relative_time':
                name = "f'ds_relative_time'"
                method = 'sample.index'

        if top:
            for shift, value in top:
                if eval(name) not in sample_columns:
                    sample[eval(name)] = eval(method)

    sample = sample.dropna().reset_index(drop=True)
    logging.debug(f'Gen time: {time() - start}')
    logging.debug(f'Sample cols: {sample.columns}')

    return sample


def generate_features_causal(info,
                             sample,
                             key):
    """
    Генерация Causal Features

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample: Данные для прогнозирования;
    :param key: Ключ пересечения
    :return:
    """
    cf = info.source['causal_factors']
    if cf['df'].empty:
        return sample

    if info.forecast['two_dates_for_cf']:
        causal_df = cf['df'].loc[cf['df']['key'] == key].pivot_table(index=['key', 'ds'], columns='features_col',
                                                                     values=cf['num_column'], aggfunc='sum')
        causal_df.columns = causal_df.columns.droplevel(0)
        causal_df.reset_index(inplace=True)
    else:
        causal_df = cf['df'].loc[cf['df']['key'] == key]
        causal_df.reset_index(inplace=True, drop=True)

    sample['key'] = key
    sample = pd.merge(sample, causal_df, how='left', left_on=['ds', 'key'], right_on=['ds', 'key'])

    # В фрейме sample могут образоваться пропуски в данных.
    # Вероятная причина: в Causal Factors значения заполнены не для всех дат.
    # Пропуски заполним нулями из предположения, что CF не действуют в данные моменты времени.
    sample = sample.fillna(0)
    sample = sample.drop(['key'], axis=1)

    return sample


def select_features(info, sample, key):
    dp_df = get_forecast_info(info, key)
    selection_strategy = dp_df['selection_strategy']
    cf = info.source['causal_factors']

    # Для одного и того же исходного ряда может выполняться несколько итераций генерации признаков.
    # На каждой итерации признаки должны быть одинаковыми, поэтому они хранятся в кортеже info.
    if 'empty' not in cf['list_of_columns'].get(key, ['empty']):
        if selection_strategy == 'PCA':
            sample = select_features_pca(info, sample, key, fitted=True)
        else:
            sample = sample[cf['list_of_columns'][key]]
    else:
        unique_cf = [col for col in sample.columns if col not in ['key', 'ds', 'y']]

        if selection_strategy in ['Backward', 'Forward', 'Use all']:
            sample = select_features_t_stat(info, sample, key, unique_cf)
        elif selection_strategy == 'k_best':
            sample = select_features_k_best(info, sample, key, unique_cf)
        elif selection_strategy == 'PCA':
            sample = select_features_pca(info, sample, key, unique_cf)
        else:
            logging.warning(f'Группа {key}. Выбрана некорректная стратегия отбора признаков - {selection_strategy}. '
                            'Доступные стратегии отбора: ["Use all", "Backward", "Forward", "k-Best", "PCA"].'
                            'Будут использованы все признаки.')
        cf['list_of_columns'][key] = sample.columns
    return sample


def select_features_t_stat(info, sample, key, unique_cf):
    """
    Отбор признаков по t-статистике

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample: Фрейм с данными;
    :param unique_cf: Список уникальных признаков;
    :return: Возвращает фрейм с данными.
    """
    start = time()
    dp_df = get_forecast_info(info, key)
    selection_strategy = dp_df['selection_strategy']
    y = np.array(sample['y'])

    if selection_strategy == 'Backward':
        logging.debug('Стратегия отбора признаков "Backward"')
        temp_cf = unique_cf.copy()
        f_threshold = dp_df['f_to_leave']
    elif selection_strategy == 'Forward':
        logging.debug('Стратегия отбора признаков "Forward"')
        temp_cf = []
        f_threshold = dp_df['f_to_enter']
    else:
        logging.debug('Стратегия отбора признаков "Use all"')
        temp_cf = unique_cf

    flag = 1  # Был ли удален признак на предыдущей итерации. Для пропуска лишних пересчетов t_value. 0 - не удалялся
    if selection_strategy in ['Backward', 'Forward']:
        # Определим фрейм "Force include" - fi_df
        fi_df = info.source['cf_force_include']['df']
        if fi_df.empty:
            fi_cf = []
        else:
            fi_df = fi_df[fi_df['key'] == key]
            fi_cf = list(fi_df['cf'])

        for cf in unique_cf:
            if selection_strategy == 'Forward':
                temp_cf.append(cf)

            # Если в файле "cf_force_include" информации по данному CF не обнаружено, то force_include = 0
            if cf not in fi_cf:
                force_value = 0
            else:
                force_value = fi_df.at[fi_df[fi_df['cf'] == cf].index[0], 'force_include']

            if not force_value:
                if selection_strategy == 'Backward' and flag == 0:
                    pass
                else:
                    # Save the t-test results
                    # logging.debug(f'количество признаков в выборке: {len(temp_cf)}')
                    t_values = calculate_feature_t_value(y, sample[temp_cf])

                if t_values[cf] ** 2 <= f_threshold:
                    flag = 1
                    sample.drop(columns=cf, inplace=True)
                    temp_cf.remove(cf)
                else:
                    flag = 0

            # Данный признак должен быть включен, стратегия "Forward" и признака еще нет в списке "temp_cf"
            elif selection_strategy == 'Forward' and cf not in temp_cf:
                temp_cf.append(cf)

    # Сделаем итоговую оценку, которая попадет в результаты
    t_values = calculate_feature_t_value(y, sample[temp_cf])
    dp_df = info.source['detailed_preferences']['df']
    dp_df.loc[dp_df[dp_df['key'] == key].index[0], ['t_values']] = [dict(t_values)]
    logging.debug(f't_stat_time: {time() - start}')
    return sample


def select_features_k_best(info, sample, key, unique_cf):
    k = round(len(sample[unique_cf].columns) / 3)
    selection_model = SelectKBest(score_func=mutual_info_regression, k=k)
    selection_model.fit(sample[unique_cf], sample[['y']])
    best_features = selection_model.get_feature_names_out()
    return sample[np.append(['ds', 'y'], best_features)]


def select_features_pca(info, sample, key, unique_cf=[],
                        fitted=False):
    """
    Отбор признаков методом главных компонент (Principal Component Analysis, PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    Рекомендуется использовать после масштабирования данных. Исходные названия признаков не сохранятся.
    Подробнее: https://scikit-learn.org/stable/modules/decomposition.html#pca

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample: Фрейм с данными;
    :param key: Ключ группы;
    :param unique_cf: Список уникальных признаков;
    :param fitted: Флаг - есть ли уже обученный анализатор для данной группы;
    :return: Возвращает фрейм с данными.
    """
    if fitted:
        cf = info.source['causal_factors']['pca'][key]
        model = cf['model']
        columns = cf['columns']
        unique_cf = cf['source_features']

    else:
        # Вычислим минимальное количество основных компонент, необходимых для сохранения 95% объясненной дисперсии
        model = get_pca(sample[unique_cf])
        cum_sum = np.cumsum(model.explained_variance_ratio_)
        d = np.argmax(cum_sum >= 0.95) + 1

        model = get_pca(sample[unique_cf], d)
        columns = [f'component_{col_number}' for col_number in range(d)]
        info.source['causal_factors']['pca'] = {key: {'model': model,
                                                      'columns': columns,
                                                      'source_features': unique_cf,
                                                      }
                                                }

    transformed_features = model.transform(sample[unique_cf])
    sample[columns] = transformed_features

    return sample[['ds', 'y'] + columns]


def scale_features(info, sample, key):
    dp_df = get_forecast_info(info, key)
    scaling_strategy = dp_df['scaling_strategy_features']

    unique_cf = [col for col in sample.columns if col not in ['key', 'ds', 'y']]
    parameters = dp_df['scaling_parameters_features']
    if scaling_strategy == "Standard":
        model = StandardScaler(**parameters)
    elif scaling_strategy == "MinMax":
        model = MinMaxScaler(**parameters)
    elif scaling_strategy == "Power Transform":
        model = PowerTransformer(**parameters)
    else:
        logging.warning(
            f'Группа {key}. Выбрана некорректная стратегия масштабирования признаков - {scaling_strategy}'
            '. Доступные стратегии: ["Standard", "MinMax", "Power Transform", "Don\'t scale"].'
            'Данные масштабироваться не будут.')
        return sample
    model.fit(sample[unique_cf])

    transformed_features = model.transform(sample[unique_cf])
    sample[unique_cf] = transformed_features
    return sample


def scale_target(info, target: pd.Series, key, unscale=False):
    dp_df = get_forecast_info(info, key)
    scaling_strategy = dp_df['scaling_strategy_target']
    cf = info.source['causal_factors']['scaling_target']
    y = np.asarray(target).reshape(-1, 1)

    if unscale:
        model = cf[key]['model']
        # unique_cf = cf[key]['unique_cf']
        transformed_target = model.inverse_transform(y, copy=True)
    elif 'empty' not in cf.get(key, ['empty']):
        model = cf[key]['model']
        transformed_target = model.transform(y)
    else:
        parameters = dp_df['scaling_parameters_target']
        if scaling_strategy == "Standard":
            model = StandardScaler(**parameters)
        elif scaling_strategy == "MinMax":
            model = MinMaxScaler(**parameters)
        elif scaling_strategy == "Power Transform":
            model = PowerTransformer(**parameters)
        else:
            logging.warning(f'Группа {key}. Выбрана некорректная стратегия масштабирования целевого признака - '
                            f'{scaling_strategy}'
                            '. Доступные стратегии: ["Standard", "MinMax", "Power Transform", "Don\'t scale"].'
                            'Данные масштабироваться не будут.')
            return y
        model.fit(y)
        cf[key] = {'model': model}
        transformed_target = model.transform(y)
    return transformed_target
