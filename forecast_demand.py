import logging
from time import time

import pandas as pd

from checking_functions import check_data_correctness
from forecast_functions import get_warm_result, create_constant_forecast, get_cold_result, \
    create_empty_forecast
from om_pandas import dataframe_to_list, calculate_dates
from postprocessing import remove_outliers
from preparing_functions import create_info, prepare_save_result, prepare_fact, load_and_prepare_data, \
    prepare_detailed_preferences, prepare_causal_factors, create_quality_file, detect_methods, prepare_correlation
from service_functions import Info, Mode

# Подавим вывод ненужных сообщений от библиотек
logger1 = logging.getLogger('cmdstanpy')
logger2 = logging.getLogger('chardet')
logger3 = logging.getLogger('numba')

for log in [logger1, logger2, logger3]:
    log.addHandler(logging.NullHandler())
    log.propagate = False
    log.setLevel(logging.CRITICAL)


def prepare_for_training(info):
    dp_df = load_and_prepare_data(info, 'detailed_preferences', info.source['detailed_preferences']['group_column'],
                                  True)
    prepare_detailed_preferences(info, dp_df)

    quality = create_quality_file(info)
    fact_df = load_and_prepare_data(info, 'fact', info.source['group_by_columns'])

    return fact_df, quality


def fit_model(info, df, quality):
    """
    Выполняет подготовку данных к обучению, направляет данные в функции,
    соответствующие режиму прогнозирования

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param df: Фрейм с фактическими данными;
    :param quality: Именованный кортеж. Содержит фреймы для внесения информации о прогнозе;
    :return: Возвращает именованный кортеж. Содержит прогноз (forecast), качество и параметры моделей (quality).
    """
    if df.empty:
        message = 'Отсутствуют фактические данные'
        logging.exception(message)
        return

    # df = check_on_significant_zero(df)
    group_by_columns = info.source['group_by_columns']

    # Сгруппируем данные с созданием списка групп
    grouped_list = dataframe_to_list(df, group_by_columns, invert_columns=False)

    # Создадим датафрейм соответствующий выходному файлу
    forecasting_mode = info.forecast['mode']
    grouped_list_len = len(grouped_list)

    quality_file = quality.quality_file
    for iteration, fact in enumerate(grouped_list):
        logging.info(f'\nГруппа {iteration + 1} из {grouped_list_len}')

        group, test_part, true_periods = prepare_fact(info, fact, quality_file)

        if true_periods is None:
            create_empty_forecast(info, group_by_columns, quality_file)
            continue
        logging.info(f'Группирующие колонки: {group_by_columns}. Ключ группы: {group.key}')

        # Определим параметры моделей и граничные даты
        dates = calculate_dates(group.fact['ds'], info, group.key)
        detect_methods(info, group.key)
        data_correctness = check_data_correctness(info, group, dates, quality_file)
        if not data_correctness:
            create_empty_forecast(info, group_by_columns, quality_file)
            continue
        # проверки ряда на постоянство значений
        if group.test['y'][:-test_part].nunique() == 1 or group.test['y'][-test_part:].nunique() == 1:
            message_detailed = f'Группа {group.key}. Постоянная обучающая часть данных. '

            if group.fact['y'].nunique() == 1:
                message_detailed = f'Группа {group.key}. Постоянный весь фактический ряд. '

            fact_future_forecast = create_constant_forecast(info, group, dates, quality_file, message_detailed)

        # Теплый старт
        elif forecasting_mode == Mode.warm:
            fact_future_forecast = get_warm_result(info, group, dates, quality)

        # Холодный старт
        elif forecasting_mode == Mode.cold:
            # fact_future_forecast = get_cold_fourier_mlr_result(info, group, dates, methods, quality.quality_file)
            fact_future_forecast = get_cold_result(info, group, dates, quality_file)

        fact_future_forecast = remove_outliers(info, fact_future_forecast, group.key)

        fact_future_forecast[group_by_columns] = group.group_items
        # Удалить последние наблюдения, если они были выколоты
        fact_periods = info.source['detailed_preferences']['df']['periods_to_forecast'].values[0]
        if fact_periods != true_periods:
            fact_future_forecast = pd.concat([fact_future_forecast[:fact_future_forecast.shape[0] - fact_periods],
                                              fact_future_forecast[fact_future_forecast.shape[0] - true_periods - 1:]])

        if info.forecast['join_output_result']:
            fact_forecast = fact_future_forecast.loc[:2, :]
            future_forecast = fact_future_forecast

        elif not info.forecast['join_output_result']:
            fact_forecast = fact_future_forecast.loc[fact_future_forecast['ds'] <= dates.test_end]
            future_forecast = fact_future_forecast.loc[fact_future_forecast['ds'] > dates.test_end]

        prepare_save_result(info, group_by_columns, fact_forecast, future_forecast, quality_file)
        quality_file = pd.DataFrame(columns=quality_file.columns)


def forecast_demand(fact, **kwargs):
    """
    Функция выполняет загрузку исходных данных, прогноз и сохранение итоговых файлов

    :param kwargs: Словарь параметров, поступающих в скрипт
    """
    start_time = time()
    if not kwargs.get('fact', ''):
        kwargs['fact'] = fact
    file_info = create_info(kwargs)

    info = Info(file_info['source'], file_info['forecast'], file_info['output'])
    fact_df, quality = prepare_for_training(info)

    # Приходит один файл
    fit_model(info, fact_df, quality)
    logging.info(f'Время затраченное на обучение и прогнозирование: {time() - start_time}')
    logging.shutdown()
