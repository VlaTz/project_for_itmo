import logging
import warnings
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count, Value, Process, Queue, get_context
from time import time, sleep
from typing import NoReturn

import numpy as np
import pandas as pd
import psutil

from checking_functions import check_data_correctness
from filling_functions import fill_quality_file
from forecast_functions import create_constant_forecast, get_warm_result, get_cold_result, \
    create_empty_forecast
from om_pandas import dataframe_to_list, calculate_dates
from postprocessing import remove_outliers
from preparing_functions import prepare_fact, prepare_save_result, create_df, \
    prepare_detailed_preferences, prepare_causal_factors, create_quality_file, create_info, detect_methods
from service_functions import Mode

np.seterr(all='raise')

warnings.filterwarnings('error')

Info = namedtuple('Info', ['source', 'forecast', 'output'])

# Подавим вывод ненужных сообщений от библиотек
logger1 = logging.getLogger('cmdstanpy')
logger2 = logging.getLogger('prophet')
logger3 = logging.getLogger('chardet')
logger4 = logging.getLogger('numba')

for log in [logger1, logger2, logger3, logger4]:
    log.addHandler(logging.NullHandler())
    log.propagate = False
    log.setLevel(logging.CRITICAL)

memory_info = psutil.virtual_memory()
MEMORY_USED, MEMORY_PERCENT = memory_info.used, memory_info.percent


def iterative_fit(fact_list, info, quality, group_by_columns, forecasting_mode):
    iteration, fact_list = fact_list
    logging.debug(f'Итерация {iteration + 1}, Количество пересечений в обработке - {len(fact_list)}')

    for i, fact in enumerate(fact_list):
        try:
            quality.quality_file.drop(quality.quality_file.index, inplace=True)
            group, test_part, true_periods = prepare_fact(info, fact, quality.quality_file)

            if true_periods == None:
                create_empty_forecast(info, group_by_columns, quality.quality_file)
                continue

            logging.info(group.key)
            # Определим параметры моделей и граничные даты
            dates = calculate_dates(group.fact['ds'], info, group.key)
            detect_methods(info, group.key)
            data_correctness = check_data_correctness(info, group, dates, quality.quality_file)
            if not data_correctness:
                create_empty_forecast(info, group_by_columns, quality.quality_file)
                continue

            methods = detect_methods(info, group.key)

            # проверки ряда на постоянство значений
            if group.test['y'][:-test_part].nunique() == 1 or group.test['y'][-test_part:].nunique() == 1:
                message_detailed = f'Группа {group.key}. Постоянная обучающая часть данных. '

                if group.fact['y'].nunique() == 1:
                    message_detailed = f'Группа {group.key}. Постоянный весь фактический ряд. '

                fact_future_forecast = create_constant_forecast(info, group, dates, quality.quality_file,
                                                                message_detailed)

            # Теплый старт
            elif forecasting_mode == Mode.warm:
                fact_future_forecast = get_warm_result(info, group, dates, quality)

            # Холодный старт
            elif forecasting_mode == Mode.cold:
                fact_future_forecast = get_cold_result(info, group, dates, quality.quality_file)

            fact_future_forecast = remove_outliers(info, fact_future_forecast, group.key)

            fact_future_forecast[group_by_columns] = group.group_items
            # Удалить последние наблюдения, если они были выколоты
            fact_periods = info.source['detailed_preferences']['df']['periods_to_forecast'].values[0]
            if fact_periods != true_periods:
                fact_future_forecast = pd.concat([fact_future_forecast[:fact_future_forecast.shape[0] - fact_periods],
                                                  fact_future_forecast[
                                                  fact_future_forecast.shape[0] - true_periods - 1:]])

            if info.forecast['join_output_result']:
                fact_forecast = fact_future_forecast.loc[:2, :]
                future_forecast = fact_future_forecast

            elif not info.forecast['join_output_result']:
                fact_forecast = fact_future_forecast.loc[fact_future_forecast['ds'] <= dates.test_end]
                future_forecast = fact_future_forecast.loc[fact_future_forecast['ds'] > dates.test_end]

            prepare_save_result(info, group_by_columns, fact_forecast, future_forecast, quality.quality_file)

        except Exception as e:
            if isinstance(e, ValueError) and 'Too many splits' in e.__str__():
                logging.exception(f' {e}')
                break
            logging.exception(f' {e}')
            message = 'Ошибка прогноза'
            message_detailed = (f'Непредвиденная ошибка в ходе выполнения скрипта. {e} ')
            logging.warning(message_detailed)
            fill_quality_file(quality.quality_file, list(fact.loc[0, info.source['group_by_columns']]),
                              0, 0, {}, error=message, detailed_error=message_detailed)
            create_empty_forecast(info, group_by_columns, quality.quality_file)
            continue


def track_memory_usage(interval, memory_peak, memory_percent):
    while True:
        memory_info = psutil.virtual_memory()
        temp_peak, temp_percent = memory_info.used, memory_info.percent
        if temp_peak > memory_peak.value:
            memory_peak.value = temp_peak
            memory_percent.value = temp_percent
        sleep(interval)


def calculate_cpus(grouped_list_len):
    cpus = (cpu_count() - 3 if grouped_list_len > cpu_count() - 3 else grouped_list_len)
    memory_info = psutil.virtual_memory()
    temp_usage, temp_percent = memory_info.used, memory_info.percent
    diff_temp_usage, diff_temp_percent = temp_usage - MEMORY_USED, temp_percent - MEMORY_PERCENT
    if diff_temp_percent == 0:
        cpus_calc = cpus
    else:
        cpus_calc = cpus

    if cpus_calc >= cpus:
        return cpus
    elif cpus_calc < 1:
        logging.error(f'Из-за риска падения скрипта по памяти текущая итерация прогнозирования будет завершена. '
                      f'Рекомендуем сократить допустимый размер файлов при разбиении в функции "split_files" '
                      f'или изменить входные файлы')
    else:
        logging.warning(f'Из-за риска падения скрипта по памяти количество процессов было сокращено '
                        f'с {cpus} до {cpus_calc}. Рекомендуем сократить допустимый размер файлов при разбиении '
                        f'в функции "split_files" или изменить входные файлы')
    return cpus_calc


def create_pool(cpus, info, quality, group_by_columns, forecasting_mode, fit_list):
    if cpus > 1:
        # Параллельная обработка файлов пересечений
        with get_context("spawn").Pool(cpus) as p:
            p.map(partial(iterative_fit,
                          info=info,
                          quality=quality,
                          group_by_columns=group_by_columns,
                          forecasting_mode=forecasting_mode), enumerate(fit_list))
            p.close()
            p.join()
            p.terminate()
    else:
        # Код на случай отладки скрипта в последовательном режиме
        for i, one_list in enumerate(fit_list):
            iterative_fit((i, one_list),
                          info=info,
                          quality=quality,
                          group_by_columns=group_by_columns,
                          forecasting_mode=forecasting_mode)


def prepare_for_training(info):
    start = time()

    q = Queue()
    dp_p = Process(target=create_df,
                   args=(info, q, 'detailed_preferences', info.source['detailed_preferences']['group_column'], True,),
                   name='dp')
    jobs = [dp_p]

    # Подготовим фрейм с Causal Factors при необходимости
    if info.forecast['use_causal_factors']:
        cf_p = Process(target=create_df,
                       args=(info, q, 'causal_factors', info.source['causal_factors']['group_column'], True,),
                       name='cf')
        cf_desc_p = Process(target=create_df,
                            args=(info, q, 'cf_desc', info.source['cf_desc']['features_col'],),
                            name='cf_desc')

        jobs.extend([cf_p, cf_desc_p])
    fact_p = Process(target=create_df,
                     args=(info, q, 'fact', info.source['group_by_columns'],),
                     name='fact')
    jobs.append(fact_p)

    dict_of_df = {}
    for process in jobs:
        process.start()
    for _ in range(len(jobs)):
        dict_of_df.update(q.get())
    for process in jobs:
        process.join()
    for process in jobs:
        process.close()
    q.close()
    logging.debug(f'load_dp_cf_fact_time: {time() - start}')

    prepare_detailed_preferences(info, dict_of_df['detailed_preferences'])
    if info.forecast['use_causal_factors']:
        prepare_causal_factors(info, dict_of_df['causal_factors'], dict_of_df['cf_desc'])
    quality = create_quality_file(info)

    return dict_of_df['fact'], quality


def fit_model(info, df, quality):
    """
    Выполняет подготовку данных к обучению, направляет данные в функции,
    соответствующие режиму прогнозирования

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :return: Возвращает именованный кортеж. Содержит прогноз (forecast), качество и параметры моделей (quality).
    """
    if df.empty:
        message = 'Отсутствуют фактические данные'
        logging.exception(message)
        return

    group_by_columns = info.source['group_by_columns']

    # Сгруппируем данные с созданием списка групп
    grouped_list = dataframe_to_list(df, group_by_columns, invert_columns=False)

    # Создадим датафрейм соответствующий выходному файлу
    forecasting_mode = info.forecast['mode']

    grouped_list_len = len(grouped_list)

    cpus = calculate_cpus(grouped_list_len)
    if cpus < 1:
        return NoReturn
    logging.info(f'Для решения задачи будет запущено {cpus} процессов')

    # Настроим мониторинг потребления оперативной памяти
    memory_peak = Value('d', 1)
    memory_percent = Value('d', 0.00)
    memory_process = Process(target=track_memory_usage,
                             args=(0.1, memory_peak, memory_percent))  # Check memory every 0.1 seconds
    memory_process.start()

    if grouped_list_len >= 200:
        num_in_split = int(np.ceil(grouped_list_len / cpus))
    elif grouped_list_len >= 100:
        num_in_split = 5
    else:
        num_in_split = 1

    num_of_splits = int(np.ceil(grouped_list_len / num_in_split))
    fit_list = [grouped_list[split * num_in_split: split * num_in_split + num_in_split]
                for split in range(num_of_splits)]
    fit_list = [one_list for one_list in fit_list if one_list]

    logging.info(f'Количество пересечений в текущей части данных - {grouped_list_len}')
    logging.info(f'Пересечения сгруппированы по {len(fit_list[0])}. Общее количество групп - {len(fit_list)}')
    logging.info(f'Далее будет выполнено обучение и прогнозирование')

    create_pool(cpus, info, quality, group_by_columns, forecasting_mode, fit_list)

    memory_process.terminate()  # Terminate memory tracking process
    memory_process.join()  # Wait for the memory process to finish

    logging.info(f'Пиковая нагрузка RAM: {round(memory_peak.value / (1024 ** 3), 3)} GB')
    logging.info(f'Процент от доступной RAM: {memory_percent.value} %')

    logging.info(f'Обучение и прогнозирование текущего файла завершено')


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
    logging.info(f'Время затраченное на обучение и прогнозирование данной части: {time() - start_time}')
