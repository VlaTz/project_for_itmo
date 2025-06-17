import logging
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd

from filling_functions import fill_quality_file


def check_file_extension(path):
    extension = os.path.splitext(path)[-1].lower()

    if extension == '.csv':
        right_path = path

    elif extension == '.zip':
        path_without_extension = "".join(os.path.splitext(path)[:-1])
        right_path = f'{path_without_extension}.csv'

        with ZipFile(path, 'r') as zip_cf:
            file_name = zip_cf.infolist()[0].orig_filename
            zip_cf.extract(file_name, '.')
            os.rename(file_name, right_path)

    return right_path


def check_data_file(info,
                    df,
                    time_column,
                    sample_name):
    """
    Проверяет соответствие названий столбцов входных данных и конфигурационного файла

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param df: Фрейм с данными;
    :param time_column: Колонка времени;
    :param sample_name: Название выборки
    """

    specified_columns = info.source['group_by_columns'].copy()
    specified_columns.append(info.source[sample_name]['target_column'])
    specified_columns.extend(time_column)

    df_cols = df.columns
    for expected_col in specified_columns:
        if expected_col not in df_cols:
            message = f'The "{expected_col}" column was not found'
            logging.exception(message)
            return KeyError(message)
    else:
        return True


def check_result(info, result):
    """
    Проверка на отрицательные и нулевые значения, с заменой последних

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param result: Результаты прогнозирования
    :return: Проверенный датафрейм и сообщение
    """
    # Проверка наличия отрицательных значений
    message = ''
    message_detailed = ''
    if not info.forecast['negative_values_acceptable']:
        if any(value < 0 for value in result['y_pred']):
            message = 'Ошибка прогноза'
            message_detailed = 'Присутствуют отрицательные значения. '
            logging.info(message_detailed)

    return result, message, message_detailed


def check_data_correctness(info,
                           group,
                           dates,
                           quality_file):
    """
    Проверка корректности данных с точки зрения наполненности и постоянства интервалов.
    Записывает "Error" и описание ошибки в соответствующее поле

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param group: Именованный кортеж. Содержит информацию о фактических данных (fact),
    тестовых данных (test), значениях колонок, определяющих группу (group_items), ключ (key);
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param quality_file: Датафрейм с качеством и параметрами;
    :return: Возвращает "True", когда данные корректны, иначе - "False"
    """
    step = info.source['fact']['time_column']['step']
    ts_length = len(group.fact)

    dp = info.source['detailed_preferences']
    dp_df = dp["df"]
    dp_index = dp_df[dp_df["key"] == group.key].index

    # Определим количество наблюдений ряда, которые можно обрезать (15%)
    dp_df.loc[dp_index, 'amount_of_pruning'] = int(np.floor(ts_length * 0.15))

    message = ''
    message_detailed = ''

    # Проверим наличие пропусков в датах
    detected_methods = [method for method in info.forecast['detected_methods'][group.key].keys()]
    dates_interval = pd.date_range(start=min(group.fact['ds']), end=max(group.fact['ds']), freq=step)
    differences = dates_interval.difference(group.fact['ds'])

    if not differences.empty and ('sarima' in detected_methods or 'prophet' in detected_methods):
        message = 'Ошибка исходных данных'
        message_detailed += f'Группа {group.key}.Обнаружены прощенные даты, ' \
                            f'что не подходит для методов SARIMA и Prophet.'

    n_periods = dp_df.loc[dp_index, 'forecasting_step']  # Шаг прогнозирования
    n_splits = 6

    if n_splits < 5:
        message = 'Ошибка исходных данных'
        message_detailed = f'Недостаточно наблюдений для заданного горизонта прогнозирования.\n' \
                           f'Количество наблюдений - {len(group.fact)}, Шаг прогнозирования - {n_periods}'

    if not message:
        return True
    else:
        logging.warning(message_detailed)
        fill_quality_file(quality_file, group.group_items, 0, 0, {},
                          error=message, detailed_error=message_detailed)
        return False


def check_file_existence(files,
                         ignore_additional=True):
    """
    Проверка наличия файлов в директории

    :path: Путь к файлу
    """
    dct = files.copy()
    if ignore_additional:
        for key in ['meta_parameters_in', 'meta_model_in']:
            del dct[key]

    for parameter, path in dct.items():
        if parameter.split('_')[-1] != 'out':
            if os.path.exists(path):
                return True
            else:
                message = f'File not found in path "{path}"'
                return FileNotFoundError(message)
