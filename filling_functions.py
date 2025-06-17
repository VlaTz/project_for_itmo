import json

import pandas as pd

from service_functions import NpEncoder


def fill_quality_file(quality_file,
                      group_items,
                      quality,
                      durbin_watson,
                      parameters,
                      model='',
                      error='',
                      detailed_error=''):
    """
    Добавляет значения в файл с качеством и параметрами

    :param quality_file: Датафрейм с качеством и параметрами;
    :param group_items: Список значений, определяющих уникальную группу;
    :param quality: Качество;
    :param durbin_watson: Статистика Дарбина-Уотсона;
    :param parameters: Параметры модели;
    :param model: Название модели;
    :param error: Текст ошибки;
    :param detailed_error: Текст ошибки подробный;
    """
    parameters = json.dumps(parameters, ensure_ascii=False, cls=NpEncoder)
    quality_items = group_items.copy()
    if model == '':
        model = ' '
    if error == '':
        error = detailed_error = ' '
    if detailed_error == '' and quality == 0:
        detailed_error = 'Слишком высокое качество. Возможно, допущена ошибка'
    quality_items.extend([quality, durbin_watson, parameters, model, error, detailed_error])
    quality_file.loc[len(quality_file.index)] = quality_items


def fill_with_zeros(info,
                    fact_forecast,
                    future_forecast,
                    column_group):
    """
    Проставляет нули для дат, которых нет в прогнозных фреймах, но есть в выборке времени в модели.

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param fact_forecast: Фрейм прогноза на фактический период;
    :param future_forecast: Фрейм прогноза на будущий период;
    :param column_group: Лист группирующих колонок;
    :return: Фреймы прогноза на фактический период и на будущее
    """
    start, end, time_type = info.forecast['start_time_subset'], info.forecast['end_time_subset'], \
        info.source['fact']['time_column']['step']
    zero_df = pd.DataFrame({'ds': pd.date_range(start, end, freq=time_type)})
    group_name = fact_forecast.loc[0, column_group]

    if info.forecast['join_output_result']:
        samples = [pd.DataFrame(), future_forecast]
    else:
        samples = [fact_forecast, future_forecast]

    for i, sample in enumerate(samples):
        if not sample.empty:
            sample = sample.merge(zero_df, how='right').fillna(0)
            sample[column_group] = group_name
            samples[i] = sample
        else:
            samples[0] = fact_forecast

    return samples[0], samples[1]
