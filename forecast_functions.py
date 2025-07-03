import json
import logging

import pandas as pd

from checking_functions import check_result
from filling_functions import fill_quality_file
from metrics import calculate_quality
from preparing_functions import prepare_save_result
from preprocessing import generate_features
from service_functions import get_forecast_info


def make_forecast_by_strategy(info, sample, mode, dates,
                              method, group_name):
    key = "".join(group_name)
    dp_df = info.source['detailed_preferences']['df']
    dp_index = dp_df[dp_df['key'] == key].index[0]
    strategy = dp_df.at[dp_index, 'strategy']
    if str(method).split()[1].endswith('_predict'):
        method_name = str(method).split()[1][:-8]
    if method_name == 'fourier_mlr':
        fourier_mlr_columns = info.forecast['fourier_mlr_columns']

    if strategy == 'Direct simplified':
        result, params, model = method(info,
                                       key,
                                       sample,
                                       mode,
                                       dates)
    elif strategy == 'Recursive':
        info.forecast['last_iter'] = False

        temp_result = pd.DataFrame(columns=['ds', 'y_pred'])

        forecasting_step = dp_df.at[dp_index, 'forecasting_step']
        number_of_cycles = dp_df.at[dp_index, 'number_of_cycles']
        fractional_part = dp_df.at[dp_index, 'fractional_part']

        for cycle in range(1, number_of_cycles + 1):
            logging.debug(f'Итерация {cycle} из {number_of_cycles}')
            if cycle == number_of_cycles:
                info.forecast['last_iter'] = True

            if cycle == 2:
                mode = 'warm'
                info.forecast['specified_parameters'][key] = params

            result, params, model = method(info,
                                           key,
                                           sample,
                                           mode,
                                           dates)

            if cycle == number_of_cycles and fractional_part > 0:
                cycle_result = result[-round(fractional_part * forecasting_step):]
            else:
                cycle_result = result[-forecasting_step:]

            temp_result = pd.concat([temp_result, cycle_result], ignore_index=True)

            # Дальше добавление прогноза к обучающей выборке
            cycle_result = cycle_result[['ds', 'y_pred']]
            cycle_result.columns = ['ds', 'y']

            # if method in info.forecast['additional_sample_methods']:
            sample = sample.fact.copy()
            sample = pd.concat([sample, cycle_result], ignore_index=True)

            if cycle != number_of_cycles:
                sample = generate_features(info, sample, dates, key, 'warm')

        temp_result = temp_result.set_index(result.index[-len(temp_result):])
        result.loc[result.index[-len(temp_result):], :] = temp_result

    return result, params, model


def run_method_warm(info,
                    dates,
                    method,
                    sample,
                    group_name):
    """
    Определяет выборку, с которой необходимо запускать метод прогнозирования

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param method: Метод прогнозирования;
    :param sample: Данные для прогнозирования;
    :return: Результат и параметры прогнозирования
    """
    key = ''.join(group_name)
    sample = generate_features(info, sample, dates, key, 'warm')
    result, params, model = make_forecast_by_strategy(info, sample, 'warm', dates, method, group_name)

    if info.forecast['use_causal_factors']:
        if not info.source['causal_factors']['df'].empty:
            pass

    return result, params, model


def get_warm_result(info,
                    group,
                    dates,
                    quality):
    """
    Направляет данные в модели для обучения в теплом режиме

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param group: Именованный кортеж. Содержит информацию о фактических данных (fact),
    тестовых данных (test), значениях колонок, определяющих группу (group_items), ключ (key);
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param quality: Именованный кортеж с файлом качества и параметрами моделей;
    :return: Датафрейм с результатами прогнозирования
    """
    ind = list(quality.quality_info['key']).index(group.key)

    dp_df = get_forecast_info(info, group.key)
    methods = info.forecast['detected_methods'][group.key]
    metric = dp_df['quality_metric']

    # Определим модель и ее параметры для данной группы
    method_name_column = info.forecast['warm_mode_info']['method_name_column']
    method_name = quality.quality_info.loc[ind, method_name_column]

    parameters_column = info.forecast['warm_mode_info']['parameters_column']
    group_parameters = quality.quality_info.loc[ind, parameters_column]
    group_parameters = json.loads(group_parameters)
    group_parameters = group_parameters[method_name]
    info.forecast['specified_parameters'][group.key] = group_parameters

    method = methods[method_name]
    test_result, test_params, model = run_method_warm(info,
                                                      dates,
                                                      method,
                                                      group.test,
                                                      group.group_items)
    test_result = test_result.reset_index(drop=True)

    time_weight_coef = dp_df['time_weight_coef']

    test_index = group.fact.y[-dp_df['test_part']:].index
    model_quality, dw_value = calculate_quality(group.fact.y[test_index],
                                                test_result.y_pred[test_index],
                                                metric=metric,
                                                time_weight_coef=time_weight_coef,
                                                calculate_durbin_watson=True)

    result, params, model = run_method_warm(info,
                                            dates,
                                            method,
                                            group.fact,
                                            group.group_items)

    # Очистим список CF в кортеже info
    info.source['causal_factors']['list_of_columns'][group.key] = ['empty']

    param_dict = {method_name: group_parameters}

    result, message, message_detailed = check_result(info, result)

    quality_message = info.forecast['quality_message'][group.key]
    if quality_message:
        message_detailed += (f'{quality_message}: {info.forecast["quality_message_detailed"][group.key]}; '
                             f'{message: message_detailed}')

    fill_quality_file(quality.quality_file, group.group_items, model_quality,
                      dw_value, param_dict, method_name, message, message_detailed)

    return result


def get_cold_result(info,
                    group,
                    dates,
                    quality_file):
    """
    Направляет данные в модели для обучения в холодном режиме

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param group: Именованный кортеж. Содержит информацию о фактических данных (fact),
    тестовых данных (test), значениях колонок, определяющих группу (group_items), ключ (key);
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param quality_file: Датафрейм с качеством и параметрами;
    :return: Датафрейм с результатами прогнозирования
    """
    quality = []
    durbin_watson = []
    parameters = {}
    names = []

    ml_sample = generate_features(info, group.test, dates, group.key)
    dp_df = get_forecast_info(info, group.key)
    metric = dp_df['quality_metric']
    time_weight_coef = dp_df['time_weight_coef']
    methods = info.forecast['detected_methods'][group.key]

    test_dates = group.fact.ds[-dp_df['test_part']:]
    target_test = group.fact.loc[group.fact['ds'].isin(test_dates), 'y']

    for method_name, method in methods.items():
        names.append(method_name)
        test_result, test_params, model = make_forecast_by_strategy(info,
                                                                    ml_sample,
                                                                    'cold',
                                                                    dates,
                                                                    method,
                                                                    group.group_items)

        test_quality, dw = calculate_quality(target_test,
                                             test_result.loc[test_result['ds'].isin(test_dates), 'y_pred'],
                                             metric=metric,
                                             time_weight_coef=time_weight_coef,
                                             calculate_durbin_watson=True)

        quality.append(test_quality)
        durbin_watson.append(dw)
        parameters.update({method_name: test_params})

    logging.debug(f'Ошибки моделей: {dict(zip(names, quality))}')

    # Определим наиболее качественную модель для данной группы
    def find_closest(lst, k):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]

    best_quality = find_closest(quality, 0)
    best_index = quality.index(best_quality)
    dw_value = durbin_watson[best_index]
    best_model_name = names[best_index]
    best_method = methods[best_model_name]

    info.forecast['specified_parameters'][group.key] = parameters[best_model_name]

    logging.debug(f'Лучшая модель - {best_model_name}')
    logging.debug(f'Качество по метрике {metric} - {best_quality}')
    logging.debug('Построение итоговой модели...')

    result, params, model = run_method_warm(info,
                                            dates,
                                            best_method,
                                            group.fact,
                                            group.group_items)

    # Очистим список CF в кортеже info
    if info.forecast['use_causal_factors']:
        info.source['causal_factors']['list_of_columns'][group.key] = ['empty']

    result, message, message_detailed = check_result(info, result)

    quality_message = info.forecast['quality_message'].get(group.key, '')
    if quality_message:
        message_detailed += (f'{quality_message}: {info.forecast["quality_message_detailed"][group.key]}; '
                             f'{message: message_detailed}')

    fill_quality_file(quality_file, group.group_items, best_quality, dw_value, parameters,
                      best_model_name, message, message_detailed)

    logging.debug('Построение итоговой модели завершено')

    return result


def create_empty_forecast(info, group_by_columns, quality_file):
    # cf_info = info.output['cf_info']
    # cf_columns = cf_info['group_columns'] + [cf_info['cf_column'],
    #                                          cf_info['t_value_column'],
    #                                          cf_info['mlr_coefficient_column']]
    # cf_out_df = pd.DataFrame(columns=[cf_columns])
    # info.output['cf_info']['df'] = cf_out_df
    prepare_save_result(info, group_by_columns, pd.DataFrame(), pd.DataFrame(), quality_file)


def create_constant_forecast(info,
                             group,
                             dates,
                             quality_file,
                             message_detailed):
    """
    Функция для построения прогноза константных рядов. Постоянное значение копируется на горизонт прогнозирования

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param group: Именованный кортеж. Содержит информацию о фактических данных (fact),
    тестовых данных (test), значениях колонок, определяющих группу (group_items), ключ (key);
    :param dates: Именованный кортеж. Содержит информацию о граничных датах;
    :param quality_file: Датафрейм с качеством и параметрами;
    :return: Возвращает прогноз.
    """
    sample_end = max(group.fact['ds'])
    # Увеличим выборку на прогнозируемые даты
    forecast_range = pd.date_range(start=sample_end,
                                   freq=info.source['fact']['time_column']['step'],
                                   periods=dates.n_periods + 1)

    forecast_df = pd.DataFrame({'ds': forecast_range,
                                'y': 0})

    sample = pd.concat([group.fact, forecast_df[1:]]).reset_index(drop=True)
    result = sample.copy()[['ds']]

    result['y_pred'] = sample.at[min(sample.index), 'y']
    dp_df = info.source['detailed_preferences']['df']
    dp_index = dp_df[dp_df['key'] == group.key].index[0]

    message = 'Ошибка исходных данных'
    message_detailed += ('Исходные данные полностью или частично состоят из одинаковых значений, '
                         'поэтому прогноз также постоянная.')
    logging.info(message_detailed)
    fill_quality_file(quality_file, group.group_items, 0,
                      2, {}, 'constant_value', message, message_detailed)

    return result
