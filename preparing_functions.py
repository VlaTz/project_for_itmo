import json
import os
from collections import namedtuple

from checking_functions import check_data_file, check_file_existence, check_file_extension
from filling_functions import fill_quality_file, fill_with_zeros
from om_pandas import file_to_dataframe, dt_to_om_weeks, datetime_to_om_time, convert_columns, convert_dates
from preprocessing import scale_target
from service_functions import (find_delimiter_encoding, create_key, QualityFile,
                               detect_time_type)
from stats_methods import *


def detect_methods(info, key):
    dp_df, dp_index = get_forecast_info(info, key, get_index=True)
    selected_models = dp_df['selected_models']
    all_methods = info.forecast['all_methods']
    func_name = [method for method in selected_models if method in all_methods]

    if not func_name:
        logging.warning(f'В настройках не обнаружены методы прогнозирования. '
                        f'Пожалуйста, выберите один из следующих: {all_methods}. '
                        f'Для данной группы будет использован метод по умолчанию - "fourier_mlr"')
        func_name = ['fourier_mlr']
    methods = {method: eval(f'{method}_predict') for method in func_name}
    info.forecast['detected_methods'][key] = methods


def load_and_prepare_data(info,
                          sample_name,
                          group_columns=None,
                          need_key=False,
                          sample_path=''):
    """
    Загружает и преобразует данные

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param sample_name: Название выборки;
    :param group_columns: Список колонок, определяющих группу;
    :param need_key: Флаг, который определяет создавать ключ или нет;
    :param sample_path: Путь к файлу;
    :return: Фрейм с данными
    """
    logging.info(f'Загрузка данных {sample_name}')
    sample_dict = info.source[sample_name]

    path = sample_dict['path']
    if sample_path:
        path = sample_path
    # Импорт и форматирование с помощью библиотеки om_pandas
    if not sample_dict.get('encoding', ''):
        sample_dict['encoding'], sample_dict['delimiter'] = find_delimiter_encoding(path)
    num_column = sample_dict.get('num_column', None)

    df = file_to_dataframe(file_path=path,
                           file_type=sample_dict['type'],
                           encoding=sample_dict['encoding'],
                           sep=sample_dict['delimiter'])

    if not group_columns:
        group_columns = info.source['group_by_columns']

    if sample_name == 'fact' and df.empty:
        return df

    if sample_name == 'causal_factors' and df.empty:
        return df

    if sample_name not in ['target_intersection', 'cf_force_include', 'causal_factors']:
        time_column = sample_dict['time_column']['name']
        time_type = sample_dict['time_column']['time_type']

        # Преобразование дат
        df = convert_dates(info, df, num_column, group_columns, time_column, time_type)

    if sample_name == 'fact' or sample_name == 'future':
        res_check = check_data_file(info, df, time_column, sample_name)
        if res_check:
            # Переименуем столбцы с датой и целевым признаком
            target_column = sample_dict['target_column']
            df = df.rename(columns={target_column: 'y', time_column[0]: 'ds'})

            # Оставим только необходимые столбцы
            data_columns = group_columns.copy()
            data_columns.extend(['ds', 'y'])
            df = df[data_columns]
        else:
            raise res_check

    if need_key:
        df = create_key(df, group_columns)

    return df.reset_index(drop=True)


def reformat_forecast(sample, mapping, group_columns, time_column, target, value_columns, time_type):
    df = sample.copy()
    result_columns = group_columns.copy()
    result_columns.extend([time_column, target] + value_columns)

    if time_type == 'om_weeks':
        df = dt_to_om_weeks(df,
                            'ds',
                            ['y_pred'] + value_columns,
                            group_columns,
                            mapping)
    else:
        df['ds'] = datetime_to_om_time(df['ds'], time_type)

    df = df.rename(columns={'ds': time_column, 'y_pred': target})

    return df[result_columns]


def make_forecast_df(info,
                     column_group,
                     fact_forecast,
                     future_forecast):
    """
    Формирует итоговые датафреймы с прогнозом, в соответствии с пользовательскими настройками

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param fact_forecast: Результаты прогнозирования значений исходного ряда;
    :param future_forecast: Результаты прогнозирования значений прогнозного ряда;
    :return: Именованный кортеж. Содержит прогнозы для исходного периода и будущего,
    качество и параметры моделей (quality).
    """

    future_forecast.reset_index(drop=True, inplace=True)
    result_dict = {'Fitted history': fact_forecast,
                   'Forecast': future_forecast}

    # Названия колонок целевого признака и времени
    target_column_fact = info.output['data']['target_column_fact']
    target_column_future = info.output['data']['target_column_future']

    fact_time = info.output['data']['fact_time_column']
    future_time = info.output['data']['future_time_column']

    if info.forecast['join_output_result']:
        future_time = info.source['joined_output_time_col']

    mapping_df = info.source.get('mapping_df', None)
    value_columns = []

    for result_name, result_df in result_dict.items():
        try:
            result_df.loc[0, column_group[0]]
        except KeyError:
            message = f'{result_name} не содержит результатов'
            logging.warning(message)
            fact_forecast = pd.DataFrame(columns=column_group + [fact_time, target_column_fact] + value_columns)
            future_forecast = pd.DataFrame(columns=column_group + [future_time, target_column_future] + value_columns)
            return fact_forecast, future_forecast

    fact_forecast, future_forecast = fill_with_zeros(info, fact_forecast, future_forecast, column_group)

    time_type = info.source['fact']['time_column']['time_type']
    fact_forecast = reformat_forecast(fact_forecast, mapping_df, column_group, fact_time,
                                      target_column_fact, value_columns, time_type)
    future_forecast = reformat_forecast(future_forecast, mapping_df, column_group, future_time,
                                        target_column_future, value_columns, time_type)

    return fact_forecast, future_forecast


def create_quality_file(info):
    """
    Формирует кортеж с качеством и информацией о параметрах моделей (только в теплом режиме)

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :return: Возвращает кортеж
    """
    forecasting_mode = info.forecast["mode"]
    column_group, quality_columns = info.source["group_by_columns"].copy(), info.source["group_by_columns"].copy()
    quality_columns.extend(['quality', 'durbin_watson', 'parameters', 'best_model', 'error', 'error_description'])
    quality_file = pd.DataFrame(columns=quality_columns)

    if forecasting_mode == Mode.warm:
        warm_file = info.forecast['warm_mode_info']['file']
        quality_info = file_to_dataframe(file_path=warm_file['path'],
                                         file_type=warm_file['type'],
                                         encoding=warm_file['encoding'],
                                         sep=warm_file['delimiter'])
        quality_info['key'] = quality_info[column_group].agg(''.join, axis=1)
    elif forecasting_mode == Mode.cold or forecasting_mode == Mode.fitted_model:
        quality = QualityFile(quality_file)

    info.forecast['quality_file'] = quality_file

    return quality


def prepare_quality_file(info,
                         quality):
    """
    Формирует файл с параметрами моделей и качеством, в соответствии с требованиями пользователя

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param quality: Именованный кортеж с файлом качества и параметрами моделей;
    :return: Датафрейм с параметрами моделей и качеством
    """
    metadata = info.output['metadata']
    quality_file = quality
    quality_file = quality_file.rename(columns={'quality': metadata['quality_column'],
                                                'parameters': metadata['model_info_column'],
                                                'durbin_watson': metadata['durbin_watson_column'],
                                                'best_model': metadata['model_name_column'],
                                                'error': metadata['error_column_name'],
                                                'error_description': metadata['error_description_column_name']})

    return quality_file


def prepare_detailed_preferences(info, dp_df):
    start = time()
    dp = info.source['detailed_preferences']

    updated_cols = ['quality_metric', 'eval_metric', 'time_weight_coef',
                    'scaling_strategy_target', 'outlier_threshold',
                    'periods_to_forecast', 'scaling_parameters_target',
                    'start_history', 'end_history', 'min_history_periods', 'selected_models',

                    # Статистические параметры:
                    'rol_mean_min_window_size', 'rol_mean_max_window_size', 'exp_smoothing_min_alpha',
                    'exp_smoothing_max_alpha', 'exp_smoothing_step', 'holt_min_alpha', 'holt_max_alpha',
                    'holt_min_beta', 'holt_max_beta', 'holt_step', 'holt_winters_min_seasonality',
                    'holt_winters_max_seasonality', 'holt_winters_trend_types', 'holt_winters_seasonal_types',
                    'polynomial_min_degrees', 'polynomial_max_degrees', 'theil_sen_min_degrees',
                    'theil_sen_max_degrees', 'ransac_min_degrees', 'ransac_max_degrees', 'huber_min_degrees',
                    'huber_max_degrees', 'lasso_min_alpha', 'lasso_max_alpha', 'lasso_step', 'ridge_min_alpha',
                    'ridge_max_alpha', 'ridge_step', 'elastic_net_min_alpha', 'elastic_net_max_alpha',
                    'elastic_net_min_l1', 'elastic_net_max_l1', 'elastic_net_step', 'croston_tsb_min_alpha',
                    'croston_tsb_max_alpha', 'croston_tsb_min_beta', 'croston_tsb_max_beta', 'croston_tsb_step'
                    ]

    new_names = {value: key for key, value in dp.items() if key in updated_cols}
    dp_df = dp_df.rename(columns=new_names)

    int_cols = ['periods_to_forecast', 'min_history_periods']
    float_cols = ['time_weight_coef', 'outlier_threshold']
    json_cols = ['eval_metric', 'selected_models', 'holt_winters_trend_types', 'holt_winters_seasonal_types',
                 'scaling_parameters_target']

    dp_df[int_cols] = dp_df[int_cols].astype(int)
    dp_df[float_cols] = dp_df[float_cols].astype(float)

    for col in json_cols:
        dp_df[col] = dp_df[col].apply(lambda x: json.loads(x))

    # Адаптация под MVP
    dp_df['strategy'] = 'Direct simplified'
    dp_df['forecasting_step'] = dp_df['periods_to_forecast']
    dp_df['settings_horizon'] = dp_df['periods_to_forecast']

    dp_df.loc[dp_df['strategy'] == 'Direct simplified', 'forecasting_step'] = dp_df['periods_to_forecast']
    dp_df['number_of_cycles'] = np.ceil(dp_df['periods_to_forecast'] / dp_df['forecasting_step'])
    dp_df['fractional_part'] = np.ceil(dp_df['periods_to_forecast'] % dp_df['forecasting_step'])

    additional_cols = ['test_part', 'amount_of_pruning', 'mean', 'std']
    for col in additional_cols:
        dp_df[col] = np.nan
    dp_df[['test_part', 'amount_of_pruning']] = dp_df[['test_part', 'amount_of_pruning']].astype('Int64')

    dp['df'] = dp_df

    logging.debug(f'prepare_detailed_preferences_time: {time() - start}')


def prepare_fact(info,
                 fact,
                 quality_file):
    """
    Подготовка фактических данных

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param fact: Датафрейм с исходным временным рядом;
    :param quality_file: Датафрейм с качеством и параметрами;
    :return: Возвращает именованный кортеж "group". Содержит информацию о фактических данных (fact),
    тестовых данных (test), значениях колонок, определяющих группу (group_items), ключ (key)
    """
    message = ''

    if not fact['ds'].is_monotonic_increasing:
        fact = fact.sort_values(by='ds')

    fact.reset_index(drop=True, inplace=True)

    # Названия колонок и значения постоянных значений сохраним в отдельные переменные
    group_items = list(fact.loc[0, info.source['group_by_columns']])
    key = ''.join(group_items)
    dp_df = info.source['detailed_preferences']['df']

    if key in dp_df['key'].unique():
        dp_index = dp_df.loc[dp_df['key'] == key].index[0]
        fact = fact.loc[(fact['ds'] >= dp_df.at[dp_index, 'start_history']) &
                        (fact['ds'] <= dp_df.at[dp_index, 'end_history'])]

        # Добавить параметр в настройки, контролирующий минимальную длину истории
        min_history_periods = dp_df.at[dp_index, 'min_history_periods']

        if not fact.empty and len(fact) >= min_history_periods:
            fact = fact.reset_index(drop=True)
        else:
            message = 'Ошибка исходных данных'
            message_detailed = (f'Группа {key}. Недостаточная длина ряда ({len(fact)}) после ограничения по заданным '
                                f'началу и концу истории: '
                                f'{dp_df.at[dp_index, "start_history"]} - {dp_df.at[dp_index, "end_history"]}. '
                                f'Минимальная длина истории для данного пересечения: {min_history_periods}')
            logging.warning(message_detailed)
            fill_quality_file(quality_file, group_items, 0, 0,
                              {}, error=message, detailed_error=message_detailed)
            return None, None, None
    else:
        # Если настройки прогнозирования не обнаружены
        message = 'Ошибка исходных данных'
        message_detailed = f'Группа {key}. Настройки прогнозирования не обнаружены. '
        logging.warning(message_detailed)
        info.forecast['quality_message'][key] = message
        info.forecast['quality_message_detailed'][key] = message_detailed

        logging.warning(message_detailed)
        fill_quality_file(quality_file, group_items, 0, 0,
                          {}, error=message, detailed_error=message_detailed)
        return None, None, None

    # Оставим в выборке только столбец с датой и целевым признаком
    fact = fact[['ds', 'y']]

    # Адаптация под MVP
    true_periods = dp_df.loc[dp_index, 'periods_to_forecast']

    if fact['ds'].max() != dp_df.at[dp_index, 'end_history']:
        dp_df.loc[dp_index, 'periods_to_forecast'] += len(
            pd.date_range(fact['ds'].max(), dp_df.at[dp_index, 'end_history']))
    horizon = dp_df.loc[dp_index, 'periods_to_forecast']

    if not info.source['start_test_period']:
        test_part = int(np.ceil(len(fact) * 0.15))
    else:
        time_type = info.source['fact']['time_column']['time_type']
        start_test_period = convert_dates(info, pd.DataFrame({'ds': [info.source['start_test_period']]}), [], [],
                                          ['ds'], time_type).loc[0, 'ds']
        test_part = len(fact[fact.ds >= start_test_period].ds)
    test_part = (test_part if test_part < horizon else horizon)
    test = fact[:-test_part]
    dp_df.loc[dp_index, 'test_part'] = test_part

    # Для определения выбросов в функции "remove_outliers"
    dp_df.loc[dp_index, 'mean'] = np.mean(fact['y'])
    dp_df.loc[dp_index, 'std'] = np.std(fact['y'])

    if dp_df.loc[dp_index, 'scaling_strategy_target'] != 'Don\'t scale':
        test['y'] = scale_target(info, test['y'], key)
        fact['y'] = scale_target(info, fact['y'], key)

    GroupInfo = namedtuple('GroupInfo', ['fact', 'test', 'group_items', 'key'])
    group = GroupInfo(fact, test, group_items, key)

    return group, test_part, true_periods


def prepare_save_result(info, group_by_columns, fact_forecast, future_forecast, quality_file):
    # Объединим результаты прогнозирования с соответствующими датафреймами
    fact_forecast, future_forecast = make_forecast_df(info, group_by_columns, fact_forecast, future_forecast)
    quality_file = prepare_quality_file(info, quality_file)

    if info.forecast['use_causal_factors']:
        Result = namedtuple('result', ['fact_forecast', 'future_forecast', 'quality', 'cf_info'])
        result = Result(fact_forecast, future_forecast, quality_file, info.output['cf_info']['df'])
    else:
        Result = namedtuple('result', ['fact_forecast', 'future_forecast', 'quality'])
        result = Result(fact_forecast, future_forecast, quality_file)

    save_result(info, result)


def save_result(info,
                result):
    """Создает файл-прогноз и файл с качеством и параметры моделей

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :param result: Именованный кортеж. Содержит прогноз (forecast), качество и параметры моделей (quality);
    """

    result_info = info.output['data']['file']

    mode = 'a'
    header = False
    if not os.path.isfile(result_info['future_path']):
        mode = 'w'
        header = True

    result.future_forecast.to_csv(result_info['future_path'], sep=info.source['fact']['delimiter'],
                                  encoding=info.source['fact']['encoding'], index=False, mode=mode, header=header)

    if not info.forecast['join_output_result'] and info.forecast['append_model_estimation_for_known_periods']:
        result.fact_forecast.to_csv(result_info['fact_path'], sep=info.source['fact']['delimiter'],
                                    encoding=info.source['fact']['encoding'], index=False, mode=mode, header=header)
    else:
        pd.DataFrame().to_csv(result_info['fact_path'], sep=info.source['fact']['delimiter'],
                              encoding=info.source['fact']['encoding'], index=False, mode=mode, header=header)

    if info.forecast['use_causal_factors']:
        logging.debug(f'Создание файла с информацией по Causal Factors...')
        result_info = info.output['cf_info']['file']
        result.cf_info.to_csv(result_info['path'], sep=result_info['delimiter'],
                              encoding=result_info['encoding'], index=False, mode=mode, header=header)
        info.output['cf_info']['df'] = pd.DataFrame()

    logging.debug(f'Создание файла c качеством для каждой группы...')
    result_info = info.output['metadata']['file']
    result.quality.to_csv(result_info['path'], sep=result_info['delimiter'],
                          encoding=result_info['encoding'], index=False, mode=mode, header=header)
    logging.debug(f'Создание файла c качеством для каждой группы завершено\n'
                  f'Качество сохранено в файле - {result_info["path"]}\n')


def get_config_info(input_info):
    """
    Проверяет конфигурационный файл

    :param input_info: Словарь параметров переданных в скрипт;
    :return: Именованный кортеж с параметрами обработки и прогнозирования.
    """

    encoding, delimiter = find_delimiter_encoding(input_info['forecasting_config'])
    df = pd.read_csv(filepath_or_buffer=input_info['forecasting_config'], sep=delimiter, encoding=encoding)

    bool_col = ['Включить значения модели по фактическому периоду',
                'Объединить прогноз факта и будущего в один файл']
    date_cols = ['Начало выборки времени', 'Конец выборки времени']
    df = convert_columns(df, bool_col, 'bool')
    df = convert_columns(df, date_cols, 'om_date')

    # Проверка корректности заданного интервала наблюдений
    step, time_type = detect_time_type(df.loc[0, 'Шаг времени'])
    acceptable_step = ['YS', 'MS', 'W', 'W-MON', 'D', 'H', 'T', 'S']
    try:
        acceptable_step.index(step)
    except KeyError:
        message = f'The "{step}" step is not allowed. Please choose one of the options:' \
                  f'"YS", "MS", "W", "W-MON", "D", "H", "T", "S"'
        logging.exception(message)
        raise KeyError(message)

    mode = 'cold'

    ConfigInfo = namedtuple('ConfigInfo',
                            ['df', 'step', 'time_type', 'encoding', 'delimiter',
                             'mode'])
    config_info = ConfigInfo(df, step, time_type, encoding, delimiter, mode)
    return config_info


def prepare_cf_force_include(info):
    fi = info.source['cf_force_include']
    features_col = fi['features_col']
    force_col = fi['num_column'][0]
    fi_df = load_and_prepare_data(info, 'cf_force_include', fi['group_column'], need_key=True)
    fi_df.rename(columns={features_col: 'cf', force_col: 'force_include'}, inplace=True)
    fi['df'] = fi_df


def prepare_causal_factors(info, cf_df, cf_desc_df):
    """
    Подготовка Causal Factors в виде фрейма Pandas. Результат сохраняется в именованном кортеже info

    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    """
    start = time()
    cf = info.source['causal_factors']
    cf_desc = info.source['cf_desc']
    value_col = cf_desc['num_column']
    cf_df = cf_df.rename(columns={cf['features_col']: 'features_col_x'})
    cf_desc_df = cf_desc_df.drop(columns=[cf_desc['fragmented_features_col']])
    cf['num_column'] = value_col

    if info.forecast['two_dates_for_cf']:
        rename_col = {cf_desc['time_column']['name'][0]: 'start',
                      cf_desc['time_column']['name'][1]: 'end'}
    else:
        rename_col = {cf_desc['time_column']['name'][0]: 'ds'}
    rename_col[cf_desc['features_col']] = 'features_col'
    cf_desc_df = cf_desc_df.rename(columns=rename_col)

    if info.forecast['two_dates_for_cf']:
        cf_desc_df['ds'] = [pd.date_range(s, e, freq='d') for s, e in
                            zip(cf_desc_df['start'], cf_desc_df['end'])]

    # Объединим фрейм с causal factors со справочником cf для добавления интервалов и коэффициентов
    cf_df = pd.merge(cf_df, cf_desc_df, how='left',
                     left_on='features_col_x', right_on='features_col').drop(columns='features_col_x')
    cf_df.dropna(inplace=True)

    if not info.forecast['two_dates_for_cf'] and not cf_df.empty:
        cf_df = pd.pivot(cf_df, index=['ds', 'key'], columns='features_col', values=value_col)
        cf_df.columns = cf_df.columns.map(lambda x: x[1])
    elif info.forecast['two_dates_for_cf'] and not cf_df.empty:
        cf_df = cf_df.explode('ds').drop(['start', 'end'], axis=1)

    cf['df'] = cf_df
    prepare_cf_force_include(info)
    logging.debug(f'prepare_causal_factors_time: {time() - start}')


def prepare_correlation(info):
    dp = info.source['detailed_preferences']
    dp_df = dp['df']
    if np.sum(dp_df['use_source_global_corr']) > 0:
        gc = info.source['global_correlation']
        corr_path = gc['path']
        corr_encoding, corr_delimiter = find_delimiter_encoding(corr_path)
        corr_df = file_to_dataframe(file_path=corr_path,
                                    file_type='om_csv',
                                    encoding=corr_encoding,
                                    sep=corr_delimiter)
        corr_df['key'] = corr_df[gc['group_column']].agg(';'.join, axis=1)
        corr_df.rename(columns={gc['feature_column']: 'cf', gc['top_features']: 'top_features'}, inplace=True)

        # Оставим корреляцию для тех пересечений, где она потребуется
        dp_corr_keys = list(dp_df.loc[dp_df['use_source_global_corr'] == 1, 'key'])
        corr_df = corr_df[corr_df['key'].isin(dp_corr_keys)]
        gc['corr_df'] = corr_df[['key', 'cf', 'top_features']]


def create_info(input_info):
    """
    Функция преобразует входные данные

    :param input_info: Словарь параметров переданных в скрипт;
    :return: возвращает словарь с информацией о параметрах прогнозирования.
    """
    logging.info('Создание информационного файла')
    start = time()
    config_info = get_config_info(input_info)
    config_df = config_info.df.loc[0, :]
    file_encoding, file_delimiter = find_delimiter_encoding(input_info['forecasting_config'])
    fact_path = check_file_extension(input_info['fact'])
    fact_encoding, fact_delimiter = find_delimiter_encoding(input_info['forecasting_config'])
    file_info = {
        'source': {
            'fact': {
                'path': fact_path,
                'type': 'om_csv',
                'encoding': fact_encoding,
                'delimiter': fact_delimiter,
                'target_column': config_df['Целевая колонка факта входящая'],
                'num_column': [config_df['Целевая колонка факта входящая']],
                'time_column': {
                    'name': [config_df['Измерение времени входящее']],
                    'step': config_info.step,
                    'time_type': config_info.time_type
                }
            },
            'group_by_columns': json.loads(config_df['Список прочих измерений']),
            'start_test_period': config_df['Начало тестового периода'],
        },
        'forecast': {
            'mode': config_info.mode,
            'use_causal_factors': 0,
            'two_dates_for_cf': 1,
            'join_output_result': config_df['Объединить прогноз факта и будущего в один файл'],
            'append_model_estimation_for_known_periods':
                config_df['Включить значения модели по фактическому периоду'],
            'negative_values_acceptable': False,
            'last_iter': True,
            'fourier_mlr_columns': {},
            'quality_message': {},
            'quality_message_detailed': {},
            'specified_parameters': {},
            'start_time_subset': config_df['Начало выборки времени'],
            'end_time_subset': config_df['Конец выборки времени'],
            'all_methods': ['linear_regression', 'theil_sen', 'ransac', 'lasso', 'ridge', 'elastic_net', 'polynomial',
                            'huber', 'exp_smoothing', 'holt', 'holt_winters', 'rol_mean', 'croston_tsb', 'hyperbolic',
                            'const'],
            'detected_methods': {},
        },
        'output': {
            'data': {
                'file': {
                    'fact_path': f"{input_info['forecast_fact_out']}",
                    'future_path': f"{input_info['forecast_future_out']}",
                    'type': 'om_csv',
                    'encoding': file_encoding,
                    'delimiter': file_delimiter},
                'target_column_fact': config_df['Целевая колонка факта исходящая'],
                'target_column_future': config_df['Целевая колонка прогноза исходящая'],
                'fact_time_column': config_df['Измерение времени факта исходящее'],
                'future_time_column': config_df['Измерение времени прогноза исходящее']},
            'metadata': {
                'file': {
                    'path': f"{input_info['meta_parameters_out']}",
                    'type': 'om_csv',
                    'encoding': config_info.encoding,
                    'delimiter': config_info.delimiter},
                'description_column': config_df['Название колонки для описания прогноза'],
                'metric_column': config_df['Название колонки для метрики качества'],
                'hyperparams_metric': config_df['Название колонок для метрик подбора гиперпараметров'],
                'quality_column': config_df['Название колонки для ошибки'],
                'durbin_watson_column': config_df['Название колонки для значения Дарбина-Уотсона'],
                'model_name_column': config_df['Название колонки для лучшей модели'],
                'model_info_column': config_df['Название колонки для параметров'],
                'error_column_name': config_df['Название колонки ошибки прогноза'],
                'error_description_column_name': config_df['Название колонки описания ошибки прогноза']},
        },
    }

    if file_info.get('correlation'):
        file_info['source']['global_correlation'] = {
            'path': input_info['correlation'],
            'type': 'om_csv',
            'group_column': json.loads(config_df['Колонка(и), определяющая(ие) пересечение глобальной корреляции']),
            'feature_column': config_df['Колонка признаков глобальной корреляции'],
            'top_features': config_df['Колонка топ признаков данного типа глобальной корреляции'],
        }

    if file_info['forecast']['use_causal_factors']:

        causal_factors_path = input_info['causal_factors']
        res_check = check_file_existence({'causal_factors_path': causal_factors_path}, ignore_additional=False)
        if res_check:
            file_info['source']['causal_factors'] = {
                'path': check_file_extension(causal_factors_path),
                'type': 'om_csv',
                'list_of_columns': {},
                'scaling': {},
                'scaling_target': {},
                'features_col': config_df['Колонка признаков CF'],
                'group_column': json.loads(config_df['Колонка(и), определяющая(ие) пересечение CF']),
            }
        else:
            raise res_check

        # Force include for Causal factors
        cf_force_include_path = input_info['cf_force_include']
        res_check = check_file_existence({'cf_force_include_path': cf_force_include_path}, ignore_additional=False)

        if res_check:
            file_info['source']['cf_force_include'] = {
                'path': cf_force_include_path,
                'type': 'om_csv',
                'features_col': config_df['Колонка признаков CF_force_include'],
                'group_column': json.loads(
                    config_df['Колонка(и), определяющая(ие) пересечение CF_force_include']),
                'num_column': [config_df['Колонка Force include']],
            }
        else:
            raise res_check

        # Causal factors description
        cf_desc_path = input_info['cf_desc']
        res_check = check_file_existence({'cf_desc_path': cf_desc_path}, ignore_additional=False)

        if res_check:
            cf_desc_start = config_df['Информация_промо Колонка начала события']
            cf_desc_end = config_df['Информация_промо Колонка окончания события']
            file_info['source']['cf_desc'] = {
                'path': cf_desc_path,
                'type': 'om_csv',
                'encoding': config_info.encoding,
                'delimiter': config_info.delimiter,
                'fragmented_features_col': config_df['Информация_промо Колонка признаков'],
                'features_col': config_df['Информация_промо Колонка группы признаков'],
                'num_column': [config_df['Информация_промо Колонка коэффициента события']],
                'time_column': {
                    'name': [cf_desc_start, cf_desc_end],
                    'step': config_info.step,
                    'time_type': config_info.time_type
                }
            }
        else:
            raise res_check

        # Output CF information
        file_info['output']['cf_info'] = {
            'file': {
                'path': f"{input_info['cf_info_out']}",
                'type': 'om_csv',
                'encoding': config_info.encoding,
                'delimiter': config_info.delimiter,
            },
            'group_columns': json.loads(config_df['Колонка(и), определяющая(ие) пересечение CF output']),
            'cf_column': config_df['Колонка признаков CF output'],
            't_value_column': config_df['Колонка t-value output'],
            'mlr_coefficient_column': config_df['Колонка коэффициентов CF output'],
            'df': pd.DataFrame(),
        }


    detailed_preferences = input_info['detailed_preferences']
    res_check = check_file_existence({'detailed_preferences_path': detailed_preferences}, ignore_additional=False)

    if res_check:
        file_info['source']['detailed_preferences'] = {
            'path': detailed_preferences,
            'type': 'om_csv',
            'group_column': json.loads(
                config_df['Колонка(и), определяющая(ие) пересечение уникальных настроек']),
            'start_history': config_df['Колонка даты начала истории'],
            'end_history': config_df['Колонка даты конца истории'],
            'min_history_periods': config_df['Колонка минимальной длины истории'],
            'periods_to_forecast': config_df['Колонка уникального горизонта прогнозирования'],
            'forecasting_step': config_df['Колонка уникального шага прогнозирования'],
            'quality_metric': config_df['Колонка уникальной метрики качества'],
            'eval_metric': config_df['Колонка уникальных метрик подбора гиперпараметров'],
            'time_weight_coef': config_df['Колонка коэффициента затухания ошибки со временем'],
            'outlier_threshold': config_df['Колонка значения outlier threshold'],

            # Генерация, масштабирование и отбор признаков
            'scaling_strategy_target': config_df['Колонка стратегии масштабирования целевого признака'],
            'scaling_parameters_target': config_df['Колонка параметров масштабирования целевого признака'],
            'selected_models': config_df['Колонка выбранных моделей'],

            # Параметры для статистических методов
            'rol_mean_min_window_size': config_df['Скользящее среднее - размер окна минимум'],
            'rol_mean_max_window_size': config_df['Скользящее среднее - размер окна максимум'],
            'exp_smoothing_min_alpha': config_df['Экспоненциальное сглаживание - alpha минимум'],
            'exp_smoothing_max_alpha': config_df['Экспоненциальное сглаживание - alpha максимум'],
            'exp_smoothing_step': config_df['Экспоненциальное сглаживание - шаг'],
            'holt_min_alpha': config_df['модель Хольта - alpha минимум'],
            'holt_max_alpha': config_df['модель Хольта - alpha максимум'],
            'holt_min_beta': config_df['модель Хольта - beta минимум'],
            'holt_max_beta': config_df['модель Хольта - beta максимум'],
            'holt_step': config_df['модель Хольта - шаг'],
            'holt_winters_min_seasonality': config_df['Хольта-Винтерса - сезонность минимум'],
            'holt_winters_max_seasonality': config_df['Хольта-Винтерса - сезонность максимум'],
            'holt_winters_trend_types': config_df['Хольта-Винтерса - типы тренда'],
            'holt_winters_seasonal_types': config_df['Хольта-Винтерса - типы сезонности'],
            'polynomial_min_degrees': config_df['Полиномиальная - степень минимум'],
            'polynomial_max_degrees': config_df['Полиномиальная - степень максимум'],
            'theil_sen_min_degrees': config_df['Функция Тейла-Сена - степень минимум'],
            'theil_sen_max_degrees': config_df['Функция Тейла-Сена - степень максимум'],
            'ransac_min_degrees': config_df['RANSAC - степень минимум'],
            'ransac_max_degrees': config_df['RANSAC - степень максимум'],
            'huber_min_degrees': config_df['Модель Хубера - степень минимум'],
            'huber_max_degrees': config_df['Модель Хубера - степень максимум'],
            'lasso_min_alpha': config_df['Модель Lasso - alpha минимум'],
            'lasso_max_alpha': config_df['Модель Lasso - alpha максимум'],
            'lasso_step': config_df['Модель Lasso - шаг'],
            'ridge_min_alpha': config_df['Модель Ridge - alpha минимум'],
            'ridge_max_alpha': config_df['Модель Ridge - alpha максимум'],
            'ridge_step': config_df['Модель Ridge - шаг'],
            'elastic_net_min_alpha': config_df['Модель Elastic-Net - alpha минимум'],
            'elastic_net_max_alpha': config_df['Модель Elastic-Net - alpha максимум'],
            'elastic_net_min_l1': config_df['Модель Elastic-Net - l1 минимум'],
            'elastic_net_max_l1': config_df['Модель Elastic-Net - l1 максимум'],
            'elastic_net_step': config_df['Модель Elastic-Net - шаг'],
            'croston_tsb_min_alpha': config_df['Модель Кростона TSB - alpha минимум'],
            'croston_tsb_max_alpha': config_df['Модель Кростона TSB - alpha максимум'],
            'croston_tsb_min_beta': config_df['Модель Кростона TSB - beta минимум'],
            'croston_tsb_max_beta': config_df['Модель Кростона TSB - beta максимум'],
            'croston_tsb_step': config_df['Модель Кростона TSB - шаг'],

            'num_column': [],
            'time_column': {
                'name': [config_df['Колонка даты начала истории'],
                         config_df['Колонка даты конца истории']],
                'step': config_info.step,
                'time_type': config_info.time_type
            }
        }
    else:
        raise res_check

    if file_info['forecast']['join_output_result']:
        file_info['source']['joined_output_time_col'] = config_df['Измерение времени совмещенное исходящее']

    logging.debug(f'Create_info_time: {time() - start}')

    return file_info


def create_df(info, queue, sample_name, group, need_key=False):
    queue.put({sample_name: load_and_prepare_data(info, sample_name, group, need_key)})
