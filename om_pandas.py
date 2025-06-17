from collections import namedtuple
from re import findall

import numpy as np
import pandas as pd


def make_week_to_dt_mapping(args_for_date_mapping, kwargs_for_date_mapping,
                            column_names):
    """
    Produces a correspondence dataframe for om_day - om_week - uniformly spaced pandas datetime.

    Requires mapping file to work (OM Days in one column and corresponding OM Weeks in the other).

    If some broken non-7-days weeks are empty due to absence in the source data
    make shure NOT to include them into mapping file.

    :param args_for_date_mapping: positional arguments to convert mapping file into df.
    :param kwargs_for_date_mapping: key-word arguments to convert mapping file into df.
    :param column_names: dict, containing 'week_column' key for week column name and 'day_column' for day column name.
    """

    mapping_df = file_to_dataframe(*args_for_date_mapping, **kwargs_for_date_mapping)
    mapping_df.rename(columns={v: k for (k, v) in column_names.items()}, inplace=True)
    mapping_df.sort_values(by='day_column', inplace=True)

    if len(mapping_df.index) == 0:
        raise ValueError('Mapping dataframe has no entries')

    mapping_df['week_num'] = mapping_df.week_column.str.extract(r'W(\d*)').astype(int)

    trustable = mapping_df.query('2 <= week_num <= 51')

    if len(trustable.index) > 0:
        pivot_date = trustable.iloc[0].day_column
    else:
        pivot_date = mapping_df.iloc[0].day_column

    start_year = mapping_df.iloc[0].day_column.year
    end_year = mapping_df.iloc[-1].day_column.year + 1

    generated = pd.DataFrame()
    generated['start'] = pd.date_range(start=pivot_date - pd.tseries.offsets.DateOffset(days=21),
                                       periods=(end_year - start_year) * 53,
                                       freq='7D')
    generated['end'] = generated.start + pd.tseries.offsets.DateOffset(days=6)

    mapping_df['out_dt'] = mapping_df.day_column - pd.offsets.Week(weekday=0)
    mapping_df.loc[mapping_df.day_column.dt.dayofweek == 0, 'out_dt'] = mapping_df.day_column

    mapping_df['one'] = 1

    mapping_df['days_in_om_week'] = mapping_df.groupby(by='week_column')['one'].transform('sum')

    mapping_df['days_in_iso_week'] = mapping_df.groupby(by='out_dt')['one'].transform('sum')

    return mapping_df


def om_weeks_to_dt(full_df, om_week_column, value_columns, categorical_columns,
                   dates_mapping):
    """
    Converts dataframe with om_weeks to one on pandas datetime.
    Automatically manages allocation between mismatching weeks.
    Caution!!! It uses simple proportion for recalculation

    :param full_df: df for conversion.
    :param om_week_column: name of the column containing om_weeks.
    :param value_columns: list of numerical columns to be allocated in case of week mismatch.
    :param categorical_columns: list of columns to group by.
    :param dates_mapping: df with dates mapping produced by make_week_to_dt_mapping.
    :return:
    """

    def extract_dt_by_week(week):
        try:
            return dates_mapping.loc[dates_mapping.week_column == week].iloc[0].out_dt
        except IndexError:
            raise ValueError(f'Dates mapping does not contain week {week}')

    def safe_week_conversion(week):
        if week not in mapping_cache:
            mapping_cache[week] = extract_dt_by_week(week)
        return mapping_cache[week]

    source_columns = full_df.columns.values.tolist()

    mapping_cache = {}

    full_df['week_num'] = full_df[om_week_column].str.extract(r'W(\d*)').astype(int)
    full_df['year_num'] = full_df[om_week_column].str.extract(r'_(\d*)').astype(int) + 2000

    full_df['datetime'] = full_df.query('2 <= week_num <= 51')[om_week_column].apply(safe_week_conversion)

    unreliable = full_df.query('week_num < 2 or week_num > 51')

    full_df.drop(full_df[(full_df.week_num < 2) | (full_df.week_num > 51)].index, inplace=True)

    unreliable = unreliable.merge(dates_mapping, left_on=om_week_column, right_on='week_column')

    suffix = '_by_day'

    agg_dict = {'days_in_iso_week': 'first'}
    for col in source_columns:
        if col != om_week_column and col not in value_columns and col not in categorical_columns:
            agg_dict[col] = 'first'

    for col in value_columns:
        name = col + suffix
        unreliable[name] = unreliable[col] / unreliable.days_in_om_week
        agg_dict[name] = np.sum

    unreliable = unreliable.groupby(by=['out_dt', *categorical_columns]).agg(agg_dict).reset_index()

    for col in value_columns:
        unreliable[col] = unreliable[col + suffix] / (unreliable.days_in_iso_week / 7)

    unreliable.rename(columns={'out_dt': 'datetime'}, inplace=True)

    full_df = pd.concat([full_df, unreliable])

    full_df[om_week_column] = full_df.datetime

    full_df = full_df[source_columns]

    return full_df.sort_values(om_week_column)


def dt_to_om_weeks(full_df, dt_column, value_columns, categorical_columns,
                   dates_mapping, uniform=True):
    """
    Converts dataframe with pandas datetime om_weeks to one on.
    Automatically manages allocation between mismatching weeks.
    Caution!!! It uses simple proportion for recalculation

    :param full_df: df for conversion.
    :param dt_column: name of the column containing datetime.
    :param value_columns: list of numerical columns to be allocated in case of week mismatch.
    :param categorical_columns: list of columns to group by.
    :param dates_mapping: df with dates mapping produced by make_week_to_dt_mapping.
    :param uniform: flag for checking uniform translation of weeks or not.
    :return:
    """

    def extract_week_by_dt(dt):
        try:
            week = dates_mapping.loc[dates_mapping.out_dt == dt].iloc[0].week_column
        except IndexError:
            raise ValueError(f'Dates mapping does not contain datetime {dt}')

        week_num = int(findall(r'^W(\d*)', week)[0])
        mapping_cache[dt] = week if 2 <= week_num <= 51 else None

    def safe_week_conversion(dt):
        if dt not in mapping_cache:
            mapping_cache[dt] = extract_week_by_dt(dt)
        if mapping_cache[dt] is None:
            return np.nan
        return mapping_cache[dt]

    if uniform:
        source_columns = full_df.columns.values.tolist()

        mapping_cache = {}

        full_df['converted_weeks'] = full_df[dt_column].apply(safe_week_conversion)

        unreliable = full_df.loc[full_df.converted_weeks.isnull()]

        full_df.drop(full_df[full_df.converted_weeks.isnull()].index, inplace=True)

        unreliable = unreliable.merge(dates_mapping, left_on=dt_column, right_on='out_dt')

        suffix = '_by_day'

        agg_dict = {dt_column: 'first'}
        for col in source_columns:
            if col != dt_column and col not in value_columns and col not in categorical_columns:
                agg_dict[col] = 'first'

        for col in value_columns:
            name = col + suffix
            unreliable[name] = unreliable[col] / 7
            agg_dict[name] = np.sum

        unreliable = unreliable.groupby(by=['week_column', *categorical_columns]).agg(agg_dict).reset_index()

        for col in value_columns:
            unreliable[col] = unreliable[col + suffix]

        unreliable.rename(columns={'week_column': 'converted_weeks'}, inplace=True)

        full_df = pd.concat([full_df, unreliable])

        full_df[dt_column] = full_df.converted_weeks

        full_df = full_df[source_columns]
    else:
        full_df = full_df.merge(dates_mapping[['day_column', 'week_column']], how='left', left_on=dt_column,
                                right_on='day_column').drop(columns=['day_column', dt_column])
        full_df = full_df[['week_column'] + [i for i in full_df.columns if i != 'week_column']].rename(
            columns={'week_column': dt_column})
    return full_df


def om_time_to_datetime(series):
    """Функция для конвертации дат формата ОМ в datetime.

    Args:
        series (pd.Series): Серия с датой в формате OM.
    """
    date_example = series.iloc[0]

    if len(date_example) == 4:
        # only year
        return pd.to_datetime(series.str.lstrip('FY'), format='%y')
    elif date_example[0] == "Q":
        # quarter and year
        return pd.to_datetime(series.str.split('_').apply(lambda x: ''.join(x[::-1])))
    elif len(date_example) == 6:
        # month and year
        return pd.to_datetime(series, format='%b %y')
    elif date_example[0] == "W":
        raise ValueError('Weeks can not be converted 1 to 1. Use om_weeks_to_dt function for such a conversion')
    elif 8 <= len(date_example) <= 9:
        # day, month and year
        return pd.to_datetime(series, format='%d %b %y')
    else:
        raise ValueError(
            f"Can't convert date of unsupported format {date_example} from Optimacros format to pandas datetime")


def datetime_to_om_time(series, date_format='Months'):
    """
    Функция для конвертации дат формата datetime (dd-MM-YY) в даты формата OM.

    :param series: Серия формата datetime.
    :param date_format: Формат результата, допустимые варианты: "Years", "Months", "Weeks", "Days".
    По умолчанию 'Months'.
    :return: Серия выбранного формата из OM.
    """

    series = pd.to_datetime(series, format='%Y-%m-%d')
    if date_format in ['Years', 'om_year']:
        return 'FY' + series.dt.strftime('%y')
    elif date_format in ['Quartes', 'om_quarter']:
        return series.dt.toperiod('Q').astype(str).apply(lambda x: "".join([x[-2:], x[2:4]]))
    elif date_format in ['Months', 'om_month']:
        return series.dt.strftime('%b %y')
    elif date_format in ['Weeks', 'om_weeks']:
        raise ValueError('Weeks can not be converted 1 to 1. Use dt_to_om_weeks function for such a conversion')
    elif date_format in ['Days', 'om_days']:
        return series.dt.strftime('%d %b %y').str.lstrip('0')
    else:
        raise ValueError(f"Can't convert date to unsupported format {date_format}")


def csv_to_dataframe(filename, sep=";", encoding="UTF 8"):
    df = pd.read_csv(filepath_or_buffer=filename, sep=sep, encoding=encoding)
    df.reset_index(inplace=True)
    offset = df.loc[1].count() - df.loc[0].count()
    df.columns = df.loc[0][-1 * offset - 1:-1].index.to_list() + df.loc[0][offset:].to_list()
    df = df[1:]
    df = df.reset_index(drop=True)
    return df


def excel_to_dataframe(filename):
    df = pd.read_excel(filename)
    dim_count = df.loc[0].count() - 1
    dims = df.loc[0][0:dim_count].to_list()
    cubes = df.loc[1][dim_count:].to_list()
    df.columns = dims + cubes
    df = df[2:]
    df = df.reset_index(drop=True)
    return df


def file_to_dataframe(file_path,
                      file_type,
                      sep=";",
                      encoding="UTF 8",
                      columns=None,  # type: ignore
                      invert_columns=False,
                      index_column=None,  # type: ignore
                      columns_to_convert=None):  # type: ignore
    """
        Function for convert files to DataFrame, where dimension and cubes in columns.
        Work with multicubes, where all dimensions in rows and cubes in columns.
    """
    if file_type in ['csv', 'txt']:
        df = csv_to_dataframe(filename=file_path, sep=sep, encoding=encoding)
    elif file_type in ['om_csv', 'om_txt']:
        df = pd.read_csv(filepath_or_buffer=file_path, sep=sep, encoding=encoding)
    elif file_type in ['xlsx']:
        df = excel_to_dataframe(filename=file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_type}")
    if columns is not None:
        columns_list = columns
        if invert_columns:
            columns_list = []
            for i in df.columns.to_list():
                if i not in columns:
                    columns_list.append(i)
        df = df[columns_list]
    if columns_to_convert is not None:
        for column_type in columns_to_convert.keys():
            df = convert_columns(df, columns_to_convert[column_type], column_type)
    if index_column is not None:
        df = df.set_index(index_column)
    return df


def convert_int_om_to_pandas(series):
    series = series.replace(to_replace=r",", value=".", regex=True)
    series = series.apply(np.float64)
    series = series.apply(np.rint)
    series = series.apply(np.int64)
    return series


def convert_float_om_to_pandas(series):
    series = series.replace(to_replace=r",", value=".", regex=True)
    series = series.apply(np.float64)
    return series


def convert_boolean_om_to_pandas(series):
    if type(series.loc[0]) == str:
        series = series.apply(lambda x: True if x == 'true' else False)

    elif type(series[0]) == np.int64:
        series = series.apply(bool)
    return series


def convert_columns(df, columns, data_type):
    if data_type == 'int':
        for i in columns:
            df[i] = convert_int_om_to_pandas(df[i])
    elif data_type == 'float':
        for i in columns:
            df[i] = convert_float_om_to_pandas(df[i])
    elif data_type == 'bool':
        for i in columns:
            df[i] = convert_boolean_om_to_pandas(df[i])
    elif data_type == 'om_date':
        for i in columns:
            df[i] = om_time_to_datetime(df[i])
    elif data_type == 'date':
        for i in columns:
            df[i] = datetime_to_om_time(df[i])
    else:
        raise ValueError(f"Unsupported data type of columns: {data_type}")
    return df


def dataframe_to_list(df, columns, invert_columns=False):
    columns_list = columns
    if invert_columns:
        columns_list = []
        for i in df.columns.to_list():
            if i not in columns:
                columns_list.append(i)
    gb = df.groupby(columns_list)
    gb_list = [gb.get_group(x) for x in gb.groups.keys()]
    return gb_list


def calculate_dates(time_column,
                    info,
                    key):
    """
    Рассчитывает граничные даты

    :param time_column: Колонка с временем;
    :param info: Именованный кортеж. Содержит информацию о параметрах прогнозирования;
    :return: Именованный кортеж. Содержит информацию о граничных датах
    """
    dp = info.source['detailed_preferences']
    dp_df = dp["df"]
    dp_index = dp_df[dp_df["key"] == key].index[0]
    n_periods = dp_df.loc[dp_index, 'periods_to_forecast']

    time_step = info.source['fact']['time_column']['step']

    train_start = min(time_column)
    test_end = max(time_column)

    # Годы
    if time_step == 'YS':
        n_per_season = 1  # Тут спорный вопрос
        horizon_cv = f'{int(n_periods * 365.2468)} days'

    # Кварталы
    elif time_step == 'QS':
        n_per_season = 4  # Тут спорный вопрос
        horizon_cv = f'{int(n_periods * 91.3117)} days'

    # Месяца
    elif time_step == 'MS':
        n_per_season = 12
        horizon_cv = f'{int(n_periods * 29.3)} days'

    # Недели
    elif time_step == 'W' or time_step == 'W-MON':
        n_per_season = 4
        horizon_cv = f'{n_periods * 7} days'

    # Дни
    elif time_step == 'D':
        n_per_season = 7
        horizon_cv = f'{n_periods} days'

    # Часы
    elif time_step == 'H':
        n_per_season = 24
        horizon_cv = f'{n_periods} hours'

    # Минуты
    elif time_step == 'T' or time_step == 'min':
        n_per_season = 60
        horizon_cv = f'{n_periods} minutes'

    # Секунды
    elif time_step == 'S':
        n_per_season = 3600
        horizon_cv = f'{n_periods} seconds'

    DatesInfo = namedtuple('DatesInfo', ['n_per_season', 'train_start', 'test_end',
                                         'horizon_cv', 'time_step', 'n_periods'])
    dates = DatesInfo(n_per_season, train_start, test_end,
                      horizon_cv, time_step, n_periods)
    return dates


def convert_dates(info,
                  df,
                  num_columns,
                  group_columns,
                  time_column,
                  time_type):
    if time_type == 'om_weeks':
        mapping_df = info.source.get('mapping_df', pd.DataFrame())
        if mapping_df.empty:
            column_names = {'day_column': 'Days', 'week_column': 'Week', }
            mapping_df = make_week_to_dt_mapping([info.source['mapping_path'], 'om_csv'], {'columns': ['Days', 'Week'],
                                                                                           'columns_to_convert': {
                                                                                               'om_date': ['Days']}},
                                                 column_names)

            info.source['mapping_df'] = mapping_df
        for time_col in time_column:
            df = om_weeks_to_dt(df,
                                time_col,
                                num_columns,
                                group_columns,
                                mapping_df)
    elif time_type in ['om_days', 'om_month', 'om_quarter', 'om_year']:
        df = convert_columns(df, time_column, 'om_date')
    else:
        for series in time_column:
            df[series] = pd.to_datetime(df[series], format='%d.%m.%Y')

    return df
