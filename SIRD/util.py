from datetime import date, datetime, timedelta

import pandas as pd


def get_period(start_date, end_date, out_date_format=None):
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, get_date_format(start_date))
    if type(start_date) == date:
        start_date = convert_date_to_datetime(start_date)

    if type(end_date) == str:
        end_date = datetime.strptime(end_date, get_date_format(end_date))
    if type(end_date) == date:
        end_date = convert_date_to_datetime(end_date)

    duration = (end_date - start_date).days + 1
    period = [start_date + timedelta(days=i) for i in range(duration)]

    if out_date_format is None:
        return period
    else:
        return [elem.strftime(out_date_format) for elem in period]


def get_date_format(date: str) -> str:
    formats = ['%Y-%m-%d', '%y%m%d', '%m-%d-%Y', '%m/%d/%y']
    for format in formats:
        if validate(date, format):
            return format

    return ''


def validate(date: str, format: str) -> bool:
    try:
        if date != datetime.strptime(date, format).strftime(format):
            raise ValueError
        return True
    except ValueError:
        return False


def convert_date_to_datetime(target_date):
    return datetime(year=target_date.year, month=target_date.month, day=target_date.day)


def get_common_dates(dates1, dates2):
    start1 = datetime.strptime(dates1[0], '%Y-%m-%d')
    start2 = datetime.strptime(dates2[0], '%Y-%m-%d')
    start_date = start2 if start1 < start2 else start1

    end1 = datetime.strptime(dates1[-1], '%Y-%m-%d')
    end2 = datetime.strptime(dates2[-1], '%Y-%m-%d')
    end_date = end1 if end1 < end2 else end2

    common_dates = get_period(start_date, end_date, out_date_format='%Y-%m-%d')
    return common_dates


def get_predict_period_from_dataset(pre_info, initial_dict, y_frames):
    pre_dates = get_period(pre_info.start, pre_info.end, '%Y-%m-%d')

    initial_df = initial_dict[next(iter(initial_dict))]
    initial_dates = initial_df.index.tolist()

    predict_dates = get_common_dates(initial_dates, pre_dates)[1:-y_frames + 1]
    return predict_dates


def generate_dataframe(index, columns, index_name):
    df = pd.DataFrame(index=index, columns=columns)
    df.index.name = index_name
    return df
