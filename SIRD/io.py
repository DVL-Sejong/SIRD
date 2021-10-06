from os.path import join, abspath, dirname, split, isfile
from dataclasses import fields
from pathlib import Path
from glob import glob

import pandas as pd

ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATASET_PATH = join(ROOT_PATH, 'dataset')
RESULT_PATH = join(ROOT_PATH, 'results')
SETTING_PATH = join(ROOT_PATH, 'settings')


def load_dataset(case_name):
    regions = load_population(case_name)
    population_df = load_population(case_name)
    infectious_period_df = load_infectious_period(case_name)
    initial_dict = load_initial_dict(case_name)
    sird_dict = load_sird_dict(case_name)

    dataset = {'regions': regions, 'population': population_df,
               'infectious_period': infectious_period_df,
               'initial': initial_dict, 'sird': sird_dict}
    return dataset


def load_population(case_name):
    population_path = join(DATASET_PATH, case_name, 'population.csv')
    population_df = pd.read_csv(population_path, index_col='regions')
    return population_df


def load_regions(case_name):
    population_df = load_population(case_name)
    population_regions = population_df.index.tolist()

    initial_values_path_list = glob(join(DATASET_PATH, case_name, 'initial_values', '*.csv'))
    initial_regions = [split(elem)[1].split('.csv')[0] for elem in initial_values_path_list]

    sird_path_list = glob(join(DATASET_PATH, case_name, 'sird_data', '*.csv'))
    sird_regions = [split(elem)[1].split('.csv')[0] for elem in sird_path_list]

    regions = list(set(population_regions) & set(initial_regions) & set(sird_regions))
    regions.sort()
    return regions


def load_sird_dict(case_name):
    regions = load_regions(case_name)
    sird_dict = dict()

    for region in regions:
        sird_df = load_sird_data(case_name, region)
        sird_dict.update({region: sird_df})

    return sird_dict


def load_sird_data(case_name, region):
    sird_path = join(DATASET_PATH, case_name, 'sird_data', f'{region}.csv')
    sird_df = pd.read_csv(sird_path, index_col='date')
    return sird_df


def load_initial_dict(case_name):
    regions = load_regions(case_name)
    initial_dict = dict()

    for region in regions:
        initial_df = load_initial_data(case_name, region)
        initial_dict.update({region: initial_df})

    return initial_dict


def load_initial_data(case_name, region):
    sird_path = join(DATASET_PATH, case_name, 'initial_values', f'{region}.csv')
    sird_df = pd.read_csv(sird_path, index_col='date')
    return sird_df


def load_infectious_period(case_name):
    period_path = join(DATASET_PATH, case_name, 'infectious_period.csv')
    infectious_period_df = pd.read_csv(period_path, index_col='regions')
    return infectious_period_df


def get_result_path(predict_info, case_name, index):
    result_path = join(RESULT_PATH, case_name, predict_info.get_hash(), str(index).zfill(2))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def save_results(predict_info, case_name, index, region, result_df):
    result_path = get_result_path(predict_info, case_name, index)
    result_df.to_csv(join(result_path, f'{region}.csv'))
    print(f'saving {region} results under {result_path}')


def save_setting(param_class, class_name):
    new_param_dict = dict()
    new_param_dict.update({'hash': param_class.get_hash()})

    for field in fields(param_class):
        if field.name[0] == '_': continue
        new_param_dict.update({field.name: getattr(param_class, field.name)})

    param_df = pd.DataFrame(columns=list(new_param_dict.keys()))
    param_df = param_df.append(new_param_dict, ignore_index=True)
    param_df = param_df.set_index('hash')

    filename = f'{class_name}.csv'
    if isfile(join(SETTING_PATH, filename)):
        df = pd.read_csv(join(SETTING_PATH, filename), index_col='hash')
        if param_class.get_hash() not in df.index.tolist():
            df = param_df.append(df, ignore_index=False)
            df.to_csv(join(SETTING_PATH, filename))
            print(f'updating settings to {join(SETTING_PATH, filename)}')
    else:
        param_df.to_csv(join(SETTING_PATH, filename))
        print(f'saving settings to {join(SETTING_PATH, filename)}')
