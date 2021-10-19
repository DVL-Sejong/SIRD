from SIRD.datatype import get_country_name
from os.path import join, abspath, dirname, split, isfile
from dataclasses import fields
from pathlib import Path
from glob import glob

import pandas as pd


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATASET_PATH = join(ROOT_PATH, 'dataset')
RESULT_PATH = join(ROOT_PATH, 'results')
SETTING_PATH = join(ROOT_PATH, 'settings')


def load_links(country=None):
    link_path = join(DATASET_PATH, 'links.csv')
    link_df = pd.read_csv(link_path, index_col='country')
    return link_df.loc[get_country_name(country), :] if country is not None else link_df


def load_population(country):
    population_path = join(DATASET_PATH, get_country_name(country), 'population.csv')
    population_df = pd.read_csv(population_path, index_col='regions')
    return population_df


def load_regions(country):
    population_df = load_population(country)
    regions = population_df.index.tolist()
    regions.sort()
    return regions


def load_dataset(country, pre_info, test_info):
    regions = load_population(country)
    population_df = load_population(country)
    infectious_period_df = load_infectious_period(country)
    initial_dict = load_initial_dict(country, pre_info, test_info)
    sird_dict = load_sird_dict(country, pre_info.get_sird_info())

    dataset = {'regions': regions, 'population': population_df,
               'infectious_period': infectious_period_df,
               'initial': initial_dict, 'sird': sird_dict}
    return dataset


def load_dict(dict_path):
    path_list = glob(dict_path)

    if len(path_list) == 0:
        raise FileNotFoundError(dict_path)

    data_dict = dict()
    for target_path in path_list:
        elem_name = path_to_name(target_path)
        elem_df = pd.read_csv(target_path, index_col='date')
        data_dict.update({elem_name: elem_df})

    return data_dict


def load_sird_dict(country, sird_info):
    initial_path = join(DATASET_PATH, get_country_name(country), 'sird', sird_info.get_hash(), '*.csv')
    initial_dict = load_dict(initial_path)
    return initial_dict


def load_initial_dict(country, pre_info, test_info):
    initial_path = join(DATASET_PATH, get_country_name(country), 'initial_values',
                        f'{pre_info.get_hash()}_{test_info.get_hash()}', '*.csv')
    initial_dict = load_dict(initial_path)
    return initial_dict


def load_infectious_period(country):
    period_path = join(DATASET_PATH, get_country_name(country), 'infectious_period.csv')
    infectious_period_df = pd.read_csv(period_path, index_col='regions')
    return infectious_period_df


def get_result_path(country, result_hash, region, index):
    result_path = join(RESULT_PATH, get_country_name(country), result_hash,
                       region, str(index).zfill(4))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def save_results(country, result_hash, loader_index, region, result_df):
    result_path = get_result_path(country, result_hash, region, loader_index)
    result_df.to_csv(join(result_path, f'{region}.csv'))
    print(f'saving {region} results under {result_path}')


def save_region_result(country, result_hash, region, region_df):
    result_path = join(RESULT_PATH, get_country_name(country), result_hash)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    saving_path = join(result_path, f'{region}.csv')
    region_df.to_csv(saving_path)
    print(f'saving {region} of {get_country_name(country)} result to {saving_path}')


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


def path_to_name(file_path, extension='.csv'):
    _, file_name = split(file_path)
    file_name = file_name.split(extension)[0]
    return file_name
