from SIRD.datatype import Country, PreprocessInfo, DatasetInfo
from SIRD.io import load_links, load_initial_dict, load_dataset, load_regions
from SIRD.io import save_setting, save_region_result, save_result_dict
from SIRD.loader import DataLoader
from SIRD.sird import SIRD
from SIRD.util import get_predict_period_from_dataset, generate_dataframe

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument(
        "--country", type=str, default='italy',
        choices=['italy', 'india', 'us', 'china'],
        help="Country name"
    )

    parser.add_argument(
        "--y_frames", type=int, default=3,
        help="Number of x frames for generating dataset"
    )

    args = parser.parse_args()
    return args


def main(args):
    country = Country(args.country.upper())
    link_df = load_links(country)

    pre_info = PreprocessInfo(country=country, start=link_df['start_date'], end=link_df['end_date'],
                              increase=True, daily=True, remove_zero=True,
                              smoothing=True, window=5, divide=False)

    test_info = PreprocessInfo(country=country, start=link_df['start_date'], end=link_df['end_date'],
                               increase=False, daily=True, remove_zero=True,
                               smoothing=True, window=5, divide=False)

    initial_dict = load_initial_dict(country, pre_info, test_info)
    predict_dates = get_predict_period_from_dataset(pre_info, initial_dict, args.y_frames)
    predict_info = DatasetInfo(x_frames=0, y_frames=args.y_frames,
                               test_start=predict_dates[0], test_end=predict_dates[-1])
    save_setting(predict_info, 'data_info')

    dataset = load_dataset(country, pre_info, test_info)
    result_hash = f'{pre_info.get_hash()}_{test_info.get_hash()}_{predict_info.get_hash()}'

    result_dict = dict()
    for region in load_regions(country):
        print(f'Predict {region.upper()}')
        loader = DataLoader(predict_info, dataset, region)
        region_df = generate_dataframe([], ['susceptible', 'infected', 'recovered', 'deceased'], 'date')

        day_dict = dict()
        for loader_index in range(len(loader)):
            x, y, initial_values = loader[loader_index]
            model = SIRD(predict_info, initial_values)
            predict_df = model.predict(x)
            day_dict.update({str(loader_index).zfill(4): predict_df})
            region_df = region_df.append(predict_df.iloc[0, :])

        result_dict.update({region: day_dict})
        save_region_result(country, result_hash, region, region_df)

    save_result_dict(country, result_hash, result_dict)


if __name__ == '__main__':
    args = get_args()
    main(args)
