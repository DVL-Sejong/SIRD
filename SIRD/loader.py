from SIRD.datatype import PredictInfo, Country, PreprocessInfo
from SIRD.util import get_period, get_predict_period_from_dataset
from SIRD.io import load_regions, load_dataset, load_links, load_initial_dict
from datetime import timedelta


class DataLoader:
    def __init__(self, predict_info, dataset, region):
        self.predict_info = predict_info
        self.test_start = predict_info.test_start
        self.test_end = predict_info.test_end
        self.test_period = get_period(self.test_start, self.test_end)
        self.y_frames = predict_info.y_frames

        self.population = dataset['population'].loc[region, 'population']
        self.infectious_period = dataset['infectious_period'].loc[region, 'infectious_period']
        self.initial_data = dataset['initial'][region]
        self.sird_df = dataset['sird'][region]

        self.region = region

    def __len__(self):
        return len(self.test_period)

    def __getitem__(self, idx):
        index_date = (self.test_period[idx] + timedelta(days=-1)).strftime('%Y-%m-%d')
        y_period = get_period(self.test_period[idx],
                              self.test_period[idx] + timedelta(days=self.y_frames-1), '%Y-%m-%d')

        s = self.sird_df.loc[index_date, 'susceptible']
        i = self.sird_df.loc[index_date, 'infected']
        r = self.sird_df.loc[index_date, 'recovered']
        d = self.sird_df.loc[index_date, 'deceased']

        x = {'S': s, 'I': i, 'R': r, 'D': d, 'date': index_date}
        y = self.sird_df.loc[y_period, :]
        initial_values = {'population': self.population,
                          'infectious_period': self.infectious_period,
                          'r0': self.initial_data.loc[index_date, 'r0'],
                          'mortality_rate': self.initial_data.loc[index_date, 'mortality_rate']}

        return x, y, initial_values


if __name__ == '__main__':
    country = Country.ITALY
    link_df = load_links(country)

    pre_info = PreprocessInfo(country=country, start=link_df['start_date'], end=link_df['end_date'],
                              increase=True, daily=True, remove_zero=True,
                              smoothing=True, window=5, divide=False)

    test_info = PreprocessInfo(country=country, start=link_df['start_date'], end=link_df['end_date'],
                               increase=True, daily=True, remove_zero=True,
                               smoothing=True, window=5, divide=False)

    y_frames = 3
    initial_dict = load_initial_dict(country, pre_info, test_info)
    predict_dates = get_predict_period_from_dataset(pre_info, initial_dict, y_frames)
    predict_info = PredictInfo(y_frames=y_frames, test_start=predict_dates[0], test_end=predict_dates[-1])

    dataset = load_dataset(country, pre_info, test_info)
    loader = DataLoader(predict_info, dataset, load_regions(country)[0])

    print(loader[0])
    print()
    print(loader[1])
    print()
    print(loader[len(loader)-1])
