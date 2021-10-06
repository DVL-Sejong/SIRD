from SIRD.datatype import PredictInfo
from SIRD.io import load_regions, load_dataset

from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, predict_info, dataset, region):
        self.predict_info = predict_info
        self.test_start = predict_info.test_start
        self.test_end = predict_info.test_end
        self.y_frames = predict_info.y_frames

        self.population = dataset['population'].loc[region, 'population']
        self.infectious_period = dataset['infectious_period'].loc[region, 'infectious_period']
        self.initiate(dataset, region)

        self.region = region

    def initiate(self, dataset, region):
        self.start = (self.test_start + timedelta(days=-1)).strftime('%Y-%m-%d')
        self.end = (self.test_end + timedelta(days=self.y_frames-1)).strftime('%Y-%m-%d')

        self.r0_values = dataset['initial'][region].loc[self.start:self.end, 'r0']
        self.mortality_rates = dataset['initial'][region].loc[self.start:self.end, 'mortality_rate']
        self.sird_df = dataset['sird'][region].loc[self.start:self.end, :]

        self.len = (datetime.strptime(self.end, '%Y-%m-%d') - datetime.strptime(self.start, '%Y-%m-%d')).days + 1

    def __len__(self):
        return (self.test_end - self.test_start).days + 1

    def __getitem__(self, idx):
        population = self.population
        infectious_period = self.infectious_period
        r0 = self.r0_values.iloc[idx]
        mortality_rate = self.mortality_rates.iloc[idx]

        sird_index = self.sird_df.index.tolist()
        s = self.sird_df.loc[sird_index[idx], 'susceptible']
        i = self.sird_df.loc[sird_index[idx], 'infected']
        r = self.sird_df.loc[sird_index[idx], 'recovered']
        d = self.sird_df.loc[sird_index[idx], 'deceased']

        x = {'S': s, 'I': i, 'R': r, 'D': d}
        y = {'S': self.sird_df.loc[sird_index[idx+1]:sird_index[idx+self.y_frames], 'susceptible'],
             'I': self.sird_df.loc[sird_index[idx+1]:sird_index[idx+self.y_frames], 'infected'],
             'R': self.sird_df.loc[sird_index[idx+1]:sird_index[idx+self.y_frames], 'recovered'],
             'D': self.sird_df.loc[sird_index[idx+1]:sird_index[idx+self.y_frames], 'deceased']}
        initial_values = {'population': population, 'infectious_period': infectious_period,
                          'r0': r0, 'mortality_rate': mortality_rate}

        return x, y, initial_values


if __name__ == '__main__':
    case_name = 'US'
    predict_info = PredictInfo(y_frames=3, test_start='210101', test_end='210108')
    dataset = load_dataset(case_name)
    regions = load_regions(case_name)

    loader = DataLoader(predict_info, dataset, regions[0])
    print(loader[0])
    print()
    print(loader[1])
    print()
    print(loader[len(loader)-1])
