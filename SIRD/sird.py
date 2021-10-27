from SIRD.datatype import DatasetInfo, Country, PreprocessInfo
from SIRD.io import load_regions, load_dataset, load_links, load_initial_dict
from SIRD.io import save_setting, save_region_result, save_result_dict
from SIRD.loader import DataLoader
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from SIRD.util import get_predict_period_from_dataset, generate_dataframe


class SIRD:
    def __init__(self, predict_info, initial_values):
        self.y_frames = predict_info.y_frames

        self.population = initial_values['population']
        self.infectious_period = initial_values['infectious_period']
        self.r0 = initial_values['r0']
        self.mortality_rate = initial_values['mortality_rate']

        self.gamma = 1 / self.infectious_period
        self.mu = self.gamma * self.mortality_rate
        self.lamb = self.r0 * (self.gamma + self.mu)

    def ds_dt(self, s, i, lamb):
        return -lamb * s * i

    def di_dt(self, s, i, lamb, gamma, mu):
        return (lamb * s * i) - (gamma * i) - (mu * i)

    def dr_dt(self, i, gamma):
        return gamma * i

    def dd_dt(self, i, mu):
        return mu * i

    def equations(self, t, y0, lamb, gamma, mu):
        s, i, r, d = y0
        return [self.ds_dt(s, i, lamb),
                self.di_dt(s, i, lamb, gamma, mu),
                self.dr_dt(i, gamma),
                self.dd_dt(i, mu)]

    def predict(self, x):
        solution = solve_ivp(self.equations, (0, self.y_frames), [x['S'], x['I'], x['R'], x['D']],
                             args=(self.lamb, self.gamma, self.mu),
                             t_eval=np.linspace(0, self.y_frames, self.y_frames * 2))

        pred_start = datetime.strptime(x['date'], '%Y-%m-%d') + timedelta(days=1)
        pred_period = [(pred_start + timedelta(days=index)).strftime('%Y-%m-%d') for index in range(self.y_frames)]
        predict_df = pd.DataFrame(index=pred_period, columns=['susceptible', 'infected', 'recovered', 'deceased'])
        predict_df.index.name = 'date'

        s, i, r, d = solution.y
        predict_df.loc[:, 'susceptible'] = [s[index] for index in range(1, len(solution.t) + 1, 2)]
        predict_df.loc[:, 'infected'] = [i[index] for index in range(1, len(solution.t) + 1, 2)]
        predict_df.loc[:, 'recovered'] = [r[index] for index in range(1, len(solution.t) + 1, 2)]
        predict_df.loc[:, 'deceased'] = [d[index] for index in range(1, len(solution.t) + 1, 2)]

        return predict_df
