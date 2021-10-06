from SIRD.datatype import PredictInfo
from SIRD.io import load_population, load_sird_dict, load_initial_dict, load_infectious_period, load_regions
from SIRD.io import save_setting, save_results
from SIRD.util import get_period
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SIRD:
    def __init__(self, R0, M, P, initial_conditions, tf):
        self.R0 = R0
        self.M = M
        self.P = P
        self.initial_conditions = initial_conditions
        self.tf = tf

    def dSdt(self, S, I, lamb):
        return -lamb * S * I

    def dIdt(self, S, I, lamb, gamma, mu):
        return lamb * S * I - gamma * I - mu * I

    def dRdt(self, I, gamma):
        return gamma * I

    def dDdt(self, I, mu):
        return mu * I

    def eqns(self, t, y, lamb, gamma, mu):
        S, I, R, D = y
        return [self.dSdt(S, I, lamb),
                self.dIdt(S, I, lamb, gamma, mu),
                self.dRdt(I, gamma),
                self.dDdt(I, mu)]

    def setup(self):
        population = self.initial_conditions['population']
        cases = self.initial_conditions['cases']
        recovered = self.initial_conditions['recovered']
        deaths = self.initial_conditions['deaths']

        # Compute initial values
        self.population = population
        initial_S = (population - cases) / population
        initial_R = recovered / population
        initial_D = deaths / population
        initial_I = 1 - initial_S - initial_R - initial_D
        self.y0 = [initial_S, initial_I, initial_R, initial_D]

        # Compute coefficients
        self.gamma = 1 / self.P
        self.mu = self.gamma * self.M
        self.lamb = self.R0 * (self.gamma + self.mu)

    def solve(self):
        self.setup()

        t_span = (0, self.tf)  # tf is number of days to run simulation for, defaulting to 300

        self.soln = solve_ivp(self.eqns, t_span, self.y0,
                              args=(self.lamb, self.gamma, self.mu),
                              t_eval=np.linspace(0, self.tf, self.tf * 2))
        return self

    def plot(self, ax=None, susceptible=True):
        S, I, R, D = self.soln.y
        t = self.soln.t
        N = self.population

        print(f"For a population of {N} people, after {t[-1]:.0f} days there were:")
        print(f"{D[-1] * 100:.1f}% total deaths, or {D[-1] * N:.0f} people.")
        print(f"{R[-1] * 100:.1f}% total recovered, or {R[-1] * N:.0f} people.")
        print(
            f"At the virus' maximum {I.max() * 100:.1f}% people were simultaneously infected, or {I.max() * N:.0f} people.")
        print(f"After {t[-1]:.0f} days the virus was present in less than {I[-1] * N:.0f} individuals.")
        print()

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title("Covid-19 spread")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Number")
        if susceptible:
            ax.plot(t, S * N, label="Susceptible", linewidth=2, color='blue')
        ax.plot(t, I * N, label="Infected", linewidth=2, color='orange')
        ax.plot(t, R * N, label="Recovered", linewidth=2, color='green')
        ax.plot(t, D * N, label="Deceased", linewidth=2, color='black')
        ax.legend()

        return ax

    def predict(self):
        self.solve()
        s, i, r, d = self.soln.y
        t = self.soln.t
        population = self.population

        new_s = []
        new_i = []
        new_r = []
        new_d = []

        for index in range(1, len(t) + 1, 2):
            new_s.append(s[index] * population)
            new_i.append(i[index] * population)
            new_r.append(r[index] * population)
            new_d.append(d[index] * population)

        return new_s, new_i, new_r, new_d


def region_predict(R0, M, P, initial_conditions, tf):
    sird = SIRD(R0, M, P, initial_conditions, tf)
    s, i, r, d = sird.predict()
    sird.plot()

    return s, i, r, d


if __name__ == '__main__':
    case_name = 'US'
    population_df = load_population(case_name)

    infectious_period_df = load_infectious_period(case_name)
    sird_dict = load_sird_dict(case_name)
    initial_dict = load_initial_dict(case_name)

    predict_info = PredictInfo(y_frames=3, test_start='210101', test_end='210111')
    save_setting(predict_info, 'predict_info')

    columns = ['susceptible', 'infectious', 'recovered', 'deceased']
    test_period = get_period(predict_info.test_start, predict_info.test_end, out_date_format='%Y-%m-%d')

    result_dict = dict()

    regions = load_regions(case_name)
    for region in regions:
        print(region)
        for index, predict_start_date in enumerate(test_period):
            r0 = initial_dict[region].loc[predict_start_date, 'r0']
            m = initial_dict[region].loc[predict_start_date, 'mortality_rate']
            p = infectious_period_df.loc[region, 'infectious_period']

            initial_conditions = {'population': population_df.loc[region, 'population'],
                                  'cases': sird_dict[region].loc[predict_start_date, 'infected'],
                                  'deaths': sird_dict[region].loc[predict_start_date, 'deceased'],
                                  'recovered': sird_dict[region].loc[predict_start_date, 'recovered']}

            s, i, r, d = region_predict(r0, m, p, initial_conditions, predict_info.y_frames)
            predict_period = get_period(predict_start_date, datetime.strptime(predict_start_date, '%Y-%m-%d') + timedelta(days=predict_info.y_frames-1), out_date_format='%Y-%m-%d')

            region_df = pd.DataFrame(index=predict_period, columns=columns)
            region_df.index.name = 'date'
            region_df['susceptible'] = s
            region_df['infectious'] = i
            region_df['recovered'] = r
            region_df['deceased'] = d

            save_results(predict_info, case_name, index, region, region_df)
