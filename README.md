# SIRD
Susceptible-Infectious-Recovered-Deceased model (SIRD) 

### Dataset

We have preprocessed COVID-19 dataset of US, Italy, Chana, and India. The preprocessed dataset is committed on [here](dataset/). Raw dataset of each country can be found here: 
- US: [JHU CSSE COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data), [link](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/)
- Italy: [Dati COVID-19 Italia](https://github.com/pcm-dpc/COVID-19), [link](https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv)
- China: [JHU CSSE COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data), [link](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/)
- India: [COVID19-India API](https://github.com/covid19india/api), [link](https://api.covid19india.org/csv/latest/states.csv)

Repository for reprocessing raw data is [here](https://github.com/DVL-Sejong/COVID_DataProcessor)


### SIRD

to be updated


Algorithm for optimizing r0 value is based on this paper:

```
@techreport{fernandez2020estimating,
  title={Estimating and simulating a SIRD model of COVID-19 for many countries, states, and cities},
  author={Fern{\'a}ndez-Villaverde, Jes{\'u}s and Jones, Charles I},
  year={2020},
  institution={National Bureau of Economic Research}
}
```
