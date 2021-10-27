# SIRD
Susceptible-Infectious-Recovered-Deceased model (SIRD) 



### Dataset

Dataset for this repository can be downloaded [here](https://github.com/DVL-Sejong/COVID_DataProcessor). You must download data and preprocess the data for the model. Dataset for the model should be under `\dataset\country_name`.




### SIRD

```
$ git clone https://github.com/DVL-Sejong/SIRD.git
$ cd SIRD
$ python main.py
```

- Arguments
  - country: Italy, India, US, China are available
  - y_frames: Number of y frames for generating dataset



### Citation


Algorithm for optimizing r0 value is based on this paper:

```
@article{zhou2020preliminary,
  title={Preliminary prediction of the basic reproduction number of the Wuhan novel coronavirus 2019-nCoV},
  author={Zhou, Tao and Liu, Quanhui and Yang, Zimo and Liao, Jingyi and Yang, Kexin and Bai, Wei and Lu, Xin and Zhang, Wei},
  journal={Journal of Evidence-Based Medicine},
  volume={13},
  number={1},
  pages={3--7},
  year={2020},
  publisher={Wiley Online Library}
}
```
