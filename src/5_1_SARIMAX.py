import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

# statsmodels
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# skforecast
import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.plot import set_dark_theme
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_sarimax
from skforecast.model_selection import grid_search_sarimax

import warnings
warnings.filterwarnings('once')

color = '\033[1m\033[38;5;208m' 
print(f"{color}Versión skforecast: {skforecast.__version__}")
print(f"{color}Versión statsmodels: {statsmodels.__version__}")
print(f"{color}Versión pandas: {pd.__version__}")
print(f"{color}Versión numpy: {np.__version__}")



def load_datasets():
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train_preprocessed.pkl')
    return root, train


def create_forecaster(train, series, p, d, q, P, D, Q, m):
    forecaster = ForecasterSarimax(
        regressor=Sarimax(
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            maxiter=500),
    )
    forecaster.fit(
        y    = train['target'],
        exog = train[series]
    )
    return forecaster


def backtesting(data, train, forecaster, steps):
    cv = TimeSeriesFold(
        steps              = 24,
        initial_train_size = len(train),
        refit              = True,
    )
    metrica, predicciones = backtesting_sarimax(
        forecaster            = forecaster,
        y                     = data['target'],
        cv                    = cv,
        metric                = root_mean_squared_error,
        n_jobs                = "auto",
        suppress_warnings_fit = True,
        verbose               = False,
        show_progress         = True
    )
    return metrica, predicciones


def main():
    root, train = load_datasets()
    data = train.copy()
    end_val = '2022-08-31 23:59:59'
    train = train.loc[:end_val]

    series = ['target', 'temperature', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity'] 

    data = data[series].copy()
    data_train = train[series].copy()
    
    p, d, q = 2, 1, 1
    P, D, Q = 0, 0, 0, 0
    m = 0

    forecaster = create_forecaster(data_train, series, p, d, q, P, D, Q, m)
    metrica, predicciones = backtesting(data, data_train, forecaster, m)
    print(metrica)
    predicciones.to_pickle(root.DIR_DATA_ANALYTICS + 'SARIMAX_predictions_val.pkl')


if __name__ == "__main__":
    main()