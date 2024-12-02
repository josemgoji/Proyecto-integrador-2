import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import plotly.graph_objects as go

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
from skforecast.utils import save_forecaster
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
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            trend='c',
            maxiter=500),
    )
    forecaster.fit(
        y    = train['target'],
        exog = train[series]
    )
    return forecaster


def backtesting(data, train, forecaster, steps):
    cv = TimeSeriesFold(
        steps              = 7,
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

def save_model(forecaster,name):
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    save_forecaster(
    forecaster, 
    file_name = root.DIR_DATA_ANALYTICS + name, 
    save_custom_functions = False, 
    verbose = False
)


def main():
    root, train = load_datasets()
    data = train.copy()
    end_val = '2022-08-31 23:59:59'
    val = train.loc[end_val:]
    train = train.loc[:end_val]
    

    series = ['target', 'temperature', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity'] 

    data = data[series].copy()
    data_train = train[series].copy()
    
    p, d, q = 1, 1, 1
    P, D, Q = 0, 1, 1
    m = 24

    forecaster = create_forecaster(data_train, series, p, d, q, P, D, Q, m)
    metrica, predicciones = backtesting(data, data_train, forecaster, m)
    print(metrica)
    predicciones.rename(columns={'pred': 'target'}, inplace=True)
    predicciones.to_pickle(root.DIR_DATA_ANALYTICS + 'SARIMAX_predictions_val.pkl')

    save_model(forecaster, 'SARIMAX_model')


    fig = go.Figure()
    trace1 = go.Scatter(x=val.index, y=val['target'], name="real", mode="lines",line_color="#4EA72E")
    trace2 = go.Scatter(x=predicciones.index, y=predicciones['target'], name="predicción", mode="lines")
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(
        title="Predicción vs valores reales en test",
        yaxis_title="Generación (kWh)",
        width=600,
        height=370,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
    )
    fig.show()

if __name__ == "__main__":
    main()