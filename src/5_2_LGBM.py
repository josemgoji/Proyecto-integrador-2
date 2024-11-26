# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from skforecast.datasets import fetch_dataset
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from feature_engine.timeseries.forecasting import WindowFeatures

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from skforecast.plot import plot_residuals
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 8})

# Modelado y Forecasting
# ==============================================================================
import skforecast
import lightgbm
import sklearn
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.feature_selection import select_features
from skforecast.model_selection import TimeSeriesFold
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster
import shap

# Importar prepropcesing
from exog_creation import *

import sys
import os

def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train_preprocessed.pkl')
    return train


def def_exog_cols(datos):
    datos = datos.drop(columns=['target'])
    return list(datos.columns)

def define_forecaster():
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                    regressor        = LGBMRegressor(random_state=15926, verbose=-1),
                    lags             = 72,
                    window_features  = window_features,
                )
    return forecaster

def hyperparametros_search(datos, forecaster, fin_train,exog_cols):
    cv = TimeSeriesFold(
        steps              = 24,
        initial_train_size = len(datos[:fin_train]),
        refit              = False,
        )
    # Espacio de búsqueda de hiperparámetros
    def search_space(trial):
        search_space  = {
            'n_estimators' : trial.suggest_int('n_estimators', 600, 3000, step=100),
            'max_depth'    : trial.suggest_int('max_depth', 3, 15, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
            'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
            'lags'         : trial.suggest_categorical('lags', [24,48,72])
        } 
        return search_space

    resultados_busqueda, frozen_trial = bayesian_search_forecaster(
                                            forecaster    = forecaster,
                                            y             = datos['target'],
                                            exog          = datos[exog_cols],
                                            cv            = cv,
                                            metric        = 'mean_squared_error',
                                            search_space  = search_space,
                                            n_trials      = 20,  # Aumentar para una búsqueda más exhaustiva
                                            random_state  = 42,
                                            return_best   = True,
                                            n_jobs        = 'auto',
                                            verbose       = False,
                                            show_progress = True
                                        )
    best_params = resultados_busqueda.at[0, 'params']
    best_params = best_params | {'random_state': 15926, 'verbose': -1}
    backtesting_metric = resultados_busqueda.at[0, 'mean_squared_error']
    
  
    return best_params, backtesting_metric


def feature_selection(datos, forecaster, exog_cols):
    regressor = LGBMRegressor(random_state=15926, verbose=-1)
    
    selector = RFECV(
        estimator = regressor,
        step      = 1,
        cv        = 3,
        n_jobs    = -1
    )
    
    lags_select, window_features_select, exog_select  = select_features(
        forecaster      = forecaster,
        selector        = selector,
        y               = datos['target'],
        exog            = datos[exog_cols],
        select_only     = None,
        force_inclusion = None,
        subsample       = 0.5,
        random_state    = 42,
        verbose         = True,
    )
    
    return(lags_select,exog_select)

def final_model(datos,best_params, lags_select, exog_select,fin_train):
    
    cv = TimeSeriesFold(
        steps                 = 24,
        initial_train_size    = len(datos[:fin_train]),
        refit                 = 24,
        fixed_train_size      = False,
        gap                   = 0,
        allow_incomplete_fold = True
    )
    
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                    regressor       = LGBMRegressor(**best_params),
                    lags            = lags_select,
                    window_features = window_features
                )
    forecaster.fit(
        y    = datos['target'],
        exog = datos[exog_select]
    )
    # Backtesting con los predictores seleccionados y los datos de test
    # ==============================================================================

    metrica, predicciones = backtesting_forecaster(
                                forecaster         = forecaster,
                                y                  = datos['target'],
                                exog               = datos[exog_select],
                                cv                 = cv,
                                metric             = 'mean_squared_error',
                                n_jobs             = 'auto',
                                verbose            = False,
                                show_progress      = True,
                            )
    
    return forecaster,metrica, predicciones   

def save_model(forecaster,name):
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    save_forecaster(
    forecaster, 
    file_name = root.DIR_DATA_ANALYTICS + name, 
    save_custom_functions = True, 
    verbose = False
)
    
def create_plot(predicciones,datos):
    fig = go.Figure()
    trace1 = go.Scatter(x=datos.index, y=datos['target'], name="real", mode="lines")
    trace2 = go.Scatter(x=predicciones.index, y=predicciones['pred'], name="prediction", mode="lines")
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(
        title="Predicción vs valores reales en test",
        xaxis_title="Date time",
        yaxis_title="Demand",
        width=750,
        height=370,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
    )
    fig.show()
    
    return fig

def save_plot(fig,name):
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    fig.write_html(root.DIR_DATA_ANALYTICS + name)
        

def main():
    datos = load_datasets()
    
    datos = create_exog(datos)
    
    fin_train = '2022-08-31 23:59:00'
    
    exog_cols = def_exog_cols(datos)
    
    forecaster = define_forecaster()
    
    best_params, backtesting_metric = hyperparametros_search(datos, forecaster, fin_train, exog_cols)
    
    lags_select, exog_select = feature_selection(datos, forecaster, exog_cols)
    
    modelo, metrica, predicciones = final_model(datos,best_params, lags_select, exog_select,fin_train)

    save_model(modelo, 'tree_model')
    
    fig = create_plot(predicciones,datos[fin_train:])
    
    save_plot(fig, 'tree_model.html')