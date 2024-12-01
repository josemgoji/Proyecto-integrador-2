import sys
import os
import pandas as pd
import numpy as np
from itertools import product

from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
from skforecast.plot import set_dark_theme
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)

os.environ["KERAS_BACKEND"] = "tensorflow" # 'tensorflow', 'jax´ or 'torch'
import keras
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

if keras.__version__ > "3.0":
    if keras.backend.backend() == "tensorflow":
        import tensorflow
    elif keras.backend.backend() == "torch":
        import torch
    else:
        print("Backend not recognized. Please use 'tensorflow' or 'torch'.")

import skforecast
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model
from sklearn.preprocessing import MinMaxScaler
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries

import warnings
warnings.filterwarnings('once')



def load_datasets():
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train_preprocessed.pkl')
    return root, train


def create_model(data_train, levels, lags, steps, recurrent_units, dense_units, learning_rate):
    model = create_and_compile_model(
        series=data_train,
        levels=levels, 
        lags=lags,
        steps=steps,
        recurrent_layer="LSTM",
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        optimizer=Adam(learning_rate=learning_rate), 
        loss=MeanSquaredError()
    )
    return model


def create_forecaster(data_train, data_val, model, levels, steps, lags, epochs, batch_size):
    forecaster = ForecasterRnn(
        regressor=model,
        levels=levels,
        steps=steps,
        lags=lags,
        transformer_series=MinMaxScaler(),
        fit_kwargs={
            "epochs": epochs,             # Número de épocas para entrenar el modelo.
            "batch_size": batch_size,     # Tamaño del batch para entrenar el modelo.
            "series_val": data_val,       # Datos de validación para el entrenamiento del modelo.
        },
    )
    forecaster.fit(data_train)
    return forecaster


def backtesting(data, end_val, forecaster, levels):
    cv = TimeSeriesFold(
        steps=forecaster.max_step,
        initial_train_size=len(data.loc[:end_val, :]),
        refit=True,
    )
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        series=data,
        levels=forecaster.levels,
        cv=cv,
        metric=root_mean_squared_error,
        verbose=True,
    )
    return metrics, predictions


def training_history_plot(root, forecaster):
    import root
    fig, ax = plt.subplots(figsize=(5, 2.5))
    forecaster.plot_history(ax=ax)
    plt.savefig(root.DIR_DATA_ANALYTICS + 'LSTM_training_history.png', dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    root, train = load_datasets()
    data = train.copy()
    end_val = '2022-08-31 23:59:59'
    val = train.loc[end_val:]
    train = train.loc[:end_val]

    series = ['target', 'temperature', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity'] 
    levels = ['target']  # Serie que se quiere predecir

    data_train = train[series].copy()
    data_val = val[series].copy()

    steps = 24
    lags = 72
    recurrent_units = [128, 64]
    dense_units = [64, 32]
    learning_rate = 0.01
    epochs = 4
    batch_size = 64
    
    model = create_model(data_train, levels, lags, steps, recurrent_units, dense_units, learning_rate)
    forecaster = create_forecaster(data_train, data_val, model, levels, steps, lags, epochs, batch_size)
    metrics, predictions = backtesting(data, end_val, forecaster, levels)
    training_history_plot(root, forecaster)
    predictions.to_pickle(root.DIR_DATA_ANALYTICS + 'LSTM_predictions_val.pkl')



if __name__ == "__main__":
    main()