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
    test = pd.read_pickle(root.DIR_DATA_STAGE + 'test_preprocessed.pkl')
    return root, train, test


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
        refit=False,
    )
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        series=data,
        levels=forecaster.levels,
        cv=cv,
        metric=root_mean_squared_error,
        verbose=False,
    )
    return metrics, predictions


def create_plots(root, data, end_val, forecaster, predictions):
    # Seguimiento del entrenamiento y overfitting del modelo con mejores parametros
    fig, ax = plt.subplots(figsize=(5, 2.5))
    forecaster.plot_history(ax=ax)
    plt.savefig(root.DIR_DATA_ANALYTICS + 'LSTM_training_history.png', dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Gráfico de las predicciones vs valores reales en el conjunto de test del modelo con mejores parametros
    fig = go.Figure()
    trace1 = go.Scatter(x=data.loc[end_val:].index, y=data.loc[end_val:]['target'], name="test", mode="lines")
    trace2 = go.Scatter(x=predictions.index, y=predictions['target'], name="predicciones", mode="lines")
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(
        title="Predicciones vs valores reales en el conjunto de test",
        xaxis_title="Date time",
        yaxis_title="target",
        width=750,
        height=350,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="left",
            x=0
        )
    )
    fig.write_image(root.DIR_DATA_ANALYTICS + 'LSTM_pred_vs_real.png', width=750, height=350)


def main():
    root, train, test = load_datasets()
    data = pd.concat([train, test])
    end_val = '2022-08-31 23:59:59'
    val = train.loc[end_val:]
    train = train.loc[:end_val]

    series = ['target', 'temperature', 'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity'] 
    levels = ['target']  # Serie que se quiere predecir

    data_train = train[series].copy()
    data_val = val[series].copy()
    data_test = test[series].copy()

    steps = 24
    lags = 72
    recurrent_units = [128, 64]
    dense_units = [32, 16]
    learning_rate = 0.01
    epochs = 4
    batch_size = 64
    
    model = create_model(data_train, levels, lags, steps, recurrent_units, dense_units, learning_rate)
    forecaster = create_forecaster(data_train, data_val, model, levels, steps, lags, epochs, batch_size)
    metrics, predictions = backtesting(data, end_val, forecaster, levels)

    create_plots(root, data, end_val, forecaster, predictions)


if __name__ == "__main__":
    main()