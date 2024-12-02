import numpy as np
import pandas as pd
import sys
import os
import cloudpickle
import plotly.graph_objects as go
from skforecast.utils import load_forecaster
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_sarimax
from sklearn.metrics import root_mean_squared_error


def load_datasets():
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train_preprocessed.pkl')
    test = pd.read_pickle(root.DIR_DATA_STAGE + 'test_preprocessed.pkl')
    return root, train, test


def load_pipeline():
    """Load the saved pipeline using cloudpickle."""
    import root
    with open(root.DIR_DATA_ANALYTICS + 'pipeline.pkl', 'rb') as f:
        pipeline = cloudpickle.load(f)
    return pipeline


def unscale_data(scaler, predictions):
    placeholder = np.zeros((len(predictions), 11))
    placeholder[:, 0] = predictions['pred']
    predictions_scaled = scaler.inverse_transform(placeholder)[:, 0]
    predictions_scaled[predictions_scaled < 0] = 0
    predictions = pd.DataFrame(predictions_scaled, columns=predictions.columns, index=predictions.index)
    return predictions


def create_plots(root, val, predictions, name):
    # Gráfico de las predicciones vs valores reales en el conjunto de test del modelo con mejores parametros
    fig = go.Figure()
    trace1 = go.Scatter(x=val.index, y=val['target'], name="Real", mode="lines", line_color='#5F70EB')
    trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="Estimado", mode="lines", line_color="#4EA72E")
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Producción (kWh)",
        width=750,
        height=350,
        margin=dict(l=20, r=0, t=35, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            xanchor="right",
            x=0.99,
            y=0.99
        )
    )
    fig.write_html(root.DIR_DATA_ANALYTICS + f'{name}_pred_vs_real.html')
    
def backtesting(data, train, forecaster, steps):
    cv = TimeSeriesFold(
        steps              = 7,
        initial_train_size = len(train),
        refit              = False,
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
    
def load_model(name):
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    model = load_forecaster(root.DIR_DATA_ANALYTICS + name,verbose=True)
    return model

def main():
    root, train, test = load_datasets()
    datos = pd.concat([train, test], axis=0)
    end_train = '2023-01-31 23:59:59'

    pipeline = load_pipeline()
    scaler = pipeline['scale']
    
    
    
    forecaster = load_model('SARIMAX_model.joblib')
    
    metrica, predicciones = backtesting(datos, train, forecaster, 24)
    
    
    sarimax = unscale_data(scaler, predicciones)

    test_processed = scaler.inverse_transform(test)
    test_processed = pd.DataFrame(test_processed, columns=test.columns, index=test.index)
    
    create_plots(root, test_processed, sarimax, 'SARIMAX_test')
    
    print(scaler.inverse_transform([[0.264434, 0,0,0,0,0,0,0,0,0,0]]))
    
    import numpy as np

    # Calcular el RMS
    rms = np.sqrt(((test_processed['target'] - sarimax['pred']) ** 2).mean())
    print(f"El RMS es: {rms}")
    
    mae = (test_processed['target'] - sarimax['pred']).abs().mean()
    print(f"El MAE es: {mae}")