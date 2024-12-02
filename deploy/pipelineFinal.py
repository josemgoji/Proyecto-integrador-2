from calendar import c
from os import pipe
import pandas as pd
import pickle
from skforecast.utils import load_forecaster
from filterdf import filter_datasets
from mergedf import merging_datasets
import numpy as np
import plotly.graph_objects as go

def load_csv(input_file):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(input_file)
        
        # Verificar si el DataFrame está vacío
        if df.empty:
            raise ValueError("El archivo subido está vacío o no tiene datos válidos.")
        
        # Retornar las primeras 5 filas como tabla HTML
        # return df.head().to_html()
        return df
    except Exception as e:
        raise f"Error al cargar el archivo CSV:{e}"
    
def load_model(name):

    model = load_forecaster(name,verbose=True)
    return model

def load_pipeline():
    with open('pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

def unscale_data(scaler, predictions):
    placeholder = np.zeros((len(predictions), 11))
    placeholder[:, 0] = predictions['target']
    predictions_scaled = scaler.inverse_transform(placeholder)[:, 0]
    predictions_scaled[predictions_scaled < 0] = 0
    predictions = pd.DataFrame(predictions_scaled, columns=predictions.columns, index=predictions.index)
    return predictions

def create_plots(predictions):
    # Gráfico de las predicciones vs valores reales en el conjunto de test del modelo con mejores parametros
    fig = go.Figure()
    trace2 = go.Scatter(x=predictions.index, y=predictions['target'], name="Estimado", mode="lines", line_color="#4EA72E")
    fig.add_trace(trace2)
    fig.update_layout(
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
    return fig


def pipeline_final(texto,steps,train=None,client=None,historical_weather=None,electricity_prices=None,gas_prices=None):
    #prueba
    #texto = 'No'
    # #steps
    # steps = 24
    # #dfs
    
    # train = 'files_prueba/train_filtered.csv'
    # client = 'files_prueba/client_filtered.csv'
    # historical_weather = 'files_prueba/historical_weather_filtered.csv'
    # electricity_prices = 'files_prueba/electricity_prices_filtered.csv'
    # gas_prices = 'files_prueba/gas_prices_filtered.csv'
    pipeline = load_pipeline()
    scaler = pipeline['scale']
    
    #load model
    model = load_model('LSTM_forecaster.joblib')
        
    
    if texto == 'Si':
        pred = model.predict(steps=steps)
        
        pred = unscale_data(scaler, pred)
        
        pred_reset = pred.reset_index(drop=False)
        
        pred_reset = pred_reset.astype(str)
        
        pred_reset = pred_reset.rename(columns={'index': 'fecha'})
        
        fig = create_plots(pred)
        
        return fig , pred_reset
        
    else:
        train = load_csv(train)
        client = load_csv(client)
        historical_weather = load_csv(historical_weather)
        electricity_prices = load_csv(electricity_prices)
        gas_prices = load_csv(gas_prices)
        #filter data sets
        filter_datasets(train,client,historical_weather,electricity_prices,gas_prices)
        #merge data sets
        df = merging_datasets()
        #load pipeline
        # sclaing the data
        df_processed = pipeline.transform(df)
        
        df_processed = pd.DataFrame(df_processed, columns=df.columns, index=df.index)
    
        pred = model.predict(steps=steps, last_window=df_processed)
        
        pred = unscale_data(scaler, pred)
        
        pred_reset = pred.reset_index(drop=False)
        
        pred_reset = pred_reset.astype(str)
        
        pred_reset = pred_reset.rename(columns={'index': 'fecha'})
        
        fig = create_plots(pred)
        
        return fig , pred_reset
        

    
