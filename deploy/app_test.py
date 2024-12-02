import re
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from io import StringIO
import gradio as gr
import os
import sys
from json import load
from skforecast.utils import load_forecaster
from skforecast.preprocessing import RollingFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from exog_creation import create_exog
import contextlib
import warnings
# Función para cargar el archivo CSV y mostrar las primeras 5 filas
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

def set_datetime_index(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.asfreq('h')
    return df

def load_model(name):

    model = load_forecaster(name,verbose=True)
    return model

def load_pipeline(name):
    with open('pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline


def flujo(input_file):
    
    warnings.filterwarnings("ignore")
    
    datos = load_csv(input_file)
    
    print(datos.head())

    datos = set_datetime_index(datos)

    datos_exog = create_exog(datos)

    # Redirigir stdout a os.devnull para suprimir cualquier impresión
    sys.stdout = open(os.devnull, 'w')

    # Cargar el modelo
    forecaster = load_model('tree_model.joblib')

    # Restaurar stdout a la consola
    sys.stdout = sys.__stdout__
    
    exog_selectec = ['temperature', 'rain', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity', 'hour_sin', 'poly_month_sin__week_sin', 'poly_month_sin__week_cos', 'poly_month_sin__day_of_week_sin', 'poly_month_sin__day_of_week_cos', 'poly_month_sin__hour_sin', 'poly_month_sin__hour_cos', 'poly_month_sin__sunrise_hour_cos', 'poly_month_cos__week_sin', 'poly_month_cos__day_of_week_sin', 'poly_month_cos__day_of_week_cos', 'poly_month_cos__hour_sin', 'poly_month_cos__hour_cos', 'poly_month_cos__sunset_hour_sin', 'poly_week_sin__week_cos', 'poly_week_sin__day_of_week_sin', 'poly_week_sin__day_of_week_cos', 'poly_week_sin__hour_sin', 'poly_week_sin__hour_cos', 'poly_week_sin__sunrise_hour_cos', 'poly_week_sin__sunset_hour_cos', 'poly_week_cos__day_of_week_sin', 'poly_week_cos__day_of_week_cos', 'poly_week_cos__hour_sin', 'poly_week_cos__hour_cos', 'poly_week_cos__sunrise_hour_sin', 'poly_week_cos__sunrise_hour_cos', 'poly_week_cos__sunset_hour_sin', 'poly_day_of_week_sin__day_of_week_cos', 'poly_day_of_week_sin__hour_sin', 'poly_day_of_week_sin__hour_cos', 'poly_day_of_week_sin__sunrise_hour_sin', 'poly_day_of_week_sin__sunrise_hour_cos', 'poly_day_of_week_sin__sunset_hour_sin', 'poly_day_of_week_sin__sunset_hour_cos', 'poly_day_of_week_cos__hour_sin', 'poly_day_of_week_cos__hour_cos', 'poly_day_of_week_cos__sunrise_hour_sin', 'poly_day_of_week_cos__sunrise_hour_cos', 'poly_day_of_week_cos__sunset_hour_sin', 'poly_day_of_week_cos__sunset_hour_cos', 'poly_hour_sin__hour_cos', 'poly_hour_sin__sunrise_hour_sin', 'poly_hour_sin__sunrise_hour_cos', 'poly_hour_sin__sunset_hour_sin', 'poly_hour_sin__sunset_hour_cos', 'poly_hour_cos__sunrise_hour_sin', 'poly_hour_cos__sunrise_hour_cos', 'poly_hour_cos__sunset_hour_sin', 'poly_hour_cos__sunset_hour_cos']

    predictions = forecaster.predict(steps=24, exog  = datos_exog[exog_selectec])

    datos['target'] = predictions

    target_column = 'target'

    columns_order = [target_column] + [col for col in datos.columns if col != target_column]

    datos = datos[columns_order]

    pipeline = load_pipeline('pipeline.pkl')

    pred_scaled = pipeline.inverse_transform(datos)

    pred_scaled_df = pd.DataFrame(pred_scaled, columns=datos.columns, index=datos.index)
    
    df_reset = pred_scaled_df.reset_index()
    
    df_target = df_reset[['datetime', 'target']]
    

    return df_target.to_html()

# Crear la interfaz con Gradio
interface = gr.Interface(
    fn=flujo,  # Función principal
    inputs=gr.File(label="Sube tu archivo CSV"),  # Entrada de archivo
    outputs="html",  # Salida como tabla HTML
    title="Prediccion geenracion de energia",
    description="Sube un archivo CSV y perdice la geenracion de energia."
)

interface.launch(share = True)


