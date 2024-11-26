# Tratamiento de datos
# ==============================================================================
import re
import numpy as np
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from skforecast.datasets import fetch_dataset
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from sklearn.preprocessing import PolynomialFeatures
import sys
import os
##########################################################################################


# current_dir = os.getcwd()
# ROOT_PATH = os.path.dirname(current_dir)
# sys.path.insert(1, ROOT_PATH)
# import root
# datos = pd.read_pickle(root.DIR_DATA_STAGE + 'train.pkl')

# Variables basadas en el calendario
def calendar_features(datos):
    features_to_extract = [
    'month',
    'week',
    'day_of_week',
    'hour'
    ]   
    calendar_transformer = DatetimeFeatures(
        variables='index',
        features_to_extract=features_to_extract,
        drop_original=True,
    )
    variables_calendario = calendar_transformer.fit_transform(datos)[features_to_extract]
    
    return variables_calendario

# Variables basadas en la luz solar
def solar_features(datos):
    location = LocationInfo(
        name      = 'Taillin',
        region    = 'Estonia',
        timezone  = 'Europe/Riga',
        latitude  = 56.946285,
        longitude = 24.105078
    )
    sunrise_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunrise']
        for date in datos.index
    ]
    sunset_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunset']
        for date in datos.index
    ]
    sunrise_hour = pd.Series(sunrise_hour, index=datos.index).dt.round("h").dt.hour
    sunset_hour = pd.Series(sunset_hour, index=datos.index).dt.round("h").dt.hour
    variables_solares = pd.DataFrame({
                            'sunrise_hour': sunrise_hour,
                            'sunset_hour': sunset_hour
                        })
    variables_solares['daylight_hours'] = (
        variables_solares['sunset_hour'] - variables_solares['sunrise_hour']
    )
    variables_solares["is_daylight"] = np.where(
        (datos.index.hour >= variables_solares["sunrise_hour"])
        & (datos.index.hour < variables_solares["sunset_hour"]),
        1,
        0,
    )
    
    return variables_solares

# Unión de variables exógenas

def union_exog_features(variables_calendario, variables_solares):
    assert all(variables_calendario.index == variables_solares.index)
    variables_exogenas = pd.concat([
                            variables_calendario,
                            variables_solares
                        ], axis=1)
    
    return variables_exogenas

def ciclic_features(variables_exogenas):
    features_to_encode = [
        "month",
        "week",
        "day_of_week",
        "hour",
        "sunrise_hour",
        "sunset_hour",
    ]
    max_values = {
        "month": 12,
        "week": 52,
        "day_of_week": 6,
        "hour": 23,
        "sunrise_hour": 23,
        "sunset_hour": 23,
    }
    cyclical_encoder = CyclicalFeatures(
        variables     = features_to_encode,
        max_values    = max_values,
        drop_original = False
    )

    variables_exogenas = cyclical_encoder.fit_transform(variables_exogenas)   

    return variables_exogenas

def pol_features(variables_exogenas):
# Interacción entre variables exógenas
    transformer_poly = PolynomialFeatures(
                            degree           = 2,
                            interaction_only = True,
                            include_bias     = False
                        ).set_output(transform="pandas")
    poly_cols = [
        'month_sin', 
        'month_cos',
        'week_sin',
        'week_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'hour_sin',
        'hour_cos',
        'sunrise_hour_sin',
        'sunrise_hour_cos',
        'sunset_hour_sin',
        'sunset_hour_cos',
        'daylight_hours',
        'is_daylight',
    ]
    variables_poly = transformer_poly.fit_transform(variables_exogenas[poly_cols])
    variables_poly = variables_poly.drop(columns=poly_cols)
    variables_poly.columns = [f"poly_{col}" for col in variables_poly.columns]
    variables_poly.columns = variables_poly.columns.str.replace(" ", "__")
    assert all(variables_exogenas.index == variables_poly.index)
    variables_exogenas = pd.concat([variables_exogenas, variables_poly], axis=1)   
 
    return variables_exogenas  


def select_exog_features(variables_exogenas):
    # Selección de variables exógenas incluidas en el modelo
    exog_features = []
    # Columnas que terminan con _seno o _coseno son seleccionadas
    exog_features.extend(variables_exogenas.filter(regex='_sin$|_cos$').columns.tolist())
    
    return exog_features
     
def merge_df(datos,variables_exogenas, exog_features):
    datos = datos.merge(variables_exogenas[exog_features],
           left_index=True,
           right_index=True,
           how='left'  # Usar solo las filas que coinciden en ambos DataFrames
       )
    
    return datos
    

def create_exog(datos):
    # Read datasets
    
    ################### Train ######################
    # Prepare date columns
    variables_calendario = calendar_features(datos)
    
    #solar features
    variables_solares = solar_features(datos)
    
    # mergin variables
    variables_exogenas = union_exog_features(variables_calendario, variables_solares)
    
    # cyclical features
    variables_exogenas = ciclic_features(variables_exogenas)
    
    # polynomial features
    variables_exogenas = pol_features(variables_exogenas)
    
    # Select exog features
    exog_features = select_exog_features(variables_exogenas)
    
    # Merge datasets
    datos = merge_df(datos,variables_exogenas, exog_features)
    
    return datos
    
    
