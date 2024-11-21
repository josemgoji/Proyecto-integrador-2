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
def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train.pkl')
    test = pd.read_pickle(root.DIR_DATA_STAGE + 'test.pkl')

    return  train, test

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

# Variables basadas en temperatura
def temperature_features(datos):
    wf_transformer = WindowFeatures(
        variables = ["temperature"],
        window    = ["1D", "7D"],
        functions = ["mean", "max", "min"],
        freq      = "h",
    )
    variables_temp = wf_transformer.fit_transform(datos[['temperature']])
    
    return variables_temp

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

def union_exog_features(variables_calendario, variables_temp, variables_solares):
    assert all(variables_calendario.index == variables_temp.index)
    variables_exogenas = pd.concat([
                            variables_calendario,
                            variables_temp,
                            variables_solares
                        ], axis=1)
    # Debido a la creación de medias móviles, hay valores faltantes al principio
    # de la serie. Y debido a holiday_next_day hay valores faltantes al final.
    variables_exogenas = variables_exogenas.iloc[7 * 24:, :]
    variables_exogenas = variables_exogenas.iloc[:-24, :]
    
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
        'temperature_window_1D_mean',
        'temperature_window_1D_min',
        'temperature_window_1D_max',
        'temperature_window_7D_mean',
        'temperature_window_7D_min',
        'temperature_window_7D_max',
        'temperature',
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
    # Columnas que empiezan con temp_ son seleccionadas
    exog_features.extend(variables_exogenas.filter(regex='^temperature_.*').columns.tolist())
    # Incluir temperatura y festivos
    exog_features.extend(['temperature'])
    
    return exog_features
     
def merge_df(datos,variables_exogenas, exog_features):
    datos = datos[['target']].merge(
           variables_exogenas[exog_features],
           left_index=True,
           right_index=True,
           how='inner'  # Usar solo las filas que coinciden en ambos DataFrames
       )
    
    return datos

def save_datasets_to_pickle(datasets, paths=None):
    """Save each dataset in datasets list to the corresponding path in paths list as a pickle file."""
    if paths == None:
        import root
        paths = [
            root.DIR_DATA_STAGE + 'train_exog.pkl',
            root.DIR_DATA_STAGE + 'test_exog.pkl',
        ]

    # Create folders if not exists
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save each dataset to its respective path
    for dataset, path in zip(datasets, paths):
        dataset.to_pickle(path)
    

def main():
    # Read datasets
    train, test = load_datasets()
    
    ################### Train ######################
    # Prepare date columns
    variables_calendario = calendar_features(train)
    
    # Temperature features
    variables_temp = temperature_features(train)
    
    #solar features
    variables_solares = solar_features(train)
    
    # mergin variables
    variables_exogenas = union_exog_features(variables_calendario, variables_temp, variables_solares)
    
    # cyclical features
    variables_exogenas = ciclic_features(variables_exogenas)
    
    # polynomial features
    variables_exogenas = pol_features(variables_exogenas)
    
    # Select exog features
    exog_features = select_exog_features(variables_exogenas)
    
    # Merge datasets
    train_final = merge_df(train,variables_exogenas, exog_features)
    
    ################### Test ######################
    variables_calendario = calendar_features(test)
    
    # Temperature features
    variables_temp = temperature_features(test)
    
    #solar features
    variables_solares = solar_features(test)
    
    # mergin variables
    variables_exogenas = union_exog_features(variables_calendario, variables_temp, variables_solares)
    
    # cyclical features
    variables_exogenas = ciclic_features(variables_exogenas)
    
    # polynomial features
    variables_exogenas = pol_features(variables_exogenas)
    
    # Select exog features
    exog_features = select_exog_features(variables_exogenas)
    
    # Merge datasets
    test_final = merge_df(test,variables_exogenas, exog_features)
    
    # Save datasets
    save_datasets_to_pickle([train_final, test_final])
    
if __name__ == '__main__':
    main()
    
