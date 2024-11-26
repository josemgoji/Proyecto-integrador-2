from json import load
from skforecast.utils import load_forecaster
from skforecast.preprocessing import RollingFeatures
import pandas as pd

import sys
import os

from exog_creation import *

def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    train = pd.read_pickle(root.DIR_DATA_STAGE + 'test_preprocessed.pkl')
    return train

def load_model(name):
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    model = load_forecaster(root.DIR_DATA_ANALYTICS + name,
                    verbose=True)
    return model

def predict():
    data = load_datasets()
    
    data_last_window = data.head(144)
    
    data_last_window = create_exog(data_last_window)
    
    exog_selectec = ['temperature', 'rain', 'surface_pressure', 'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 'shortwave_radiation', 'euros_per_mwh', 'installed_capacity', 'hour_sin', 'poly_month_sin__week_sin', 'poly_month_sin__week_cos', 'poly_month_sin__day_of_week_sin', 'poly_month_sin__day_of_week_cos', 'poly_month_sin__hour_sin', 'poly_month_sin__hour_cos', 'poly_month_sin__sunrise_hour_cos', 'poly_month_cos__week_sin', 'poly_month_cos__day_of_week_sin', 'poly_month_cos__day_of_week_cos', 'poly_month_cos__hour_sin', 'poly_month_cos__hour_cos', 'poly_month_cos__sunset_hour_sin', 'poly_week_sin__week_cos', 'poly_week_sin__day_of_week_sin', 'poly_week_sin__day_of_week_cos', 'poly_week_sin__hour_sin', 'poly_week_sin__hour_cos', 'poly_week_sin__sunrise_hour_cos', 'poly_week_sin__sunset_hour_cos', 'poly_week_cos__day_of_week_sin', 'poly_week_cos__day_of_week_cos', 'poly_week_cos__hour_sin', 'poly_week_cos__hour_cos', 'poly_week_cos__sunrise_hour_sin', 'poly_week_cos__sunrise_hour_cos', 'poly_week_cos__sunset_hour_sin', 'poly_day_of_week_sin__day_of_week_cos', 'poly_day_of_week_sin__hour_sin', 'poly_day_of_week_sin__hour_cos', 'poly_day_of_week_sin__sunrise_hour_sin', 'poly_day_of_week_sin__sunrise_hour_cos', 'poly_day_of_week_sin__sunset_hour_sin', 'poly_day_of_week_sin__sunset_hour_cos', 'poly_day_of_week_cos__hour_sin', 'poly_day_of_week_cos__hour_cos', 'poly_day_of_week_cos__sunrise_hour_sin', 'poly_day_of_week_cos__sunrise_hour_cos', 'poly_day_of_week_cos__sunset_hour_sin', 'poly_day_of_week_cos__sunset_hour_cos', 'poly_hour_sin__hour_cos', 'poly_hour_sin__sunrise_hour_sin', 'poly_hour_sin__sunrise_hour_cos', 'poly_hour_sin__sunset_hour_sin', 'poly_hour_sin__sunset_hour_cos', 'poly_hour_cos__sunrise_hour_sin', 'poly_hour_cos__sunrise_hour_cos', 'poly_hour_cos__sunset_hour_sin', 'poly_hour_cos__sunset_hour_cos']
    
    columns = exog_selectec + ['target']
    
    data_last_window = data_last_window[columns]
    
    model = load_model('tree_model.joblib')

    model.predict(steps=145,
                  exog  = data_last_window[exog_selectec])    
    
    subset = data.iloc[144:148]