from json import load
import sys
import os
import pandas as pd
from datetime import timedelta

# def load_datasets():
#     """Load all datasets and return them as dataframes."""
#     current_dir = os.getcwd()
#     ROOT_PATH = os.path.dirname(current_dir)
#     sys.path.insert(1, ROOT_PATH)
#     import root
#     train = pd.read_pickle(root.DIR_DATA_STAGE + 'test_preprocessed.pkl')
#     return train


# datos = load_datasets()

# datos = datos.head(24)

# datos.drop(columns=['target'], inplace=True)

# datos.to_csv('prueba.csv')


current_dir = os.getcwd()
ROOT_PATH = os.path.dirname(current_dir)
sys.path.insert(1, ROOT_PATH)
import root

# Read datasets
client = pd.read_csv(root.DIR_DATA_RAW + 'client.csv')
electricity_prices = pd.read_csv(root.DIR_DATA_RAW + 'electricity_prices.csv')
gas_prices = pd.read_csv(root.DIR_DATA_RAW + 'gas_prices.csv')
historical_weather = pd.read_csv(root.DIR_DATA_RAW + 'historical_weather.csv')
train = pd.read_csv(root.DIR_DATA_RAW + 'train.csv')

# Crear un diccionario para organizar los DataFrames
dfs = {
    "client": client,
    "electricity_prices": electricity_prices,
    "gas_prices": gas_prices,
    "historical_weather": historical_weather,
    "train": train
}

# Fecha base
base_date = pd.Timestamp('2023-02-01')

# Definir los desfases (en días) para cada DataFrame
date_gaps = {
    "client": 3,  # Desde base_date - 3 días
    "electricity_prices": 1,  # Desde base_date - 1 día
    "gas_prices": 1,  # Desde base_date - 1 día
    "historical_weather": 2,  # Desde base_date - 2 días
}


# Aplicar el filtrado con desfases a cada DataFrame

# Filtrado de los DataFrames
client['date'] = pd.to_datetime(client['date'], errors='coerce')
client = client[client['date'] > (base_date - timedelta(days=date_gaps['client']))]
client_first_3_days = client[client['date'] <= (client['date'].min() + timedelta(days=3))]
client_first_3_days.to_csv('client_filtered.csv', index=False)  # Guardar los primeros 3 días

electricity_prices['forecast_date'] = pd.to_datetime(electricity_prices['forecast_date'], errors='coerce')
electricity_prices = electricity_prices[electricity_prices['forecast_date'] > (base_date - timedelta(days=date_gaps['electricity_prices']))]
electricity_prices_first_3_days = electricity_prices[electricity_prices['forecast_date'] <= (electricity_prices['forecast_date'].min() + timedelta(days=3))]
electricity_prices_first_3_days.to_csv('electricity_prices_filtered.csv', index=False)

gas_prices['forecast_date'] = pd.to_datetime(gas_prices['forecast_date'], errors='coerce')
gas_prices = gas_prices[gas_prices['forecast_date'] > (base_date - timedelta(days=date_gaps['gas_prices']))]
gas_prices_first_3_days = gas_prices[gas_prices['forecast_date'] <= (gas_prices['forecast_date'].min() + timedelta(days=3))]
gas_prices_first_3_days.to_csv('gas_prices_filtered.csv', index=False)

historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'], errors='coerce')
historical_weather = historical_weather[historical_weather['datetime'] > (base_date - timedelta(days=date_gaps['historical_weather']))]
historical_weather_first_3_days = historical_weather[historical_weather['datetime'] <= (historical_weather['datetime'].min() + timedelta(days=3))]
historical_weather_first_3_days.to_csv('historical_weather_filtered.csv', index=False)

train['datetime'] = pd.to_datetime(train['datetime'], errors='coerce')
train = train[train['datetime'] > base_date]
train_first_3_days = train[train['datetime'] <= (train['datetime'].min() + timedelta(days=3))]
train_first_3_days.to_csv('train_filtered.csv', index=False)


min_date = client_first_3_days['date'].min()
max_date = client_first_3_days['date'].max()