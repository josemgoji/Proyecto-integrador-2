import pandas as pd
import sys
import os



def load_datasets():
    """Load all datasets and return them as dataframes."""
    # Obtén la ruta base del directorio actual
    base_dir = os.getcwd()  # Directorio actual

    # Construye las rutas absolutas de cada archivo
    train_path = os.path.join(base_dir, 'process_files', 'generation.pkl')
    client_path = os.path.join(base_dir, 'process_files', 'client.pkl')
    historical_weather_path = os.path.join(base_dir, 'process_files', 'historical_weather.pkl')
    electricity_prices_path = os.path.join(base_dir, 'process_files', 'electricity_prices.pkl')
    gas_prices_path = os.path.join(base_dir, 'process_files', 'gas_prices.pkl')
    
    # Verifica que los archivos existan antes de intentar cargarlos
    for path in [train_path, client_path, historical_weather_path, electricity_prices_path, gas_prices_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    # Carga los archivos
    train = pd.read_pickle(train_path)
    client = pd.read_pickle(client_path)
    historical_weather = pd.read_pickle(historical_weather_path)
    electricity_prices = pd.read_pickle(electricity_prices_path)
    gas_prices = pd.read_pickle(gas_prices_path)
    
    return train, client, historical_weather, electricity_prices, gas_prices


def add_time_series_col(client, historical_weather, electricity_prices, gas_prices):
    """Add column with date where data is available."""

    client['datetime'] = pd.to_datetime(client['date']) + pd.Timedelta(days=3)
    historical_weather['datetime'] += pd.Timedelta(days=2)
    electricity_prices['datetime'] = pd.to_datetime(electricity_prices['forecast_date']) + pd.Timedelta(days=1)
    gas_prices['datetime'] = pd.to_datetime(gas_prices['forecast_date']) + pd.Timedelta(days=1)

    # Drop unnecessary columns after date adjustments
    client = client.drop(['date'], axis=1)
    electricity_prices = electricity_prices.drop(['forecast_date'], axis=1)
    gas_prices = gas_prices.drop(['forecast_date'], axis=1)

    return client, historical_weather, electricity_prices, gas_prices


def merge_datasets(train, client, historical_weather, electricity_prices, gas_prices):
    """Merge DataFrames train, client, historical weather, gas prices and electricity prices based on the datetime column."""
    merged = train.merge(historical_weather, on='datetime', how='left') \
                  .merge(electricity_prices, on='datetime', how='left')
    
    # Add dt.floor('D')
    merged['date'] = merged['datetime'].dt.floor('D')
    client['date'] = client['datetime'].dt.floor('D')
    client = client.drop('datetime', axis=1)
    gas_prices['date'] = gas_prices['datetime'].dt.floor('D')
    gas_prices = gas_prices.drop('datetime', axis=1)

    merged = merged.merge(client, on='date', how='outer') \
                   .merge(gas_prices, on='date', how='outer')

    #dreop unnecessary columns
    merged = merged.drop(['date'], axis=1)
    
    return merged


def reorder_columns(df, column_order=None):
    """Reorder columns of the DataFrame."""
    if column_order == None:
        column_order = [
            'datetime', 'target', 'temperature', 'dewpoint', 'rain', 'snowfall',
            'surface_pressure', 'cloudcover_total', 'cloudcover_low', 'cloudcover_mid', 
            'cloudcover_high', 'windspeed_10m', 'winddirection_10m', 
            'shortwave_radiation', 'direct_solar_radiation', 'diffuse_radiation',
            'lowest_price_per_mwh', 'highest_price_per_mwh', 'euros_per_mwh','eic_count', 'installed_capacity'
            ]
    return df[column_order]


def save_datasets_to_pickle(datasets, paths=None):
    """Save each dataset in datasets list to the corresponding path in paths list as a pickle file."""
    if paths == None:
        import root
        paths = [
            root.DIR_DATA_STAGE + 'merged_df.pkl',
        ]

    # Create folders if not exists
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save each dataset to its respective path
    for dataset, path in zip(datasets, paths):
        dataset.to_pickle(path)


def drop_first_3_days(df, column, threshold_column, threshold_nans=70):
    """Drop first 3 days of the dataset if the threshold is exceeded."""
    # Count null values in the threshold column
    nulos = df[threshold_column].isna().sum()
    
    # If the threshold is exceeded drop the first 3 days
    if nulos > threshold_nans:
        # Initial date
        fecha_minima = df[column].min()
        # Limit day
        limite = fecha_minima + pd.Timedelta(days=3)
        # Filter df
        df = df[df[column] >= limite]
    
    return df


def feature_selection(df):
    cols_2_drop = [ 'dewpoint','cloudcover_low','cloudcover_mid', 
                   'cloudcover_high','direct_solar_radiation',
                   'diffuse_radiation', 'lowest_price_per_mwh',
                   'highest_price_per_mwh','eic_count']
    df.drop(columns = cols_2_drop, axis = 1, inplace = True)
    return df


def set_datetime_index(df):
    df = df.set_index('datetime')
    df = df.asfreq('h')
    return df


def merging_datasets():
     # Read datasets
    train, client, historical_weather, electricity_prices, gas_prices = load_datasets()

    # Prepare date columns for merging
    client, historical_weather, electricity_prices, gas_prices = add_time_series_col(client, historical_weather, electricity_prices, gas_prices)

    # Merge datasets
    merged = merge_datasets(train, client, historical_weather, electricity_prices, gas_prices)
    
    # Reorder dataset columns
    merged = reorder_columns(merged)
    
    # Feature selection
    merged = feature_selection(merged)

    # Set datetime index
    merged = set_datetime_index(merged)

    return merged