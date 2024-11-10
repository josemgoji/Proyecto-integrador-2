import pandas as pd
import os
import sys



def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, current_dir)
    import root

    train_c = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/train_consumption.pkl')
    train_p = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/train_production.pkl')
    client = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/client.pkl')
    historical_weather = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/historical_weather.pkl')
    electricity_prices = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/electricity_prices.pkl')
    gas_prices = pd.read_pickle(root.DIR_DATA_STAGE + '1_single/gas_prices.pkl')  

    return  train_c, train_p, client, historical_weather, electricity_prices, gas_prices
        

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
    merged = train.merge(client, on='datetime', how='left') \
                  .merge(historical_weather, on='datetime', how='left') \
                  .merge(electricity_prices, on='datetime', how='left') \
                  .merge(gas_prices, on='datetime', how='left')
    return merged


def reorder_columns(df, column_order=None):
    """Reorder columns of the DataFrame."""
    if column_order == None:
        column_order = [
            'datetime', 'date', 'target', 'temperature', 'dewpoint', 'rain', 'snowfall',
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
            root.DIR_DATA_STAGE + '2_merged/consumption.pkl',
            root.DIR_DATA_STAGE + '2_merged/production.pkl',
        ]

    # Create folders if not exists
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save each dataset to its respective path
    for dataset, path in zip(datasets, paths):
        dataset.to_pickle(path)


def main():
     # Read datasets
    train_c, train_p, client, historical_weather, electricity_prices, gas_prices = load_datasets()

    # Prepare date columns for merging
    client, historical_weather, electricity_prices, gas_prices = add_time_series_col(client, historical_weather, electricity_prices, gas_prices)

    # Merge datasets
    merged_c = merge_datasets(train_c, client, historical_weather, electricity_prices, gas_prices)
    merged_p = merge_datasets(train_p, client, historical_weather, electricity_prices, gas_prices)

    # Add dt.floor('D')
    merged_c['date'] = merged_c['datetime'].dt.floor('D')
    merged_p['date'] = merged_p['datetime'].dt.floor('D')

    # Reorder dataset columns
    merged_c = reorder_columns(merged_c)
    merged_p = reorder_columns(merged_p)

    # Save datasets to pickle files
    save_datasets_to_pickle([merged_c, merged_p])


if __name__ == "__main__":
    main()