import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import sys
import os



def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, current_dir)
    import root
    
    train = pd.read_csv(root.DIR_DATA_RAW + 'train.csv')
    client = pd.read_csv(root.DIR_DATA_RAW + 'client.csv')
    historical_weather = pd.read_csv(root.DIR_DATA_RAW + 'historical_weather.csv')
    electricity_prices = pd.read_csv(root.DIR_DATA_RAW + 'electricity_prices.csv')
    gas_prices = pd.read_csv(root.DIR_DATA_RAW + 'gas_prices.csv')
    return train, client, historical_weather, electricity_prices, gas_prices


def initialize_geolocator(user_agent="county_locator"):
    """Initialize the geolocator object."""
    return Nominatim(user_agent=user_agent)


def get_county_from_coordinates(latitude, longitude, geolocator):
    """Retrieve county name from coordinates using the geolocator."""
    location = geolocator.reverse((latitude, longitude), language="en")
    if location:
        return location.raw.get("address", {}).get("county", "Unknown")
    return "Unknown"


def add_county_column(df):
    """Add county column to DataFrame based on coordinates."""
    geolocator = initialize_geolocator()
    coordinates = df[['longitude', 'latitude']].drop_duplicates()
    coordinates['county'] = coordinates.apply(
        lambda row: get_county_from_coordinates(row['latitude'], row['longitude'], geolocator), axis=1
    )
    df = df.merge(coordinates[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'])
    df = df.drop(['longitude', 'latitude'], axis=1)
    return df


def filter_estonian_counties(df):
    """Filter rows by Estonian counties and map county names to integers."""
    county_locations = [
        'Saare County', 'Võru County', 'Pärnu County', 'Valga County', 'Viljandi County', 'Tartu County',
        'Põlva County', 'Jõgeva County', 'Hiiu County', 'Lääne County', 'Rapla County', 'Järva County',
        'Harju County', 'Lääne-Viru County', 'Ida-Viru County'
    ]
    county_to_int = {
        'Saare County': 10, 'Võru County': 15, 'Pärnu County': 7, 'Valga County': 13, 'Viljandi County': 14,
        'Tartu County': 11, 'Põlva County': 8, 'Jõgeva County': 4, 'Hiiu County': 1, 'Lääne County': 6,
        'Rapla County': 9, 'Järva County': 3, 'Harju County': 0, 'Lääne-Viru County': 5, 'Ida-Viru County': 2
    }
    df = df[df['county'].isin(county_locations)]
    df.loc[:, 'county'] = df['county'].map(county_to_int)
    return df


def filter_data(train, client, weather, is_business, product_type, county_code):
    """Filter and split train data based on is_business, product_type, county_code and is_consumption."""
    train = train[
        (train['is_business'] == is_business) &
        (train['product_type'] == product_type) &
        (train['county'] == county_code)
    ]
    train = train.drop(['is_business', 'product_type', 'county'], axis=1)
    train_c = train[train['is_consumption'] == 1]; train_c = train_c.drop(['is_consumption'], axis=1)
    train_p = train[train['is_consumption'] == 0]; train_p = train_p.drop(['is_consumption'], axis=1)

    client = client[
        (client['is_business'] == is_business) &
        (client['product_type'] == product_type) &
        (client['county'] == county_code)
    ]
    client = client.drop(['is_business', 'product_type', 'county'], axis=1)

    weather = weather[weather['county'] == county_code]
    weather = weather.drop(['county'], axis=1)

    return train_c, train_p, client, weather


def fill_missing_target_values(df, column):
    """Sort by date and fill missing values by linear interpolation."""
    df = df.sort_values(by='datetime')
    df[column] = df[column].interpolate(method='linear', limit_direction='both')
    return df


def save_datasets_to_pickle(datasets, paths=None):
    """Save each dataset in datasets list to the corresponding path in paths list as a pickle file."""
    if paths == None:
        import root
        paths = [
            root.DIR_DATA_STAGE + '1_single/train_consumption.pkl',
            root.DIR_DATA_STAGE + '1_single/train_production.pkl',
            root.DIR_DATA_STAGE + '1_single/client.pkl',
            root.DIR_DATA_STAGE + '1_single/historical_weather.pkl',
            root.DIR_DATA_STAGE + '1_single/electricity_prices.pkl',
            root.DIR_DATA_STAGE + '1_single/gas_prices.pkl'
        ]
    
    # Create folders if not exists
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save each dataset to its respective path
    for dataset, path in zip(datasets, paths):
        dataset.to_pickle(path)


def main():
    # Filter parameters
    is_business, product_type, county_code = 1, 3, 0

    # Read datasets
    train, client, historical_weather, electricity_prices, gas_prices = load_datasets()

    # Drop unnecessary columns and change date columns to datetime type
    datasets_info = [
        [train, ['data_block_id', 'row_id', 'prediction_unit_id'], ['datetime']],
        [client, ['data_block_id'], ['date']],
        [historical_weather, ['data_block_id'], ['datetime']],
        [electricity_prices, ['data_block_id', 'origin_date'], ['forecast_date']],
        [gas_prices, ['data_block_id', 'origin_date'], ['forecast_date']]
    ] # [df, [drop_cols], [date_cols]]

    for df, drop_cols, date_cols in datasets_info:
        df.drop(drop_cols, axis=1, inplace=True)
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

    # Add county and filter weather data 
    historical_weather = add_county_column(historical_weather)
    historical_weather = filter_estonian_counties(historical_weather)
    # Group weather data by day
    historical_weather = historical_weather.groupby(['county', 'datetime']).agg('mean').reset_index()

    # Filter data by is_business, product_type, county_code
    train_c, train_p, client, historical_weather = filter_data(train, client, historical_weather, is_business, product_type, county_code)

    # Interpolate target missing values
    train_c = fill_missing_target_values(train_c, 'target')
    train_p = fill_missing_target_values(train_p, 'target')

    # Save datasets to pickle files
    save_datasets_to_pickle([train_c, train_p, client, historical_weather, electricity_prices, gas_prices])


if __name__ == "__main__":
    main()