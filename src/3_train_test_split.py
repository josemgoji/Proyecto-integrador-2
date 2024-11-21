import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root

    datos = pd.read_pickle(root.DIR_DATA_STAGE + 'merged_df.pkl')
    datos = datos.set_index('datetime')
    datos = datos.asfreq('h')
    
    return  datos

def train_test_split(datos, fin_train, ):
    datos_train = datos.loc[: fin_train, :]
    datos_test  = datos.loc[fin_train:, :]

    return datos_train, datos_test

def save_datasets_to_pickle(datasets, paths=None):
    """Save each dataset in datasets list to the corresponding path in paths list as a pickle file."""
    if paths == None:
        import root
        paths = [
            root.DIR_DATA_STAGE + 'train.pkl',
            root.DIR_DATA_STAGE + 'test.pkl',
        ]

    # Create folders if not exists
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save each dataset to its respective path
    for dataset, path in zip(datasets, paths):
        dataset.to_pickle(path)

def main():
    # Read datasets
    datos = load_datasets()
    
    # Prepare date columns
    fin_train = '2023-01-31 23:59:00'
    datos_train, datos_test = train_test_split(datos, fin_train)
    
    # Save datasets
    save_datasets_to_pickle([datos_train, datos_test])

if __name__ == "__main__": 
    main()

