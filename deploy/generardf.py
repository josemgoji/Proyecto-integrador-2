from json import load
import sys
import os
import pandas as pd

def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    import root
    train = pd.read_pickle(root.DIR_DATA_STAGE + 'test_preprocessed.pkl')
    return train


datos = load_datasets()

datos = datos.head(24)

datos.drop(columns=['target'], inplace=True)

datos.to_csv('prueba.csv')

