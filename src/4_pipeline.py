import pandas as pd
import sys
import os
import cloudpickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline



def load_datasets():
    """Load all datasets and return them as dataframes."""
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train.pkl')
    test = pd.read_pickle(root.DIR_DATA_STAGE + 'test.pkl')
    return  train, test


def fill_missing_values(df):
    """Sort by date and fill missing values by linear interpolation."""
    df = df.sort_values(by='datetime')
    df = df.interpolate(method='linear', limit_direction='both')
    return df


def save_pipeline_to_pickle(pipeline, file_name):
    """Save the pipeline as a pickle file."""
    import root
    with open(root.DIR_DATA_ANALYTICS + file_name, 'wb') as f:
        cloudpickle.dump(pipeline, f)


def save_datasets_to_pickle(datasets, paths=None):
    """Save each dataset in datasets list to the corresponding path in paths list as a pickle file."""
    if paths == None:
        import root
        paths = [
            root.DIR_DATA_STAGE + 'train_preprocessed.pkl',
            root.DIR_DATA_STAGE + 'test_preprocessed.pkl',
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
    pipeline = Pipeline(steps=[
        ('fill_missing', FunctionTransformer(fill_missing_values, validate=False)),
        ('scale', MinMaxScaler())
    ])
    train_processed = pipeline.fit_transform(train)
    test_processed = pipeline.transform(test)

    # Convert to DataFrame
    train_df = pd.DataFrame(train_processed, columns=train.columns, index=train.index)
    test_df = pd.DataFrame(test_processed, columns=test.columns, index=test.index)

    # Save scaler and datasets
    save_pipeline_to_pickle(pipeline, 'pipeline.pkl')
    save_datasets_to_pickle([train_df, test_df])


if __name__ == "__main__": 
    main()