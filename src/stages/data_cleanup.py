import argparse
import logging
from typing import Text
import yaml
import pandas as pd

# Constants for config keys
DATA_CLEANUP_KEY = 'data_cleanup'
DATASET_KEY = 'dataset_algerian_forest_fires'
DATASET_CLEAN_KEY = 'dataset_algerian_forest_fires_clean'

def load_config(config_path: Text) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as conf_file:
        return yaml.safe_load(conf_file)

def load_dataset(config: dict) -> pd.DataFrame:
    """Load dataset from CSV file."""
    dataset_path = config[DATA_CLEANUP_KEY][DATASET_KEY]
    return pd.read_csv(dataset_path, sep=',', header='infer')

def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Perform cleaning operations on the dataset."""
    # Rename columns to remove leading/trailing spaces
    columns_to_rename = {' RH': 'RH', ' Ws': 'Ws', 'Classes  ': 'Classes', 'Rain ': 'Rain'}
    data.rename(columns=columns_to_rename, inplace=True)

    # Remove extra spaces from 'Classes' column
    data['Classes'] = data['Classes'].str.strip()

    # Drop rows with missing values
    return data.dropna()

def transform_dataset(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ['Classes', 'Region']
    integer_columns = ['day', 'month', 'year']
    continuous_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

    # Create a copy and apply transformations to a correct type
    data_transformed = data.copy()
    data_transformed[categorical_columns] = data[categorical_columns].astype('category')
    data_transformed[integer_columns] = data[integer_columns].astype('int64')
    data_transformed[continuous_columns] = data[continuous_columns].astype('float64')

    return data_transformed

def save_dataset(data: pd.DataFrame, config: dict) -> None:
    """Save the cleaned and transformed dataset to CSV."""
    output_path = config[DATA_CLEANUP_KEY][DATASET_CLEAN_KEY]
    data.to_csv(output_path, index=False)
    logging.info(f"Cleaned dataset saved to {output_path}")

def data_cleanup(config_path: Text) -> None:
    """Main function to clean and transform the dataset."""
    config = load_config(config_path)
    data = load_dataset(config)
    data_cleaned = clean_dataset(data)
    data_transformed = transform_dataset(data_cleaned)
    save_dataset(data_transformed, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    data_cleanup(config_path=args.config)
