import dvc.api
import argparse
import pandas as pd
import yaml


# Global Constants for Config Keys
DATA_LOAD = 'data_load'
DVC_VERSION = 'dvc_version'

def load_data(config_path):
    # Load configuration
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Use dvc.api.get_url to get the data for the specified version
    data_url = dvc.api.get_url(
        path=config[DATA_LOAD]['in'],
        repo=config[DATA_LOAD]['repo'],
        rev=config[DVC_VERSION]
    )

    # Load the data and save to the output path
    data = pd.read_csv(data_url)
    data.to_csv(config[DATA_LOAD]['out'], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    load_data(config_path=args.config)