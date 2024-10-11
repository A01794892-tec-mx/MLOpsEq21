import argparse
import pandas as pd
from typing import Text
import yaml


def load_data(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = pd.read_csv(config['data_load']['in'] ,sep=',', header='infer')

    data.head()

    data.to_csv(config['data_load']['out'], index=False)
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data = load_data(config_path=args.config)