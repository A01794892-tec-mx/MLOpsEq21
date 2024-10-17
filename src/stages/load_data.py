import dvc.api
import argparse
import pandas as pd
import yaml

def load_data(config_path, repo, dvc_version):
    # Load configuration
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Use dvc.api.get_url to get the data for the specified version
    data_url = dvc.api.get_url(
        path=config['data_load']['in'],
        repo=repo,
        rev=dvc_version  # Use the version from the params.yaml file
    )

    # Load the data and save to the output path
    data = pd.read_csv(data_url)
    data.to_csv(config['data_load']['out'], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dvc-version', required=False, help="DVC version or Git tag to pull data from")
    args = parser.parse_args()

    load_data(config_path=args.config, dvc_version=args.dvc_version)