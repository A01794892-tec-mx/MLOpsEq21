import os
import subprocess
import yaml
import mlflow
import argparse

# Function to pull DVC data using a Git tag
def pull_dvc_data(tag):
    # Checkout the specific tag
    subprocess.run(["git", "checkout", tag], check=True)
    # Pull the DVC data associated with that tag
    subprocess.run(["dvc", "pull","-f"], check=True)

# Load YAML config file
def load_params(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

# Define pipeline stages
def run_pipeline_stage(stage_name, script, params):
    with mlflow.start_run(run_name=stage_name):
        mlflow.log_params(params)
        subprocess.run(["python", script, "--config", "params.yaml"], check=True)

# Main script to run the pipeline
def main(tag):

    # Pull DVC data associated with the tag
    pull_dvc_data(tag)
    
    # Set up MLFlow tracking
    mlflow.set_tracking_uri("http://localhost:5000")  # Modify with your tracking server URI
    mlflow.set_experiment(f"/pipe/test/")  # Modify with your experiment path/name


    
    # Load parameters from the YAML config
    params = load_params("params.yaml")

    # Run the pipeline stages with MLFlow tracking
    run_pipeline_stage("load_data", "src/stages/load_data.py", params['data_load'])
    run_pipeline_stage("preprocess_data", "src/stages/preprocess_data.py", params['data_preproc'])
    run_pipeline_stage("train_lr", "src/stages/train_lr.py", params['train_lr'])
    run_pipeline_stage("evaluate_model", "src/stages/evaluate_model.py", params['evaluate_model'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Git tag to pull DVC data from")
    args = parser.parse_args()
    
    # Run the pipeline with the provided tag
    main(args.tag)
