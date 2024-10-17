import os
import subprocess
import yaml
import mlflow
import argparse
import uuid
import dvc.api  # Importing DVC API to fetch data

def get_dvc_data_path(dvc_path, repo, rev):
    return dvc.api.get_url(path=dvc_path, repo=repo, rev=rev)

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
    # Set up MLFlow tracking
    mlflow.set_tracking_uri("http://localhost:5000")  # Modify with your tracking server URI
    mlflow.set_experiment(f"/forestFires/{tag}/")  # Modify with your experiment path/name

    # # Save the original branch name
    # original_branch = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()

    # # Create a temporary branch from the original branch
    # temp_branch = create_temp_branch(original_branch)

    # try:
    params = load_params("params.yaml")
    # Get DVC data paths using dvc.api.get_url based on the provided tag
    repo_url = "https://github.com/A01794892-tec-mx/MLOpsEq21.git"  # Modify with your repo URL
    # Example of how to get paths from DVC
    X_train_path = get_dvc_data_path(params['data_load']['out'], repo_url, tag)
    y_train_path = get_dvc_data_path(params['data_preproc']['out_y_train'], repo_url, tag)
    print(f"Fetched data: X_train from {X_train_path}, y_train from {y_train_path}")
    # Run the pipeline stages with MLFlow tracking
    run_pipeline_stage("load_data", "src/stages/load_data.py", params['data_load'])
    run_pipeline_stage("preprocess_data", "src/stages/preprocess_data.py", params['data_preproc'])
    run_pipeline_stage("train_lr", "src/stages/train_lr.py", params['train_lr'])
    run_pipeline_stage("evaluate_model", "src/stages/evaluate_model.py", params['evaluate_model'])

    # finally:
    #     # Switch back to the original branch and clean up the temporary branch
    #     cleanup_branch(original_branch, temp_branch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Git tag to pull DVC data from")
    args = parser.parse_args()

    # Run the pipeline with the provided tag
    main(args.tag)