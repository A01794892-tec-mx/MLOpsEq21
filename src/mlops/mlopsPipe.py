import os
import subprocess
import yaml
import mlflow
import argparse
import uuid

# Function to create a temporary Git branch
def create_temp_branch(base_branch):
    temp_branch = f"temp-branch-{uuid.uuid4().hex[:8]}"  # Create a unique temp branch name
    subprocess.run(["git", "checkout", "-b", temp_branch, base_branch], check=True)
    return temp_branch

# Function to checkout back to the original branch and clean up
def cleanup_branch(original_branch, temp_branch):
    subprocess.run(["git", "checkout", original_branch], check=True)  # Checkout back to original
    subprocess.run(["git", "branch", "-D", temp_branch], check=True)  # Delete the temp branch

# Function to pull DVC data using a Git tag
def pull_dvc_data(tag):
    subprocess.run(["git", "checkout", tag], check=True)  # Checkout to the given tag
    subprocess.run(["dvc", "pull", "-f"], check=True)  # Pull the corresponding DVC data

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
    # Set up MLFlow tracking
    mlflow.set_tracking_uri("http://localhost:5000")  # Modify with your tracking server URI
    mlflow.set_experiment(f"/pipe/test/")  # Modify with your experiment path/name

    # Save the original branch name
    original_branch = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()

    # Create a temporary branch from the original branch
    temp_branch = create_temp_branch(original_branch)

    try:
        # Pull DVC data associated with the tag
        pull_dvc_data(tag)

        # Load parameters from the YAML config
        params = load_params("params.yaml")

        # Run the pipeline stages with MLFlow tracking
        run_pipeline_stage("load_data", "src/stages/load_data.py", params['data_load'])
        run_pipeline_stage("preprocess_data", "src/stages/preprocess_data.py", params['data_preproc'])
        run_pipeline_stage("train_lr", "src/stages/train_lr.py", params['train_lr'])
        run_pipeline_stage("evaluate_model", "src/stages/evaluate_model.py", params['evaluate_model'])

    finally:
        # Switch back to the original branch and clean up the temporary branch
        cleanup_branch(original_branch, temp_branch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Git tag to pull DVC data from")
    args = parser.parse_args()

    # Run the pipeline with the provided tag
    main(args.tag)
