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


# def run_pipeline_stage(stage_name, script, params, config_path):
#     with mlflow.start_run(run_name=stage_name):
#         mlflow.log_params(params)
#         subprocess.run(["python", script, "--config", config_path], check=True)

def run_pipeline_stage(stage_name, script, config):
    with mlflow.start_run(run_name=stage_name):
        mlflow.log_params(config[stage_name])
        try:
            # Use absolute path for the script
            script_path = os.path.abspath(script)  # Convert to absolute path

            # Specify the Python executable from your virtual environment
            python_executable = os.path.join("MLOpsEq21_venv", "Scripts", "python.exe")

            result = subprocess.run(
                [python_executable, script_path, "--config", config['config_path']],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if stage_name == "train_lr":
                run_id = mlflow.active_run().info.run_id  # Get the current run ID
                mlflow.register_model(f"runs:/{run_id}/model", f"{config['train_lr']['model_LR']['modelName']}_{config['dvc_version']}")  # Register the model
            print(result.stdout)  # Print standard output
        except subprocess.CalledProcessError as e:
            print(f"Error in stage '{stage_name}': {e}")
            print(f"Command output: {e.output}")  # This will give you more insight
            print(f"Error output: {e.stderr}")  # Print error output for more details
            raise  # Optionally re-raise the error to stop the pipeline


# Main script to run the pipeline
def main(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Set up MLFlow tracking
    mlflow.set_tracking_uri(config['mlflow']['host'])  # Modify with your tracking server URI
    mlflow.set_experiment(f"/{config['mlflow']['experiment_name']}_{config['dvc_version']}/")  # Modify with your experiment path/name



    run_pipeline_stage("data_load", "src/stages/load_data.py", config)
    run_pipeline_stage("data_preproc", "src/stages/preprocess_data.py", config)
    run_pipeline_stage("train_lr", "src/stages/train_lr.py", config)
    run_pipeline_stage("evaluate_model", "src/stages/evaluate_model.py", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    # parser.add_argument('--repo', required=False, help="Git repo")
    # parser.add_argument('--dvc_version', required=False, help="DVC version or Git tag to pull data from")
    args = parser.parse_args()

    # Run the pipeline with the provided tag
    main(config_path=args.config)