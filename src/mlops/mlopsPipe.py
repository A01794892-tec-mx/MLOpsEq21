import os
import subprocess
import yaml
import mlflow
import argparse
import json
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
        #mlflow.log_params({'data_version': config['dvc_version']})
        #mlflow.log_params(config[stage_name])
        mlflow.set_tag("data_version", config['dvc_version'])
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
                mlflow.log_params(config[stage_name])
                run_id = mlflow.active_run().info.run_id  # Get the current run ID
                mlflow.register_model(f"runs:/{run_id}/model", f"{config['train_lr']['model_LR']['modelName']}",tags={"data_version": config['dvc_version']})  # Register the model
                            
            if stage_name == "evaluate_model":
                # Load evaluation metrics from the generated JSON file
                metrics_path = f"{config['evaluate_model']['out']}/evaluation_metrics_{config['dvc_version']}.json"
                with open(metrics_path) as metrics_file:
                    evaluation_metrics = json.load(metrics_file)
                    # Log metrics to MLflow
                    mlflow.log_metric("accuracy_train", evaluation_metrics["accuracy_train"])
                    mlflow.log_metric("accuracy_test", evaluation_metrics["accuracy_test"])

                    # Log confusion matrix and classification report
                    #mlflow.log_dict(evaluation_metrics["classification_report"], "classification_report.json")
                    #mlflow.log_dict({"confusion_matrix": evaluation_metrics["confusion_matrix"]}, "confusion_matrix.json")
            
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
    mlflow.set_experiment(f"/{config['mlflow']['experiment_name']}/")  # Modify with your experiment path/name



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