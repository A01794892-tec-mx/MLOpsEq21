import os
import subprocess
import yaml
import mlflow
import argparse
import json
import dvc.api  # Importing DVC API to fetch data

VENV_DIR = "MLOpsEq21_venv"
PYTHON_SCRIPT_DIR = "Scripts"
PYTHON_EXECUTABLE_NAME = "python.exe"
PYTHON_EXECUTABLE = os.path.join(VENV_DIR, PYTHON_SCRIPT_DIR, PYTHON_EXECUTABLE_NAME)

CONFIG_PATH_KEY = 'config_path'
DVC_VERSION_KEY = 'dvc_version'
MLFLOW_HOST_KEY = 'mlflow'
MLFLOW_EXPERIMENT_KEY = 'experiment_name'
TRAIN_LR_KEY = 'train_lr'
EVALUATE_MODEL_KEY = 'evaluate_model'
MODEL_LR_KEY = 'model_LR'
MODEL_NAME_KEY = 'modelName'
OUT_KEY = 'out'
HOST_KEY = 'host'

#Hyperparameters
C_KEY = 'C'
PENALTY_KEY = 'penalty'
SOLVER_KEY = 'solver'
MAX_ITER_KEY = 'max_iter'
RANDOM_STATE_KEY = 'random_state'

STAGES = ["data_load", "data_preproc", "train_lr", "evaluate_model"]
SCRIPTS = ["src/stages/load_data.py", "src/stages/preprocess_data.py", "src/stages/train_lr.py", "src/stages/evaluate_model.py"]

def get_dvc_data_path(dvc_path, repo, rev):
    return dvc.api.get_url(path=dvc_path, repo=repo, rev=rev)

def load_params(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def run_script(script, config):
    # Use absolute path for the script
    script_path = os.path.abspath(script)
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, script_path, "--config", config[CONFIG_PATH_KEY]],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running script '{script}': {e}")
        print(f"Command output: {e.output}")
        print(f"Error output: {e.stderr}")
        raise

def run_pipeline_stage(stage_name, script, config):
    with mlflow.start_run(run_name=stage_name):
        mlflow.set_tag("data_version", config[DVC_VERSION_KEY])
        result = run_script(script, config)

        if stage_name == TRAIN_LR_KEY:
            mlflow.log_params(config[TRAIN_LR_KEY])

            mlflow.log_param('C', config[TRAIN_LR_KEY][MODEL_LR_KEY][C_KEY])
            mlflow.log_param('penalty', config[TRAIN_LR_KEY][MODEL_LR_KEY][PENALTY_KEY])
            mlflow.log_param('solver', config[TRAIN_LR_KEY][MODEL_LR_KEY][SOLVER_KEY])
            mlflow.log_param('max_iter', config[TRAIN_LR_KEY][MODEL_LR_KEY][MAX_ITER_KEY])
            mlflow.log_param('random_state', config[TRAIN_LR_KEY][MODEL_LR_KEY][RANDOM_STATE_KEY])
            
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/model", f"{config[TRAIN_LR_KEY][MODEL_LR_KEY][MODEL_NAME_KEY]}", tags={"data_version": config[DVC_VERSION_KEY]})

        if stage_name == EVALUATE_MODEL_KEY:
            # Load evaluation metrics from the generated JSON file
            metrics_path = f"{config[EVALUATE_MODEL_KEY][OUT_KEY]}/evaluation_metrics_{config[DVC_VERSION_KEY]}.json"
            with open(metrics_path) as metrics_file:
                evaluation_metrics = json.load(metrics_file)
                # Log metrics to MLflow
                mlflow.log_metric("accuracy_train", evaluation_metrics["accuracy_train"])
                mlflow.log_metric("accuracy_test", evaluation_metrics["accuracy_test"])

def main(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Set up MLFlow tracking
    mlflow.set_tracking_uri(config[MLFLOW_HOST_KEY][HOST_KEY])
    mlflow.set_experiment(f"/{config[MLFLOW_HOST_KEY][MLFLOW_EXPERIMENT_KEY]}/")

    for stage_name, script in zip(STAGES, SCRIPTS):
        run_pipeline_stage(stage_name, script, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(config_path=args.config)
