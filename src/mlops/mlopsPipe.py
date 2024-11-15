import os
import subprocess
import yaml
import mlflow
import argparse
import json
import dvc.api

# Paths and Keys
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

# Hyperparameters
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
    print(f"""########   Running {stage_name} stage #######""")
    # Set up MLflow experiment
    mlflow.set_tracking_uri(config[MLFLOW_HOST_KEY][HOST_KEY])
    mlflow.set_experiment(f"/{config[MLFLOW_HOST_KEY][MLFLOW_EXPERIMENT_KEY]}/")
    
    if stage_name == TRAIN_LR_KEY:
        run_script(script, config)
    
    else:
        with mlflow.start_run(run_name=stage_name):
            mlflow.set_tag("data_version", config[DVC_VERSION_KEY])
            run_script(script, config)

            if stage_name == EVALUATE_MODEL_KEY:
                # Load evaluation metrics from the generated JSON file
                metrics_path = f"{config[EVALUATE_MODEL_KEY][OUT_KEY]}/evaluation_metrics_{config[DVC_VERSION_KEY]}.json"
                with open(metrics_path) as metrics_file:
                    evaluation_metrics = json.load(metrics_file)
                    mlflow.log_metric("accuracy_train", evaluation_metrics["accuracy_train"])
                    mlflow.log_metric("accuracy_test", evaluation_metrics["accuracy_test"])
    
    print(f"Stage {stage_name} completed successfully.\n")

def main(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    for stage_name, script in zip(STAGES, SCRIPTS):
        run_pipeline_stage(stage_name, script, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(config_path=args.config)
