import argparse
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import mlflow
import mlflow.sklearn  # For registering sklearn models
import os

TRAIN_LR_KEY = 'train_lr'
MODEL_LR_KEY = 'model_LR'
MODEL_NAME_KEY = 'modelName'
MLFLOW_KEY = 'mlflow'
EXPERIMENT_NAME_KEY = 'experiment_name'
MLFLOW_HOST_KEY = 'mlflow'
MLFLOW_EXPERIMENT_KEY = 'experiment_name'

# Hyperparameters
C_KEY = 'C'
PENALTY_KEY = 'penalty'
SOLVER_KEY = 'solver'
MAX_ITER_KEY = 'max_iter'
RANDOM_STATE_KEY = 'random_state'

def train_lr(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow']['host'])
    
    # Get experiment name from config
    experiment_name = f"/{config[MLFLOW_HOST_KEY][MLFLOW_EXPERIMENT_KEY]}/"
    mlflow.set_experiment(experiment_name)


    # Start MLflow run
    with mlflow.start_run(run_name="train_lr") as run:
        # Set data version tag for the run
        mlflow.set_tag("data_version", config['dvc_version'])

        # Load training data
        X_train = pd.read_csv(config[TRAIN_LR_KEY]['in_X_train'])
        y_train = pd.read_csv(config[TRAIN_LR_KEY]['in_y_train'])

        # Train Logistic Regression model
        model = LogisticRegression(
            penalty=config[TRAIN_LR_KEY][MODEL_LR_KEY][PENALTY_KEY],
            C=config[TRAIN_LR_KEY][MODEL_LR_KEY][C_KEY],
            solver=config[TRAIN_LR_KEY][MODEL_LR_KEY][SOLVER_KEY],
            max_iter=config[TRAIN_LR_KEY][MODEL_LR_KEY][MAX_ITER_KEY],
            random_state=config[TRAIN_LR_KEY][MODEL_LR_KEY][RANDOM_STATE_KEY]
        )
        model.fit(X_train, np.ravel(y_train))

        # Log the model to MLflow
        artifact_path = "model"
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path=artifact_path
        )

        # Register the model in the Model Registry
        run_id = run.info.run_id
        model_name = config[TRAIN_LR_KEY][MODEL_LR_KEY][MODEL_NAME_KEY]
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",
            name=model_name
        )

        # Add the `data_version` tag to the registered model version
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=registered_model.name,
            version=registered_model.version,
            key="data_version",
            value=config['dvc_version']
        )

        # Log model parameters
        mlflow.log_params(config[TRAIN_LR_KEY][MODEL_LR_KEY])

        print(f"Model trained and registered in MLflow under experiment: {experiment_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model registered with version: {registered_model.version}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    train_lr(config_path=args.config)