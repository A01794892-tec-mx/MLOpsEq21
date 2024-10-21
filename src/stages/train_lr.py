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

#Hyperparameters
C_KEY = 'C'
PENALTY_KEY = 'penalty'
SOLVER_KEY = 'solver'
MAX_ITER_KEY = 'max_iter'
RANDOM_STATE_KEY = 'random_state'


def train_lr(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    # Start MLflow run
    with mlflow.start_run():  # Remove experiment setting here
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

        model_dir = config[TRAIN_LR_KEY]['out']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the model locally
        model_path = f"{model_dir}/{config[TRAIN_LR_KEY][MODEL_LR_KEY][MODEL_NAME_KEY]}_{config['dvc_version']}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=config[TRAIN_LR_KEY][MODEL_LR_KEY][MODEL_NAME_KEY])

        # Log model parameters (optional)
        mlflow.log_params(config[TRAIN_LR_KEY][MODEL_LR_KEY])

        mlflow.log_param('C', config[TRAIN_LR_KEY][MODEL_LR_KEY][C_KEY])
        mlflow.log_param('penalty', config[TRAIN_LR_KEY][MODEL_LR_KEY][PENALTY_KEY])
        mlflow.log_param('solver', config[TRAIN_LR_KEY][MODEL_LR_KEY][SOLVER_KEY])
        mlflow.log_param('max_iter', config[TRAIN_LR_KEY][MODEL_LR_KEY][MAX_ITER_KEY])
        mlflow.log_param('random_state', config[TRAIN_LR_KEY][MODEL_LR_KEY][RANDOM_STATE_KEY])

        # Optionally log the model path
        mlflow.log_artifact(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    train_lr(config_path=args.config)