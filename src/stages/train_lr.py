import argparse
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import mlflow
import mlflow.sklearn  # For registering sklearn models

def train_lr(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    # Start MLflow run
    with mlflow.start_run():  # Remove experiment setting here
        # Load training data
        X_train = pd.read_csv(config['train_lr']['in_X_train'])
        y_train = pd.read_csv(config['train_lr']['in_y_train'])
        
        # Train Logistic Regression model
        model = LogisticRegression(
            penalty=config['train_lr']['model_LR']['penalty'],
            C=config['train_lr']['model_LR']['C'],
            solver=config['train_lr']['model_LR']['solver'],
            max_iter=config['train_lr']['model_LR']['max_iter'],
            random_state=config['train_lr']['model_LR']['random_state']
        )
        model.fit(X_train, np.ravel(y_train))

        # Save the model locally
        model_path = f"{config['train_lr']['model_LR']['out']}/{config['train_lr']['model_LR']['modelName']}_{config['dvc_version']}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=config['train_lr']['model_LR']['modelName'])

        # Log model parameters (optional)
        mlflow.log_params(config['train_lr']['model_LR'])

        # Optionally log the model path
        mlflow.log_artifact(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    train_lr(config_path=args.config)