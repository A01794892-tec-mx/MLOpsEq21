import argparse
from typing import Text
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def train_lr(config_path: Text) -> None:
    # Start a new MLflow run if none exists
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Load the config file
        with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)

        # Load training data
        X_train = pd.read_csv(config['train_lr']['in_X_train'])
        y_train = pd.read_csv(config['train_lr']['in_y_train'])

        # Logistic Regression model training
        modeloRL = LogisticRegression(penalty=config['train_lr']['model_LR']['penalty'],
                                      C=config['train_lr']['model_LR']['C'],
                                      solver=config['train_lr']['model_LR']['solver'],
                                      max_iter=config['train_lr']['model_LR']['max_iter'],
                                      random_state=config['train_lr']['model_LR']['random_state'])

        modeloRL.fit(X_train, np.ravel(y_train))

        # Save the model locally
        model_path = f"{config['train_lr']['model_LR']['out']}/{config['train_lr']['model_LR']['modelName']}_{config['data_load']['dvc_version']}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(modeloRL, f)

        # Log the model signature
        signature = infer_signature(X_train, modeloRL.predict(X_train))

        # Log the model in MLflow along with its signature
        mlflow.sklearn.log_model(
            modeloRL, 
            "logistic_regression_model", 
            signature=signature
        )

    finally:
        # Make sure to end the run at the end of the script
        mlflow.end_run()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_lr(config_path=args.config)
