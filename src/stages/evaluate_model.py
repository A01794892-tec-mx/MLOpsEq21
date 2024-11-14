import argparse
from typing import Text
from sklearn import metrics
import yaml
import pandas as pd
import json
import os
import mlflow.pyfunc

# Global Constants for Config Keys
TRAIN_LR_KEY = 'train_lr'
MODEL_LR_KEY = 'model_LR'
MODEL_NAME_KEY = 'modelName'
MLFLOW_KEY = 'mlflow'
EXPERIMENT_NAME_KEY = 'experiment_name'
EVALUATE_MODEL = 'evaluate_model'
IN_X_TRAIN = 'in_X_train'
IN_X_TEST = 'in_X_test'
IN_Y_TRAIN = 'in_y_train'
IN_Y_TEST = 'in_y_test'
EVALUATE_MODEL_OUT = 'out'

def evaluate_model(config_path: Text) -> None:
    try:
        with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")
        return

    # Fetch the latest model from MLflow Model Registry
    mlflow.set_tracking_uri(config['mlflow']['host'])
    client = mlflow.tracking.MlflowClient()

    try:
        model_name = config[TRAIN_LR_KEY][MODEL_LR_KEY][MODEL_NAME_KEY]
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(model_versions, key=lambda mv: int(mv.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model: {model_name}, version: {latest_version.version}")
    except Exception as e:
        print(f"Error: Failed to retrieve model from MLflow. {e}")
        return

    # Load datasets
    try:
        X_train = pd.read_csv(config[EVALUATE_MODEL][IN_X_TRAIN])
        X_test = pd.read_csv(config[EVALUATE_MODEL][IN_X_TEST])
        y_train = pd.read_csv(config[EVALUATE_MODEL][IN_Y_TRAIN]).squeeze()
        y_test = pd.read_csv(config[EVALUATE_MODEL][IN_Y_TEST]).squeeze()
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return
    except pd.errors.EmptyDataError as e:
        print(f"Error: Data file is empty. {e}")
        return

    # Make predictions
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    except ValueError as e:
        print(f"Error: Failed to make predictions. {e}")
        return

    # Calculate evaluation metrics
    try:
        accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
        accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
        classification_report = metrics.classification_report(y_test, y_pred_test, output_dict=True)
    except ValueError as e:
        print(f"Error: Failed to calculate evaluation metrics. {e}")
        return

    # Print evaluation metrics
    print(f"Accuracy on Training Set: {accuracy_train:.4f}")
    print(f"Accuracy on Test Set: {accuracy_test:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred_test))

    # Create a dictionary to hold all metrics
    evaluation_metrics = {
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "confusion_matrix": confusion_matrix.tolist(),  # Convert NumPy array to list for JSON serialization
        "classification_report": classification_report
    }

    # Save the metrics to a JSON file
    output_dir = config[EVALUATE_MODEL][EVALUATE_MODEL_OUT]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics_path = os.path.join(output_dir, f"evaluation_metrics_{config['dvc_version']}.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
        print(f"Evaluation metrics saved to {metrics_path}")
    except IOError as e:
        print(f"Error: Failed to save metrics to file. {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained logistic regression model.")
    parser.add_argument('--config', required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    evaluate_model(config_path=args.config)
