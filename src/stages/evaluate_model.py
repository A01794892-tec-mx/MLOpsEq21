import pickle
import argparse
from typing import Text
from sklearn import metrics
import yaml
import pandas as pd
import json
import os

def evaluate_model(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    # Load the trained model
    model_path = f"{config['train_lr']['out']}/{config['train_lr']['model_LR']['modelName']}_{config['dvc_version']}.pkl"
    with open(model_path, 'rb') as f:
        modelLR = pickle.load(f)

    # Load datasets
    X_train = pd.read_csv(config['evaluate_model']['in_X_train'])
    X_test  = pd.read_csv(config['evaluate_model']['in_X_test'])
    y_train = pd.read_csv(config['evaluate_model']['in_y_train'])
    y_test  = pd.read_csv(config['evaluate_model']['in_y_test'])

    # Make predictions
    y_pred_trainRL = modelLR.predict(X_train)
    y_pred_testRL = modelLR.predict(X_test)

    # Calculate evaluation metrics
    accuracy_train = metrics.accuracy_score(y_train, y_pred_trainRL)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_testRL)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_testRL)
    classification_report = metrics.classification_report(y_test, y_pred_testRL, output_dict=True)

    # Print evaluation metrics
    print(f"Accuracy on Training Set: {accuracy_train}")
    print(f"Accuracy on Test Set: {accuracy_test}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred_testRL))


    # Create a dictionary to hold all metrics
    evaluation_metrics = {
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "confusion_matrix": confusion_matrix.tolist(),  # Convert NumPy array to list for JSON serialization
        "classification_report": classification_report
    }

    # Save the metrics to a JSON file
    output_dir = config['evaluate_model']['out']  # Output directory for metrics
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics_path = os.path.join(output_dir, f"evaluation_metrics_{config['dvc_version']}.json")
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    evaluate_model(config_path=args.config)