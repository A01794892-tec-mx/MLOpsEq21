import unittest
import pandas as pd
import yaml
import pickle
import json
import os
from unittest.mock import patch
from src.stages.evaluate_model import evaluate_model
from sklearn.linear_model import LogisticRegression

class TestEvaluateModelFunction(unittest.TestCase):
    """Functional test for the evaluate_model function to ensure end-to-end behavior."""

    def setUp(self):
        # Sample configuration dictionary with input/output paths and model parameters
        self.config = {
            'train_lr': {
                'out': 'models',
                'model_LR': {
                    'modelName': 'LogisticRegressionModel'
                }
            },
            'dvc_version': 'v1.0.6',
            'evaluate_model': {
                'in_X_train': 'data/processed/X_train.csv',
                'in_X_test': 'data/processed/X_test.csv',
                'in_y_train': 'data/processed/y_train.csv',
                'in_y_test': 'data/processed/y_test.csv',
                'out': 'reports'
            },
            'mlflow': {  # Added mlflow configuration
                'host': 'http://localhost:5000',
                'experiment_name': 'TestEvaluateModelExperiment'
            }
        }

        # Save the configuration to a YAML file
        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Create sample data files for X_train, X_test, y_train, and y_test
        self.sample_X_train = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=['feature1', 'feature2'])
        self.sample_X_test = pd.DataFrame([[0.5, 0.6], [0.7, 0.8]], columns=['feature1', 'feature2'])
        self.sample_y_train = pd.DataFrame([0, 1], columns=['target'])
        self.sample_y_test = pd.DataFrame([0, 1], columns=['target'])

        os.makedirs('data/processed', exist_ok=True)
        self.sample_X_train.to_csv(self.config['evaluate_model']['in_X_train'], index=False)
        self.sample_X_test.to_csv(self.config['evaluate_model']['in_X_test'], index=False)
        self.sample_y_train.to_csv(self.config['evaluate_model']['in_y_train'], index=False)
        self.sample_y_test.to_csv(self.config['evaluate_model']['in_y_test'], index=False)

        # Train and save a basic logistic regression model
        model = LogisticRegression()
        model.fit(self.sample_X_train, self.sample_y_train.values.ravel())
        os.makedirs(self.config['train_lr']['out'], exist_ok=True)
        model_path = os.path.join(
            self.config['train_lr']['out'],
            f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl"
        )
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def test_evaluate_model_functional(self):
        """Run a functional test on evaluate_model, checking metrics calculation and saving."""
        evaluate_model(config_path=self.config_path)

        metrics_path = os.path.join(
            self.config['evaluate_model']['out'],
            f"evaluation_metrics_{self.config['dvc_version']}.json"
        )
        self.assertTrue(os.path.exists(metrics_path), "Metrics file should be created")

        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        self.assertIn("accuracy_train", metrics_data)
        self.assertIn("accuracy_test", metrics_data)
        self.assertIn("confusion_matrix", metrics_data)
        self.assertIn("classification_report", metrics_data)
        self.assertAlmostEqual(metrics_data["accuracy_train"], 0.5, places=4)
        self.assertAlmostEqual(metrics_data["accuracy_test"], 0.5, places=4)
        
    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        for path in [
            self.config['evaluate_model']['in_X_train'],
            self.config['evaluate_model']['in_X_test'],
            self.config['evaluate_model']['in_y_train'],
            self.config['evaluate_model']['in_y_test']
        ]:
            if os.path.exists(path):
                os.remove(path)

        model_path = os.path.join(
            self.config['train_lr']['out'],
            f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl"
        )
        if os.path.exists(model_path):
            os.remove(model_path)

        metrics_path = os.path.join(
            self.config['evaluate_model']['out'],
            f"evaluation_metrics_{self.config['dvc_version']}.json"
        )
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

if __name__ == "__main__":
    unittest.main()