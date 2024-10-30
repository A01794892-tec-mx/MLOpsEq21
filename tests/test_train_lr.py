"""
TestTrainLRFunction

This test suite performs functional tests for the `train_lr` function, ensuring end-to-end behavior, 
robust error handling, and verification of output. The primary goal of these tests is to validate 
the functionality, correctness, and reliability of the model training pipeline, which includes 
handling configurations, MLflow logging, and saving models.

Test cases included:
- `test_train_lr_functional`: Validates full functionality, ensuring that the model is trained, 
   saved correctly, and MLflow logs essential parameters and artifacts.
- `test_invalid_model_params`: Confirms that an invalid configuration, such as unsupported penalty, 
   results in an appropriate error.
- `test_missing_input_files`: Ensures that the function raises an error if essential input data files 
   are missing.
- `test_output_directory_creation`: Verifies that the function creates the output directory for saving 
   the model if it does not already exist.
"""

import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml
import os
import pickle
import mlflow
from src.stages.train_lr import train_lr


class TestTrainLRFunction(unittest.TestCase):
    """Comprehensive test suite for the train_lr function to ensure robust behavior."""

    def setUp(self):
        self.config = {
            'train_lr': {
                'in_X_train': 'data/processed/X_train.csv',
                'in_y_train': 'data/processed/y_train.csv',
                'out': 'models',
                'model_LR': {
                    'modelName': 'LogisticRegressionModel',
                    'penalty': 'l2',
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 100,
                    'random_state': 42
                }
            },
            'dvc_version': 'v1.0.6'
        }

        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        self.sample_X_train = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=['feature1', 'feature2'])
        self.sample_y_train = pd.DataFrame([0, 1], columns=['target'])

        os.makedirs('data/processed', exist_ok=True)
        self.sample_X_train.to_csv(self.config['train_lr']['in_X_train'], index=False)
        self.sample_y_train.to_csv(self.config['train_lr']['in_y_train'], index=False)

        mlflow_tracking_dir = "mlruns"
        trash_dir = os.path.join(mlflow_tracking_dir, ".trash")
        os.makedirs(trash_dir, exist_ok=True)

        self.experiment_name = "DefaultTestExperiment"
        mlflow.set_experiment(self.experiment_name)

    def test_train_lr_functional(self):
        """Run a functional test on train_lr, checking model saving and MLflow logging."""
        train_lr(config_path=self.config_path)

        model_dir = self.config['train_lr']['out']
        model_name = f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl"
        model_path = os.path.join(model_dir, model_name)
        self.assertTrue(os.path.exists(model_path), "Model file was not created")

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            self.assertIsInstance(model, LogisticRegression, "The model should be a LogisticRegression instance")

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        self.assertGreater(len(runs), 0, "No MLflow runs found")
        latest_run = runs[0]
        
        self.assertEqual(latest_run.data.params['C'], str(self.config['train_lr']['model_LR']['C']))
        self.assertEqual(latest_run.data.params['penalty'], self.config['train_lr']['model_LR']['penalty'])
        self.assertEqual(latest_run.data.params['solver'], self.config['train_lr']['model_LR']['solver'])
        self.assertEqual(latest_run.data.params['max_iter'], str(self.config['train_lr']['model_LR']['max_iter']))
        self.assertEqual(latest_run.data.params['random_state'], str(self.config['train_lr']['model_LR']['random_state']))

    def test_invalid_model_params(self):
        """Test that train_lr raises an error for invalid model parameters."""
        self.config['train_lr']['model_LR']['penalty'] = 'unsupported_penalty'
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        with self.assertRaises(ValueError, msg="train_lr should raise a ValueError for invalid penalty"):
            train_lr(config_path=self.config_path)

    def test_missing_input_files(self):
        """Test that train_lr raises an error if input data files are missing."""
        os.remove(self.config['train_lr']['in_X_train'])
        os.remove(self.config['train_lr']['in_y_train'])

        with self.assertRaises(FileNotFoundError, msg="train_lr should raise a FileNotFoundError for missing input files"):
            train_lr(config_path=self.config_path)

    def test_output_directory_creation(self):
        """Verify that train_lr creates output directory if it doesn't exist."""
        model_dir = self.config['train_lr']['out']
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, file))

        train_lr(config_path=self.config_path)
        self.assertTrue(os.path.exists(model_dir), "Output directory should be created if it doesn't exist")

    def tearDown(self):
        os.remove(self.config_path)
        if os.path.exists(self.config['train_lr']['in_X_train']):
            os.remove(self.config['train_lr']['in_X_train'])
        if os.path.exists(self.config['train_lr']['in_y_train']):
            os.remove(self.config['train_lr']['in_y_train'])

        model_path = os.path.join(self.config['train_lr']['out'], f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)

        mlflow_tracking_dir = "mlruns"
        if os.path.exists(mlflow_tracking_dir):
            import shutil
            shutil.rmtree(mlflow_tracking_dir)

if __name__ == "__main__":
    unittest.main()