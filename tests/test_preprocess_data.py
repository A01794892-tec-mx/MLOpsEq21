"""
TestTrainLRFunction

This test suite is designed to comprehensively test the `train_lr` function, which is responsible for training a 
logistic regression model, saving it to disk, and logging essential training details in MLflow. The suite covers 
various aspects of functionality, robustness, and error handling to ensure the model training process works as 
expected under different scenarios.

The test cases included in this suite are:

1. `test_train_lr_functional`: Performs a full functional test to validate that the logistic regression model 
   is trained with correct parameters, saved in the designated output path, and that relevant parameters are 
   logged accurately in MLflow.
2. `test_invalid_model_params`: Tests that `train_lr` raises an appropriate error when given an unsupported 
   model parameter (e.g., an invalid penalty term), verifying that configuration validation is functioning.
3. `test_missing_input_files`: Ensures the function raises a `FileNotFoundError` if input data files are 
   missing, which would prevent model training from proceeding.
4. `test_output_directory_creation`: Verifies that `train_lr` creates the specified output directory if it 
   doesn't already exist, allowing the model file to be saved in a previously nonexistent path.
"""


import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml
import os
import pickle
import mlflow
from src.stages.train_lr import train_lr  # Adjust path as needed

class TestTrainLRFunction(unittest.TestCase):

    def setUp(self):
        # Mock configuration and sample data setup
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

        # Save configuration to a YAML file
        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Prepare sample data and paths
        self.sample_X_train = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=['feature1', 'feature2'])
        self.sample_y_train = pd.DataFrame([0, 1], columns=['target'])
        os.makedirs('data/processed', exist_ok=True)
        self.sample_X_train.to_csv(self.config['train_lr']['in_X_train'], index=False)
        self.sample_y_train.to_csv(self.config['train_lr']['in_y_train'], index=False)

        # Ensure MLflow tracking and `.trash` directories exist
        os.makedirs("mlruns/.trash", exist_ok=True)

        # Set up a test-specific MLflow experiment
        self.experiment_name = "DefaultTestExperiment"
        mlflow.set_experiment(self.experiment_name)

    def test_train_lr_functional(self):
        train_lr(config_path=self.config_path)

        # Verify model file creation
        model_dir = self.config['train_lr']['out']
        model_name = f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl"
        model_path = os.path.join(model_dir, model_name)
        self.assertTrue(os.path.exists(model_path))

        # Check model type
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            self.assertIsInstance(model, LogisticRegression)

        # Verify MLflow logging
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        self.assertGreater(len(runs), 0)
        latest_run = runs[0]
        
        # Check parameter logging
        self.assertEqual(latest_run.data.params['C'], str(self.config['train_lr']['model_LR']['C']))
        self.assertEqual(latest_run.data.params['penalty'], self.config['train_lr']['model_LR']['penalty'])
        self.assertEqual(latest_run.data.params['solver'], self.config['train_lr']['model_LR']['solver'])
        self.assertEqual(latest_run.data.params['max_iter'], str(self.config['train_lr']['model_LR']['max_iter']))
        self.assertEqual(latest_run.data.params['random_state'], str(self.config['train_lr']['model_LR']['random_state']))

    def test_invalid_model_params(self):
        self.config['train_lr']['model_LR']['penalty'] = 'unsupported_penalty'
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        with self.assertRaises(ValueError):
            train_lr(config_path=self.config_path)

    def test_missing_input_files(self):
        os.remove(self.config['train_lr']['in_X_train'])
        os.remove(self.config['train_lr']['in_y_train'])

        with self.assertRaises(FileNotFoundError):
            train_lr(config_path=self.config_path)

    def test_output_directory_creation(self):
        model_dir = self.config['train_lr']['out']
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, file))

        train_lr(config_path=self.config_path)
        self.assertTrue(os.path.exists(model_dir))

    def tearDown(self):
        os.remove(self.config_path)
        if os.path.exists(self.config['train_lr']['in_X_train']):
            os.remove(self.config['train_lr']['in_X_train'])
        if os.path.exists(self.config['train_lr']['in_y_train']):
            os.remove(self.config['train_lr']['in_y_train'])

        model_path = os.path.join(self.config['train_lr']['out'], f"{self.config['train_lr']['model_LR']['modelName']}_{self.config['dvc_version']}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)

        # Clean up MLflow's tracking data (optional, based on local setup)
        mlflow_tracking_dir = "mlruns"
        if os.path.exists(mlflow_tracking_dir):
            import shutil
            shutil.rmtree(mlflow_tracking_dir)

if __name__ == "__main__":
    unittest.main()
