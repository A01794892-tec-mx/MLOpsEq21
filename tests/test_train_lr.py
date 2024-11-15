import unittest
import pandas as pd
import os
import yaml
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from src.stages.train_lr import train_lr

class TestTrainLR(unittest.TestCase):
    """Functional tests for the train_lr function."""

    def setUp(self):
        self.config = {
            'train_lr': {
                'in_X_train': 'data/processed/X_train.csv',
                'in_y_train': 'data/processed/y_train.csv',
                'out': 'models',  # Not used for saving locally but kept for compatibility
                'model_LR': {
                    'modelName': 'LogisticRegressionModel',
                    'penalty': 'l2',
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 100,
                    'random_state': 42
                }
            },
            'mlflow': {
                'host': 'http://localhost:5000',
                'experiment_name': 'TestExperiment'
            },
            'dvc_version': 'v1.0.6'
        }

        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Create sample training data
        self.X_train = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=['feature1', 'feature2'])
        self.y_train = pd.DataFrame([0, 1], columns=['target'])

        os.makedirs('data/processed', exist_ok=True)
        self.X_train.to_csv(self.config['train_lr']['in_X_train'], index=False)
        self.y_train.to_csv(self.config['train_lr']['in_y_train'], index=False)

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_train_lr_functional(self, mock_client, mock_register_model, mock_log_model, mock_start_run, mock_set_experiment, mock_set_tracking_uri):
        """Run a functional test for train_lr, ensuring proper MLflow logging and model registration."""

        # Mock MLflowClient methods
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_registered_model = MagicMock()
        mock_registered_model.name = self.config['train_lr']['model_LR']['modelName']
        mock_registered_model.version = "1"
        mock_register_model.return_value = mock_registered_model

        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Run the train_lr function
        train_lr(config_path=self.config_path)

        # Assertions for MLflow interactions
        mock_set_tracking_uri.assert_called_once_with(self.config['mlflow']['host'])
        mock_set_experiment.assert_called_once_with(f"/{self.config['mlflow']['experiment_name']}/")
        mock_start_run.assert_called_once_with(run_name="train_lr")
        mock_log_model.assert_called_once_with(
            sk_model=unittest.mock.ANY,  # Checking that a model object is passed
            artifact_path="model"
        )
        mock_register_model.assert_called_once_with(
            model_uri=f"runs:/test_run_id/model",
            name=self.config['train_lr']['model_LR']['modelName']
        )
        client_instance.set_model_version_tag.assert_called_once_with(
            name=self.config['train_lr']['model_LR']['modelName'],
            version="1",
            key="data_version",
            value=self.config['dvc_version']
        )

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        for path in [
            self.config['train_lr']['in_X_train'],
            self.config['train_lr']['in_y_train']
        ]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    unittest.main()
