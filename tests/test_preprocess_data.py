import unittest
import pandas as pd
import os
import yaml
from unittest.mock import patch, MagicMock
from src.stages.preprocess_data import data_pre_proc

class TestDataPreprocessing(unittest.TestCase):
    """Test suite for the data_pre_proc function."""

    def setUp(self):
        # Configuration setup
        self.config = {
            'data_preproc': {
                'in': 'data/raw/sample_data.csv',
                'out_X_train': 'data/processed/X_train.csv',
                'out_X_test': 'data/processed/X_test.csv',
                'out_y_train': 'data/processed/y_train.csv',
                'out_y_test': 'data/processed/y_test.csv'
            },
            'base': {
                'random_state': 42
            }
        }

        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Sample dataset
        self.sample_data = pd.DataFrame({
            'Temperature': [30, 25, 20, 35],
            'RH': [45, 50, 55, 40],
            'Ws': [5, 10, 15, 20],
            'Rain': [0, 1, 0, 1],
            'FFMC': [85, 90, 95, 80],
            'DMC': [60, 65, 70, 75],
            'DC': [200, 210, 220, 230],
            'ISI': [10, 15, 20, 25],
            'BUI': [50, 55, 60, 65],
            'FWI': [5, 10, 15, 20],
            'Classes': ['fire', 'not fire', 'fire', 'not fire'],
            'Region': ['Region1', 'Region2', 'Region1', 'Region2'],
            'day': [1, 2, 3, 4],
            'month': [6, 7, 8, 9],
            'year': [2020, 2021, 2022, 2023]
        })

        # Create raw data directory and save sample data
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        self.sample_data.to_csv(self.config['data_preproc']['in'], index=False)

    def test_data_preproc_functional(self):
        """Test the functional behavior of data_pre_proc."""
        data_pre_proc(config_path=self.config_path)

        # Check that output files exist
        self.assertTrue(os.path.exists(self.config['data_preproc']['out_X_train']), "X_train.csv was not created")
        self.assertTrue(os.path.exists(self.config['data_preproc']['out_X_test']), "X_test.csv was not created")
        self.assertTrue(os.path.exists(self.config['data_preproc']['out_y_train']), "y_train.csv was not created")
        self.assertTrue(os.path.exists(self.config['data_preproc']['out_y_test']), "y_test.csv was not created")

        # Validate the content of the processed files
        X_train = pd.read_csv(self.config['data_preproc']['out_X_train'])
        X_test = pd.read_csv(self.config['data_preproc']['out_X_test'])
        y_train = pd.read_csv(self.config['data_preproc']['out_y_train'])
        y_test = pd.read_csv(self.config['data_preproc']['out_y_test'])

        self.assertFalse(X_train.empty, "X_train is empty")
        self.assertFalse(X_test.empty, "X_test is empty")
        self.assertFalse(y_train.empty, "y_train is empty")
        self.assertFalse(y_test.empty, "y_test is empty")

    def test_missing_input_file(self):
        """Test that data_pre_proc raises an error when the input file is missing."""
        os.remove(self.config['data_preproc']['in'])

        with self.assertRaises(FileNotFoundError):
            data_pre_proc(config_path=self.config_path)

    @patch("builtins.print")
    def test_output_directory_creation(self, mock_print):
        """Test that output directories are created automatically."""
        for key in ['out_X_train', 'out_X_test', 'out_y_train', 'out_y_test']:
            if os.path.exists(self.config['data_preproc'][key]):
                os.remove(self.config['data_preproc'][key])

        data_pre_proc(config_path=self.config_path)

        # Check if output files exist
        for key in ['out_X_train', 'out_X_test', 'out_y_train', 'out_y_test']:
            self.assertTrue(os.path.exists(self.config['data_preproc'][key]), f"{key} was not created")

    def tearDown(self):
        # Clean up files and directories
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        if os.path.exists('data/raw'):
            for file in os.listdir('data/raw'):
                os.remove(os.path.join('data/raw', file))
            os.rmdir('data/raw')

        if os.path.exists('data/processed'):
            for file in os.listdir('data/processed'):
                os.remove(os.path.join('data/processed', file))
            os.rmdir('data/processed')

if __name__ == "__main__":
    unittest.main()
