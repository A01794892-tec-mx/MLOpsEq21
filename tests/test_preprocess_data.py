import unittest
import pandas as pd
import os
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.stages.preprocess_data import data_pre_proc

class TestDataPreprocessing(unittest.TestCase):
    """Functional tests for the data_pre_proc function."""

    def setUp(self):
        self.config = {
            'data_preproc': {
                'in': 'data/raw/test_data.csv',
                'out_X_train': 'data/processed/X_train.csv',
                'out_X_test': 'data/processed/X_test.csv',
                'out_y_train': 'data/processed/y_train.csv',
                'out_y_test': 'data/processed/y_test.csv'
            },
            'base': {
                'random_state': 42
            }
        }

        # Save configuration to YAML
        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Create sample raw data
        self.raw_data = pd.DataFrame({
            'Region': ['Region1', 'Region2', 'Region1', 'Region2'],
            'Temperature': [30.0, 25.0, 35.0, 28.0],
            'RH': [40, 50, 60, 70],
            'Ws': [10, 15, 20, 25],
            'Rain': [0.0, 0.1, 0.0, 0.2],
            'FFMC': [85.0, 90.0, 88.0, 87.0],
            'DMC': [10.0, 15.0, 12.0, 13.0],
            'DC': [40.0, 50.0, 45.0, 42.0],
            'ISI': [5.0, 6.0, 7.0, 4.0],
            'BUI': [20.0, 25.0, 22.0, 23.0],
            'FWI': [3.0, 4.0, 5.0, 2.0],
            'Classes': ['fire', 'not fire', 'fire', 'not fire'],
            'day': [1, 2, 3, 4],
            'month': [6, 6, 6, 6],
            'year': [2023, 2023, 2023, 2023]
        })

        os.makedirs('data/raw', exist_ok=True)
        self.raw_data.to_csv(self.config['data_preproc']['in'], index=False)

    def test_data_preproc_functional(self):
        """Run a functional test on data_pre_proc, ensuring proper outputs."""
        data_pre_proc(config_path=self.config_path)

        # Check that output files are created
        for output in [
            self.config['data_preproc']['out_X_train'],
            self.config['data_preproc']['out_X_test'],
            self.config['data_preproc']['out_y_train'],
            self.config['data_preproc']['out_y_test']
        ]:
            self.assertTrue(os.path.exists(output), f"Output file {output} was not created.")

        # Validate the content of processed files
        X_train = pd.read_csv(self.config['data_preproc']['out_X_train'])
        X_test = pd.read_csv(self.config['data_preproc']['out_X_test'])
        y_train = pd.read_csv(self.config['data_preproc']['out_y_train'])
        y_test = pd.read_csv(self.config['data_preproc']['out_y_test'])

        self.assertGreater(len(X_train), 0, "X_train should not be empty.")
        self.assertGreater(len(X_test), 0, "X_test should not be empty.")
        self.assertGreater(len(y_train), 0, "y_train should not be empty.")
        self.assertGreater(len(y_test), 0, "y_test should not be empty.")

    def test_missing_input_file(self):
        """Test that data_pre_proc raises an error if the input file is missing."""
        os.remove(self.config['data_preproc']['in'])

        with self.assertRaises(FileNotFoundError, msg="Expected a FileNotFoundError for missing input file."):
            data_pre_proc(config_path=self.config_path)

    def test_output_directory_creation(self):
        """Verify that data_pre_proc creates output directories if they do not exist."""
        for output in [
            self.config['data_preproc']['out_X_train'],
            self.config['data_preproc']['out_X_test'],
            self.config['data_preproc']['out_y_train'],
            self.config['data_preproc']['out_y_test']
        ]:
            if os.path.exists(output):
                os.remove(output)

        data_pre_proc(config_path=self.config_path)

        for output in [
            self.config['data_preproc']['out_X_train'],
            self.config['data_preproc']['out_X_test'],
            self.config['data_preproc']['out_y_train'],
            self.config['data_preproc']['out_y_test']
        ]:
            self.assertTrue(os.path.exists(output), f"Output file {output} was not created.")

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        if os.path.exists(self.config['data_preproc']['in']):
            os.remove(self.config['data_preproc']['in'])

        for output in [
            self.config['data_preproc']['out_X_train'],
            self.config['data_preproc']['out_X_test'],
            self.config['data_preproc']['out_y_train'],
            self.config['data_preproc']['out_y_test']
        ]:
            if os.path.exists(output):
                os.remove(output)

if __name__ == "__main__":
    unittest.main()
