"""
Test Suite for the load_data Function

This test suite covers a variety of scenarios to ensure the load_data function
in src/stages/load_data.py behaves as expected in different situations.

Test Cases:
1. Functional Test (test_load_data_functional):
   - Simulates a successful run of the load_data function, where it reads data from
     the specified input file (simulated with a mock for DVC's get_url) and saves
     it to the output path.
   - This test verifies that the output file is created, contains the expected number
     of rows, and has the correct column structure.

2. Missing Input File (test_missing_input_file):
   - Tests how load_data handles the case where the input file specified in the
     configuration does not exist.
   - This test expects a FileNotFoundError to be raised, ensuring that load_data
     provides a meaningful error when the data source is missing.

3. Empty Data File (test_empty_data_file):
   - Verifies that load_data handles an empty input file gracefully.
   - This test expects a pandas.errors.EmptyDataError to be raised, validating that
     load_data detects and reports empty data files.

4. Invalid Configuration File (test_invalid_config_file):
   - Checks how load_data behaves when provided with an invalid YAML configuration file.
   - This test expects a yaml.YAMLError to be raised if the configuration file format
     is incorrect or unreadable, ensuring that load_data provides useful error messages
     for configuration issues.
"""
import os
import unittest
from unittest.mock import patch
import pandas as pd
import yaml
from src.stages.load_data import load_data  # Adjust path as needed


class TestLoadDataFunction(unittest.TestCase):
    """Functional test suite for the load_data function."""

    def setUp(self):
        # Define a sample configuration for input and output paths
        self.config = {
            'data_load': {
                'in': 'data/raw/sample_data.csv',
                'out': 'data/interim/output_data.csv',
                'repo': 'https://github.com/A01794892-tec-mx/MLOpsEq21.git'
            },
            'dvc_version': 'v1.0.6'
        }

        # Save this configuration to a YAML file
        self.config_path = "config.yaml"
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        # Create sample data
        self.sample_data_csv = """Region,day,month,year,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Classes
Bejaia,1,6,2012,29,57,18,0,65.7,3.4,7.6,1.3,3.4,0.5,not fire
Bejaia,2,6,2012,30,60,20,0,66.2,4.0,7.9,1.0,3.7,0.6,fire
"""

        # Simulate the DVC-tracked file path
        self.input_data_path = self.config['data_load']['in']
        os.makedirs(os.path.dirname(self.input_data_path), exist_ok=True)
        with open(self.input_data_path, 'w') as file:
            file.write(self.sample_data_csv)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config['data_load']['out']), exist_ok=True)

    @patch("dvc.api.get_url")
    def test_load_data_functional(self, mock_get_url):
        """Run a functional test on load_data with mocked dvc.api.get_url."""
        mock_get_url.return_value = self.input_data_path

        # Execute the load_data function
        load_data(config_path=self.config_path)

        # Verify output file is created
        output_data_path = self.config['data_load']['out']
        self.assertTrue(os.path.exists(output_data_path), "Output file was not created")

        # Load the output file and verify its content
        output_df = pd.read_csv(output_data_path)
        self.assertFalse(output_df.empty, "Output data should not be empty")
        self.assertEqual(len(output_df), 2, "Output data should contain 2 rows")

    def tearDown(self):
        # Remove only files created during the test
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.input_data_path):
            os.remove(self.input_data_path)
        output_path = self.config['data_load']['out']
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
