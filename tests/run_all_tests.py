# tests/run_all_tests.py
import unittest
import sys
import os

# Get the project root by going up one directory from tests
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Discover and load all test cases from the tests directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(start_dir=os.path.join(project_root, "tests"), pattern="test_*.py")

# Run the test suite
test_runner = unittest.TextTestRunner(verbosity=2)
test_runner.run(test_suite)
