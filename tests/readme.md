
# Test Suite

## Table of Contents
1. [Introduction](#introduction)
2. [Test Suites](#test-suites)
   - [1. Load Data (`load_data`)](#1-load-data-load_data)
   - [2. Data Preprocessing (`data_pre_proc`)](#2-data-preprocessing-data_pre_proc)
   - [3. Model Training (`train_lr`)](#3-model-training-train_lr)
   - [4. Model Evaluation (`evaluate_model`)](#4-model-evaluation-evaluate_model)
3. [Conclusion](#conclusion)

---

## Introduction

The goal of this effort was to develop test suites for four main functions within our ML pipeline to guarantee 
correct functionality and robust error handling. Each test suite contains a collection of test cases designed to:
- Verify core functionality of each function.
- Ensure correct handling of errors and missing files.
- Validate logging and directory creation operations.
- Provide isolation for each test to avoid dependencies on external factors.

---

## Test Suites

### 1. Load Data (`load_data`)
The `load_data` function is responsible for reading data from a specified DVC-tracked CSV file and saving it 
to an output location.

#### Test Cases
- **Functional Test**: Checks that `load_data` reads data correctly and saves it to the output path.
- **Error Handling**: Ensures proper handling of cases where input files are missing or paths are incorrect.

#### Description:
```python
"""
TestLoadDataFunction

This test suite validates the `load_data` function by checking its ability to read data from a specified 
DVC-tracked CSV file and save it to a designated output location. The suite includes functional tests and 
error handling checks.

Test cases:
1. Functional test to ensure data is correctly loaded and saved.
2. Error handling test for missing input files or incorrect paths.

Setup:
- Creates sample configuration and CSV files to simulate the data load process.

Teardown:
- Removes files created during the setup and test process to ensure test isolation.
"""
```

---

### 2. Data Preprocessing (`data_pre_proc`)
The `data_pre_proc` function performs essential preprocessing on input data, including scaling, encoding, 
and PCA transformation.

#### Test Cases
- **Functional Test**: Ensures data is correctly processed and split into training and testing sets.
- **Data Integrity Check**: Confirms that processed files maintain the correct format, structure, and number of features.
- **Empty Data File Handling**: Verifies that the function raises appropriate errors when data files are empty or missing.

#### Description:
```python
"""
TestDataPreProcFunction

This test suite covers the `data_pre_proc` function to ensure data preprocessing operations, including 
scaling, encoding, and PCA transformations, are functioning correctly.

Test cases:
1. Functional test to validate that data preprocessing and splitting occur correctly.
2. Data integrity checks to ensure processed data files maintain correct structure and format.
3. Error handling for empty or missing input files.

Setup:
- Generates sample data and configuration files to simulate the preprocessing workflow.

Teardown:
- Cleans up all generated files and directories to maintain a clean test environment.
"""
```

---

### 3. Model Training (`train_lr`)
The `train_lr` function is responsible for training a logistic regression model, saving it to disk, and logging 
parameters and artifacts to MLflow.

#### Test Cases
- **Functional Test**: Validates end-to-end model training, saving, and MLflow logging.
- **Invalid Parameter Handling**: Ensures that invalid parameters in the configuration raise an appropriate error.
- **Missing Input Files**: Verifies that missing input files result in a `FileNotFoundError`.
- **Output Directory Creation**: Confirms the function creates output directories if they do not exist.

#### Description:
```python
"""
TestTrainLRFunction

This test suite verifies the functionality of the `train_lr` function, focusing on:

1. Full functional test, ensuring model training, saving, and MLflow logging work as expected.
2. Validation of error handling for invalid model parameters.
3. Verification of appropriate error handling when input files are missing.
4. Ensuring the output directory is created if it doesn't exist.

Setup:
- Creates sample input data files and a configuration file.
- Configures an MLflow experiment specifically for testing.

Teardown:
- Removes all files and directories created during setup and test runs.
"""
```

---

### 4. Model Evaluation (`evaluate_model`)
The `evaluate_model` function loads a trained model, generates predictions, calculates metrics, and saves 
evaluation results.

#### Test Cases
- **Functional Test**: Verifies correct evaluation metrics are calculated and saved.
- **Missing Model File**: Ensures a `FileNotFoundError` is raised when the model file is missing.
- **Empty Data File Handling**: Checks that empty input files raise an appropriate error.
- **Metrics Verification**: Confirms the accuracy and confusion matrix metrics are within expected ranges.

#### Description:
```python
"""
TestEvaluateModelFunction

This suite tests the `evaluate_model` function by verifying that model predictions, accuracy metrics, and 
error handling work as expected. 

Test cases:
1. Functional test to ensure evaluation metrics are calculated and saved correctly.
2. Missing model file test to check if the function raises a `FileNotFoundError`.
3. Empty data file handling to verify errors for empty input files.
4. Metrics verification to confirm accuracy and confusion matrices meet expected values.

Setup:
- Creates sample data, configuration, and a logistic regression model file.

Teardown:
- Cleans up all files and directories created during the test runs.
"""
```

---

## Conclusion

The test suite provides a robust validation framework for the critical functions in the model training pipeline, including data loading, preprocessing, model training, and evaluation. By implementing functional and error-handling tests, this suite helps ensure each function operates reliably and manages errors appropriately, contributing to a stable and predictable pipeline.

---
