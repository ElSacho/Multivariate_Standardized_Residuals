# Multivariate Conformal Prediction via Conformalized Gaussian Scoring

This repository provides tools for constructing and evaluating conformal prediction sets for multivariate regression, implementing our proposed **Multivariate Conformal Prediction via Conformalized Gaussian Scoring** method. It accompanies our paper:

**Multivariate Conformal Prediction via Conformalized Gaussian Scoring**.

## Installation

Clone this repository and install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Overview

The package is organized as follows:

### **code/**  
Contains the core implementation of Gaussian-Scoring methods:
- `gaussian_predictor_levelsets.py`: Implements the `GaussianPredictorLevelsets` class for learning and evaluating Gaussian-Scoring on complete data.
- `gaussian_predictor_missing_outputs.py`: Implements the `GaussianPredictorMissingValues` class for learning and evaluating Gaussian-Scoring with missing output values.
- `example_of_usage.ipynb`: Jupyter notebook demonstrating how to use the implemented methods.

### **experiments/**  
Contains all resources related to experimental evaluations:
- **code/**: Scripts to reproduce the experiments described in the paper.  
  To run an experiment, open a terminal in this folder and execute:
  ```bash
  python generate_experiment_<EXPERIMENT_NAME>.py <parameter_folder_name>
  ```
  Example:
  ```bash
  python generate_experiment_projection.py taxi01
  ```

- **parameters/**: JSON files specifying hyperparameters for various strategies.
- **figs/**: Plots and figures used in the paper.
- **results/**: Saved outputs from the experiments.

To generate a figure used in the paper, run the corresponding `make_fig_<EXPERIMENT_NAME>.ipynb` notebook.


## Using `GaussianPredictorLevelsets` & `GaussianPredictorMissingValues`

The `GaussianPredictorLevelsets` class requires two models as input:
- A **center model**, which should ideally be pre-trained.
- A **matrix model**, used to fit the covariance matrix. 

Those two models should be writen in PyTorch. The **matrix model** should return a SDP matrix. 

### Basic usage for full output

```python
from code.gaussian_predictor_levelsets import GaussianPredictorLevelsets

# Initialize with pre-trained models
predictor = GaussianPredictorLevelsets(center_model, matrix_model)

# Fit the models to data
predictor.fit(trainloader)

# Conformalize the prediction sets
predictor.conformalize(calibrationloader, alpha=alpha)

# Get volume and coverage
volume = predictor.get_volume(x_test)
coverage = predictor.get_coverage(testloader)
```

### Basic usage for missing outputs

```python
from code.gaussian_predictor_missing_outputs import GaussianPredictorMissingValues

# Initialize with pre-trained models
predictor = GaussianPredictorMissingValues(center_model, matrix_model)

# Fit the models to data
predictor.fit(trainloader)

# Conformalize the prediction sets
predictor.conformalize(calibrationloader, alpha=alpha)

# Get volume and coverage
volume = predictor.get_volume(x_test)
coverage = predictor.get_coverage(testloader)
```

### Basic usage for revealed outputs

```python
from code.gaussian_predictor_levelsets import GaussianPredictorLevelsets

# Initialize with pre-trained models
predictor = GaussianPredictorLevelsets(center_model, matrix_model)

# Fit the models to data
predictor.fit(trainloader)

# Conformalize the prediction sets
predictor.conformalize_with_knowned_idx(x_calibration=x_calibration_tensor, 
                                            y_calibration=y_calibration_tensor, 
                                            alpha = alpha, 
                                            idx_knowned=idx_knowned)

# Get volume and coverage
volume = predictor.get_coverage_condition_on_idx(x_test_tensor, y_test_tensor)
coverage = predictor.get_averaged_volume_condition_on_idx(x_test_tensor, y_test_tensor[:, idx_knowned]).item()
```

### Basic usage for projection of the outputs
```python
from code.gaussian_predictor_levelsets import GaussianPredictorLevelsets

# Initialize with pre-trained models
predictor = GaussianPredictorLevelsets(center_model, matrix_model)

# Fit the models to data
predictor.fit(trainloader)

# Conformalize the prediction sets
predictor.conformalize_linear_projection(
                                            projection_matrix=projection_matrix_tensor,
                                            x_calibration=x_calibration_tensor, 
                                            y_calibration=y_calibration_tensor, 
                                            alpha = alpha
                                            )

# Get volume and coverage
volume = predictor.get_coverage_projection(x_test_tensor, y_test_tensor)
coverage = predictor.get_averaged_volume_projection(x_test_tensor).item()
```

## Citation
If you use this repository for research purposes, please cite our paper:

**Minimum Volume Conformal Sets for Multivariate Regression**.

For any questions, feel free to contact us.