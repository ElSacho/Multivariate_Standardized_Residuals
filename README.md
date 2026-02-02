# Multivariate Standardized Residuals for Conformal Prediction

This repository provides tools for constructing and evaluating conformal prediction sets for multivariate regression, implementing our proposed **Multivariate Standardized Residuals** method. It accompanies our paper:

**Multivariate Standardized Residuals for Conformal Prediction**.

## Installation

Clone this repository and install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Overview

The package is organized as follows:

### **code/**  
Contains the core implementation of standardized residuals methods:
- `standardized_residuals.py`: Implements the `StandardizedResiduals` class for learning and evaluating Mahalanobis on complete or incomplete data.
- `example_of_usage_with_pretrained_model.ipynb`: Jupyter notebook demonstrating how to use the implemented methods with a pretrained model to predict f(x)
- `example_of_usage_joint_distribution.ipynb`: Jupyter notebook demonstrating how to use the implemented methods by jointly learning mean and covariance.

### **experiments/**  
Contains all resources related to experimental evaluations:
- **code/**: Scripts to reproduce the experiments described in the paper.  
  To run an experiment, open a terminal in this folder and execute:
  ```bash
  python generate_experiment_<EXPERIMENT_NAME>.py <parameter_folder_name> <parameter_seed_number> 
  ```
  Example:
  ```bash
  python generate_experiment_projection.py taxi 1
  ```

- **parameters/**: JSON files specifying hyperparameters for various strategies.
- **figs/**: Plots and figures used in the paper.
- **results/**: Saved outputs from the experiments.

To generate a figure used in the paper, run the corresponding `make_fig_<EXPERIMENT_NAME>.ipynb` notebook.


## Using `StandardizedResiduals`

The `StandardizedResiduals` class can either be used on top of an existing pointwise predictor f(x), or by jointly learning f(x) and the covariance matrix.

The covariance matrix can be learned by a diagonal + low rank approximation when the dimension of the output is large by initializing with the value "mode="low_rank". Default is "full_cholesky".

### Basic usage for existing pointwise predictor

Add a center_model parameter to the StandardizedResiduals class. The other parameters are the input_dim (feature dimension), output_dim, hidden_dim (for the hidden MLP), num_layers (for the hidden MLP aswell).

```python
StandardizedResiduals(input_dim, 
                            output_dim,
                            hidden_dim = hidden_dim,
                            num_layers = num_layers,
                            center_model = center_model,
                            mode="full_cholesky" # or low_rank
                            )
```
The center_model should work with a __call__ function and be compatible with batches, such that 

```python
center_pred = self.center_model(bx)
```
works. Its about can either be numpy or torch tensor.

### Basic usage for learning the pointwise predictor jointly

Similar to the previous section, except that no center_model is provided (center_model=None)

```python
from code.standardized_residuals import StandardizedResiduals

standardized_residuals = StandardizedResiduals(input_dim, 
                            output_dim,
                            hidden_dim = hidden_dim,
                            num_layers = num_layers,
                            mode="full_cholesky" # or low_rank
                            )

standardized_residuals.fit(trainloader, 
                    stoploader,
                    num_epochs=num_epochs,
                    lr=lr,
                    verbose = 2
                    )

# Conformalize the prediction sets
standardized_residuals.conformalize(x=x_calibration_tensor, y=y_calibration_tensor, alpha = alpha)

# Get volume and coverage
volume = predictor.get_average_volume(x_test)
coverage = predictor.get_coverage(testloader)
```

Once this model has been learned (it automatically detects missing outputs), the extensions can be performed as followed:

#### Missing outputs

```python

standardized_residuals.conformalize_missing(x = x_calibration_tensor,
                                            y = y_calibration_nan_tensor, 
                                            alpha = alpha
                                            )

coverage_with_nan = standardized_residuals.get_coverage_missing(x_test_tensor, y_test_nan_tensor)
```

#### Basic usage for revealed outputs

```python
idx_knowned = np.array([0]) # Change the the idx number that are revealed

standardized_residuals.conformalize_revealed(idx_revealed = idx_knowned,
                                            x = x_calibration_tensor, 
                                            y = y_calibration_tensor, 
                                            alpha = alpha
                                            )

coverage = standardized_residuals.get_coverage_revealed(x_test_tensor, y_test_tensor)
volumes  = standardized_residuals.get_average_volume_revealed(x_test_tensor, y_test_tensor[:idx_knowned])

print("Coverage:", coverage)
print("Average Volume:", volumes)
```

#### Basic usage for projection of the outputs

```python
projection_matrix_tensor =  torch.randn((2, output_dim), dtype=dtype) # Change to your projection matrix

standardized_residuals.conformalize_projection(
                                            projection_matrix = projection_matrix_tensor,
                                            x = x_calibration_tensor, 
                                            y = y_calibration_tensor, 
                                            alpha = alpha
                                            )

coverage = standardized_residuals.get_coverage_projection(x_test_tensor, y_test_tensor)
volumes  = standardized_residuals.get_average_volume_projection(x_test_tensor)

print("Coverage:", coverage)
print("Average Volume:", volumes)
```

## Citation
If you use this repository for research purposes, please cite our paper:

**Multivariate Standardized Residuals for Conformal Prediction**.

For any questions, feel free to contact us.