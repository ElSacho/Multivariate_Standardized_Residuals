import json
import pickle
import argparse


import numpy as np
import matplotlib.pyplot as plt

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from network import *

from plot_data import *
from data_loading import *

import json

from missing_value_pred_lambda import *
from missing_value_pred_sigma import *

seed_everything(42)

parser = argparse.ArgumentParser(description="Script avec argument config_name")
parser.add_argument("config_name", type=str, help="Nom de la configuration")


args = parser.parse_args()
config_name = args.config_name

config_path = "../parameters/" + config_name + ".json"
with open(config_path, 'r') as file : 
    parameters = json.load(file)

tab_alpha = [0.1, 0.05]

resuls_alpha = {}
results_coverage = {alpha: {} for alpha in tab_alpha}
results_volume = {alpha: {} for alpha in tab_alpha}
results_losses = {}

def save_results():
    for alpha in tab_alpha:
        save_path_volume = f"../results/volume_missing_full/full_{config_name}_alpha_{alpha}.pkl"
        save_path_coverage = f"../results/coverage_missing_full/full_{config_name}_alpha_{alpha}.pkl"

        with open(save_path_volume, "wb") as f:
            pickle.dump(results_volume[alpha], f)

        with open(save_path_coverage, "wb") as f:
            pickle.dump(results_coverage[alpha], f)

        mean_results = {}
        for key in results_volume[alpha][0].keys():
            values = [results_volume[alpha][exp][key] for exp in results_volume[alpha]]
            values_sorted = sorted(values)
            mean_results[key] = np.mean(values_sorted)

        print(f"[Alpha={alpha}] Mean volumes:", mean_results)

    save_path_losses = f"../results/tab_results_missing_full/full_{config_name}.pkl"
    with open(save_path_losses, "wb") as f:
            pickle.dump(results_losses, f) 


for experiment in range(parameters["n_experiments"]):
    seed_everything(experiment)

    print(f"Experiment {experiment}/{parameters['n_experiments']}")

    prop_train = parameters["prop_train"]
    prop_calibration = parameters["prop_calibration"]

    # Load Data
    load_path = "../../data/processed_data_3Dmin/" + parameters["load_name"] + ".npz"
    X, Y = load_data(load_path)


    normalize = parameters["normalize"]
    splits = [parameters["prop_train"], parameters["prop_stop"], parameters["prop_calibration"], parameters["prop_test"]]

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64

    subsets = split_and_preprocess(X, Y, splits=splits, normalize=normalize)

    x_train, y_train, x_calibration, y_calibration, x_test, y_test, x_stop, y_stop = subsets["X_train"], subsets["Y_train"], subsets["X_calibration"], subsets["Y_calibration"], subsets["X_test"], subsets["Y_test"], subsets["X_stop"], subsets["Y_stop"]

    print("X_train shape:", x_train.shape, "Y_train shape:", y_train.shape)
    print("X_cal shape:", x_calibration.shape, "Y_cal shape:", y_calibration.shape)
    print("X_test shape:", x_test.shape, "Y_test shape:", y_test.shape)
    print("X_stop shape:", x_stop.shape, "Y_stop shape:", y_stop.shape)

    
    d = x_train.shape[1]
    k = y_train.shape[1]

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_calibration = x_calibration.shape[0]
    n_stop = x_stop.shape[0]

    hidden_dim = parameters["hidden_dim"]
    hidden_dim_matrix = parameters["hidden_dim_matrix"]
    n_hidden_layers = parameters["n_hidden_layers"]
    n_hidden_layers_matrix = parameters["n_hidden_layers_matrix"]


    num_epochs = parameters["num_epochs"]

    lr_center = parameters["lr_center"]   
    lr_matrix = parameters["lr_matrix"]

    batch_size = parameters["batch_size"]

    use_lr_scheduler = parameters["use_lr_scheduler"]
    keep_best = parameters["keep_best"] 

    idx_knowned = np.array(parameters["idx_knowned"])

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64


    min_nan = 1
    max_nan = k-1

    y_train_nan = add_nan(y_train, min_nan=min_nan, max_nan=max_nan)
    y_stop_nan = add_nan(y_stop, min_nan=min_nan, max_nan=max_nan)
    y_calibration_nan = add_nan(y_calibration, min_nan=min_nan, max_nan=max_nan)
    y_test_nan = add_nan(y_test, min_nan=min_nan, max_nan=max_nan)


    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_no_nan_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_nan, dtype=torch.float32)
    x_stop_tensor = torch.tensor(x_stop, dtype=torch.float32)
    y_stop_no_nan_tensor = torch.tensor(y_stop, dtype=torch.float32)
    y_stop_tensor = torch.tensor(y_stop_nan, dtype=torch.float32)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float32)
    y_calibration_tensor = torch.tensor(y_calibration_nan, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_nan, dtype=torch.float32)
    y_test_tensor_without_nan = torch.tensor(y_test, dtype=torch.float32)


    
    center_model = NetworkWithMissingOutputs(d, k, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers).to(dtype)
    if parameters["parameterisation"] == "Cholesky":
        matrix_model = CholeskyMatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
    else:
        matrix_model = MatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_no_nan_tensor), batch_size= batch_size, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_no_nan_tensor), batch_size= batch_size, shuffle=True)
    calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor), batch_size= batch_size, shuffle=True)

    center_model.fit_and_plot(trainloader, 
                                stoploader, 
                                num_epochs, 
                                keep_best = True, 
                                lr = lr_center, 
                                verbose=False)     

    gaussian_predictor_lambda = GaussianPredictorMissingValuesLambda(center_model, matrix_model, dtype=dtype)

    gaussian_predictor_lambda.fit(trainloader, 
                            stoploader, 
                            num_epochs = num_epochs,
                            lr_center_models = 0.0,
                            lr_matrix_models = lr_matrix,
                            use_lr_scheduler = use_lr_scheduler,
                            verbose = 1,
                            stop_on_best = keep_best
                            )

    train_losses_lambda = gaussian_predictor_lambda.tab_train_loss
    stop_losses_lambda  = gaussian_predictor_lambda.tab_stop_loss

    results_losses[experiment] = {"CE_train_lambda" : train_losses_lambda, 
                                "CE_stop_lambda" : stop_losses_lambda,
                                }
        
    for alpha in tab_alpha:
    
        gaussian_predictor_lambda.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        coverage_with_nan_lambda = gaussian_predictor_lambda.get_coverage(x_test_tensor, y_test_tensor)
        coverage_full_vector_lambda = gaussian_predictor_lambda.get_coverage(x_test_tensor, y_test_tensor_without_nan)
        volume_lambda = gaussian_predictor_lambda.get_averaged_volume(x_test_tensor)

        results_volume[alpha][experiment] = {
            "volume_lambda": volume_lambda
        }

        results_coverage[alpha][experiment] = {
            "coverage_lambda": coverage_with_nan_lambda,
            "coverage_full_vector_lambda": coverage_full_vector_lambda,
        }

        print(f"[Alpha={alpha}] - \nCoverage with NaN (Lambda): {coverage_with_nan_lambda}, \nCoverage Full Vector (Lambda): {coverage_full_vector_lambda}, \nVolume (Lambda): {volume_lambda}")
        
    save_results()
    print(f"Results saved for experiment with seed {experiment}")

