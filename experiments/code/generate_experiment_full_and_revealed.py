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

from gaussian_predictor_likelihood import *
from gaussian_predictor_levelsets import *
from gaussian_trainer import *
from OT_predictor import *
from MVCS import *
from covariances import *


seed_everything(42) 

parser = argparse.ArgumentParser(description="Script avec argument config_name")
parser.add_argument("config_name", type=str, help="Nom de la configuration")


args = parser.parse_args()
config_name = args.config_name

print('config_name:', config_name)

config_path = "../parameters/" + config_name + ".json"
with open(config_path, 'r') as file : 
    parameters = json.load(file)

tab_alpha = [0.1, 0.05]

resuls_alpha = {}

results_coverage = {alpha: {} for alpha in tab_alpha}
results_volume = {alpha: {} for alpha in tab_alpha}

results_coverage_known = {alpha: {} for alpha in tab_alpha}
results_volume_known = {alpha: {} for alpha in tab_alpha}

results_losses = {}

def save_results():
    for alpha in tab_alpha:
        save_path_volume = f"../results/volume/gaussian_{config_name}_alpha_{alpha}.pkl"
        save_path_coverage = f"../results/coverage/gaussian_{config_name}_alpha_{alpha}.pkl"

        with open(save_path_volume, "wb") as f:
            pickle.dump(results_volume[alpha], f)

        with open(save_path_coverage, "wb") as f:
            pickle.dump(results_coverage[alpha], f)

        save_path_volume_known = f"../results/volume_knowned/gaussian_{config_name}_alpha_{alpha}.pkl"
        save_path_coverage_known = f"../results/coverage_knowned/gaussian_{config_name}_alpha_{alpha}.pkl"
        with open(save_path_volume_known, "wb") as f:
            pickle.dump(results_volume_known[alpha], f)
        with open(save_path_coverage_known, "wb") as f:
            pickle.dump(results_coverage_known[alpha], f)

        mean_results = {}
        for key in results_volume[alpha][0].keys():
            values = [results_volume[alpha][exp][key] for exp in results_volume[alpha]]
            values_sorted = sorted(values)
            mean_results[key] = np.mean(values_sorted)

        print(f"[Alpha={alpha}] Mean volumes:", mean_results)
        print("")

        mean_results = {}
        for key in results_volume_known[alpha][0].keys():
            values = [results_volume_known[alpha][exp][key] for exp in results_volume_known[alpha]]
            values_sorted = sorted(values)
            mean_results[key] = np.mean(values_sorted)
        print(f"[Alpha={alpha}] Mean known volumes:", mean_results)

    save_path_losses = f"../results/tab_results/gaussian_{config_name}.pkl"
    with open(save_path_losses, "wb") as f:
            pickle.dump(results_losses, f) 

for experiment in range(parameters["n_experiments"]):
    seed_everything(experiment)
    print(f"Experiment {experiment}/{parameters['n_experiments']}")

    prop_train = parameters["prop_train"]
    prop_calibration = parameters["prop_calibration"]

    
    load_path = "../../data/processed_data/" + parameters["load_name"] + ".npz"
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
    idx_unknowned = np.setdiff1d(np.arange(k), idx_knowned) 

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64


    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    x_stop_tensor = torch.tensor(x_stop, dtype=dtype)
    y_stop_tensor = torch.tensor(y_stop, dtype=dtype)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=dtype)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=dtype)
    x_test_tensor = torch.tensor(x_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)

    
    center_model = Network(d, k, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers).to(dtype)
    if parameters["parameterisation"] == "Cholesky":
        matrix_model = CholeskyMatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
    else:
        matrix_model = MatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size, shuffle=True)
    calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor), batch_size= batch_size, shuffle=True)

    gaussian_predictor = GaussianTrainer(center_model, matrix_model, dtype=dtype)

    center_model.fit_and_plot(trainloader, 
                              stoploader, 
                              num_epochs, 
                              keep_best = True, 
                              lr = lr_center, 
                              verbose=False)

    gaussian_predictor.fit(trainloader, 
                            stoploader, 
                            num_epochs = num_epochs,
                            lr_center_models = 0.0,
                            lr_matrix_models = lr_matrix,
                            use_lr_scheduler = use_lr_scheduler,
                            verbose = 1,
                            stop_on_best = keep_best
                            )
    
    train_losses = gaussian_predictor.tab_train_loss
    stop_losses  = gaussian_predictor.tab_stop_loss
    
    results_losses[experiment] = {"CE_train" : train_losses, 
                                "CE_stop" : stop_losses
                                }
    
    center_model = gaussian_predictor.center_model
    matrix_model = gaussian_predictor.matrix_model
    
    gaussian_level_sets = GaussianPredictorLevelsets(center_model, matrix_model, dtype=dtype)
    gaussian_likelihood = GaussianPredictorLikelihood(center_model, matrix_model, dtype=dtype)

    one_covariance_predictor = CovariancePredictor(gaussian_predictor.center_model)
    one_covariance_predictor.fit(trainloader, stoploader, num_epochs = num_epochs, lr_model = lr_center)
    one_covariance_predictor.fit_cov(trainloader, x_train_tensor, y_train_tensor)

    if len(x_calibration) < 1000:
        ot_predictor = OTPredictor(center_model, int(len(x_calibration) * 0.1) )
    else:
        print("Using all calibration points for OT")
        ot_predictor = OTPredictor(center_model, -1)
        
    for alpha in tab_alpha:        
        #### WITH LEVEL SETS #####
        gaussian_level_sets.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_ls = gaussian_level_sets.get_averaged_volume(x_test_tensor)
        coverage_ls = gaussian_level_sets.get_coverage(x_test_tensor, y_test_tensor)

        gaussian_level_sets.conformalize_with_knowned_idx(x_calibration=x_calibration_tensor, 
                                                    y_calibration=y_calibration_tensor, 
                                                    alpha = alpha, 
                                                    idx_knowned=idx_knowned)

        volume_with_bayes_levelsets = gaussian_level_sets.get_averaged_volume_condition_on_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_with_bayes_levelsets = gaussian_level_sets.get_coverage_condition_on_idx(x_test_tensor, y_test_tensor)

        volume_without_bayes_levelsets = gaussian_level_sets.get_averaged_volume_condition_on_idx_without_bayes(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_without_bayes_levelsets = gaussian_level_sets.get_coverage(x_test_tensor, y_test_tensor)


        ##### WITH LIKELIHOOD #####
        gaussian_likelihood.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_l = gaussian_likelihood.get_averaged_volume(x_test_tensor)
        coverage_l = gaussian_likelihood.get_coverage(x_test_tensor, y_test_tensor)

        gaussian_likelihood.conformalize_with_knowned_idx(x_calibration=x_calibration_tensor, 
                                                    y_calibration=y_calibration_tensor, 
                                                    alpha = alpha, 
                                                    idx_knowned=idx_knowned)


        volume_with_bayes_likelihood = gaussian_likelihood.get_averaged_volume_condition_on_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_with_bayes_likelihood = gaussian_likelihood.get_coverage_condition_on_idx(x_test_tensor, y_test_tensor)

        volume_without_bayes_likelihood = gaussian_likelihood.get_averaged_volume_condition_on_idx_without_bayes(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_without_bayes_likelihood = gaussian_likelihood.get_coverage(x_test_tensor, y_test_tensor)
        

        ##### WITH ONE COVARIANCE #####
        one_covariance_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_one = one_covariance_predictor.get_averaged_volume(x_test_tensor)
        coverage_one = one_covariance_predictor.get_coverage(x_test_tensor, y_test_tensor)
        
        volume_one_covariance_known = one_covariance_predictor.get_averaged_volume_condition_on_idx_without_bayes(x_test_tensor, y_test_tensor[:, idx_knowned], idx_knowned)
        coverage_one_covariance_known = one_covariance_predictor.get_coverage(x_test_tensor, y_test_tensor)


        ##### WITH OT #####
        volume_ot = -1.
        coverage_ot = -1.
        volume_ot_known = -1.
        coverage_ot_known = -1.
        ot_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_ot = ot_predictor.get_averaged_volume(x_test_tensor)
        coverage_ot = ot_predictor.get_coverage(x_test_tensor, y_test_tensor)        

        ot_predictor.conformalize_with_knowned_idx(idx_knowned=idx_knowned)
        volume_ot_known = ot_predictor.get_averaged_volume_with_knowned_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_ot_known = coverage_ot


        #### WITH MVCS #####
        if parameters["parameterisation"] == "Cholesky":
            mvcs_matrix_model = CholeskyMatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
        else:
            mvcs_matrix_model = MatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
        
        MVCS_predictor = MVCSPredictor(center_model, mvcs_matrix_model, dtype=dtype)

        MVCS_predictor.fit(trainloader,
                        stoploader,
                        alpha,
                        num_epochs = num_epochs,             # The total number of epochs
                        num_epochs_mat_only = num_epochs,    # The first 100 epochs are used to train the matrix model only
                        lr_model = 0.0,            # The learning rate for the center model
                        lr_matrix_model = lr_matrix,      # The learning rate for the matrix model
                        verbose = 1,                  # The verbosity level (0 : No verbose; 1: Print the loss 10 times or 2: Print the loss at each epoch)
                        )
        
        MVCS_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_MVCS = MVCS_predictor.get_averaged_volume(x_test=x_test_tensor)
        coverage_MVCS = MVCS_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)

        MVCS_predictor.conformalize_with_knowned_idx(idx_knowned)
        volume_MVCS_knowned = MVCS_predictor.get_averaged_volume_with_knowned_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_MVCS_knowned = coverage_MVCS

        results_volume[alpha][experiment] = {
            "volume_levelsets": volume_ls,
            "volume_likelihood": volume_l,
            "volume_one": volume_one,
            "volume_ot": volume_ot,
            "volume_MVCS": volume_MVCS,
        }

        results_coverage[alpha][experiment] = {
            "coverage_levelset": coverage_ls,
            "coverage_likelihood": coverage_l,
            "coverage_one": coverage_one,
            "coverage_ot": coverage_ot,
            "coverage_MVCS": coverage_MVCS,
        }

        results_volume_known[alpha][experiment] = {
            "volume_with_bayes_levelsets": volume_with_bayes_levelsets,
            "volume_without_bayes_levelsets": volume_without_bayes_levelsets,
            "volume_with_bayes_likelihood": volume_with_bayes_likelihood,
            "volume_without_bayes_likelihood": volume_without_bayes_likelihood,
            "volume_one_covariance_known": volume_one_covariance_known,
            "volume_ot_known": volume_ot_known,
            "volume_MVCS_knowned": volume_MVCS_knowned
        }

        results_coverage_known[alpha][experiment] = {
            "coverage_with_bayes_levelsets": coverage_with_bayes_levelsets,
            "coverage_without_bayes_levelsets": coverage_without_bayes_levelsets,
            "coverage_with_bayes_likelihood": coverage_with_bayes_likelihood,
            "coverage_without_bayes_likelihood": coverage_without_bayes_likelihood,
            "coverage_one_covariance_known": coverage_one_covariance_known,
            "coverage_ot_known": coverage_ot_known,
            "coverage_MVCS_knowned": coverage_MVCS_knowned
        }

        print("")
        print(f"[Alpha={alpha}]")
        print(f"Volume levelset : {volume_ls:.3f}, \nVolume likelihood: {volume_l:.3f}, \nVolume one covariance: {volume_one:.3f}, \nVolume OT: {volume_ot:.3f}, \nVolume MVCS: {volume_MVCS:.3f}")
        print(f"Coverage levelset : {coverage_ls:.3f}, \nCoverage likelihood: {coverage_l:.3f}, \nCoverage one covariance: {coverage_one:.3f}, \nCoverage OT: {coverage_ot:.3f}, \nCoverage MVCS: {coverage_MVCS:.3f}")        
        print("")

        print("Knowned : ")
        print(f"[Alpha={alpha}]")
        print(f"Volume bayes levelsets: {volume_with_bayes_levelsets:.3f}, \nVolume without Bayes levelsets: {volume_without_bayes_levelsets:.3f}, \nVolume with Bayes likelihood: {volume_with_bayes_likelihood:.3f}, \nVolume without Bayes likelihood: {volume_without_bayes_likelihood:.3f}, \nVolume one covariance: {volume_one_covariance_known:.3f}, \nVolume OT: {volume_ot_known:.3f} \n Volume MVCS: {volume_MVCS_knowned:.3f}")
        print(f"Coverage bayes levelsets: {coverage_with_bayes_levelsets:.3f}, \Coverage without Bayes likelihood: {coverage_without_bayes_levelsets:.3f}, \nCoverage with Bayes likelihood: {coverage_with_bayes_likelihood:.3f}, \nCoverage without Bayes likelihood: {coverage_without_bayes_likelihood:.3f}, \nCoverage one covariance: {coverage_one_covariance_known:.3f}, \nCoverage OT: {coverage_ot_known:.3f} \n Coverage MVCS: {coverage_MVCS_knowned:.3f}")
        print("")

    save_results()



