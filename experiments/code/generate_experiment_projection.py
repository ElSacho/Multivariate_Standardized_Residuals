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

from covmetrics import ERT, WSC

from approaches.gaussian_predictor_likelihood import *
from approaches.gaussian_predictor_levelsets import *
from approaches.gaussian_trainer import *
from approaches.OT_predictor import *
from approaches.MVCS import *
from approaches.covariances import *

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("config_name", type=str, help="Nom de la configuration")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    config_name = args.config_name
    experiment = args.experiment_index  # Use this as seed

    config_path = "../parameters/" + config_name + ".json"
    with open(config_path, 'r') as file : 
        parameters = json.load(file)

    tab_alpha = [0.1, 0.05]


    seed_everything(experiment)
    
    print(f"Experiment {experiment}/10")

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

    
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    projection_matrix = torch.randn((2, output_dim), dtype=dtype)


    batch_size = parameters["batch_size"]


    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64

    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    x_stop_tensor = torch.tensor(x_stop, dtype=dtype)
    y_stop_tensor = torch.tensor(y_stop, dtype=dtype)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=dtype)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=dtype)
    x_test_tensor = torch.tensor(x_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size, shuffle=True)
    

    gaussian_trainer = GaussianTrainer(input_dim, 
                                output_dim,
                                hidden_dim = parameters.get("hidden_dim", 256),
                                num_layers = parameters.get("n_hidden_layers", 3)
                                )

    gaussian_trainer.fit(trainloader, 
                        stoploader,
                        num_epochs=parameters.get("num_epochs", 300),
                        lr=parameters.get("lr", 1e-3),
                        verbose = 2
                        )
    
    center_model = gaussian_trainer.center_model
    matrix_model = gaussian_trainer.matrix_model

    gaussian_level_sets = GaussianPredictorLevelsets(center_model, matrix_model, dtype=dtype)
    gaussian_likelihood = GaussianPredictorLikelihood(center_model, matrix_model, dtype=dtype)

    

    if len(x_calibration) < 1000:
        ot_predictor = OTPredictor(center_model, int(len(x_calibration) * 0.1) )
    else:
        print("Using all calibration points for OT")
        ot_predictor = OTPredictor(center_model, -1)
        
    for alpha in tab_alpha:   
        #### WITH LEVEL SETS #####
        gaussian_level_sets.conformalize_linear_projection(projection_matrix, x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_ls = gaussian_level_sets.get_averaged_volume_projection(x_test_tensor)
        coverage_ls = gaussian_level_sets.get_coverage_projection(x_test_tensor, y_test_tensor)
        z_ls = gaussian_level_sets.get_cover_projection(x_test_tensor, y_test_tensor)
        ERT_ls = ERT().evaluate(x_test_tensor, z_ls, alpha)
        WSC_ls = WSC().evaluate(x_test_tensor, z_ls, alpha)
        print("ERT_ls: ", ERT_ls)


        ##### WITH LIKELIHOOD #####
        gaussian_likelihood.conformalize_linear_projection(projection_matrix, x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_l = gaussian_likelihood.get_averaged_volume_projection(x_test_tensor)
        coverage_l = gaussian_likelihood.get_coverage_projection(x_test_tensor, y_test_tensor)
        z_l = gaussian_likelihood.get_cover_projection(x_test_tensor, y_test_tensor)
        ERT_l = ERT().evaluate(x_test_tensor, z_l, alpha)
        WSC_l = WSC().evaluate(x_test_tensor, z_l, alpha)
        print("ERT_l: ", ERT_l)

        ##### WITH ONE COVARIANCE #####
        class CenterProjectedModel:
            def __init__(self, center_model, projection_matrix):
                self.center_model = center_model
                self.projection_matrix = projection_matrix

            def __call__(self, x):
                full_f_x = self.center_model(x)
                return torch.einsum('rk,nk->nr', projection_matrix, full_f_x)
            
        center_projected_model = CenterProjectedModel(center_model, projection_matrix)

        y_train_proj_tensor = torch.einsum('rk,nk->nr', projection_matrix, y_train_tensor)
        y_calibration_proj_tensor = torch.einsum('rk,nk->nr', projection_matrix, y_calibration_tensor)
        y_test_proj_tensor = torch.einsum('rk,nk->nr', projection_matrix, y_test_tensor)
        one_covariance_predictor = CovariancePredictor(center_projected_model)
        one_covariance_predictor.fit_cov_projected(trainloader, x_train_tensor, y_train_proj_tensor, projection_matrix=projection_matrix)
        one_covariance_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_proj_tensor, alpha = alpha)
        volume_one = one_covariance_predictor.get_averaged_volume(x_test_tensor)
        coverage_one = one_covariance_predictor.get_coverage(x_test_tensor, y_test_proj_tensor)
        z_one = torch.tensor(one_covariance_predictor.get_cover(x_test_tensor, y_test_proj_tensor), dtype=torch.float32)
        ERT_one = ERT().evaluate(x_test_tensor, z_one, alpha)  
        WSC_one = WSC().evaluate(x_test_tensor, z_one, alpha)  
        print("ERT_one: ", ERT_one)

        ##### WITH OT #####
        volume_ot = -1.
        coverage_ot = -1.
        ERT_ot = -1.
        WSC_ot = -1.
        if parameters["load_name"] != "energy":
            ot_predictor.conformalize_linear_projection(projection_matrix=projection_matrix, x_calibration=x_calibration_tensor,  y_calibration=y_calibration_tensor,  alpha = alpha)
            volume_ot = ot_predictor.get_averaged_volume_projection(x_test_tensor)
            coverage_ot = ot_predictor.get_coverage_projection(x_test_tensor, y_test_tensor)
            z_ot = torch.tensor(ot_predictor.get_cover_projection(x_test_tensor, y_test_tensor), dtype=torch.float32)
            ERT_ot = ERT().evaluate(x_test_tensor, z_ot, alpha)
            WSC_ot = WSC().evaluate(x_test_tensor, z_ot, alpha)
            print("ERT_ot: ", ERT_ot)

        file_path = Path("../results/results_projection.csv")

        new_row = {
            "dataset": config_name,
            "alpha": alpha,
            "experiment": experiment,
            "volume_levelset": np.array(volume_ls),
            "volume_likelihood": np.array(volume_l),
            "volume_one": np.array(volume_one),
            "volume_ot": np.array(volume_ot),
            "coverage_levelset": coverage_ls,
            "coverage_likelihood": coverage_l,
            "coverage_one": coverage_one,
            "coverage_ot": coverage_ot,
            "ERT_levelset": ERT_ls,
            "ERT_likelihood": ERT_l,
            "ERT_one": ERT_one,
            "ERT_ot": ERT_ot,
            "WSC_levelset": WSC_ls,
            "WSC_likelihood": WSC_l,
            "WSC_one": WSC_one,
            "WSC_ot": WSC_ot
        }

        if file_path.exists():
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file_path, index=False)

        print("")
        print(f"[Alpha={alpha}]")
        print(f"Volume levelset : {volume_ls:.3f}, \nVolume likelihood: {volume_l:.3f}, \nVolume one covariance: {volume_one:.3f}, \nVolume OT: {volume_ot:.3f}")
        print(f"Coverage levelset : {coverage_ls:.3f}, \nCoverage likelihood: {coverage_l:.3f}, \nCoverage one covariance: {coverage_one:.3f}, \nCoverage OT: {coverage_ot:.3f}")
        print(f"ERT levelset : {ERT_ls:.3f}, \n ERT likelihood: {ERT_l:.3f}, \n ERT one covariance: {ERT_one:.3f}, \n ERT OT: {ERT_ot:.3f}")
        print("")
    

if __name__=="__main__":
    main()