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

from approaches.gaussian_predictor_levelsets import *
from approaches.gaussian_trainer import *

from covmetrics import *

import pandas as pd
from pathlib import Path


def main():
    tab_alpha = [0.1, 0.05]

    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("config_name", type=str, help="Nom de la configuration")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    config_name = args.config_name
    experiment = args.experiment_index  # Use this as seed

    print('config_name:', config_name)

    config_path = "../parameters/" + config_name + ".json"
    with open(config_path, 'r') as file : 
        parameters = json.load(file)


    seed_everything(experiment)
    print(f"Experiment {experiment}/10")

    prop_train = parameters["prop_train"]
    prop_calibration = parameters["prop_calibration"]


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

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_calibration = x_calibration.shape[0]
    n_stop = x_stop.shape[0]

    batch_size = parameters["batch_size"]

    idx_knowned = np.array(parameters["idx_knowned"])
    idx_unknowned = np.setdiff1d(np.arange(output_dim), idx_knowned) 

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64


    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    x_stop_tensor = torch.tensor(x_stop, dtype=dtype)
    y_stop_tensor = torch.tensor(y_stop, dtype=dtype)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=dtype)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=dtype)
    x_test_tensor = torch.tensor(x_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)


    y_train_nan = add_nan(y_train, min_nan=1, max_nan=output_dim-1)
    y_train_nan_tensor = torch.tensor(y_train_nan, dtype=dtype)

    y_stop_nan = add_nan(y_stop, min_nan=1, max_nan=output_dim-1)
    y_stop_nan_tensor = torch.tensor(y_stop_nan, dtype=dtype)

    y_calibration_nan = add_nan(y_calibration, min_nan=1, max_nan=output_dim-1)
    y_calibration_nan_tensor = torch.tensor(y_calibration_nan, dtype=dtype)

    y_test_nan = add_nan(y_test, min_nan=1, max_nan=output_dim-1)
    y_test_nan_tensor = torch.tensor(y_test_nan, dtype=dtype)


    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size, shuffle=True)
    calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_nan_tensor), batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_nan_tensor), batch_size= batch_size, shuffle=True)

    gaussian_trainer = GaussianTrainer(input_dim, 
                                    output_dim,
                                    hidden_dim = parameters.get("hidden_dim", 256),
                                    num_layers = parameters.get("n_hidden_layers", 3)
                                    )

    gaussian_trainer.fit(trainloader, 
                        stoploader,
                        num_epochs=parameters.get("num_epochs", 300),
                        lr=parameters.get("lr", 1e-3),
                        verbose = 2,
                        NaN_Values = True
                        )


    center_model = gaussian_trainer.center_model
    matrix_model = gaussian_trainer.matrix_model


    for alpha in tab_alpha:

        gaussian_predictor_missing_outputs = GaussianPredictorMissingValues(center_model, matrix_model, dtype=dtype)
        gaussian_predictor_missing_outputs.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_nan_tensor, alpha = alpha)

        coverage_with_nan    = gaussian_predictor_missing_outputs.get_coverage(x_test_tensor, y_test_nan_tensor)
        coverage_full_vector = gaussian_predictor_missing_outputs.get_coverage(x_test_tensor, y_test_tensor)

        print("Coverage with NaN:", coverage_with_nan)
        print("Coverage full vector:", coverage_full_vector)

        

        file_path = Path("../results/results_missing.csv")

        new_row = {
            "dataset": config_name,
            "alpha": alpha,
            "experiment": experiment,
            "alpha": alpha,
            "coverage_nan": coverage_with_nan,
            "coverage_full": coverage_full_vector,
        }

        if file_path.exists():
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file_path, index=False)

if __name__=="__main__":
    main()