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
import pandas as pd
from pathlib import Path

import sys
import os

# --- BLOC DE CONFIGURATION DU PATH ---
# 1. On récupère le chemin absolu du dossier où se trouve ce script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On construit le chemin vers le dossier qui CONTIENT 'moc' (c'est-à-dire 'multi')
# Structure attendue : current_dir/approaches/multi/moc
path_to_add = os.path.join(current_script_dir, "approaches", "multi")

# 3. DEBUG : On vérifie si ce dossier existe vraiment
if not os.path.exists(path_to_add):
    print(f"ERREUR CRITIQUE : Le dossier n'existe pas : {path_to_add}")
    print(f"Vérifie l'arborescence. Dossier actuel : {os.getcwd()}")
else:
    print(f"Succès : Ajout de {path_to_add} au Python Path")
    # 4. On l'insère en PREMIER (index 0) pour qu'il soit prioritaire
    sys.path.insert(0, path_to_add)
# -------------------------------------

# Maintenant, l'import
try:
    from moc.models.trainers.lightning_trainer import get_lightning_trainer
except ImportError as e:
    print("\n--- DIAGNOSTIC D'ERREUR ---")
    print(f"L'import a échoué : {e}")
    print("Vérifie que :")
    print("1. Le dossier 'approaches/multi/moc' existe.")
    print("2. Le dossier 'approaches/multi/moc' contient un fichier '__init__.py'.")
    print("3. Le dossier 'approaches/multi/moc/models' contient un fichier '__init__.py'.")
    sys.exit(1)


from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import C_HDR

import time

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("config_name", type=str, help="Nom de la configuration")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    config_name = args.config_name
    experiment_index = args.experiment_index  # Use this as seed

    seed_everything(experiment_index)

    print('config_name:', config_name)

    config_path = "../parameters/" + config_name + ".json"
    with open(config_path, 'r') as file : 
        parameters = json.load(file)

    tab_alpha = [0.1, 0.05]

    
    print(f"Experiment {experiment_index}/10")
    
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

    
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    hidden_dim_matrix = parameters["hidden_dim_matrix"]
    n_hidden_layers_matrix = parameters["n_hidden_layers_matrix"]

    num_epochs = parameters["num_epochs"]
    lr_matrix = parameters["lr_matrix"]
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

    one_covariance_predictor = CovariancePredictor(center_model)
    one_covariance_predictor.fit_cov(trainloader, x_train_tensor, y_train_tensor)

    if len(x_calibration) < 1000:
        ot_predictor = OTPredictor(center_model, int(len(x_calibration) * 0.1) )
    else:
        print("Using all calibration points for OT")
        ot_predictor = OTPredictor(center_model, -1)

    model = MQF2LightningModule(input_dim, output_dim)
    trainer = get_lightning_trainer( max_epochs=parameters.get("max_epochs_multi", 100) )
    trainer.fit(model, train_dataloaders=trainloader)
        
    for alpha in tab_alpha:  
        x_and_y_r_test_tensor = torch.cat([x_test_tensor, y_test_tensor[:, idx_knowned]], dim=1)
        # y_h_test_tensor = y_test_tensor[:, idx_unknowned]
        #### WITH LEVEL SETS #####
        start = time.time()
        gaussian_level_sets.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        end = time.time()
        time_ls = end - start
        volume_ls = gaussian_level_sets.get_averaged_volume(x_test_tensor)
        coverage_ls = gaussian_level_sets.get_coverage(x_test_tensor, y_test_tensor)
        z_test_ls = gaussian_level_sets.get_cover(x_test_tensor, y_test_tensor)
        # ERT_ls = ERT(LGBMClassifier, n_estimators=10_000, learning_rate=0.02, subsample=0.75, subsample_freq=1, num_leaves=50,
        #                    random_state=0, early_stopping_round=300, min_child_samples=40, min_child_weight=1e-7, verbosity=-1).evaluate(x_test_tensor, z_test_ls, alpha)            
        ERT_ls = ERT().evaluate(x_test_tensor, z_test_ls, alpha)
        WSC_ls = WSC().evaluate(x_test_tensor, z_test_ls, alpha)
        print('ERT_ls: ', ERT_ls)

        gaussian_level_sets.k = output_dim
        start = time.time()
        gaussian_level_sets.conformalize_with_knowned_idx(x_calibration=x_calibration_tensor, 
                                                    y_calibration=y_calibration_tensor, 
                                                    alpha = alpha, 
                                                    idx_knowned=idx_knowned)
        end = time.time()
        time_with_bayes_ls = end - start

        volume_with_bayes_levelsets = gaussian_level_sets.get_averaged_volume_condition_on_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_with_bayes_levelsets = gaussian_level_sets.get_coverage_condition_on_idx(x_test_tensor, y_test_tensor)
        z_with_bayes_levelsets = gaussian_level_sets.get_cover_condition_on_idx(x_test_tensor, y_test_tensor)
        ERT_with_bayes_ls = ERT().evaluate(x_and_y_r_test_tensor, z_with_bayes_levelsets, alpha)
        WSC_with_bayes_ls = WSC().evaluate(x_and_y_r_test_tensor, z_with_bayes_levelsets, alpha)
        print('ERT_with_bayes_ls: ', ERT_with_bayes_ls)

        volume_without_bayes_levelsets = gaussian_level_sets.get_averaged_volume_condition_on_idx_without_bayes(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_without_bayes_levelsets = gaussian_level_sets.get_coverage(x_test_tensor, y_test_tensor)
        ERT_without_bayes_ls = ERT().evaluate(x_and_y_r_test_tensor, z_test_ls, alpha)
        WSC_without_bayes_ls = WSC().evaluate(x_and_y_r_test_tensor, z_test_ls, alpha)
        

        ##### WITH ONE COVARIANCE #####
        start = time.time()
        one_covariance_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        end = time.time()
        time_one = end - start
        time_one_known = time_one
        volume_one = one_covariance_predictor.get_averaged_volume(x_test_tensor)
        coverage_one = one_covariance_predictor.get_coverage(x_test_tensor, y_test_tensor)
        z_test_one = one_covariance_predictor.get_cover(x_test_tensor, y_test_tensor)
        ERT_one = ERT().evaluate(x_test_tensor, z_test_one, alpha)
        WSC_one = WSC().evaluate(x_test_tensor, z_test_one, alpha)
        print('ERT_one: ', ERT_one)
        
        volume_one_covariance_known = one_covariance_predictor.get_averaged_volume_condition_on_idx_without_bayes(x_test_tensor, y_test_tensor[:, idx_knowned], idx_knowned)
        coverage_one_covariance_known = one_covariance_predictor.get_coverage(x_test_tensor, y_test_tensor)
        ERT_one_known = ERT().evaluate(x_and_y_r_test_tensor, z_test_one, alpha)
        WSC_one_known = WSC().evaluate(x_and_y_r_test_tensor, z_test_one, alpha)

        ##### WITH OT #####
        volume_ot = -1.
        coverage_ot = -1.
        volume_ot_known = -1.
        coverage_ot_known = -1.
        ERT_ot = -1.
        WSC_ot = -1.
        ERT_ot_known = ERT_ot
        WSC_ot_known = WSC_ot
        if parameters["load_name"] != "energy":
            start = time.time()
            ot_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
            end = time.time()
            time_ot = end - start
            volume_ot = ot_predictor.get_averaged_volume(x_test_tensor)
            coverage_ot = ot_predictor.get_coverage(x_test_tensor, y_test_tensor)
            z_test_ot = torch.tensor(ot_predictor.get_cover(x_test_tensor, y_test_tensor), dtype=torch.float32)
            ERT_ot = ERT().evaluate(x_test_tensor, z_test_ot, alpha)
            WSC_ot = WSC().evaluate(x_test_tensor, z_test_ot, alpha)
            print('ERT_ot: ', ERT_ot)

            start = time.time()
            ot_predictor.conformalize_with_knowned_idx(idx_knowned=idx_knowned)
            end = time.time()
            time_ot_known = end - start
            volume_ot_known = ot_predictor.get_averaged_volume_with_knowned_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
            coverage_ot_known = coverage_ot
            ERT_ot_known = ERT().evaluate(x_and_y_r_test_tensor, z_test_ot, alpha)
            WSC_ot_known = WSC().evaluate(x_and_y_r_test_tensor, z_test_ot, alpha)

        #### WITH MVCS #####
        mvcs_matrix_model = PSDMatrixPredictor(input_dim, output_dim, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix).to(dtype)
        
        MVCS_predictor = MVCSPredictor(center_model, mvcs_matrix_model, dtype=dtype)

        MVCS_predictor.fit(trainloader,
                        stoploader,
                        alpha,
                        num_epochs = num_epochs,             # The total number of epochs
                        num_epochs_mat_only = num_epochs,    # The first 100 epochs are used to train the matrix model only
                        lr_model = 0.0,                      # The learning rate for the center model
                        lr_matrix_model = lr_matrix,         # The learning rate for the matrix model
                        verbose = 1,                         # The verbosity level (0 : No verbose; 1: Print the loss 10 times or 2: Print the loss at each epoch)
                        )
        
        start = time.time()
        MVCS_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        end = time.time()
        time_MVCS = end - start
        volume_MVCS = MVCS_predictor.get_averaged_volume(x_test=x_test_tensor)
        coverage_MVCS = MVCS_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)
        z_test_MVCS = MVCS_predictor.get_cover(x_test_tensor, y_test_tensor)
        ERT_mvcs = ERT().evaluate(x_test_tensor, z_test_MVCS, alpha)
        WSC_mvcs = WSC().evaluate(x_test_tensor, z_test_MVCS, alpha)
        print('ERT_mvcs: ', ERT_mvcs)

        start = time.time()
        MVCS_predictor.conformalize_with_knowned_idx(idx_knowned)
        end = time.time()
        time_MVCS_knowned = end - start
        volume_MVCS_knowned = MVCS_predictor.get_averaged_volume_with_knowned_idx(x_test_tensor, y_test_tensor[:, idx_knowned])
        coverage_MVCS_knowned = coverage_MVCS
        ERT_mvcs_knowned = ERT().evaluate(x_and_y_r_test_tensor, z_test_MVCS, alpha)
        WSC_mvcs_knowned = WSC().evaluate(x_and_y_r_test_tensor, z_test_MVCS, alpha)


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # ignore tous les warnings dans ce bloc

            calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size=1, shuffle=True, num_workers=1)
            start = time.time()
            conformalizer = C_HDR(calibrationloader, model)
            end = time.time()
            time_HDR = end - start
            
            cover = compute_coverage_indicator(conformalizer, alpha, x_test_tensor, y_test_tensor)
            print("cover")
            all_volumes_HDR = compute_log_region_size(conformalizer, model, alpha, x_test_tensor, n_samples=100)
            volume_HDR = torch.mean(torch.exp(all_volumes_HDR)**(1/output_dim)).item()
            print("volume", volume_HDR)
            coverage_HDR = torch.mean(cover).item()

            print('coverage', coverage_HDR)

            ERT_HDR = ERT().evaluate(x_test_tensor, cover, alpha)
            WSC_HDR = WSC().evaluate(x_test_tensor, cover, alpha)

            ERT_HDR_known = ERT().evaluate(x_and_y_r_test_tensor, cover, alpha)
            WSC_HDR_known = WSC().evaluate(x_and_y_r_test_tensor, cover, alpha)


            print("\n\n##########################################\n\n")



        
        file_path = Path("../results/results_full.csv")

        new_row = {
            "dataset": config_name,
            "experiment": experiment_index,
            "alpha": alpha,
            "volume_levelset": np.array(volume_ls),
            "volume_one": np.array(volume_one),
            "volume_ot": np.array(volume_ot),
            "volume_MVCS": np.array(volume_MVCS),
            "volume_HDR": np.array(volume_HDR),
            "coverage_levelset": coverage_ls,
            "coverage_one": coverage_one,
            "coverage_ot": coverage_ot,
            "coverage_MVCS": coverage_MVCS,
            "coverage_HDR": coverage_HDR,
            "ERT_levelset": ERT_ls,
            "ERT_one": ERT_one,
            "ERT_ot": ERT_ot,
            "ERT_MVCS": ERT_mvcs,
            "ERT_HDR": ERT_HDR,
            "WSC_levelset": WSC_ls,
            "WSC_one": WSC_one,
            "WSC_ot": WSC_ot,
            "WSC_MVCS": WSC_mvcs,
            "WSC_HDR": WSC_HDR,
            "time_levelset": time_ls,
            "time_one": time_one,
            "time_ot": time_ot,
            "time_MVCS": time_MVCS,
            "time_HDR": time_HDR,
        }

        if file_path.exists():
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file_path, index=False)

        
        file_path_revealed = Path("../results/results_revealed.csv")

        new_row_revealed = {
            "dataset": config_name,
            "experiment": experiment_index,
            "alpha": alpha,
            "volume_with_bayes_levelset": np.array(volume_with_bayes_levelsets),
            "volume_without_bayes_levelset": np.array(volume_without_bayes_levelsets),
            "volume_one_covariance_known": np.array(volume_one_covariance_known),
            "volume_ot_known": np.array(volume_ot_known),
            "volume_MVCS_known": np.array(volume_MVCS_knowned),
            "volume_HDR": np.array(volume_HDR),
            "coverage_with_bayes_levelset": coverage_with_bayes_levelsets,
            "coverage_without_bayes_levelset": coverage_without_bayes_levelsets,
            "coverage_one_covariance_known": coverage_one_covariance_known,
            "coverage_ot_known": coverage_ot_known,
            "coverage_MVCS_known": coverage_MVCS_knowned,
            "coverage_HDR_known": coverage_HDR,
            "ERT_with_bayes_levelset": ERT_with_bayes_ls,
            "ERT_without_bayes_levelset": ERT_without_bayes_ls,
            "ERT_one_covariance_known": ERT_one_known,
            "ERT_ot_known": ERT_ot_known,
            "ERT_MVCS_known": ERT_mvcs_knowned,
            "ERT_HDR_known": ERT_HDR_known,
            "WSC_with_bayes_levelset": WSC_with_bayes_ls,
            "WSC_without_bayes_levelset": WSC_without_bayes_ls,
            "WSC_one_covariance_known": WSC_one_known,
            "WSC_ot_known": WSC_ot_known,
            "WSC_MVCS_known": WSC_mvcs_knowned,
            "WSC_HDR_known": WSC_HDR_known,
            "time_with_bayes_levelset": time_with_bayes_ls,
            "time_one_covariance_known": time_one_known,
            "time_ot_known": time_ot_known,
            "time_MVCS_known": time_MVCS_knowned,
        }

        if file_path_revealed.exists():
            df = pd.read_csv(file_path_revealed)
            df = pd.concat([df, pd.DataFrame([new_row_revealed])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row_revealed])

        df.to_csv(file_path_revealed, index=False)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()