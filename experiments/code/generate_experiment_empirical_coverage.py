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
from generate_data import *

import json

from approaches.gaussian_predictor_likelihood import *
from approaches.gaussian_predictor_levelsets import *
from approaches.gaussian_trainer import *
from approaches.OT_predictor import *
from approaches.MVCS import *
from approaches.covariances import *

import sys
import os
from pathlib import Path


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

def compute_batched_coverage(predictor, x, y, alpha, batch_size=100):
    n = x.shape[0]
    coverages = []
    
    # On découpe x et y en petits paquets
    for i in range(0, n, batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        
        # On calcule l'indicateur pour ce petit paquet seulement
        # Note: adaptez l'appel selon votre fonction compute_coverage_indicator
        res = compute_coverage_indicator(predictor, alpha, x_batch, y_batch)
        coverages.append(res)
    
    # On regroupe tout et on fait la moyenne à la fin
    return torch.cat(coverages).float().mean().item()

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



def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("config_name", type=str, help="Nom de la configuration")
    parser.add_argument("experiment", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    config_name = args.config_name
    experiment = args.experiment  # Use this as seed

    seed_everything(experiment)

    print('config_name:', config_name)

    config_path = "../parameters/" + config_name + ".json"
    with open(config_path, 'r') as file : 
        parameters = json.load(file)

    tab_alpha = [0.1, 0.05]
    n_samples_evaluation = 1_000
    n_iter_empirical = 100

    resuls_alpha = {}

    results_coverage = {alpha: {} for alpha in tab_alpha}
    results_volume = {alpha: {} for alpha in tab_alpha}
    results_empirical_coverage = {alpha: {} for alpha in tab_alpha}

    def save_results(alpha, experiment, results_volume, results_coverage, results_empirical_coverage):
        save_path_volume = f"../results/volume_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"
        save_path_coverage = f"../results/coverage_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"
        save_path_empirical = f"../results/empirical_conditional_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"

        # Création des dossiers si nécessaire
        for path in [save_path_volume, save_path_coverage, save_path_empirical]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_path_volume = f"../results/volume_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"
        save_path_coverage = f"../results/coverage_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"
        save_path_empirical = f"../results/empirical_conditional_synthetic/{config_name}_alpha_{alpha}_{experiment}.pkl"

        with open(save_path_volume, "wb") as f:
            pickle.dump(results_volume, f)

        with open(save_path_coverage, "wb") as f:
            pickle.dump(results_coverage, f)

        with open(save_path_empirical, "wb") as f:
            pickle.dump(results_empirical_coverage, f)


    
        
    print(f"Experiment {experiment}/{parameters['n_experiments']}")

    prop_train = parameters["prop_train"]
    prop_calibration = parameters["prop_calibration"]

    class RadiusTransformation2:
        def __init__(self, d, beta=None):
            if beta is None:
                beta = np.random.randn(d)
            self.beta = beta

        def get(self, x):
            return ((( np.linalg.norm(x)) / 2.0 + (np.dot(self.beta, x))**2 / 10) / 2 ) / .5 + 0.15
        
    class RadiusTransformationFixed:
        def __init__(self, d, beta=None):
            if beta is None:
                beta = np.random.randn(d)
            self.beta = beta

        def get(self, x):
            return 1.
        
    class NonLinearFunction2:
        def __init__(self, d, k, beta=None):
            if beta is None:
                beta = np.random.randn(d, k)
            self.beta = beta
            self.proj  = np.zeros((d, k))
            self.proj[0, 0] = 1.0
            self.proj[1, 1] = 1.0

        def get(self, x):
            nonlinear_term = np.sin(np.dot(x, self.beta)) + 0.5 * np.tanh(np.dot(x**2, self.beta)) + np.dot(x, self.proj)
            return nonlinear_term * 2
        
    d = 4
    k = 4

    pert = parameters["perturbation"]

    n_train = 30_000
    # n_train = 1000
    n_anchors = 10 if parameters["perturbation_type"] == "tx" else 1

    f_star = NonLinearFunction2(d, k)
    radius = RadiusTransformation2(d) if parameters["perturbation_type"] == "tx" else RadiusTransformationFixed(d)
    local_perturbation = LocalPerturbation(d, k, n_anchors=n_anchors, radius_transformation=radius)
    data_generator = DataGenerator(d, k, pert, f_star=f_star, local_perturbation=local_perturbation, covariance_matrix=np.eye(k), bias = False, seed = 42)

    X, Y = data_generator.generate(n_train)

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

    input_dim = d
    output_dim = k

    hidden_dim_matrix = parameters["hidden_dim_matrix"]
    n_hidden_layers_matrix = parameters["n_hidden_layers_matrix"]

    num_epochs = parameters["num_epochs"]
    lr_matrix = parameters["lr_matrix"]
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
        results_volume = {}
        results_coverage = {}
        results_empirical_coverage = {}
        
        #### WITH LEVEL SETS #####
        gaussian_level_sets.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_ls = gaussian_level_sets.get_averaged_volume(x_test_tensor)
        coverage_ls = gaussian_level_sets.get_coverage(x_test_tensor, y_test_tensor)

        ##### WITH LIKELIHOOD #####
        gaussian_likelihood.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_l = gaussian_likelihood.get_averaged_volume(x_test_tensor)
        coverage_l = gaussian_likelihood.get_coverage(x_test_tensor, y_test_tensor)


        ##### WITH ONE COVARIANCE #####
        one_covariance_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_one = one_covariance_predictor.get_averaged_volume(x_test_tensor)
        coverage_one = one_covariance_predictor.get_coverage(x_test_tensor, y_test_tensor)

        ##### WITH OT #####
        volume_ot = -1.
        coverage_ot = -1.
        ot_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_ot = ot_predictor.get_averaged_volume(x_test_tensor)
        coverage_ot = ot_predictor.get_coverage(x_test_tensor, y_test_tensor) 

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
        
        MVCS_predictor.conformalize(x_calibration=x_calibration_tensor, y_calibration=y_calibration_tensor, alpha = alpha)
        volume_MVCS = MVCS_predictor.get_averaged_volume(x_test=x_test_tensor)
        coverage_MVCS = MVCS_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)

        calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size= parameters["batch_size"], shuffle=True, num_workers=1)
        conformalizer = C_HDR(calibrationloader, model, n_samples=100)
        
        print("Conformalized")
        cover = compute_coverage_indicator(conformalizer, alpha, x_test_tensor, y_test_tensor)
        print("Cover ok")
        all_volumes_HDR = compute_log_region_size(conformalizer, model, alpha, x_test_tensor, n_samples=100)
        volume_HDR = torch.mean(torch.exp(all_volumes_HDR)**(1/output_dim)).item()
        print("Volume :", volume_HDR)
        coverage_HDR = torch.mean(cover).item()
        print("Coverage", coverage_HDR)


    
        results_volume = {
            "experiment": experiment,
            "volume_levelsets": volume_ls,
            "volume_likelihood": volume_l,
            "volume_one": volume_one,
            "volume_ot": volume_ot,
            "volume_MVCS": volume_MVCS,
            "volume_HDR": volume_HDR,
        }

        results_coverage[experiment] = {
            "experiment": experiment,
            "coverage_levelset": coverage_ls,
            "coverage_likelihood": coverage_l,
            "coverage_one": coverage_one,
            "coverage_ot": coverage_ot,
            "coverage_MVCS": coverage_MVCS,
            "coverage_HDR": coverage_HDR,
        }

        print("")
        print(f"[Alpha={alpha}]")
        print(f"Volume levelset : {volume_ls:.3f}, \nVolume likelihood: {volume_l:.3f}, \nVolume one covariance: {volume_one:.3f}, \nVolume OT: {volume_ot:.3f}, \nVolume MVCS: {volume_MVCS:.3f}")
        print(f"Coverage levelset : {coverage_ls:.3f}, \nCoverage likelihood: {coverage_l:.3f}, \nCoverage one covariance: {coverage_one:.3f}, \nCoverage OT: {coverage_ot:.3f}, \nCoverage MVCS: {coverage_MVCS:.3f}")        
        print("")


        tab_conditional_levelsets = np.zeros(n_iter_empirical)
        tab_conditional_likelihood = np.zeros(n_iter_empirical)
        tab_conditional_one_covariance = np.zeros(n_iter_empirical)
        tab_conditional_ot = np.zeros(n_iter_empirical)
        tab_conditional_mvcs = np.zeros(n_iter_empirical)
        tab_conditional_HDR = np.zeros(n_iter_empirical)

        for i in range(n_iter_empirical):
            with torch.no_grad():
                x_specific, y_specific = data_generator.generate_specific_x(n_samples_evaluation)
                x_specific = torch.from_numpy(x_specific).to(dtype)
                y_specific = torch.from_numpy(y_specific).to(dtype)
                tab_conditional_levelsets[i] = gaussian_level_sets.get_coverage(x_specific, y_specific)
                tab_conditional_likelihood[i] = gaussian_likelihood.get_coverage(x_specific, y_specific)
                tab_conditional_one_covariance[i] = one_covariance_predictor.get_coverage(x_specific, y_specific)
                tab_conditional_ot[i] = ot_predictor.get_coverage(x_specific, y_specific)
                tab_conditional_mvcs[i] = MVCS_predictor.get_coverage(x_specific, y_specific)
                tab_conditional_HDR[i] = compute_batched_coverage(
            conformalizer, x_specific, y_specific, alpha, batch_size=100
        )
                print("Conditional coverage :",tab_conditional_levelsets[i], tab_conditional_likelihood[i],tab_conditional_one_covariance[i] , tab_conditional_ot[i], tab_conditional_mvcs[i] , tab_conditional_HDR[i])
                del x_specific, y_specific

        results_empirical_coverage = {
            "experiment": experiment,
            "empirical_conditional_levelsets": tab_conditional_levelsets,
            "empirical_conditional_likelihood": tab_conditional_likelihood,
            "empirical_conditional_one_covariance": tab_conditional_one_covariance,
            "empirical_conditional_ot": tab_conditional_ot,
            "empirical_conditional_mvcs": tab_conditional_mvcs,
            "empirical_conditional_HDR": tab_conditional_HDR,
        }

        save_results(alpha, experiment, results_volume, results_coverage, results_empirical_coverage)



if __name__=="__main__":
    main()