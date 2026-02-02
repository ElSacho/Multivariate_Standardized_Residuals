import torch
from torch.utils.data import TensorDataset, DataLoader
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import *
from data_loading import *
import json

def main():

    # --- Load config and data ---
    alpha = 0.1
    config_name = "house"
    config_path = f"experiments/parameters/{config_name}.json"
    with open(config_path, 'r') as file:
        parameters = json.load(file)

    load_path = f"data/processed_data/{parameters['load_name']}.npz"
    X, Y = load_data(load_path)

    # --- Split and preprocess ---
    splits = [0.7, 0.1, 0.1, 0.1]
    subsets = split_and_preprocess(X, Y, splits=splits, normalize=True)
    x_train, y_train = subsets["X_train"], subsets["Y_train"]
    x_calibration, y_calibration = subsets["X_calibration"], subsets["Y_calibration"]
    x_test, y_test = subsets["X_test"], subsets["Y_test"]
    x_stop, y_stop = subsets["X_stop"], subsets["Y_stop"]

    # --- Convert to tensors ---
    dtype = torch.float64 if parameters["dtype"] == "float64" else torch.float32
    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=dtype)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=dtype)
    x_test_tensor = torch.tensor(x_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)
    x_stop_tensor = torch.tensor(x_stop, dtype=dtype)
    y_stop_tensor = torch.tensor(y_stop, dtype=dtype)

    # --- DataLoaders ---
    batch_size_model = parameters["batch_size"]
    trainloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size_model, shuffle=True, num_workers=10)
    calibrationloader = DataLoader(TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size=batch_size_model, shuffle=True, num_workers=10)
    testloader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size_model, shuffle=True, num_workers=10)
    stoploader = DataLoader(TensorDataset(x_stop_tensor, y_stop_tensor), batch_size=batch_size_model, shuffle=True, num_workers=10)

    # --- Model & Trainer ---
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    model = MQF2LightningModule(input_dim, output_dim)
    trainer = get_lightning_trainer(max_epochs=2)

    print("Training model...")
    trainer.fit(model, train_dataloaders=trainloader)
    print("Model trained!")

    # --- Conformal evaluation ---
    alpha = 0.1
    conformalizer = M_CP(calibrationloader, model)
    test_batch = next(iter(testloader))
    x, y = test_batch
    print(x.shape)
    print(x)
    coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
    volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
    print("Coverage:", coverage)
    print("Volume:", volume)


#     conformalizers = {
#     'M-CP': M_CP,
#     'DR-CP': DR_CP,
#     'C-HDR': C_HDR,
#     'PCP': PCP,
#     'HD-PCP': HD_PCP,
#     'C-PCP': C_PCP,
#     'CP2-PCP-Linear': CP2_PCP_Linear,
#     'L-CP': L_CP,
#     'HDR-H': HDR_H,
#     'L-H': L_H,
#     'STDQR': STDQR,
#     'CopulaCPTS': CopulaCPTS,
# }


if __name__ == "__main__":
    main()
