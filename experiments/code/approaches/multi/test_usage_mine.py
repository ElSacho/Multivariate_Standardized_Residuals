import torch
from torch.utils.data import TensorDataset, DataLoader
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP
from data_loading import *
import json

class L_CP_Scorer:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, x_calibration, y_calibration):
        """Get the scores for the calibration data."""
        with torch.no_grad():
            calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration, y_calibration), batch_size= 32, shuffle=True, num_workers=10)
            L_CP(calibrationloader, self.model)
            scores = L_CP.get_score(x_calibration, y_calibration)
        return scores
    
    def conformalize(self, x_calibration, y_calibration):
        with torch.no_grad():
                calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration, y_calibration), batch_size= 32, shuffle=True, num_workers=10)
                self.conformalizer = L_CP(calibrationloader, self.model)
    def get_volumes(self, x, nu_conformal):
        with torch.no_grad():        
            compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
        

    # --- Conformal evaluation ---
    alpha = 0.1
    conformalizer = L_CP(calibrationloader, model)
    test_batch = next(iter(testloader))
    x, y = test_batch
    coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
    volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
    print("Coverage:", coverage)
    print("Volume:", volume)