import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import math

from scipy.stats import chi2

def compute_loss_one_gaussian(y, centers, Sigmas):
    
    tab_dist = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    tab_log_det = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    
    for i in range(y.shape[0]):
        mask = ~torch.isnan(y[i])
        Sigmas_extracted = Sigmas[i, mask][:, mask]

        y_clean = y[i, mask]
        centers_clean = centers[i, mask]

        residuals = y_clean - centers_clean

        try:
            Sigma_inv = torch.linalg.inv(Sigmas_extracted)
        except:
            k = Sigmas_extracted.shape[-1]
            regularized = Sigmas_extracted + 1e-6 * torch.eye(k, device=Sigmas_extracted.device)
            Sigma_inv = torch.linalg.inv(regularized)

        result = residuals.T @ Sigma_inv @ residuals

        norm_squared = (result ** 2).sum(dim=0)

        tab_dist[i] = norm_squared
        tab_log_det[i] = torch.log(torch.sqrt(torch.det(Sigmas_extracted)))

    all_loss = tab_log_det + 0.5 * tab_dist 
    loss = all_loss.mean()
    
    return loss


def get_mahalanobis_distances(y, centers, Sigmas):
    """
    Computes the elliptical scores \|Lambda(y - centers)\| only on the dimensions where y is NaN,
    and also returns the number of observed (non-NaN) dimensions per row.

    Parameters:
        y (tensor): (n, k)
        centers (tensor): (n, k)
        Sigmas (tensor): (n, k, k)

    Returns:
        tab_dist (tensor): (n,)
        n_obs_per_row (list[int]): number of observed (non-NaN) dimensions per row
    """
    mask = ~torch.isnan(y)  # shape (n, k)
    n_obs_per_row = mask.sum(dim=1)
    n_obs_per_row = (n_obs_per_row).tolist()

    tab_dist = []

    for i in range(y.shape[0]):
        mask = ~torch.isnan(y[i])
        
        y_clean = y[i, mask]
        centers_clean = centers[i, mask]

        Sigmas_masked = Sigmas[i, mask][:, mask]
        Sigmas_masked_inv = torch.linalg.inv(Sigmas_masked)
        residuals = y_clean - centers_clean
        
        norm_squared = residuals.T @ Sigmas_masked_inv @ residuals

        tab_dist.append(norm_squared)

    tab_dist = torch.stack(tab_dist)
    return tab_dist, n_obs_per_row

def get_score(y, centers, Sigmas):
    norm_squared, deg_liberty = get_mahalanobis_distances(y, centers, Sigmas)
    quantiles = chi2.cdf(norm_squared, df=deg_liberty)
    return torch.tensor(quantiles)


class GaussianPredictorMissingValuesSigma:
    def __init__(self, center_model, matrix_model, dtype=torch.float32):
        """
        Parameters
        ----------
        center_model : torch.nn.Module
            The model that represents the center of the sets.
        matrix_model : torch.nn.Module
            The model that represents the matrix A of the ellipsoids. The matrix Lambda is obtained as the product of A by its transpose.
        dtype : torch.dtype, optional
            The data type of the tensors. The default is torch.float32.
        """
        self.center_model = center_model
        self.matrix_model = matrix_model
        self.q = torch.tensor(2.0, dtype=dtype)
        self._nu_conformal = None
        self.dtype = dtype

    def get_Sigmas(self, x):
        """
        Compute the matrix Lambda = A @ A^T for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        Sigmas : torch.Tensor
            The matrix Sigma = A(x) @ A(x)^T of shape (n, k, k)
        """
         
        mat_A = self.matrix_model(x)
        Sigmas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
        
        return Sigmas
    
    def get_centers(self, x):
        """
        Compute the centers of the ellipsoids for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        centers : torch.Tensor
            The centers of the ellipsoids of shape (n, k)
        """

        return self.center_model(x)
    
    @property
    def nu_conformal(self):
        return self._nu_conformal
    
    @property
    def idx_knowned(self):
        return self._idx_knowned
    
    @property
    def idx_unknown(self):
        return self._idx_unknown
    
    @property
    def nu_conformal_conditional(self):
        return self._nu_conformal_conditional

    def fit(self,
            trainloader,
            stoploader,
            num_epochs=1000,
            lr_center_models=0.001,
            lr_matrix_models=0.001,
            use_lr_scheduler=False,
            verbose=0,
            stop_on_best=True,
        ):
        """"
        Parameters
        
        trainloader : torch.utils.data.DataLoader
            The DataLoader of the training set.
        stoploader : torch.utils.data.DataLoader
            The DataLoader of the validation set : Warning: do not use the calibration set as a stopping criterion as you would lose the coverage property.
        num_epochs : int, optional
            The total number of epochs. The default is 1000.
        lr_center_models : float, optional
            The learning rate for the model. The default is 0.001.
        lr_matrix_models : float, optional
            The learning rate for the matrix model. The default is 0.001.
        use_lr_scheduler : bool, optional
            Whether to use a learning rate scheduler. The default is False.
        verbose : int, optional
            The verbosity level. The default is 0.
        stop_on_best : bool, optional   
            Whether to stop on the best model. The default is False.
            """

        if stop_on_best:
            self.best_stop_loss = np.inf
            self.best_model_weight = None
            self.best_matrix_weight = None
            self.best_q = None

        self.matrix_model = copy.deepcopy(self.matrix_model) 
        self.center_model = copy.deepcopy(self.center_model) 

        optimizer = torch.optim.Adam([
            {'params': self.center_model.parameters(), 'lr': lr_center_models},
            {'params': self.matrix_model.parameters(), 'lr': lr_matrix_models},
        ])

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.tab_train_loss = []
        self.tab_stop_loss = []

        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            for x, y in trainloader:    
                optimizer.zero_grad()

                Sigmas = self.get_Sigmas(x)    # (n, k, k)
                centers = self.get_centers(x)    # (n, k)

                loss = compute_loss_one_gaussian(y, centers, Sigmas)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            epoch_loss = self.eval(trainloader)
            self.tab_train_loss.append(epoch_loss.item())

            epoch_stop_loss = self.eval(stoploader)
            self.tab_stop_loss.append(epoch_stop_loss.item())

            if stop_on_best and self.best_stop_loss > epoch_stop_loss.item():
                if verbose == 2:
                    print(f"New best stop loss: {epoch_stop_loss.item()}")
                self.best_stop_loss = epoch_stop_loss
                self.best_center_model_weight = copy.deepcopy(self.center_model.state_dict()) 
                self.best_matrix_weight = copy.deepcopy(self.matrix_model.state_dict()) 
                
            if verbose != 0:
                if epoch % print_every == 0:
                    print(f"Epoch {epoch}: Loss = {epoch_loss.item()} - Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")

            if use_lr_scheduler:
                scheduler.step()

        epoch_loss = self.eval(trainloader)
        epoch_stop_loss = self.eval(stoploader)
        if stop_on_best:
            self.load_best_model()
        epoch_loss = self.eval(trainloader)
        epoch_stop_loss = self.eval(stoploader)
        if verbose != 0:
            print(f"Final Loss = {epoch_loss.item()} - Final Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")
        
        self.k = Sigmas.shape[-1]

    def load_best_model(self):
        """
        Load the best model.    
        """
        if self.best_center_model_weight is not None:
            self.center_model.load_state_dict(self.best_center_model_weight)
            self.matrix_model.load_state_dict(self.best_matrix_weight)
        else:
            raise ValueError("You must call the `fit` method with the `stop_on_best` parameter set to True.")

    def eval(self,
             dataloader):
        """"
        Evaluate the loss on a given DataLoader.
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader of the dataset on which to evaluate the loss.
        """
        
        with torch.no_grad():
            loss = 0.0
            for x, y in dataloader:
                Sigmas = self.get_Sigmas(x)
                centers = self.get_centers( x )
                loss += compute_loss_one_gaussian(y, centers, Sigmas)
            return loss / len(dataloader)
        

    ##########################################################
    ############# CONFORMALIZATION - CLASSIC #################
    ##########################################################
    
    def conformalize(self, calibrationloader=None, x_calibration=None, y_calibration=None, alpha=0.1):
        """
        Compute the quantile value to conformalize the ellipsoids on a unseen dataset.

        Parameters
        ----------
        calibrationloader : torch.utils.data.DataLoader, optional
            The DataLoader of the calibration set. The default is None.
        x_calibration : torch.Tensor, optional
            The input tensor of the calibration set. The default is None.  The shape is (n, d).
        y_calibration : torch.Tensor, optional
            The output tensor of the calibration set. The default is None. The shape is (n, k).
        alpha : float, optional
            The level of the confidence sets. The default is 0.1.    
        """
        with torch.no_grad():
            if calibrationloader is not None:
                # Case where we use a DataLoader
                empty = True

                for x, y in calibrationloader:
                    centers = self.get_centers(x)
                    Sigmas = self.get_Sigmas(x)

                    if empty:
                        centers_calibration = centers
                        y_calibration = y
                        Sigmas_calibration = Sigmas
                        empty = False
                    else:
                        Sigmas_calibration = torch.cat((Sigmas_calibration, Sigmas), dim=0)
                        centers_calibration = torch.cat((centers_calibration, centers), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                centers_calibration = self.get_centers(x_calibration)
                Sigmas_calibration = self.get_Sigmas(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = y_calibration.shape[0]
            p = int(np.ceil((n+1)*(1-alpha)))
            if p > n:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            scores = get_score(y_calibration, centers_calibration, Sigmas_calibration)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal = scores[p]

            return self._nu_conformal
        
    def get_radius_full_information(self):
        return np.sqrt(chi2.ppf(self._nu_conformal.item(), df=self.k))
        
    def get_one_volume(self, x):
        with torch.no_grad():
            
            Sigmas = self.get_Sigmas(x).detach().numpy()

            Lambdas = get_Lambdas_from_Sigmas(Sigmas)

            k = Lambdas.shape[-1]
            
            radius = self.get_radius_full_information()

            volumes = (2*math.gamma(1/2 + 1))**k / math.gamma(k/2 + 1) / np.linalg.det(Lambdas/radius)

            return volumes**(1/k)  # Volume of the ellipsoid is the k-th root of the determinant of the covariance matrix

    def get_averaged_volume(self, x):
        with torch.no_grad():
            tab_volumes = self.get_one_volume(x)
            return np.mean(tab_volumes)

    def get_coverage(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Sigmas = self.get_Sigmas(x_test)
            
            scores = get_score(y_test, centers, Sigmas)
            coverage = np.average(scores < self._nu_conformal.item())
            
            return coverage
        
def get_Lambdas_from_Sigmas(Sigma):
    # if Torch
    if isinstance(Sigma, torch.Tensor):
        # Inverse in batch
        Lambdas_squared = torch.linalg.inv(Sigma)
        # Eigendecomposition (symmetric)
        eigenvalues, eigenvectors = torch.linalg.eigh(Lambdas_squared)
        # sqrt of eigenvalues
        eigenvalues_sqrt = torch.sqrt(eigenvalues)
        # Construct Lambdas using batch matrix multiplication
        Lambdas = eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-1, -2)
        return Lambdas

    # if numpy
    elif isinstance(Sigma, np.ndarray):
        # Inverse in batch
        Lambdas_squared = np.linalg.inv(Sigma)
        # Eigendecomposition (symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(Lambdas_squared)
        # sqrt of eigenvalues
        eigenvalues_sqrt = np.sqrt(eigenvalues)
        # Construct Lambdas using batch matrix multiplication
        Lambdas = np.matmul(eigenvectors, np.matmul(np.expand_dims(np.eye(Sigma.shape[-1]), axis=0) * eigenvalues_sqrt[..., None, :], np.transpose(eigenvectors, (0, 2, 1))))
        return Lambdas

    else:
        raise ValueError("The type of Sigma is not recognized.")
