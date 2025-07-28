import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math


def get_ellipse_scores(y, centers, Lambdas):
    # return the norms \|Lambda(y-centers)\|
    residuals = y - centers
    vec = torch.einsum("nri,ni->nr", Lambdas, residuals ) # (n , r)
    scores = torch.linalg.norm(vec, ord=2, dim=1)
    return scores

def get_lambdas_cov(y_train, f_x):
    k = y_train.shape[1]
    residuals = y_train - f_x
    cov = torch.atleast_2d(torch.cov(residuals.T))
    Lambda_cov_torch = torch.linalg.inv(cov)
    eigenvalues, eigenvectors = torch.linalg.eigh(Lambda_cov_torch)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    Lambda_cov_torch = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return Lambda_cov_torch


class CovariancePredictor:
    def __init__(self, model):
        self._model = model
        self._nu_covariance = None
        self._Lambda_cov = None
        self._q_fix = torch.tensor(2, dtype=torch.float32)

    @property
    def nu_conformal(self):
        return self._nu_conformal
        
    @property
    def Lambda_cov(self):
        return self._Lambda_cov
    
    @property
    def model(self):
        return self._model
    
    def fit(self, trainloader = None, stoploader = None, num_epochs = 100, lr_model = 0.01):
        self._model.fit_and_plot(trainloader, stoploader, epochs = num_epochs, lr = lr_model)
        
    def fit_cov(self, trainloader = None, x_train = None, y_train = None):
        with torch.no_grad():
            if trainloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in trainloader:
                    f_x = self._model(x)
                    
                    if empty:
                        f_x_train = f_x
                        y_train = y
                        empty = False
                    else:
                        f_x_train = torch.cat((f_x_train, f_x), 0)
                        y_train = torch.cat((y_train, y), 0)
            
            elif x_train is not None and y_train is not None:
                # Case where we directly give tensors
                f_x_train = self._model(x_train)
            
            self._Lambda_cov = get_lambdas_cov(y_train, f_x_train)
    
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
        alpha : float
            The level of the confidence sets. The default is 0.1.    
        """
        with torch.no_grad():
            if calibrationloader is not None:
                # Case where we use a DataLoader
                empty = True

                for x, y in calibrationloader:
                    centers = self.get_centers(x)
                    Lambdas = self.get_Lambdas(x)

                    if empty:
                        centers_calibration = centers
                        y_calibration = y
                        Lambdas_calibration = Lambdas
                        empty = False
                    else:
                        Lambdas_calibration = torch.cat((Lambdas_calibration, Lambdas), dim=0)
                        centers_calibration = torch.cat((centers_calibration, centers), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                centers_calibration = self.get_centers(x_calibration)
                Lambdas_calibration = self.get_Lambdas(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = y_calibration.shape[0]
            p = int(np.ceil((n+1)*(1-alpha)))
            if p > n:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            scores = get_ellipse_scores(y_calibration, centers_calibration, Lambdas_calibration)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal = scores[p]

            return self._nu_conformal
        

    def get_one_volume(self, x):
        with torch.no_grad():
            
            Lambdas = self.get_Lambdas(x).detach().numpy()

            k = Lambdas.shape[-1]
            
            radius = self.nu_conformal.item()

            volumes = (2*math.gamma(1/2 + 1))**k / math.gamma(k/2 + 1) / np.linalg.det(Lambdas/radius)

            return volumes**(1/k)  # Volume of the ellipsoid, normalized by the dimension
        
    def get_averaged_volume(self, x):
        with torch.no_grad():
            volumes = self.get_one_volume(x)
            return np.mean(volumes)

    def get_coverage(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)
            
            scores = get_ellipse_scores(y_test, centers, Lambdas)
            coverage = np.average(scores < self._nu_conformal.item())
            
            return coverage
        
    ##########################################################
    ######### CONFORMALIZATION - WITH IDX KNOWNED ############
    ##########################################################

    def get_Lambdas(self, x_test):
        if self._Lambda_cov is None:
            raise ValueError("You must call the `fit` method before.")
        k = self._Lambda_cov.shape[1]
        return self._Lambda_cov.unsqueeze(0).expand(x_test.shape[0], k, k).clone()
    
    def get_centers(self, x_test):
        if self._model is None:
            raise ValueError("You must call the `fit` method before.")
        centers = self._model(x_test)
        return centers
                
    
    def get_volume_condition_on_idx_without_bayes(self, x, y_r, idx_knowned):
        if len(x.shape) == 1:   
            x = x.unsqueeze(0)
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        with torch.no_grad():
            centers = self.get_centers(x)
            Lambdas = self.get_Lambdas(x)

            k = Lambdas.shape[-1]  # Dimension of the full space
            r = len(idx_knowned)
            s = k - r  # Dimension of the subspace

            idx_unknown = [i for i in range(k) if i not in idx_knowned]
            idx_unknown = torch.tensor(idx_unknown, device=Lambdas.device)

            f_x_r = centers[:, idx_knowned]  # (n, r)

            residual_r = y_r - f_x_r
            
            # Compute A = M.T @ M
            M = Lambdas @ Lambdas
            
            idx_u = torch.tensor(idx_unknown, device=M.device)
            idx_k = torch.tensor(idx_knowned, device=M.device)

            M_ss = M[:, idx_u][:, :, idx_u]   # bloc (s,s)
            M_sr = M[:, idx_u][:, :, idx_k]   # bloc (s,r)
            M_rr = M[:, idx_k][:, :, idx_k]   # bloc (r,r)


            c = torch.einsum('nsr,nr->ns', M_sr, residual_r)  # (n, s)

            term1 = self._nu_conformal**2

            term2 = torch.einsum('nr,nr->n', residual_r, torch.einsum('nri, ni->nr', M_rr, residual_r ) )  # (n, r)

            term3 = torch.einsum('ns,ns->n', c, torch.einsum('nsi, ni->ns', torch.linalg.inv(M_ss), c) )  # (n, s)
            
            tau = ( term1 - term2 + term3 )

            tau_reshaped = tau.view(-1, 1, 1)  # reshape explicite pour un broadcasting correct
            volume = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**s \
                    / torch.lgamma(torch.tensor(s / 2 + 1)).exp() \
                    / torch.sqrt(torch.det(M_ss / tau_reshaped))
            
            # IDX where tau <= 0 : volume = 0
            volume[tau <= 0] = 0.
            volume[torch.isnan(volume)] = 0
            return volume**(1/s)
        
    def get_averaged_volume_condition_on_idx_without_bayes(self, x, y_r, idx_knowned):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx_without_bayes(x, y_r, idx_knowned)
            return torch.mean(tab_volumes)
        

    ##########################################################
    ############ CONFORMALIZATION - PROJECTION ###############
    ##########################################################

    def conformalize_linear_projection(self, projection_matrix):
        self.projection_matrix = projection_matrix
    
    def get_volume_projection(self, x):
        with torch.no_grad():
            Lambdas = self.get_Lambdas(x)
            centers = self.get_centers(x)

            _, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, self.projection_matrix)

            k_new = new_Lambdas.shape[-1]
            
            volumes = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**k_new \
                    / torch.lgamma(torch.tensor(k_new / 2 + 1)).exp() \
                    / torch.det(new_Lambdas / self._nu_conformal )

            return volumes**(1/k_new)  # Volume of the ellipsoid, normalized by the dimension
        
    def get_averaged_volume_projection(self, x):
        with torch.no_grad():
            tab_volumes = self.get_volume_projection(x)
            return torch.mean(tab_volumes)
        
    def get_coverage_projection(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, self.projection_matrix)

            y_projection = torch.einsum('rk,nk->nr', self.projection_matrix, y_test)

            scores = get_ellipse_scores(y_projection, new_centers, new_Lambdas)
            coverage = np.average(scores < self._nu_conformal.item())
            
            return coverage

def get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, projection_matrice):
    new_mean = torch.einsum('rk,nk->nr', projection_matrice, centers)  # (n, r)
    Sigmas = get_Sigmas_from_Lambdas(Lambdas) # (n, k, k)
    new_Sigmas = torch.einsum('ri,nij,sj->nrs', projection_matrice, Sigmas, projection_matrice)
    new_Lambdas = get_Lambdas_from_Sigmas(new_Sigmas)
    return new_mean, new_Lambdas
    

def get_Sigmas_from_Lambdas(Lambdas):
    # If torch : 
    if isinstance(Lambdas, torch.Tensor):    
        Sigma_squared = torch.linalg.inv(Lambdas)
        Sigma = Sigma_squared @ Sigma_squared
        return Sigma
    # If numpy :
    elif isinstance(Lambdas, np.ndarray):
        Sigma_squared = np.linalg.inv(Lambdas)
        Sigma = Sigma_squared @ Sigma_squared
        return Sigma
    else:
        raise ValueError("The type of Lambdas is not recognized.")

def get_Lambdas_from_Sigmas(Sigma):
    # if Torch
    if isinstance(Sigma, torch.Tensor):
        Lambdas_squared = torch.linalg.inv(Sigma)
        eigenvalues, eigenvectors = torch.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = torch.sqrt(eigenvalues)
        Lambdas = eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-1, -2)
        return Lambdas
    # if numpy
    elif isinstance(Sigma, np.ndarray):
        Lambdas_squared = np.linalg.inv(Sigma)
        eigenvalues, eigenvectors = np.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = np.sqrt(eigenvalues)
        Lambdas = eigenvectors @ np.diag(eigenvalues_sqrt) @ eigenvectors.T
        return Lambdas
    else:
        raise ValueError("The type of Sigma is not recognized.")

