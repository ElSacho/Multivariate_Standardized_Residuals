import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import math

from scipy.stats import chi2

def get_Lambdas_extracted(Lambdas, y):
    Sigmas = get_Sigmas_from_Lambdas(Lambdas)
    for i in range(Sigmas.shape[0]):
        mask = ~torch.isnan(y[i])
        Sigmas_masked = Sigmas[i, mask][:, mask]
        Lambdas_extracted = get_Lambdas_from_Sigmas(Sigmas_masked)
        Lambdas_masked = torch.zeros_like(Lambdas)
        Lambdas_masked[i, mask][:, mask] = Lambdas_extracted
    return Lambdas_extracted, Lambdas_masked

def compute_loss_one_gaussian_with_Sigmas(y, centers, Lambdas):
    Sigmas = get_Sigmas_from_Lambdas(Lambdas)
    tab_dist = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    tab_log_det = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
    
    for i in range(y.shape[0]):
        mask = ~torch.isnan(y[i])
        Sigmas_extracted = Sigmas[i, mask][:, mask]
        Lambdas_extracted = get_Lambdas_from_Sigmas(Sigmas_extracted)

        y_clean = y[i, mask]
        centers_clean = centers[i, mask]

        residuals = y_clean - centers_clean

        # result = Lambdas_extracted @ residuals
        result = residuals.T @ torch.linalg.inv(Sigmas_extracted) @ residuals

        norm_squared = (result ** 2).sum(dim=0)

        tab_dist[i] = norm_squared
        tab_log_det[i] = torch.linalg.slogdet(Sigmas_extracted)[1]
        if tab_log_det[i].isnan():
            print("nan")
            print("Lambdas_extracted : ", Lambdas_extracted)
            print("Sigmas_extracted : ", Sigmas_extracted)
            print("Lambda : ", Lambdas[i])
            print("Sigmas : ", Sigmas[i])
            print("y_clean : ", y_clean)
            print("centers_clean : ", centers_clean)
            print("residuals : ", residuals)
            print("result : ", result)
            print("norm_squared : ", norm_squared)
            print("tab_log_det[i] : ", tab_log_det[i])
            raise ValueError("Log determinant is NaN")
        
    all_loss = tab_log_det + 0.5 * tab_dist 
    loss = all_loss.mean()
    
    return loss

def compute_loss_one_gaussian(y, centers, Lambdas):
    # Mask for non-NaN values in y
    mask = ~torch.isnan(y)  # shape (n, k)

    # Replace NaNs in y and centers with 0 (so the residual is well-defined)
    y_clean = torch.where(mask, y, torch.zeros_like(y))
    centers_clean = torch.where(mask, centers, torch.zeros_like(centers))

    # Compute residuals
    residuals = y_clean - centers_clean  # shape (n, k)

    # Apply the mask to the residuals (set ignored components to 0)
    residuals = residuals * mask  # still (n, k)

    # Apply Lambdas: assuming Lambdas has shape (n, k, k)
    result = torch.einsum('bij,bj->bi', Lambdas, residuals)

    # Also mask the result before computing the norm (partial norm)
    result = result * mask

    # Compute the norm taking into account only valid components
    norm_squared = (result ** 2).sum(dim=1)

    # Log-determinant
    logdets = torch.linalg.slogdet(Lambdas)[1]

    # Loss
    all_loss = -logdets + 0.5 * norm_squared 
    loss = all_loss.mean()

    return loss


def get_mahalanobis_distances(y, centers, Lambdas):
    """
    Computes the elliptical scores \|Lambda(y - centers)\| only on the dimensions where y is NaN,
    and also returns the number of observed (non-NaN) dimensions per row.

    Parameters:
        y (tensor): (n, k)
        centers (tensor): (n, k)
        Lambdas (tensor): (n, k, k)

    Returns:
        scores (tensor): (n,)
        n_obs_per_row (list[int]): number of observed (non-NaN) dimensions per row
    """

    mask = ~torch.isnan(y)  # shape (n, k)
    n_obs_per_row = mask.sum(dim=1)
    n_obs_per_row = (n_obs_per_row).tolist()

    Sigmas = get_Sigmas_from_Lambdas(Lambdas)

    tab_dist = []

    for i in range(y.shape[0]):
        mask = ~torch.isnan(y[i])
        
        y_clean = y[i, mask]
        centers_clean = centers[i, mask]

        Sigmas_masked = Sigmas[i, mask][:, mask]
        Lambdas_masked = get_Lambdas_from_Sigmas(Sigmas_masked)
    
        residuals = y_clean - centers_clean
        
        result = Lambdas_masked @ residuals

        norm_squared = (result ** 2).sum(dim=0)
        tab_dist.append(norm_squared)

    tab_dist = torch.stack(tab_dist)
    return tab_dist, n_obs_per_row

def get_score(y, centers, Lambdas):
    norm_squared, deg_liberty = get_mahalanobis_distances(y, centers, Lambdas)
    quantiles = chi2.cdf(norm_squared, df=deg_liberty)
    return torch.tensor(quantiles)


class GaussianPredictorMissingValues:
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

    def get_Lambdas(self, x):
        """
        Compute the matrix Lambda = A @ A^T for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        Lambdas : torch.Tensor
            The matrix Lambda = A(x) @ A(x)^T of shape (n, k, k)
        """
         
        mat_A = self.matrix_model(x)
        Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
        
        return Lambdas
    
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
            self.best_center_model_weight = copy.deepcopy(self.center_model.state_dict()) 
            self.best_matrix_weight = copy.deepcopy(self.matrix_model.state_dict()) 

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

                Lambdas = self.get_Lambdas(x)    # (n, k, k)
                centers = self.get_centers(x)    # (n, k)

                loss = compute_loss_one_gaussian(y, centers, Lambdas)
                
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
        
        self.k = Lambdas.shape[-1]

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
                Lambdas = self.get_Lambdas(x)
                centers = self.get_centers( x )
                loss += compute_loss_one_gaussian(y, centers, Lambdas)
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

            scores = get_score(y_calibration, centers_calibration, Lambdas_calibration)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal = scores[p]

            return self._nu_conformal
        
    def get_radius_full_information(self):
        return np.sqrt(chi2.ppf(self._nu_conformal.item(), df=self.k))
        

    def get_one_volume(self, x):
        with torch.no_grad():
            
            Lambdas = self.get_Lambdas(x).detach().numpy()

            k = Lambdas.shape[-1]
            
            radius = self.get_radius_full_information()

            volumes = (2*math.gamma(1/2 + 1))**k / math.gamma(k/2 + 1) / np.linalg.det(Lambdas/radius)

            return volumes**(1/k)  # Volume of the ellipsoid, normalized by the radius
        
    def get_averaged_volume(self, x):
        with torch.no_grad():
            tab_volumes = self.get_one_volume(x)
            return np.mean(tab_volumes)


    def get_coverage(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)
            
            scores = get_score(y_test, centers, Lambdas)
            coverage = np.average(scores < self._nu_conformal.item())
            
            return coverage
        
    ##########################################################
    ######### CONFORMALIZATION - WITH IDX KNOWNED ############
    ##########################################################
                
    def conformalize_with_knowned_idx(self, x_calibration=None, y_calibration=None, idx_knowned = None, alpha=None):
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
        idx_knowned : list[int]
            The indices of the knowned dimensions. The default is None.
        alpha : float
            The level of the confidence sets. The default is 0.1.    
        """
        if alpha is None:
            raise ValueError("You need to provide a value for alpha.")
        if idx_knowned is None:
            raise ValueError("You need to provide the indices of knowned indices.")    
        
        self._idx_knowned = idx_knowned
        try:
            idx_unknown = np.setdiff1d(np.arange(self.k), idx_knowned) 
        except:
            centers_calibration = self.get_centers(x_calibration[0])
            self.k = centers_calibration.shape[-1]
            idx_unknown = np.setdiff1d(np.arange(self.k), idx_knowned) 
        self._idx_unknown = idx_unknown
        
        with torch.no_grad():    
            if x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                centers_calibration = self.get_centers(x_calibration)
                Lambdas_calibration = self.get_Lambdas(x_calibration)
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = len(y_calibration)
            p = int(np.ceil((n+1)*(1-alpha)))
            if p > n:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            y_r_calibration = y_calibration[:, self._idx_knowned]
            y_s_calibration = y_calibration[:, self._idx_unknown]

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_knowned_idx(y_r_calibration, centers_calibration, Lambdas_calibration, idx_knowned)
                        
            scores = get_score(y_s_calibration, new_centers, new_Lambdas)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal_conditional = scores[p]
            
            return self._nu_conformal_conditional
     
    def get_volume_condition_on_idx(self, x, y_r):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        with torch.no_grad():
            centers = self.get_centers(x)
            Lambdas = self.get_Lambdas(x)

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_knowned_idx(y_r, centers, Lambdas, self._idx_knowned)
 
            s = len(self._idx_unknown)

            volume = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**s \
                    / torch.lgamma(torch.tensor(s / 2 + 1)).exp() \
                    / torch.det(new_Lambdas / self.nu_conformal_conditional )

            volume[torch.isnan(volume)] = 0

            return volume

    def get_volume_condition_on_idx_without_bayes(self, x, y_r):
        if len(x.shape) == 1:   
            x = x.unsqueeze(0)
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        with torch.no_grad():
            centers = self.get_centers(x)
            Lambdas = self.get_Lambdas(x)

            k = Lambdas.shape[-1]  # Dimension of the full space
            r = len(self.idx_knowned)
            s = k - r  # Dimension of the subspace

            f_x_r = centers[:, self.idx_knowned]  # (n, r)

            residual_r = y_r - f_x_r
            
            # Compute A = M.T @ M
            M = Lambdas @ Lambdas
            
            idx_u = torch.tensor(self.idx_unknown, device=M.device)
            idx_k = torch.tensor(self.idx_knowned, device=M.device)

            M_ss = M[:, idx_u][:, :, idx_u]   # bloc (s,s)
            M_sr = M[:, idx_u][:, :, idx_k]   # bloc (s,r)
            M_rr = M[:, idx_k][:, :, idx_k]   # bloc (r,r)

            # c = M_sr @ residual_r
            c = torch.einsum('nsr,nr->ns', M_sr, residual_r)  # (n, s)

            term1 = self.nu_conformal**2

            term2 = torch.einsum('nr,nr->n', residual_r, torch.einsum('nri, ni->nr', M_rr, residual_r ) )  # (n, r)

            term3 = torch.einsum('ns,ns->n', c, torch.einsum('nsi, ni->ns', torch.linalg.inv(M_ss), c) )  # (n, s)
            
            tau = ( term1 - term2 + term3 )

            tau_reshaped = tau.view(-1, 1, 1)  # explicit reshape for correct broadcasting
            volume = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**s \
                    / torch.lgamma(torch.tensor(s / 2 + 1)).exp() \
                    / torch.sqrt(torch.det(M_ss / tau_reshaped))
            
            # IDX where tau <= 0 : volume = 0
            volume[tau <= 0] = 0.
            volume[torch.isnan(volume)] = 0
            return volume
        
    def get_averaged_volume_condition_on_idx(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx(x, y_r)
            return torch.mean(tab_volumes)
    
    def get_averaged_volume_condition_on_idx_without_bayes(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx_without_bayes(x, y_r)
            return torch.mean(tab_volumes)
        
    def get_coverage_condition_on_idx(self, x_test, y_test):
        with torch.no_grad():
            centers_test = self.get_centers(x_test)
            Lambdas_test = self.get_Lambdas(x_test)

            y_r_test = y_test[:, self._idx_knowned]
            y_s_test = y_test[:, self._idx_unknown]

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_knowned_idx(y_r_test, centers_test, Lambdas_test, self._idx_knowned)
     
            scores = get_score(y_s_test, new_centers, new_Lambdas)

            # Compter le nombre de points où p(y) > tau
            coverage = np.average(scores < self._nu_conformal_conditional.item())
            
            return coverage
        
    def get_point_inside_condition_on_idx(self, x_test, y_test):
        with torch.no_grad():
            centers_test = self.get_centers(x_test)
            Lambdas_test = self.get_Lambdas(x_test)

            y_r_test = y_test[:, self._idx_knowned]
            y_s_test = y_test[:, self._idx_unknown]

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_knowned_idx(y_r_test, centers_test, Lambdas_test, self._idx_knowned)
     
            scores = get_score(y_s_test, new_centers, new_Lambdas)

            # Compter le nombre de points où p(y) > tau
            pt_inside = np.array(scores < self._nu_conformal_conditional.item())
            
            return pt_inside



##########################################################
#############          CLASSIC           #################
##########################################################

def get_density_gaussian(y, centers, Lambdas):
    residuals = y - centers
    Lambda = Lambdas
    det = torch.det(Lambda)
    result = torch.einsum('bij,bj->bi', Lambda, residuals)
    norm_result = torch.exp(- torch.norm(result, dim=1)**2 / 2 ) * det / (2 * np.pi)**(y.shape[1] / 2)
    
    return norm_result


##########################################################
#############        IDX KNOWNED         #################
##########################################################

def get_new_centers_Lambdas_with_knowned_idx(y_r, centers, Lambdas, idx_known):
    k = Lambdas.shape[-1]
    idx_unknown = np.setdiff1d(np.arange(k), idx_known) 

    Sigmas = get_Sigmas_from_Lambdas(Lambdas) # (n_models, n, k, k)
    Sigma_rr = Sigmas[:, idx_known, :][:, :, idx_known]  #  (n, r, r)
    Sigma_ss = Sigmas[:, idx_unknown, :][:, :, idx_unknown]
    Sigma_sr = Sigmas[:, idx_unknown, :][:, :, idx_known]

    f_x = centers
    
    f_x_r = f_x[:, idx_known]   # (n, r)
    f_x_s = f_x[:, idx_unknown]

    Sigma_rr_inv = torch.linalg.inv(Sigma_rr)  # (n, r, r)

    W = torch.einsum('nsr, nru -> nsu', Sigma_sr, Sigma_rr_inv)  # (n, s, r)    
    
    # Residual
    residual = y_r - f_x_r  # (n, r)

    # Correction to the mean
    correction = torch.einsum('nsi,ni->ns', W, residual)

    # Updated mean
    new_mean = f_x_s + correction  # (n, s)

    # Correction to the covariance
    Sigma_sr_T = Sigma_sr.transpose(-1, -2)  # (1, n, r, s)

    correction_cov = torch.einsum('nsr,nri->nsi', W, Sigma_sr_T) # (n, s, s)

    # Updated covariance
    new_Sigmas = Sigma_ss - correction_cov  # (n, s, s)

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
        try :
            Lambdas_squared = torch.linalg.inv(Sigma)
        except:
            k = Sigma.shape[-1]
            regularized = Sigma + 1e-6 * torch.eye(k, device=Sigma.device)
            Lambdas_squared = torch.linalg.inv(regularized)
        eigenvalues, eigenvectors = torch.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = torch.sqrt(eigenvalues)
        Lambdas = eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-1, -2)
        return Lambdas
    # if numpy
    elif isinstance(Sigma, np.ndarray):
        try:
            Lambdas_squared = np.linalg.inv(Sigma)
        except:
            k = Sigma.shape[-1]
            regularized = Sigma + 1e-6 * np.eye(k)
            Lambdas_squared = np.linalg.inv(regularized)
        Lambdas_squared = np.linalg.inv(Sigma)
        eigenvalues, eigenvectors = np.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = np.sqrt(eigenvalues)
        Lambdas = eigenvectors @ np.diag(eigenvalues_sqrt) @ eigenvectors.T
        return Lambdas
    else:
        raise ValueError("The type of Sigma is not recognized.")


