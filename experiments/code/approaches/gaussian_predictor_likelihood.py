import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

def compute_loss_one_gaussian(y, centers, Lambdas):
    norm_result = 0
    Lambdas = Lambdas
    f_x = centers
    logdets = torch.linalg.slogdet(Lambdas)[1]
    residuals = y - f_x
    result = torch.einsum('bij,bj->bi', Lambdas, residuals)
    norm_result = torch.norm(result, dim=1)
    all_loss = - logdets + 1/2 * (norm_result ** 2)
    loss = all_loss.mean()
    return loss


def get_volume_section(Lambdas, known_idx, y_r, f_x_r, radius):
    """
    Get the volume of the section of the ellipsoid defined by the known coordinates.
    
    Parameters:
        Lambdas (numpy.ndarray): A (n , k, k) positive semidefinite matrix.
        known_idx (numpy.ndarray): A vector specifying the fixed Lambdas coordinates.
        y_r (numpy.ndarray): The known coordinates of the points.
        f_x_r (numpy.ndarray): The centers of the ellipsoids at the known coordinates.
        radius (float): The radius of the ellipsoid.

    Returns:
        float: Volume of the section of the ellipsoid.
    """
    k = Lambdas.shape[-1]  # Dimension of the full space
    r = len(known_idx)
    s = k - r  # Dimension of the subspace
    idx_unknown = np.setdiff1d(np.arange(k), known_idx)

    residual_r = y_r - f_x_r
    
    M = Lambdas @ Lambdas
    
    idx_u = torch.tensor(idx_unknown, device=M.device)
    idx_k = torch.tensor(known_idx, device=M.device)

    M_ss = M[:, idx_u][:, :, idx_u]   # bloc (s,s)
    M_sr = M[:, idx_u][:, :, idx_k]   # bloc (s,r)
    M_rr = M[:, idx_k][:, :, idx_k]   # bloc (r,r)

    c = torch.einsum('nsr,nr->ns', M_sr, residual_r)  # (n, s)

    term1 = radius**2

    term2 = torch.einsum('nr,nr->n', residual_r, torch.einsum('nri, ni->nr', M_rr, residual_r ) )  

    term3 = torch.einsum('ns,ns->n', c, torch.einsum('nsi, ni->ns', torch.linalg.inv(M_ss), c) )  
    
    tau = ( term1 - term2 + term3 )

    tau_reshaped = tau.view(-1, 1, 1)  # explicitly reshape for broadcasting
    volume = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**s \
            / torch.lgamma(torch.tensor(s / 2 + 1)).exp() \
            / torch.sqrt(torch.det(M_ss / tau_reshaped))
    
    # IDX where tau <= 0 : volume = 0
    volume[tau <= 0] = 0.

    return volume



class GaussianPredictorLikelihood:
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
         
        Lambdas = self.matrix_model(x)
        
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
            if p < 0:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            scores = - get_density_gaussian(y_calibration, centers_calibration, Lambdas_calibration)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal = scores[p]

            return self._nu_conformal
        
    def get_cover(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)
            scores = - get_density_gaussian(y_test, centers, Lambdas)

            # Compter le nombre de points où p(y) > tau
            cover = scores < self._nu_conformal.item()
            
            return cover*1.0
        
        
    def get_coverage(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)
            scores = - get_density_gaussian(y_test, centers, Lambdas)

            # Compter le nombre de points où p(y) > tau
            coverage = np.average(scores < self._nu_conformal.item())
            
            return coverage
                
    def get_radius(self, x):

        Lambdas = self.get_Lambdas(x)
        k = Lambdas.shape[-1]
        constant = - (2 * np.pi)**( k / 2)
        det = torch.det(Lambdas)
        
        radius_conformal = torch.sqrt( -2 * torch.log( constant / det * self.nu_conformal ) )
        
        return radius_conformal
    
    def get_all_volume(self, x):
        with torch.no_grad():
            Lambdas = self.get_Lambdas(x)
            k = Lambdas.shape[-1]
            
            radius = self.get_radius(x)

            radius = torch.nan_to_num(radius, nan=0.0)

            volumes = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**k \
                    / torch.lgamma(torch.tensor(k / 2 + 1)).exp() \
                    / torch.det( Lambdas / radius[:, None, None] )
            
            volumes = torch.nan_to_num(volumes, nan=0.0)

            return volumes**(1/k)


    def get_averaged_volume(self, x):
        with torch.no_grad():
            tab_volumes = self.get_all_volume(x)
            return torch.mean(tab_volumes)

    ##########################################################
    ######### CONFORMALIZATION - WITH IDX KNOWNED ############
    ##########################################################
        
    def conformalize_with_knowned_idx(self, x_calibration=None, y_calibration=None, idx_knowned = np.array([0]), alpha=None):
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
        idx_knowned : numpy.ndarray
            The indices of the known dimensions. The default is np.array([0]).
        alpha : float, optional
            The level of the confidence sets. The default is 0.1.    
        """
        self._idx_knowned = idx_knowned
        try:
            idx_unknown = np.setdiff1d(np.arange(self.k), idx_knowned) 
        except:
            centers_calibration = self.get_centers(x_calibration[0])
            self.k = centers_calibration.shape[-1]
            idx_unknown = np.setdiff1d(np.arange(self.k), idx_knowned) 
        
        self._idx_unknown = idx_unknown

        if alpha is None:
            raise ValueError("You need to provide a value for alpha.")
        with torch.no_grad():    
            if x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                centers_calibration = self.get_centers(x_calibration)
                Lambdas_calibration = self.get_Lambdas(x_calibration)
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = y_calibration.shape[0]
            p = int(np.ceil((n+1)*(1-alpha))) 
            if p > n:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            scores = - get_conditional_density_estimations_gaussian(y_calibration, centers_calibration, Lambdas_calibration, idx_knowned)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal_conditional = scores[p]
            self._idx_knowned = idx_knowned
            return self._nu_conformal_conditional
        
    def get_radius_with_knowned_idx(self, x, y_r):
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        assert len(x.shape) == 2, "x should be of shape (n, d)"
        assert len(y_r.shape) == 2, "x should be of shape (n, d)"
        
        if self._nu_conformal_conditional is None:
            raise ValueError("You need to call the `conformalize_conditional` method before.")
        
        Lambdas = self.get_Lambdas(x) # (n, k, k)

        Sigmas = get_Sigmas_from_Lambdas(Lambdas) #  (n, k, k)

        Sigma_rr = Sigmas[:, self.idx_knowned, :][:, :, self.idx_knowned]  # ( n, r, r)
        
        Lambda_rr = get_Lambdas_from_Sigmas(Sigma_rr)    # ( n, r, r)
        
        dim_revealed = len(self.idx_knowned) # r
                
        q_alpha = self.nu_conformal_conditional

        f_x = self.get_centers(x)  # (n, k)
        
        f_x_r = f_x[:, self.idx_knowned]   # (n, r)

        k = f_x.shape[-1]

        vec = torch.einsum("nri,ni->nr", Lambda_rr, (y_r - f_x_r) ) # (n , r)
        
        exp = torch.exp(-torch.linalg.norm(vec, ord=2, dim=1) ** 2 / 2)  # (n, )
        
        dets = torch.linalg.det(Lambda_rr) / torch.linalg.det(Lambdas)
        
        log = torch.log(- q_alpha * (2 * torch.pi) ** ((k - dim_revealed) / 2) * dets) - torch.linalg.norm(vec, ord=2, dim=1) ** 2 / 2

        radius = torch.sqrt( - 2 * log  )  # (n, )

        return radius
            
    def get_volume_condition_on_idx(self, x, y_r):
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        if len(x.shape) == 1:   
            x = x.unsqueeze(0)
        with torch.no_grad():
            centers = self.get_centers(x)
            Lambdas = self.get_Lambdas(x)

            f_x_r = centers[:, self.idx_knowned]  # (n, r)

            radius_condition_on_idx = self.get_radius_with_knowned_idx(x, y_r)

            tab_volume = get_volume_section(Lambdas, self.idx_knowned, y_r, f_x_r, radius_condition_on_idx)
            
            tab_volume[torch.isnan(tab_volume)] = 0

            s = centers.shape[-1] - len(self.idx_knowned)  

            return tab_volume**(1/s)
        
    def get_volume_condition_on_idx_without_bayes(self, x, y_r):
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)
        if len(x.shape) == 1:   
            x = x.unsqueeze(0)
        with torch.no_grad():
            centers = self.get_centers(x)
            Lambdas = self.get_Lambdas(x)

            f_x_r = centers[:, self.idx_knowned]  # (n, r)

            radius = self.get_radius(x)

            tab_volume = get_volume_section(Lambdas, self.idx_knowned, y_r, f_x_r, radius)
            
            tab_volume[torch.isnan(tab_volume)] = 0
            
            s = centers.shape[-1] - len(self.idx_knowned)  

            return tab_volume**(1/s)
                
    def get_averaged_volume_condition_on_idx(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx(x, y_r)
            return torch.mean(tab_volumes)
    
    def get_averaged_volume_condition_on_idx_without_bayes(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx_without_bayes(x, y_r)
            return torch.mean(tab_volumes)
        
    def get_cover_condition_on_idx(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)

            scores = - get_conditional_density_estimations_gaussian(y_test, centers, Lambdas, self.idx_knowned)
            cover = scores < self._nu_conformal_conditional.item()
            
            return cover*1.0
        

    def get_coverage_condition_on_idx(self, x_test, y_test):
        with torch.no_grad():
            centers = self.get_centers(x_test)
            Lambdas = self.get_Lambdas(x_test)

            scores = - get_conditional_density_estimations_gaussian(y_test, centers, Lambdas, self.idx_knowned)
            coverage = np.average(scores < self._nu_conformal_conditional.item())
            
            return coverage
        
    ##########################################################
    ########### CONFORMALIZATION - PROJECTION ################
    ##########################################################

    def conformalize_projection(self, projection_idx, x_calibration=None, y_calibration=None, alpha=0.1):
        if alpha is None:
            raise ValueError("You need to provide a value for alpha.")
        
        self._projection_idx = projection_idx

        projection_matrix = torch.zeros((len(projection_idx), self.k), device=self.dtype)
        for i, idx in enumerate(projection_idx):
            projection_matrix[i, idx] = 1

        self.conformalize_linear_projection(projection_matrix, x_calibration=x_calibration, y_calibration=y_calibration, alpha=alpha)


    def conformalize_linear_projection(self, projection_matrix, x_calibration=None, y_calibration=None, alpha=0.1):
        if alpha is None:
            raise ValueError("You need to provide a value for alpha.")
        
        self._projection_matrix = projection_matrix
        
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

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers_calibration, Lambdas_calibration, self._projection_matrix)
            y_projection = torch.einsum('rk,nk->nr', self._projection_matrix, y_calibration)

            scores = - get_density_gaussian(y_projection, new_centers, new_Lambdas)
            scores = torch.sort(scores, descending=False).values
            self._nu_conformal_projection = scores[p]
            
            return self._nu_conformal_projection
        
    def get_radius_given_Lambdas(self, Lambdas, nu):

        k = Lambdas.shape[-1]
        constant = - (2 * np.pi)**( k / 2)
        det = torch.det(Lambdas)
        
        radius_conformal = torch.sqrt( -2 * torch.log( constant / det * nu ) )
        
        return radius_conformal
    
    def get_radius_projection(self, x):
        Lambdas = self.get_Lambdas(x)
        centers = self.get_centers(x)

        _, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, self._projection_matrix)

        k = new_Lambdas.shape[-1]
        constant = - (2 * np.pi)**( k / 2)
        det = torch.det(new_Lambdas)
        
        radius_conformal = torch.sqrt( -2 * torch.log( constant / det * self._nu_conformal_projection ) )
        
        return radius_conformal

    def get_volume_projection(self, x):
        with torch.no_grad():
            Lambdas = self.get_Lambdas(x)
            centers = self.get_centers(x)

            _, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, self._projection_matrix)

            k_new = new_Lambdas.shape[-1]

            radius = self.get_radius_given_Lambdas(new_Lambdas, self._nu_conformal_projection)

            radius = torch.nan_to_num(radius, nan=0.0)

            volumes = (2 * torch.lgamma(torch.tensor(1/2 + 1)).exp())**k_new \
                    / torch.lgamma(torch.tensor(k_new / 2 + 1)).exp() \
                    / torch.det(new_Lambdas / radius[:, None, None] )
            
            volumes = torch.nan_to_num(volumes, nan=0.0)

            return volumes**(1/k_new)
        
    def get_averaged_volume_projection(self, x):
        with torch.no_grad():
            tab_volumes = self.get_volume_projection(x)
            return torch.mean(tab_volumes)
        
    def get_coverage_projection(self, x_test, y_test):
        with torch.no_grad():    
            if x_test is not None and x_test is not None:
                # Case where we directly give tensors
                centers_test = self.get_centers(x_test)
                Lambdas_test = self.get_Lambdas(x_test)
            else:
                raise ValueError("You need to provide a `testloader`, or `x_test` and `y_test`.")

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers_test, Lambdas_test, self._projection_matrix)
            y_projection = torch.einsum('rk,nk->nr', self._projection_matrix, y_test)

            scores = - get_density_gaussian(y_projection, new_centers, new_Lambdas)

            # Compter le nombre de points où p(y) > tau
            coverage = np.average(scores < self._nu_conformal_projection.item())
            
            return coverage

    def get_cover_projection(self, x_test, y_test):
        with torch.no_grad():    
            if x_test is not None and x_test is not None:
                # Case where we directly give tensors
                centers_test = self.get_centers(x_test)
                Lambdas_test = self.get_Lambdas(x_test)
            else:
                raise ValueError("You need to provide a `testloader`, or `x_test` and `y_test`.")

            new_centers, new_Lambdas = get_new_centers_Lambdas_with_linear_projection(centers_test, Lambdas_test, self._projection_matrix)
            y_projection = torch.einsum('rk,nk->nr', self._projection_matrix, y_test)

            scores = - get_density_gaussian(y_projection, new_centers, new_Lambdas)

            # Compter le nombre de points où p(y) > tau
            cover = scores < self._nu_conformal_projection.item()
            
            return cover*1.0



def get_density_gaussian(y, centers, Lambdas):
    norm_result = 0
    residuals = y - centers
    det = torch.det(Lambdas)
    result = torch.einsum('bij,bj->bi', Lambdas, residuals)
    norm_result = torch.exp(- torch.norm(result, dim=1)**2 / 2 ) * det / (2 * np.pi)**(y.shape[1] / 2)
    return norm_result

def get_marginal_density_gaussian(y, centers, Lambdas, idx):
    y_knowned = y[:, idx]
    centers_knowned = centers[:, idx]
    Sigma = get_Sigmas_from_Lambdas(Lambdas)
    Sigma_knowned = Sigma[:, idx][:, :, idx]
    Lambdas_knowned = get_Lambdas_from_Sigmas(Sigma_knowned)

    return get_density_gaussian(y_knowned, centers_knowned, Lambdas_knowned)

def get_conditional_density_estimations_gaussian(y, centers, Lambdas, idx_knowned):    
    joint_density = get_density_gaussian(y, centers, Lambdas)
    marginal_density = get_marginal_density_gaussian(y, centers, Lambdas, idx_knowned)
    conditional_density = joint_density / marginal_density
    return conditional_density

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
        try:
            Lambdas_squared = torch.linalg.inv(Sigma)
        except:
            Lambdas_squared = torch.linalg.inv(Sigma + 1e-6 * torch.eye(Sigma.shape[-1], device=Sigma.device))
        eigenvalues, eigenvectors = torch.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = torch.sqrt(eigenvalues)
        Lambdas = eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-1, -2)
        return Lambdas
    # if numpy
    elif isinstance(Sigma, np.ndarray):
        try:
            Lambdas_squared = np.linalg.inv(Sigma)
        except:
            Lambdas_squared = np.linalg.inv(Sigma + 1e-6 * np.eye(Sigma.shape[-1]))
        eigenvalues, eigenvectors = np.linalg.eigh(Lambdas_squared)
        eigenvalues_sqrt = np.sqrt(eigenvalues)
        Lambdas = eigenvectors @ np.diag(eigenvalues_sqrt) @ eigenvectors.T
        return Lambdas
    else:
        raise ValueError("The type of Sigma is not recognized.")


def get_new_centers_Lambdas_with_linear_projection(centers, Lambdas, projection_matrice):
    new_mean = torch.einsum('rk,nk->nr', projection_matrice, centers)  # (n, r)
    Sigmas = get_Sigmas_from_Lambdas(Lambdas) # (n, k, k)
    new_Sigmas = torch.einsum('ri,nij,sj->nrs', projection_matrice, Sigmas, projection_matrice)
    new_Lambdas = get_Lambdas_from_Sigmas(new_Sigmas)
    return new_mean, new_Lambdas