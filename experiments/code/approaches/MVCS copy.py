import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cvxpy as cp
import math



def sample_sphere(n_points, dim):
    """
    Sample points uniformly on the surface of a sphere in d dimensions.
    
    Parameters
    ----------
    n_points : int
        The number of points to sample.
    dim : int
        The dimension of the sphere.
    
    Returns
    -------
    points : np.ndarray
        An array of shape (n_points, dim) containing the sampled points.
    """
    points = np.random.randn(n_points, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

class MVCSPredictor:
    def __init__(self, center_model, matrix_model, q=torch.tensor(2.0, requires_grad=True), dtype=torch.float32):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model that represents the center of the sets.
        matrix_model : torch.nn.Module
            The model that represents the matrix A of the ellipsoids. The matrix Lambda is obtained as the product of A by its transpose.
        q : torch.Tensor, optional
            The q parameter of the q-norm. The default is torch.tensor(2.0, requires_grad=True).
        dtype : torch.dtype, optional
            The data type of the tensors. The default is torch.float32.
        """

        self.center_model = center_model
        self.matrix_model = matrix_model
        self.q = q
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
    
    @property
    def nu_conformal(self):
        return self._nu_conformal

    def fit(self,
            trainloader,
            stoploader,
            alpha,
            num_epochs=1000,
            num_epochs_mat_only=50,
            lr_model=0.001,
            lr_q=0.001,
            lr_matrix_model=0.001,
            use_lr_scheduler=True,
            verbose=0,
            stop_on_best=True,
            loss_strategy="log_volume",
            use_epsilon=False
        ):
        """"
        Parameters
        
        trainloader : torch.utils.data.DataLoader
            The DataLoader of the training set.
        stoploader : torch.utils.data.DataLoader
            The DataLoader of the validation set : Warning: do not use the calibration set as a stopping criterion as you would lose the coverage property.
        alpha : float
            The level of the confidence sets.
        num_epochs : int, optional
            The total number of epochs. The default is 1000.
        num_epochs_mat_only : int, optional
            The number of epochs where only the matrix model is trained. The default is 50.
        lr_model : float, optional
            The learning rate for the model. The default is 0.001.
        lr_q : float, optional
            The learning rate for the q parameter. The default is 0.001.
        lr_matrix_model : float, optional
            The learning rate for the matrix model. The default is 0.001.
        use_lr_scheduler : bool, optional
            Whether to use a learning rate scheduler. The default is False.
        verbose : int, optional
            The verbosity level. The default is 0.
        stop_on_best : bool, optional   
            Whether to stop on the best model. The default is False.
        loss_strategy : str, optional
            The strategy to compute the loss. The default is "exact_volume".
        use_epsilon : bool, optional
            Whether to use the epsilon parameter. The default is False.
            """
        
        if stop_on_best:
            self.best_stop_loss = np.inf
            self.best_model_weight = copy.deepcopy(self.center_model.state_dict())
            self.best_lambdas_weight = copy.deepcopy(self.matrix_model.state_dict())
            self.best_q = self.q.item()

        self.alpha = alpha

        optimizer = torch.optim.Adam([
            {'params': self.center_model.parameters(), 'lr': lr_model},  # Learning rate for self.center_model
            {'params': self.matrix_model.parameters(), 'lr': lr_matrix_model},  # Learning rate for self.matrix_model
            {'params': self.q, 'lr': lr_q}  # Learning rate for q
        ])

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.tab_train_loss = []
        self.tab_stop_loss = []

        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1

        last_model_weight = self.center_model.state_dict()
        last_lambdas_weight = self.matrix_model.state_dict()
        last_q = self.q.item()

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            for x, y in trainloader:    
                optimizer.zero_grad()

                Lambdas = self.get_Lambdas(x)
    
                if epoch < num_epochs_mat_only:
                    with torch.no_grad():
                        f_x = self.center_model(x)
                else:
                    f_x = self.center_model(x)


                loss = compute_loss_MVCS(y, f_x, Lambdas, self.q, alpha, loss_strategy=loss_strategy, use_epsilon=use_epsilon)

                if torch.isnan(loss):
                    self.center_model.load_state_dict(last_model_weight)
                    self.matrix_model.load_state_dict(last_lambdas_weight)
                    self.q = torch.tensor(last_q, requires_grad=True)
                    break

                loss.backward()
                optimizer.step()
                epoch_loss += loss

                last_model_weight = copy.deepcopy(self.center_model.state_dict())
                last_lambdas_weight = copy.deepcopy(self.matrix_model.state_dict())
                last_q = self.q.item()

            epoch_loss = self.eval(trainloader, loss_strategy)
            self.tab_train_loss.append(epoch_loss.item())

            epoch_stop_loss = self.eval(stoploader, loss_strategy)
            self.tab_stop_loss.append(epoch_stop_loss.item())

            if stop_on_best and self.best_stop_loss > epoch_stop_loss.item():
                if verbose == 2:
                    print(f"New best stop loss: {epoch_stop_loss.item()}")
                self.best_stop_loss = epoch_stop_loss
                self.best_model_weight = copy.deepcopy(self.center_model.state_dict())
                self.best_lambdas_weight = copy.deepcopy(self.matrix_model.state_dict())
                self.best_q = self.q.item()
                self.load_best_model()
                            
                
            if verbose != 0:
                if epoch % print_every == 0:
                    print(f"Epoch {epoch}: Loss = {epoch_loss.item()} - Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")

            if use_lr_scheduler:
                scheduler.step()

        epoch_loss = self.eval(trainloader, loss_strategy)
        epoch_stop_loss = self.eval(stoploader, loss_strategy)
        if stop_on_best:
            self.load_best_model()
        epoch_loss = self.eval(trainloader, loss_strategy)
        epoch_stop_loss = self.eval(stoploader, loss_strategy)
        if verbose != 0:
            print(f"Final Loss = {epoch_loss.item()} - Final Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")

    def load_best_model(self):
        """
        Load the best model.    
        """
        if self.best_model_weight is not None:
            self.center_model.load_state_dict(self.best_model_weight)
            self.matrix_model.load_state_dict(self.best_lambdas_weight)
            self.q = torch.tensor(self.best_q, requires_grad=True)
        else:
            raise ValueError("You must call the `fit` method with the `stop_on_best` parameter set to True.")

    def eval(self,
             dataloader, loss_strategy):
        """"
        Evaluate the loss on a given DataLoader.
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader of the dataset on which to evaluate the loss.
        loss_strategy : str
            The strategy to compute the loss.
        """
        
        with torch.no_grad():
            loss = 0.0
            empty = True
            for x, y in dataloader:
                Lambdas = self.get_Lambdas(x)
                f_x = self.center_model( x )

                if empty:
                    f_x_eval = f_x
                    y_eval = y
                    Lambdas_eval = Lambdas
                    empty = False

                else:
                    Lambdas_eval = torch.cat((Lambdas_eval, Lambdas), dim=0)
                    f_x_eval = torch.cat((f_x_eval, f_x), 0)
                    y_eval = torch.cat((y_eval, y), 0)

            loss = compute_loss_MVCS(y_eval, f_x_eval, Lambdas_eval, self.q, self.alpha, loss_strategy = loss_strategy)
            return loss
    
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
                    f_x = self.center_model(x)
                    Lambdas = self.get_Lambdas(x)

                    if empty:
                        f_x_calibration = f_x
                        y_calibration = y
                        Lambdas_calibration = Lambdas
                        empty = False
                    else:
                        Lambdas_calibration = torch.cat((Lambdas_calibration, Lambdas), dim=0)
                        f_x_calibration = torch.cat((f_x_calibration, f_x), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                f_x_calibration = self.center_model(x_calibration)
                Lambdas_calibration = self.get_Lambdas(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = y_calibration.shape[0]
            p = n - int(np.ceil((n+1)*(1-alpha))) - 1
            if p < 0:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            self._nu_conformal = get_p_value_mu(y_calibration, f_x_calibration, Lambdas_calibration, p, self.q)

    def get_volumes(self, testloader=None, x_test=None):
        """
        Compute the volumes of the confidence sets on a given dataset.

        Parameters
        ----------
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None. The shape is (n, d).

        Returns
        -------
        volumes : torch.Tensor
            The volumes of each confidence sets. The shape is (n,).
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")

            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, _ in testloader:
                    Lambdas = self.get_Lambdas(x)
                    if empty:
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                    
            elif x_test is not None:
                # Case where we directly give tensors
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You need to either provide a `testloader`, or `x_test`.")
            
            return calculate_all_volumes(Lambdas_test, self._nu_conformal, self.q)

    def get_averaged_volume(self, testloader=None, x_test=None):
        """
        Compute the averaged volume of the confidence sets on a given dataset.

        Parameters
        ----------
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None.

        Returns
        -------
        volume : torch.Tensor
            The averaged volume of the confidence sets.
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, _ in testloader:
                    Lambdas = self.get_Lambdas(x)
                    if empty:
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                    
            elif x_test is not None:
                # Case where we directly give tensors
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You need to either provide a `testloader`, or `x_test`.")
            
            return calculate_averaged_volume(Lambdas_test, self._nu_conformal, self.q)
    
    
    def get_coverage(self, x_test=None, y_test=None, testloader=None):  
        """
        Compute the coverage of the confidence sets on a given dataset.
        
        Parameters
        ----------
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None. The shape is (n, d).
        y_test : torch.Tensor, optional
            The output tensor of the test set. The default is None. The shape is (n, k).
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.

        returns
        -------
        coverage : float
            The coverage of the confidence sets (between 0 and 1).
        """
        with torch.no_grad(): 
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")    
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in testloader:
                    f_x = self.center_model(x)
                    Lambdas = self.get_Lambdas(x)

                    if empty:
                        f_x_test = f_x
                        y_test = y
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                        f_x_test = torch.cat((f_x_test, f_x), 0)
                        y_test = torch.cat((y_test, y), 0)

            elif x_test is not None and y_test is not None:
                # Case where x_test and y_test are given as tensors
                f_x_test = self.center_model(x_test)
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You must provide either `testloader` or both `x_test` and `y_test`.")
            
            coverage = calculate_coverage_MVCS(y_test, f_x_test, Lambdas_test, self._nu_conformal, self.q).mean().item()
                
            return coverage
        
    def get_cover(self, x, y):
        return self.is_inside(x, y)*1.0

    
    def is_inside(self, x, y):
        """"
            Check if a given point is inside the confidence set.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor of shape (n, d)
            y : torch.Tensor  
                The output tensor of shape (n, k)

            Returns
            ------- 
            is_inside : bool
                Whether the point is inside the confidence set or not.
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")

            f_x = self.center_model(x)
            Lambdas = self.get_Lambdas(x)
            
            norm_values = torch.tensor([calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), self.q) for i in range(len(y))])  # (n,)
        
            return norm_values <= self.nu_conformal
        
    ##########################################################
    #############        PROJECTION          #################
    ##########################################################

    def conformalize_linear_projection(self, projection_matrix):
            """
            Compute the quantile value to conformalize the ellipsoids on a unseen dataset.

            Parameters
            ----------
            """
            self.projection_matrix = projection_matrix

    def get_volume_projection(self, x_test, n_samples = 10000):
        raise Exception("This method is not implemented yet.")
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)

        with torch.no_grad():
            sphere = sample_sphere(n_samples, x_test.shape[1])
            sphere = torch.tensor(sphere, dtype=x_test.dtype, device=x_test.device) 
            scaling = self.nu_conformal * torch.pow(torch.sum(torch.abs(sphere) ** self.q, dim=1), -1 / self.q)
            scaled_points = sphere * scaling.unsqueeze(1)  # (n_samples, d)

            tab_volume = torch.zeros(x_test.shape[0])
            
            for i in range(x_test.shape[0]):
                center = self.center_model(x_test[i:i+1])[0]
                Lambdas = self.get_Lambdas(x_test[i:i+1])[0]
                Lambdas_inv = torch.linalg.inv(Lambdas)

                transformed_points = torch.einsum('ij, kj -> ki', Lambdas_inv, scaled_points)  # (n_samples, d)
                
                ellipsoid_points = transformed_points + center

                m = np.array( torch.min(ellipsoid_points, dim=0).values )
                M = np.array( torch.max(ellipsoid_points, dim=0).values )

                v = m + np.random.random((n_samples, ellipsoid_points.shape[-1]))*(M-m) 

                scale = np.prod(M-m) 

                v = torch.tensor(v, dtype=x_test.dtype, device=x_test.device)
                
                ctp = 0
                for j in range(n_samples):
                    val = calculate_norm_q(Lambdas @ (v[j] - center), self.q).item()
                    if val < self.nu_conformal.item():
                        ctp += 1
                    else:
                        z = self.projection_matrix @ v[j]
                        _, norm = solve_weighted_projection(Lambdas, center, self.projection_matrix, z, p=self.q)
                        if norm < self.nu_conformal.item():
                            ctp += 1

                MCMC = ctp / n_samples

                tab_volume[i] = scale * MCMC

        return tab_volume
        
    def get_averaged_volume_projection(self, x):
        with torch.no_grad():
            tab_volumes = self.get_volume_projection(x)
            return torch.mean(tab_volumes)


    def get_coverage_projection(self, x_test, y_test):
        Lambdas = self.get_Lambdas(x_test)
        f_x = self.center_model(x_test)
        
        ctp = 0
        for i in range(y_test.shape[0]):
            val = calculate_norm_q(Lambdas[i] @ (y_test[i] - f_x[i]), self.q).item()
            if val < self.nu_conformal.item():
                ctp += 1
            else:
                z = self.projection_matrix @ y_test[i]
                _, norm = solve_weighted_projection(Lambdas[i], f_x[i], self.projection_matrix, z, p=self.q)
                if norm < self.nu_conformal.item():
                    ctp += 1

        return ctp/len(y_test)



    ##########################################################
    #############        IDX KNOWNED         #################
    ##########################################################
    
    def conformalize_with_knowned_idx(self, idx_knowned):
        self.idx_knowned = idx_knowned 


    def get_averaged_volume_with_knowned_idx(self, x_test, y_r, n_samples=10000):
        with torch.no_grad():
            tab_volumes = self.get_volume_condition_on_idx_without_bayes(x_test, y_r, n_samples)
            return torch.mean(tab_volumes)
        
    def get_volume_condition_on_idx_without_bayes(self, x_test, y_r, n_samples=10000):
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        if len(y_r.shape) == 1:
            y_r = y_r.unsqueeze(0)

        k = self.center_model(x_test[[0]]).shape[-1]
        with torch.no_grad():

            sphere = sample_sphere(n_samples, k)
            sphere = torch.tensor(sphere, dtype=x_test.dtype, device=x_test.device) 
            scaling = self.nu_conformal * torch.pow(torch.sum(torch.abs(sphere) ** self.q, dim=1), -1 / self.q)
            scaled_points = sphere * scaling.unsqueeze(1)  # (n_samples, d)

            tab_volume = torch.zeros(x_test.shape[0])
            
            for i in range(x_test.shape[0]):
                center = self.center_model(x_test[i:i+1])[0]
                Lambdas = self.get_Lambdas(x_test[i:i+1])[0]
                Lambdas_inv = torch.linalg.inv(Lambdas)

                transformed_points = torch.einsum('ij, kj -> ki', Lambdas_inv, scaled_points)  # (n_samples, d)
                ellipsoid_points = transformed_points + center

                m = np.array( torch.min(ellipsoid_points, dim=0).values )
                M = np.array( torch.max(ellipsoid_points, dim=0).values )
                
                idx_unknown = np.setdiff1d(np.arange(center.shape[-1]), self.idx_knowned)
                v = m + np.random.random((n_samples, center.shape[-1]))*(M-m) 
                v[:, self.idx_knowned] = y_r[i]
                v = torch.tensor(v, dtype=x_test.dtype, device=x_test.device)
                scale = np.prod(M[idx_unknown]-m[idx_unknown]) 
                norm_values = torch.tensor([calculate_norm_q(Lambdas @ (v[i] - center), self.q) for i in range(len(v))])  # (n,)
                MCMC = np.mean( norm_values.detach().cpu().numpy() <= self.nu_conformal.item() ) 
                tab_volume[i] = scale * MCMC
                
            s = center.shape[-1] - len(self.idx_knowned)  

            return tab_volume**(1/s)
            






import torch
import torch.nn as nn
import numpy as np

def get_p_indice_mu(y, f_x, Lambdas, alpha, q):
    """
        Get the p-th indice of the \|Lambda(y - f(x))\|_q for a coverage 1-alpha.

        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            alpha (float): Coverage proportion.
            q (float): Order of the norm.

        Returns:
            (int): The p-th indice of the residuals.
    """
    p = int(alpha * y.shape[0])
    values = []

    for i in range(y.shape[0]):
        # val = torch.linalg.norm(Lambda @ (y[i] - mu), ord=q).item()
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return p_indice

def get_alpha_value_mu(y, f_x, Lambdas, alpha, q):
    """
        Get the p-th value of the \|Lambda(y - f(x))\|_q for a coverage 1-alpha.
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            p (int): Proportion of the dataset to consider.
            q (float): Order of the norm.
            
        Returns:
            (float): The p-th value of the residuals.   
    """
    p = int(alpha * y.shape[0])
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]

def get_p_value_mu(y, f_x, Lambdas, p, q):
    """
        Get the p-th value of the \|Lambda(y - f(x))\|_q for a coverage with n - p + 1 points .
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            p (int): Proportion of the dataset to consider.
            q (float): Order of the norm.
            
        Returns:
            (float): The p-th value of the residuals.   
    """
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]

def H(q, k):
    """
    Computes the function H(q, k) where q is a PyTorch tensor. It is the log of the volume of the unit ball in q-norm.
    
    Parameters:
        q (torch.Tensor): Input tensor for q.
        k (float or torch.Tensor): Scalar or tensor for k.
    
    Returns:
        torch.Tensor: Result of the computation.
    """
    term1 = k * torch.special.gammaln(1 + 1 / q)
    term2 = torch.special.gammaln(1 + k / q)
    term3 = k * torch.log(torch.tensor(2.0))
    return term1 - term2 + term3

def calculate_norm_q(z, q):
    """
    Calculates the q-norm of a vector z manually in PyTorch.
    
    Parameters:
        z (torch.Tensor): Input vector (1D tensor).
        q (torch.Tensor): Order of the norm (q > 0).
    
    Returns:
        torch.Tensor: The q-norm of the input vector.
    """
    if q <= 0:
        raise ValueError("The order of the norm (q) must be greater than 0.")
    
    # Compute the absolute values of the vector elements raised to the power q
    abs_powers = torch.abs(z) ** q
    
    # Sum up the values
    sum_abs_powers = torch.sum(abs_powers)
    
    # Take the q-th root
    norm_q = sum_abs_powers ** (1 / q)
    
    return norm_q

def calculate_averaged_volume(Lambdas, nu, q):
    """
        Calculates the average volumes of the ellipsoids in the batch.

        Parameters:
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            (float): The average volume of the ellipsoids.
    """
    all_volumes = calculate_all_volumes(Lambdas, nu, q)

    return all_volumes.mean().item()

def calculate_all_volumes(Lambdas, nu, q):
    """
        Calculates the volumes of the ellipsoids for each sample in the batch.

        Parameters:
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            torch.Tensor: The volumes of the ellipsoids for each sample in the batch - (BatchSize , ).
    """
    k = Lambdas.shape[1]
    _, logdet = torch.linalg.slogdet(Lambdas)
    volumes = - logdet + k * torch.log(nu) + H(q, k)
    
    return torch.exp(volumes)**(1/k)  # Return the k-th root of the volume to get the average volume per sample

def compute_loss_MVCS(y, f_x, Lambdas, q, alpha, loss_strategy="log_volume", use_epsilon=False):
    """
        Computes the Adaptative MVCS loss function.
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            q (float): Order of the norm.
            alpha (float): Proportion of the dataset to consider.
            loss_strategy (str): Strategy to compute the loss.
            use_epsilon (bool): Whether to add epsilon to the determinant.

        Returns:
            torch.Tensor: The loss value
    """
    k = y.shape[1]
    with torch.no_grad():
        idx_p = get_p_indice_mu(y, f_x, Lambdas, alpha, q)
    if loss_strategy == "exact_volume":
        det = torch.linalg.det(Lambdas)
        if use_epsilon:
            det = det + 1e-8
        # loss =  (1 / det).mean() * calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q) ** k * torch.exp(H(q, k))
        loss = (1 / det).mean() * torch.exp( k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k) )
    elif loss_strategy == "log_volume": 
        det = torch.linalg.det(Lambdas)
        if use_epsilon:
            det = det + 1e-8
        loss = torch.log( (1 / det).mean() ) + k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k)
    else:
        raise ValueError("The strategy must be either 'exact_volume' or 'log'.")
    return loss

def calculate_coverage_MVCS(y, f_x, Lambdas, nu, q):
    """
        Calculates the coverage of the ellipsoids for each sample in the batch.

        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            torch.Tensor: The coverage of the ellipsoids.
    """
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    count = torch.sum(values < nu)

    return count/len(values)



def solve_weighted_projection(Lambda, fx, P, z, p=2):
    """
    Solves: min_y || Lambda @ (y - fx) ||_p s.t. P @ y == z

    Args:
        Lambda: (k, k) PSD matrix
        fx: (k,) vector
        P: (d, k) matrix
        z: (d,) vector
        p: norm degree (e.g. 1, 2, 'inf')

    Returns:
        y_opt: Optimal solution vector y
    """
    Lambda = Lambda.detach().cpu().numpy()
    fx = fx.detach().cpu().numpy()
    P = P.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    p = p.item()

    k = fx.shape[0]
    y = cp.Variable(k)

    diff = Lambda @ (y - fx)
    objective = cp.norm(diff, p)
    constraints = [P @ y == z]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization failed with status: {prob.status}")

    return y.value, prob.value
    
class SimpleTabularMVCS(nn.Module):
    """
    Robust Backbone: ResNet-MLP for Tabular Data
    """
    def __init__(self, num_cont, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.first_layer = nn.Linear(num_cont, hidden_dim)
        
        # Residual Blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = x + block(x)  # Skip connection
        return self.norm(x)



class RobustCovarianceHeadMVCS(nn.Module):
    def __init__(self, input_dim, y_dim, init_sigma=1.0, init_p=2.0):
        super().__init__()
        self.y_dim = y_dim
        
        # --- Trainable p (Shape parameter) ---
        # We constrain p > 1 using Softplus: p = 1 + softplus(p_raw)
        # Inverse softplus initialization: log(exp(target) - 1)
        target_val = max(init_p - 1.0, 1e-4)
        init_raw = math.log(math.exp(target_val) - 1)
        self.p_raw = nn.Parameter(torch.tensor(init_raw))

        # --- Covariance Parameters ---
        if y_dim > 10:
            self.mode = 'low_rank'
            rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_mu = nn.Linear(input_dim, y_dim)
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * rank)
            self.rank = rank
        else:
            self.mode = 'full_cholesky'
            self.fc_mu = nn.Linear(input_dim, y_dim)
            num_chol = (y_dim * (y_dim + 1)) // 2
            self.fc_chol = nn.Linear(input_dim, num_chol)
            self.register_buffer('tril_indices', torch.tril_indices(y_dim, y_dim))
            
            with torch.no_grad():
                diag_mask = (self.tril_indices[0] == self.tril_indices[1])
                inv_softplus = math.log(math.exp(init_sigma) - 1)
                self.fc_chol.bias[diag_mask] = inv_softplus

    @property
    def p(self):
        """Returns the constrained p value (p > 1)"""
        return self.p_raw

    def forward(self, x):
        if self.mode == 'low_rank':
            B = x.shape[0]
            mu = self.fc_mu(x)
            D = torch.exp(self.fc_log_diag(x)) + 1e-6
            V = self.fc_factors(x).view(B, self.y_dim, self.rank)
            return (mu, D, V)
        else:
            B = x.shape[0]
            mu = self.fc_mu(x)
            chol_flat = self.fc_chol(x)
            L = torch.zeros(B, self.y_dim, self.y_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = chol_flat
            diag_idx = torch.arange(self.y_dim, device=x.device)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
            return (mu, L)

    def _log_lambda_constant(self, k, p):
        """
        Calculates log lambda(B ||.||p (1)) with differentiable p.
        """
        # Ensure k is a tensor for consistency, though usually fixed
        k_t = torch.tensor(k, device=p.device, dtype=p.dtype)
        
        # All operations here support autograd through p
        term1 = k_t * torch.special.gammaln(1 + 1 / p)
        term2 = torch.special.gammaln(1 + k_t / p)
        term3 = k_t * torch.log(torch.tensor(2.0, device=p.device))
        
        return term1 - term2 + term3

    def loss(self, params, y_target, alpha):
        """
        Args:
            params: Output from forward()
            y_target: Ground truth
            r (int): Rank statistic index. Defaults to Median.
        """
        B = y_target.shape[0]
        r = int(alpha * y_target.shape[0])
        
        k = self.y_dim
        current_p = self.p  # This is a Tensor

        if self.mode == 'low_rank':
            mu, D, V = params
            residual = y_target - mu
            
            # --- 1. Log Determinant (Low Rank) ---
            inv_D = 1.0 / D
            W = V * torch.sqrt(inv_D).unsqueeze(-1)
            M = torch.eye(self.rank, device=V.device).unsqueeze(0) + torch.bmm(W.transpose(1, 2), W)
            L_M = torch.linalg.cholesky(M)
            
            log_det_sigma = torch.sum(torch.log(D), dim=1) + 2 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), dim=1)
            
            # --- 2. Whitened Residuals ---
            Sigma = torch.diag_embed(D) + torch.bmm(V, V.transpose(1, 2))
            L_full = torch.linalg.cholesky(Sigma)
            z = torch.linalg.solve_triangular(L_full, residual.unsqueeze(-1), upper=False).squeeze(-1)
            
            # FIX: Manual p-norm for learnable p
            # whitened_norm = torch.norm(z, p=current_p, dim=1) <-- CAUSES ERROR
            whitened_norm = (torch.abs(z).pow(current_p).sum(dim=1)).pow(1.0 / current_p)

        else:
            # --- Full Cholesky Mode ---
            mu, L = params
            residual = (y_target - mu).unsqueeze(-1)
            
            log_det_sigma = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)

            # print("residual", residual)
            # print("L", L)
            
            z = torch.linalg.solve_triangular(L, residual, upper=False).squeeze(-1)
            
            # FIX: Manual p-norm for learnable p
            # whitened_norm = torch.norm(z, p=current_p, dim=1) <-- CAUSES ERROR

            # print("z",z)
            
            whitened_norm = (torch.abs(z).pow(current_p).sum(dim=1)).pow(1.0 / current_p)

        # --- 3. Term 1 ---
        log_det_lambda = -0.5 * log_det_sigma
        term1 = torch.logsumexp(log_det_lambda, dim=0) 
        
        # --- 4. Term 2 ---
        # Select the top r values
        sorted_norms, sorted_idx = torch.sort(whitened_norm, descending=True)

        idx_r = sorted_idx[r - 1]      # index in original whitened_norm
        sigma_r = whitened_norm[idx_r] # value at that index

        term2 = k * torch.log(sigma_r + 1e-8)

        # --- 5. Term 3 ---
        term3 = self._log_lambda_constant(k, current_p)

        # print("term 1", term1)
        # print("term 2", term2)
        # print("term 3", term3)

        return term1 + term2 + term3
    


class CenterPredictorMVCS:
    def __init__(self, head, backbone):
        self.head = head
        self.backbone = backbone

    def __call__(self, x):
        self.backbone.eval()
        self.head.eval()
        with torch.no_grad():
            feats = self.backbone(x)
            params = self.head(feats)
            return params[0]  # Return mu

class MatrixPredictorMVCS:
    def __init__(self, head, backbone):
        self.head = head
        self.backbone = backbone

    def __call__(self, x):
        self.backbone.eval()
        self.head.eval()
        with torch.no_grad():
            feats = self.backbone(x)
            params = self.head(feats)
            # Return full covariance reconstruction
            if self.head.mode == 'low_rank':
                mu, D, V = params
                Sigma = torch.diag_embed(D) + torch.bmm(V, V.transpose(1, 2))
                return Sigma
            else:
                mu, L = params
                return torch.bmm(L, L.transpose(1, 2))

# --- Main Wrapper Class ---
class MVCSRobustTabularModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.backbone = SimpleTabularMVCS(input_dim, hidden_dim, num_layers, dropout)
        # Initialize head with trainable p (defaulting to ~2.0)
        self.head = RobustCovarianceHeadMVCS(hidden_dim, output_dim, init_p=2.0)
        
        # Placeholders for inference models
        self.center_model = None
        self.matrix_model = None

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def fit(self,
            trainloader,
            stoploader,
            alpha=0.1,           # Rank statistic index (Hyperparameter)
            num_epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            verbose=-1
            ):
        """
        Args:
            r (int): The rank index for the robust loss (e.g., int(0.9 * batch_size)).
                     If None, defaults to Median (Batch // 2).
            NaN_Values (bool): If True, uses loss_with_missing handling NaNs in targets.
        """
        
        optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        best_stop_loss = float("inf")
        best_backbone_state = None
        best_head_state = None

        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1
        else:
            print_every = num_epochs + 1

        for epoch in range(1, num_epochs + 1):
            self.backbone.train()
            self.head.train()
            total_loss = 0.0

            for bx, by in trainloader:
                # Move data to device
                device = next(self.backbone.parameters()).device
                bx, by = bx.to(device), by.to(device)

                optimizer.zero_grad()

                feats = self.backbone(bx)
                preds = self.head(feats)

                loss = self.head.loss(preds, by, alpha)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(trainloader)
        
            # Validation / Early Stopping
            self.backbone.eval()
            self.head.eval()
            total_stop_loss = 0.0

            with torch.no_grad():
                for bx, by in stoploader:
                    bx, by = bx.to(device), by.to(device)
                    
                    feats = self.backbone(bx)
                    preds = self.head(feats)

                    loss = self.head.loss(preds, by, alpha)

                    total_stop_loss += loss.item()

                avg_stop_loss = total_stop_loss / len(stoploader)

            if verbose != -1 and epoch % print_every == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} -- Stop Loss: {avg_stop_loss:.4f} -- Best: {best_stop_loss:.4f}")
                
            # Save best model
            if avg_stop_loss < best_stop_loss:
                best_stop_loss = avg_stop_loss
                best_backbone_state = copy.deepcopy(self.backbone.state_dict())
                best_head_state = copy.deepcopy(self.head.state_dict())
            
        # Restore best model
        if best_backbone_state is not None:
            self.backbone.load_state_dict(best_backbone_state)
            self.head.load_state_dict(best_head_state)

        if verbose != -1:
            print(f"Best stop loss achieved: {best_stop_loss:.4f}")

        # Initialize inference wrappers
        self.center_model = CenterPredictorMVCS(self.head, self.backbone)
        self.matrix_model = MatrixPredictorMVCS(self.head, self.backbone)
