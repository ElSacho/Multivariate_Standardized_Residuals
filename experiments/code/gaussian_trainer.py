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

class GaussianTrainer:
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
            weight_decay = 0.0
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
            {
                'params': self.center_model.parameters(),
                'lr': lr_center_models,
                'weight_decay': weight_decay
            },
            {
                'params': self.matrix_model.parameters(),
                'lr': lr_matrix_models,
                'weight_decay': weight_decay
            },
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

        if self.best_stop_loss == np.inf:
            print('retraining')
            self.matrix_model.re_init_weights()
            self.fit(trainloader,
            stoploader,
            num_epochs=num_epochs,
            lr_center_models=lr_center_models,
            lr_matrix_models=lr_matrix_models,
            use_lr_scheduler=use_lr_scheduler,
            verbose=verbose,
            stop_on_best=stop_on_best,
            weight_decay = weight_decay)

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
        