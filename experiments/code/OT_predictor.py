import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import ot_functions 
import seaborn as sns 
from time import time

import pandas as pd 
import seaborn as sns
from sklearn.neighbors import NearestNeighbors 

from matplotlib import cm

from utils import seed_everything

seed_everything(42)


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



class OTPredictor(nn.Module):
    def __init__(self, center_model, n_neighbors, seed=42):
        """
        Parameters            
        ----------
        center_model : nn.Module
            The model that represents the center of the sets.
        n_neighbors : int
            The number of neighbors to consider for the OT prediction.
        seed : int, optional
            The random seed for reproducibility. The default is 42.

        """
        super(OTPredictor, self).__init__()
        seed_everything(seed)
        self.center_model = center_model
        self.n_neighbors = n_neighbors
 
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

    
    ##########################################################
    ############# CONFORMALIZATION - CLASSIC #################
    ##########################################################
    
    def conformalize(self, x_calibration, y_calibration, alpha=0.1):
        """
        Compute the quantile value to conformalize the ellipsoids on a unseen dataset.

        Parameters
        ----------
        x_calibration : torch.Tensor
            The input tensor of shape (n, d) for the calibration set.
        y_calibration : torch.Tensor
            The target tensor of shape (n, k) for the calibration set.
        alpha : float, optional
            The significance level for the conformal prediction. The default is 0.1.
        """

        ## Evaluate scores on Calibration data and on Test data 

        f_x = self.get_centers(x_calibration)
        res_cal = y_calibration - f_x 

        res_cal = res_cal.detach().cpu().numpy()
        x_calibration = x_calibration.detach().cpu().numpy()
        y_calibration = y_calibration.detach().cpu().numpy()
        
        ## COMPUTE MK quantiles on calibration data and get metrics on test data 
        self.parameters_MK = ot_functions.MultivQuantileTreshold_Adaptive(res_cal,x_calibration, self.n_neighbors,1-alpha)
        
    
    def get_all_volumes(self, x_test):
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        if n == -1:
            psi,Y = ot_functions.psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume = np.ones(len(x_test)) * volume_QR

            k = scores_cal_1.shape[-1]  # Number of outputs

            return torch.tensor(tab_volume)**(1/k)

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,Y = ot_functions.psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume[i] = volume_QR
        
        k = scores_cal_1.shape[-1]  # Number of outputs

        return torch.tensor(tab_volume)**(1/k)
        
    def get_averaged_volume(self, x):
        with torch.no_grad():
            tab_volumes = self.get_all_volumes(x)
            return torch.mean(tab_volumes)


    def get_coverage(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        if n == -1:
            prop = 0
            psi,psi_star = ot_functions.learn_psi(mu,scores_cal_1) 

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = ot_functions.RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
                prop += indic_content 
            
            prop = prop / len(x_test)
            return prop
        
        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ot_functions.ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

            indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            prop += indic_content 

        prop = prop / len(x_test)
        
        return prop
    

    def get_inside(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors
        inside = []

        if n == -1:
            
            psi,psi_star = ot_functions.learn_psi(mu,scores_cal_1) 

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = ot_functions.RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
                inside.append(indic_content)
            
            return inside
        
        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ot_functions.ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

            indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            inside.append(indic_content)
        
        return inside
    
    def get_contour(self, x_test, n_points=100):
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        x_test = x_test.detach().cpu().numpy()
        sphere = np.array([np.cos(2*np.pi*np.arange(n_points)/n_points),np.sin(2*np.pi*np.arange(n_points)/n_points)]).T 
        quantile_contours = []

        if self.n_neighbors == -1:
            Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
            data_knn, psi_star = ot_functions.get_psi_star(x_test[0], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)
            
            Q0 = ot_functions.T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
            
            for i in range(len(x_test)):
                center = self.get_centers(torch.tensor(x_test[i])).detach().numpy()
                contour = Q0 + center  # Add the center to the contour
                contour = np.concatenate((contour, contour[0:1, :]), axis=0)
                quantile_contours.append( contour ) 
            quantile_contours = np.array(quantile_contours) 
            return quantile_contours

        for i in range(len(x_test)):
            Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
            data_knn, psi_star = ot_functions.get_psi_star(x_test[i], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)

            Q0 = ot_functions.T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
            center = self.get_centers(torch.tensor(x_test[i])).detach().numpy()

            Q0 = Q0 + center  # Add the center to the contour
            Q0 = np.concatenate((Q0, Q0[0:1, :]), axis=0)
            quantile_contours.append( Q0 ) 
        quantile_contours = np.array(quantile_contours) 
        return quantile_contours
    

    ##########################################################
    ############## CONFORMAL - PROJECTION ####################
    ##########################################################

    
    def conformalize_linear_projection(self, projection_matrix, x_calibration=None, y_calibration=None, alpha=0.1):
        """
        Compute the quantile value to conformalize the ellipsoids on a unseen dataset.

        Parameters
        ----------
        """
        self.projection_matrix = projection_matrix

        f_x = self.get_centers(x_calibration)
        res_cal = y_calibration - f_x 
        res_cal = torch.einsum('rk,nk->nr', self.projection_matrix, res_cal)

        res_cal = res_cal.detach().cpu().numpy()
        x_calibration = x_calibration.detach().cpu().numpy()
        y_calibration = y_calibration.detach().cpu().numpy()
        
        ## COMPUTE MK quantiles on calibration data and get metrics on test data 
        self.parameters_MK_proj = ot_functions.MultivQuantileTreshold_Adaptive(res_cal,x_calibration, self.n_neighbors,1-alpha)
        

    def get_volume_projection(self, x_test):
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
        n = self.n_neighbors

        if n == -1:
            psi,Y = ot_functions.psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume = np.ones(len(x_test)) * volume_QR
            
            return torch.tensor(tab_volume)**(1/scores_cal_1.shape[-1])

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,Y = ot_functions.psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume[i] = volume_QR
        
        return torch.tensor(tab_volume)**(1/scores_cal_1.shape[-1]) 
        
    def get_averaged_volume_projection(self, x):
        with torch.no_grad():
            tab_volumes = self.get_volume_projection(x)
            return torch.mean(tab_volumes)


    def get_coverage_projection(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = torch.einsum('rk,nk->nr', self.projection_matrix, res_test)
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
        n = self.n_neighbors

        if n == -1:
            prop = 0
            psi,psi_star = ot_functions.learn_psi(mu,scores_cal_1)

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = ot_functions.RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region.
                prop += indic_content   

            prop = prop / len(x_test)
            return prop

        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ot_functions.ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

            indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            prop += indic_content 

        prop = prop / len(x_test)
        
        return prop
    
    def get_contour_proj(self, x_test, n_points=100):
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        x_test = x_test.detach().cpu().numpy()
        dim = self.projection_matrix.shape[0]
        sphere = sample_sphere(n_points, dim) 
        quantile_contours = []
        for i in range(len(x_test)):
            Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
            data_knn, psi_star = ot_functions.get_psi_star(x_test[i], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)

            Q0 = ot_functions.T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
            center = self.get_centers(torch.tensor(x_test[i]))
            center = self.projection_matrix @ center  # Project the center onto the lower-dimensional space
            center = center.detach().numpy()
            Q0 = Q0 + center  # Add the center to the contour
            quantile_contours.append( Q0 ) 
        quantile_contours = np.array(quantile_contours) 
        return quantile_contours
    
    
    ##########################################################
    ############### CONFORMAL - CONDITION ####################
    ##########################################################

    
    def conformalize_with_knowned_idx(self, idx_knowned):
        self._idx_knowned = idx_knowned
        
    def get_all_volume_with_knowned_idx(self, x_test, y_r):
        f_x = self.get_centers(x_test)
        res_r = y_r - f_x[:, self.idx_knowned]  
        res_r = res_r.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        if n == -1:
            psi,residuals = ot_functions.psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR_condition_on_yr(Quantile_Treshold,mu,psi, residuals, res_r[0], self.idx_knowned) 
            tab_volume = np.ones(len(x_test)) * volume_QR

            s = scores_cal_1.shape[-1] - len(self.idx_knowned)  

            return torch.tensor(tab_volume)**(1/s)

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,residuals = ot_functions.psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = ot_functions.get_volume_QR_condition_on_yr(Quantile_Treshold,mu,psi, residuals, res_r[i], self.idx_knowned) 
            tab_volume[i] = volume_QR

        s = scores_cal_1.shape[-1] - len(self.idx_knowned)  
        
        return torch.tensor(tab_volume)**(1/s)
    
    def get_averaged_volume_with_knowned_idx(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_all_volume_with_knowned_idx(x, y_r)
            return torch.mean(tab_volumes)
        

        


        