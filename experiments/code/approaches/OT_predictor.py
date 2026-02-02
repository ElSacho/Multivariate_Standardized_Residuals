import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from time import time

import pandas as pd 
import seaborn as sns
from sklearn.neighbors import NearestNeighbors 

from matplotlib import cm

from utils import seed_everything

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
        self.parameters_MK = MultivQuantileTreshold_Adaptive(res_cal,x_calibration, self.n_neighbors,1-alpha)
        
    
    def get_all_volumes(self, x_test):
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        if n == -1:
            psi,Y = psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume = np.ones(len(x_test)) * volume_QR

            k = scores_cal_1.shape[-1]  # Number of outputs

            return torch.tensor(tab_volume)**(1/k)

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,Y = psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume[i] = volume_QR
        
        k = scores_cal_1.shape[-1]  # Number of outputs

        return torch.tensor(tab_volume)**(1/k)
        
    def get_averaged_volume(self, x):
        with torch.no_grad():
            tab_volumes = self.get_all_volumes(x)
            return torch.mean(tab_volumes)
        
    def get_cover(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        cover = []

        if n == -1:
            
            psi,psi_star = learn_psi(mu,scores_cal_1) 

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = RankFunc(s_tick,mu,psi)
                indic_content = 1.0*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
                cover.append(indic_content)
            
            return np.array(cover)
        
        for i in range(len(x_test)):
            
            ConditionalRank,psi,Y = ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            indic_content = 1.0*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            cover.append(indic_content)
 
        return np.array(cover)

    def get_coverage(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
        n = self.n_neighbors

        if n == -1:
            prop = 0
            psi,psi_star = learn_psi(mu,scores_cal_1) 

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
                prop += indic_content 
            
            prop = prop / len(x_test)
            return prop
        
        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

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
            
            psi,psi_star = learn_psi(mu,scores_cal_1) 

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
                inside.append(indic_content)
            
            return inside
        
        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

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
            data_knn, psi_star = get_psi_star(x_test[0], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)
            
            Q0 = T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
            
            for i in range(len(x_test)):
                center = self.get_centers(torch.tensor(x_test[i])).detach().numpy()
                contour = Q0 + center  # Add the center to the contour
                contour = np.concatenate((contour, contour[0:1, :]), axis=0)
                quantile_contours.append( contour ) 
            quantile_contours = np.array(quantile_contours) 
            return quantile_contours

        for i in range(len(x_test)):
            Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK 
            data_knn, psi_star = get_psi_star(x_test[i], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)

            Q0 = T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
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
        self.parameters_MK_proj = MultivQuantileTreshold_Adaptive(res_cal,x_calibration, self.n_neighbors,1-alpha)
        

    def get_volume_projection(self, x_test):
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
        n = self.n_neighbors

        if n == -1:
            psi,Y = psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR(Quantile_Treshold,mu,psi,Y) 
            tab_volume = np.ones(len(x_test)) * volume_QR
            
            return torch.tensor(tab_volume)**(1/scores_cal_1.shape[-1])

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,Y = psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR(Quantile_Treshold,mu,psi,Y) 
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
            psi,psi_star = learn_psi(mu,scores_cal_1)

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region.
                prop += indic_content   

            prop = prop / len(x_test)
            return prop

        prop = 0
        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

            indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            prop += indic_content 

        prop = prop / len(x_test)
        
        return prop
    
    def get_cover_projection(self, x_test, y_test):
        f_x = self.get_centers(x_test)
        res_test = y_test - f_x
        res_test = torch.einsum('rk,nk->nr', self.projection_matrix, res_test)
        res_test = res_test.detach().cpu().numpy()
        x_test = x_test.detach().cpu().numpy()

        Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
        n = self.n_neighbors

        cover = []
        if n == -1:
        
            psi,psi_star = learn_psi(mu,scores_cal_1)

            for i in range(len(x_test)):
                s_tick = res_test[i]

                ConditionalRank = RankFunc(s_tick,mu,psi)
                indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region.
                cover.append(indic_content)
                
            return np.array(cover)


        for i in range(len(x_test)):
            ConditionalRank,psi,Y = ConditionalRank_Adaptive(res_test[i],x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)

            indic_content = 1*(np.linalg.norm(ConditionalRank) <= Quantile_Treshold)  #Ytest is in prediction region if Stest is in quantile region. 
            cover.append(indic_content)
        
        return np.array(cover)
    
    def get_contour_proj(self, x_test, n_points=100):
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        x_test = x_test.detach().cpu().numpy()
        dim = self.projection_matrix.shape[0]
        sphere = sample_sphere(n_points, dim) 
        quantile_contours = []
        for i in range(len(x_test)):
            Quantile_Treshold,knn,scores_cal_1,mu = self.parameters_MK_proj
            data_knn, psi_star = get_psi_star(x_test[i], knn, scores_cal_1, n_neighbors=self.n_neighbors, mu=mu)

            Q0 = T0(Quantile_Treshold*sphere , data_knn ,psi_star) 
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
            psi,residuals = psi_Y(x_test[0] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR_condition_on_yr(Quantile_Treshold,mu,psi, residuals, res_r[0], self.idx_knowned) 
            tab_volume = np.ones(len(x_test)) * volume_QR

            s = scores_cal_1.shape[-1] - len(self.idx_knowned)  

            return torch.tensor(tab_volume)**(1/s)

        tab_volume = np.zeros(len(x_test))
        for i in range(len(x_test)):
            psi,residuals = psi_Y(x_test[i] ,knn,scores_cal_1,n_neighbors=n,mu=mu)
            volume_QR = get_volume_QR_condition_on_yr(Quantile_Treshold,mu,psi, residuals, res_r[i], self.idx_knowned) 
            tab_volume[i] = volume_QR

        s = scores_cal_1.shape[-1] - len(self.idx_knowned)  
        
        return torch.tensor(tab_volume)**(1/s)
    
    def get_averaged_volume_with_knowned_idx(self, x, y_r):
        with torch.no_grad():
            tab_volumes = self.get_all_volume_with_knowned_idx(x, y_r)
            return torch.mean(tab_volumes)
        
import ot
from sklearn.model_selection import train_test_split
from utils import seed_everything

########################################################################################################################################
########################################################################################################################################
## CODES TO SOLVE OPTIMAL TRANSPORT / LEARN MK QUANTILES AND RANKS : 
########################################################################################################################################
########################################################################################################################################

import numpy as np

def sample_uniform(data, n_r=6, n_s=83):
    n = data.shape[0]
    dimension = data.shape[1]
    
    n_o = (n + 1) - (n_r * n_s)
    if n_o < 0:
        raise ValueError(f"n_r * n_s ({n_r * n_s}) cannot exceed n + 1 ({n + 1}).")

    # 1. Generate n_s unit vectors uniformly distributed
    if dimension == 2:
        # For 2D, we can get perfect uniform spacing using roots of unity
        angles = np.linspace(0, 2 * np.pi, n_s, endpoint=False)
        unit_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    elif dimension == 3:
        # For 3D, we use the Fibonacci Lattice (Golden Spiral) for uniform coverage
        indices = np.arange(0, n_s) + 0.5
        phi = np.arccos(1 - 2 * indices / n_s)
        theta = np.pi * (1 + 5**0.5) * indices
        
        unit_vectors = np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi)
        ], axis=1)
        
    else:
        # For d > 3, we use a Sobol sequence or low-discrepancy sampling 
        # to be more 'uniform' than standard random Gaussian sampling.
        # Here, we use a simple normalization of a low-discrepancy-style distribution.
        # For true uniformity in high D without external libraries like scipy.stats.qmc,
        # we stick to normalized Gaussian but seed it or use a fixed Halton sequence.
        random_directions = np.random.standard_normal((n_s, dimension))
        unit_vectors = random_directions / np.linalg.norm(random_directions, axis=1, keepdims=True)

    # 2. Rest of the logic remains the same
    origin_points = np.zeros((n_o, dimension))
    radii = np.linspace(1.0 / n_r, 1.0, n_r)
    
    shells = radii[:, np.newaxis, np.newaxis] * unit_vectors[np.newaxis, :, :]
    shell_points = shells.reshape(-1, dimension)
    
    return np.vstack([origin_points, shell_points])

def sample_grid(data,positive=False, seed=42):
    ''' Sample the reference distribution.'''
    seed_everything(seed)
    n = data.shape[0]
    d = data.shape[1]
    R = np.linspace(0,1,n)
    if positive==False:
        sampler = qmc.Halton(d=d, seed=seed)
        sample_gaussian = sampler.random(n=n+1)[1:]
        sample_gaussian = norm.ppf(sample_gaussian, loc=0, scale=1)
        mu = []
        for i in range(n):
            Z = sample_gaussian[i]
            Z = Z / np.linalg.norm(Z)
            mu.append( R[i]*Z)
    else:
        mu = []
        for i in range(n):
            Z = np.random.exponential(scale=1.0,size=d) 
            Z = Z / np.sum(Z)
            mu.append( R[i]*Z)
    return(np.array(mu))

def T0(x,DATA,psi): 
    ''' Returns the image of `x` by the OT map parameterized by `psi` towards the empirical distribution of `sample_sort`.'''
    if (len(x.shape)==1):  
        to_max = (DATA @ x) - psi 
        res = DATA[np.argmax(to_max)]
    else: 
        to_max = (DATA @ x.T).T - psi 
        res = DATA[np.argmax(to_max,axis=1)]
    return(res)

def learn_psi(mu,data):
    M = ot.dist(data,mu)/2 
    res = ot.solve(M)
    g,f = res.potentials
    psi = 0.5*np.linalg.norm(mu,axis=1)**2 - f
    psi_star = 0.5*np.linalg.norm(data,axis=1)**2 - g 
    to_return = [psi,psi_star]
    return(to_return)

def RankFunc(x,mu,psi,ksi=0):
    # ksi >0 computes a smooth argmax (LogSumExp). ksi is a regularisation parameter, hence approximates the OT map. 
    if (len(x.shape)==1):  
        to_max = ((mu @ x)- psi)*ksi
        to_sum = np.exp(to_max - np.max(to_max) )
        weights = to_sum/(np.sum(to_sum)) 
        res =  np.sum(mu*weights.reshape(len(weights),1),axis=0)
    else: 
        res=[]
        for xi in x:
            to_max = ((mu @ xi)- psi)*ksi
            to_sum = np.exp(to_max - np.max(to_max) )
            weights = to_sum/(np.sum(to_sum)) 
            res.append( np.sum(mu*weights.reshape(len(weights),1),axis=0))
        res = np.array(res)
    # For exact recovery of the argsup, one can use T0. 
    if ksi == 0:
        res = T0(x,mu,psi) 
    return( res )

def QuantFunc(u,data,psi_star):
    return( T0(u,data,psi_star) )

from scipy.stats import qmc
from scipy.stats import norm 

def MultivQuantileTreshold(data,alpha = 0.9,positive=False):
    ''' To change the reference distribution towards a positive one, set positive = True.  '''
    data_calib, data_valid = train_test_split(data,test_size=0.25) 
    # Solve OT 
    mu = sample_grid(data_calib,positive=positive) 
    # mu = sample_uniform(data_calib)
    psi,psi_star = learn_psi(mu,data_calib) 
    # QUANTILE TRESHOLDS 
    n = len(data_valid) 
    Ranks_data_valid = RankFunc(data_valid,mu,psi) 
    Norm_ranks_valid = np.linalg.norm(  Ranks_data_valid ,axis=1,ord=2) 
    Quantile_Treshold = np.quantile( Norm_ranks_valid, np.min(  [np.ceil((n+1)*alpha)/n ,1] )   ) 
    return(Quantile_Treshold,mu,psi,psi_star,data_calib) 



########################################################################################################################################
########################################################################################################################################
## CODES FOR REGRESSION : 
########################################################################################################################################
########################################################################################################################################

def get_volume_QR(Quantile_Treshold,mu,psi,scores,N = int(1e4)):
    """ Monte-Carlo estimation of the quantile region of order 'Quantile_Treshold'."""
    M = np.max(scores,axis=0)
    m = np.min(scores,axis=0)
    v = m + np.random.random((N,mu.shape[1]))*(M-m) 
    scale = np.prod(M-m) 
    MCMC = np.mean(np.linalg.norm( RankFunc(v,mu,psi) ,axis=1) <= Quantile_Treshold) 
    return(MCMC*scale) 


def get_volume_QR_condition_on_yr(Quantile_Treshold,mu,psi,scores, y_r, idx_known, N = int(1e4)):
    """ Monte-Carlo estimation of the quantile region of order 'Quantile_Treshold'."""
    M = np.max(scores,axis=0)
    m = np.min(scores,axis=0)
    idx_unknown = np.setdiff1d(np.arange(mu.shape[1]), idx_known)
    v = m + np.random.random((N,mu.shape[1]))*(M-m) 

    v[:,idx_known] = y_r  # Condition on the known indices
    scale = np.prod(M[idx_unknown]-m[idx_unknown]) 
    MCMC = np.mean(np.linalg.norm( RankFunc(v,mu,psi) ,axis=1) <= Quantile_Treshold) 
    return(MCMC*scale) 

def get_contourMK(Quantile_Treshold,psi_star,scores,N=100):
    ''' get 2D quantile contours'''
    contour = []
    angles = 2*np.pi*np.linspace(0,1,N)
    for theta in angles:
        us = np.array([[np.cos(theta)][0],[np.sin(theta)][0]])
        contour.append(us) 
    contour = np.array(contour) * Quantile_Treshold
    contourMK = QuantFunc(contour,scores,psi_star)
    return(contourMK)

from sklearn.neighbors import NearestNeighbors 

def MultivQuantileTreshold_Adaptive(scores_cal,x_cal,n_neighbors, alpha = 0.9):
    ''' Conformal Quantile Regression (OT-CP+).
    Returns parameters related for MK quantiles based on a quantile function that is conditional on x_tick. 
    A neighborhood of x_tick is regarded within x_cal, the calibration data. 

    - x_cal = covariates of calibration data
    - scores = calibration scores, such as residuals, computed from predictions f(x) with same indices as in `x`
    - n_neighbors = number of neighbors for KNN 
    - alpha: confidence level in [0,1]
    ''' 
    indices_split1,indices_split2 = train_test_split(np.arange(len(x_cal)),test_size=0.5)

    # Quantile regression (and a fortiori KNN) on data 1 
    knn = None
    if n_neighbors != -1:
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(x_cal[indices_split1])

    scores_cal_1 = scores_cal[indices_split1]
        
    n_batch = n_neighbors if n_neighbors != -1 else len(scores_cal_1)

    # Conformal treshold on data 2 
    mu = sample_grid(np.zeros((n_batch,scores_cal.shape[1])),positive=False) 
    # mu = sample_uniform(np.zeros((n_batch,scores_cal.shape[1])))
    list_MK_ranks = []

    if n_neighbors == -1:
        Y = scores_cal_1
        psi,psi_star = learn_psi(mu,Y) 

        for i in range(len(indices_split2)):
            # We want the MK rank of s_tick conditional on x_tick  
            x_tick = x_cal[indices_split2][i]
            s_tick = scores_cal[indices_split2][i]

            # Solve OT 
            Ranks_data_valid = RankFunc(s_tick,mu,psi) 
            list_MK_ranks.append( Ranks_data_valid )

        # QUANTILE TRESHOLDS 
        n = len(indices_split2)
        list_MK_ranks = np.array(list_MK_ranks)
        Norm_ranks_valid = np.linalg.norm(  list_MK_ranks ,axis=1)  
        Quantile_Treshold = np.quantile( Norm_ranks_valid, np.min( [np.ceil((n+1)*alpha)/n ,1] ) ) 

        return(Quantile_Treshold,knn,scores_cal_1,mu)  

    for i in range(len(indices_split2)):
        # We want the MK rank of s_tick conditional on x_tick  
        x_tick = x_cal[indices_split2][i]
        s_tick = scores_cal[indices_split2][i]

        if n_neighbors != -1:    
            local_neighbors_test = knn.kneighbors(x_tick.reshape(1, -1), return_distance=False)
            indices_knn = local_neighbors_test.flatten()
            Y = scores_cal_1[indices_knn] #neighbors among data set 1 
        else:
            Y = scores_cal_1

        # Solve OT 
        psi,psi_star = learn_psi(mu,Y) 

        Ranks_data_valid = RankFunc(s_tick,mu,psi) 
        list_MK_ranks.append( Ranks_data_valid )

    # QUANTILE TRESHOLDS 
    n = len(indices_split2)
    list_MK_ranks = np.array(list_MK_ranks)
    Norm_ranks_valid = np.linalg.norm(  list_MK_ranks ,axis=1)  
    Quantile_Treshold = np.quantile( Norm_ranks_valid, np.min(  [np.ceil((n+1)*alpha)/n ,1] )   ) 

    return(Quantile_Treshold,knn,scores_cal_1,mu)  

def ConditionalRank_Adaptive(s_tick,x_tick,knn,scores_cal_1,n_neighbors,mu):
    ''' 
    Return parameters related MK quantiles based on a quantile function that is conditional on x_tick. A neighborhood of x_tick is regarded within x, the calibration data. 
    - s_tick = new score where  the conditional quantile function Q( s_tick / X = x_tick) is to be computed, conditionnaly on x_tick
    - scores_cal_1 = calibration scores on which knn was learnt 
    - knn: k-nearest neighbors previously fitted on covariates from same data as scores_cal_1
    ''' 
    if knn is None:
        Y = scores_cal_1  # If no knn, then we use all calibration scores
    else:
        local_neighbors_test = knn.kneighbors(x_tick.reshape(1, -1), return_distance=False)
        indices_knn = local_neighbors_test.flatten()
        Y = scores_cal_1[indices_knn]  # Calibration scores associated to k nearest neighbors of x_tick (in x) 

    # Solve OT 
    psi,psi_star = learn_psi(mu,Y) 
    
    # Conditional rank 
    ConditionalRank = RankFunc(s_tick,mu,psi)
    return(ConditionalRank,psi,Y)

def psi_Y(x_tick,knn,scores_cal_1,n_neighbors,mu):
    ''' 
    Return parameters related MK quantiles based on a quantile function that is conditional on x_tick. A neighborhood of x_tick is regarded within x, the calibration data. 
    - s_tick = new score where  the conditional quantile function Q( s_tick / X = x_tick) is to be computed, conditionnaly on x_tick
    - scores_cal_1 = calibration scores on which knn was learnt 
    - knn: k-nearest neighbors previously fitted on covariates from same data as scores_cal_1
    ''' 
    if knn is None:
        Y = scores_cal_1
    else:
        local_neighbors_test = knn.kneighbors(x_tick.reshape(1, -1), return_distance=False)
        indices_knn = local_neighbors_test.flatten()
        Y = scores_cal_1[indices_knn]  # Calibration scores associated to k nearest neighbors of x_tick (in x) 

    # Solve OT 
    psi,psi_star = learn_psi(mu,Y) 
    
    # Conditional rank 
    
    return(psi,Y)

def get_psi_star(x_tick,knn,scores_cal_1,n_neighbors,mu):
    ''' 
    Return parameters related MK quantiles based on a quantile function that is conditional on x_tick. A neighborhood of x_tick is regarded within x, the calibration data. 
    - s_tick = new score where  the conditional quantile function Q( s_tick / X = x_tick) is to be computed, conditionnaly on x_tick
    - scores_cal_1 = calibration scores on which knn was learnt 
    - knn: k-nearest neighbors previously fitted on covariates from same data as scores_cal_1
    ''' 
    if knn is None:
        data_knn = scores_cal_1  # If no knn, then we use all calibration scores
    else:
        local_neighbors_test = knn.kneighbors(x_tick.reshape(1, -1), return_distance=False)
        indices_knn = local_neighbors_test.flatten()
        data_knn = scores_cal_1[indices_knn]  # Calibration scores associated to k nearest neighbors of x_tick (in x) 

    # Solve OT 
    psi,psi_star = learn_psi(mu,data_knn) 
    
    # Conditional rank 
    
    return data_knn, psi_star
