import numpy as np
import ot
from sklearn.model_selection import train_test_split
from utils import seed_everything

seed_everything(42)


########################################################################################################################################
########################################################################################################################################
## CODES TO SOLVE OPTIMAL TRANSPORT / LEARN MK QUANTILES AND RANKS : 
########################################################################################################################################
########################################################################################################################################

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
