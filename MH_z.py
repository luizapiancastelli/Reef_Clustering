"""
Functions to run Gibbs update for cluster allocations
@author: Luiza Piancastelli
"""

import numpy
import math
from numba import jit

    
@jit(nopython=True)
def empty_vector(n_row):
    """ Creates an empty vector of n_row floats compatible with Numba. """
    a = numpy.empty(n_row, numpy.float64) 
    return(a)
    

@jit(nopython=True)
def loglik_dirichlet_j(j, p, z, rho):
    
    """Calculates the log-likelihood of observation j.
    p: data set of compositions.
    z: cluster allocations
    rho: Dirichlet mixture parameters. 
    """

    rho_j = rho[(int(z[j])-1),:]
    cat= len(rho_j)
    p_j = p[j,:]
  
    rho_lgamma = empty_vector(cat)
    for i in range(cat):
        rho_lgamma[i] = math.lgamma(rho_j[i])  
                
    l = numpy.sum((rho_j - 1)*numpy.log(p_j))
    
    loglik = l + math.lgamma(numpy.sum(rho_j)) - numpy.sum(rho_lgamma)
        
    return loglik

@jit(nopython=True)
def n_clust(z,K):
    """Counts the number of observations at each group."""
    j = 1
    counts = empty_vector(K)
    while j <= K:
        counts[j-1] = numpy.sum(z==j)
        j +=1
    return counts
    

@jit(nopython=True)
def joint_p_zj(j, p, z, rho, delta):
    """ Calculates conditional distribution in expression (2). """
    K = numpy.shape(rho)[1]
    counts = n_clust(z, K)
    al = counts + delta
    
    al_lgamma = numpy.zeros(K)
    for l in numpy.arange(0,K):
        al_lgamma[l] = math.lgamma(al[l])  

    ll = loglik_dirichlet_j(j, p, z, rho)
    prior = numpy.sum(al_lgamma) - math.lgamma(numpy.sum(al))
    
    posterior = ll + prior
    
    return posterior


@jit(nopython=True) 
def rand_choice_nb(arr, prob):
    """Samples from arr with probability prob."""
    return arr[numpy.searchsorted(numpy.cumsum(prob), numpy.random.random(), side="right")]


@jit(nopython=True) 
def input_zj(j, z_current, rho_current, p, delta):
    
    """ Calculates the Gibbs probabilities in Expression (3). """
    
    z_input = numpy.copy(z_current) 
    
    k = numpy.shape(rho_current)[0]
    
    logliks = numpy.empty(k)
    
    for l in numpy.arange(1, k+1):
        z_input[j] = l
        logliks[l-1] = joint_p_zj(j, p, z_input, rho_current, delta)
        
    l_min = numpy.min(logliks)
    l_ = logliks - l_min +1.0
    
    probs = numpy.exp(l_)/numpy.sum(numpy.exp(l_))
    
    return probs

@jit(nopython=True) 
def Gibbs_z(rho_current, z_current, p, delta):
    
    """Gibbs update of cluster allocations.
    rho_current: current Dirichlet parameter matrix.
    z_current: current cluster memberships.
    p: data set of compositions.
    delta: hyperparameter of z prior."""
    
    K = numpy.shape(rho_current)[0]
    n = len(z_current)
    
    j = 0
    z_new = numpy.copy(z_current)
    prob_matrix = numpy.empty((n, K))
    
    #Loop through all observations
    while j <= (n-1):
    
        z_current = numpy.copy(z_new)
        probs = input_zj(j, z_current, rho_current, p, delta)
        
        zj_new =  rand_choice_nb(numpy.arange(1,K+1),probs)
        z_new[j] = zj_new
        
        prob_matrix[j,] = probs
        
        j +=1

    return z_new, prob_matrix


