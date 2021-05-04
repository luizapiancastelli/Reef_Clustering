"""
Reef Clustering model code
@author: Luiza Piancastelli

"""

import os
import scipy.stats as stats
import numpy 
import math
import time
from numba import jit
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import dirichlet
import pickle

from MH_z import *
from MH_rho import *

def sim_dirichlet_mixture(rho, n):
    """ Simulates data from a Dirichlet mixture model. 
    
    rho: numpy array where the number of rows is the number of mixture components
    n: sample size
    
    Output: dictionary. 'data' is the simulated compositional data, 'z' are the cluster allocations.
    """
    k = numpy.shape(rho)[0]
    z_true = numpy.repeat(numpy.arange(1,k+1), numpy.ceil(n/k))
    z_true = z_true[:n]
    
    p_sim = numpy.empty((n,numpy.shape(rho)[1]))
    for i in range(n):
        p_sim[i,:] = stats.dirichlet.rvs(rho[(z_true[i]-1),:],1) 
    
    return {'data': p_sim, 'z': z_true}



def initial_values(p, k, m):
    """Creates a list of lenght m with initial values.
    K-means generates cluster assignments and groupwise maximum likelihood
    estimates are the initial Dirichlet parameter values. """

    scaler = preprocessing.StandardScaler() #Center the data
    p_scaled = scaler.fit_transform(pd.DataFrame(p))
    cat = numpy.shape(p)[1]
    
    index = 0
    inits = []
    while index < m:
        kmeans = KMeans(n_clusters=k, n_init = 5)
        z_init = kmeans.fit_predict(p_scaled) + 1
        
        rho_mles = numpy.reshape(numpy.repeat(0.0, k*cat), (k,cat))
        try:
            for l in numpy.arange(1,k+1):
                rho_mles[l-1,:] =  dirichlet.mle(p[z_init==l, ])
        except:
            for l in numpy.arange(1,k+1):
                rho_mles[l-1,:] =  numpy.random.uniform(low = 0.1, high = 5, size =1)
            
        alpha_init = numpy.random.uniform(1,2,1)
        beta_init = numpy.random.uniform(0.1, 0.5, 1)
        
        inits_m = [z_init, rho_mles, alpha_init, beta_init]
        inits.append(inits_m)
        index +=1

    return inits


def MCMC_dirichlet(i, inits, p, burn_in, thin, chain_output, hyperparams_dict, perc_var=0.7, sigma_a = 0.5):
    """ Runs MCMC for a Dirichlet mixture model.
    
    Input:
    i: integer from 0 to m-1. Index of the initial values list to be taken. 
    p: numpy array. Data set of compositions. Number of rows is the number of observations.
    burn_in: integer. Lenght of burn in period.
    thin: integer. Thinning period.
    chain_output: integer. Desired number of final MCMC samples.
    hyperparams_dict: dictionary. Contains the prior hyperparameters delta, gamma, phi and lambda.
    perc_var: Proportion of current parameter value set as proposal variance of rho elements. 
    sigma_a: Proposal variance parameter for alpha Metropolis-Hastings.
    
    Output: list of length 7 containing
    1: rho chain
    2: z chain 
    3: alpha chain
    4: beta chain
    5: rho elements acceptance rate
    6: alpha acceptance rate
    7: Gibbs allocation probabilities at each iteration (for Stephen's relabelling)
    """
    
    chain_size = chain_output*(thin)
    n = numpy.shape(p)[0]
    cat = numpy.shape(p)[1]
    
    # Hyperparameters
    delta = hyperparams_dict['delta']
    gamma = hyperparams_dict['gamma']
    phi = hyperparams_dict['phi']
    lamb = hyperparams_dict['lambda']
    
    # Use initial guesses
    inits_m = inits[i]
    z_init = inits_m[0]
    rho_init = inits_m[1]
    alpha_init = inits_m[2]
    beta_init = inits_m[3]
    k = numpy.shape(rho_init)[0]
    
    z_current = numpy.copy(z_init)
    rho_current = numpy.copy(rho_init)
    alpha_current = numpy.copy(alpha_init)
    beta_current = numpy.copy(beta_init)
    
    #--------------------Burn-in period-------------------
    burn_in_step = 0
    while burn_in_step < burn_in:
        
        # Update allocations
        gibbs_Z = Gibbs_z(rho_current, z_current, p, delta)
        z_current = numpy.copy(gibbs_Z[0])
        
        # Update alpha
        MHa = MH_alpha(gamma, float(alpha_current), float(beta_current), rho_current, sigma_a)
        alpha_current = numpy.copy(MHa[0])
        
        # Update beta (Gibbs)
        beta_current = numpy.random.gamma(4*k*alpha_current + phi, 1/(numpy.sum(rho_current) + lamb))

        #Update rho
        MH = MH_rho(rho_current, z_current, p, alpha_current, beta_current, perc_var)
        rho_current = numpy.copy(MH[0])
        
        burn_in_step += 1
    
        if (burn_in_step % 10000) == 0:
            print("Burn in step: ", burn_in_step)
    
    #Initialize acceptance counters
    accepted_rho = numpy.zeros((k,cat))
    accepted_alpha = 0
    
    #Store MCMC chains
    z_chain = empty(chain_output, numpy.shape(p)[0])
    rho_chain = numpy.empty((k, cat, chain_output)) 
    alpha_chain = numpy.empty((chain_output))
    beta_chain = numpy.empty((chain_output))
    probs_array = numpy.empty((chain_output, n, k))
    
    #Iterate
    step = 0
    while step <= (chain_size -1):
        
        # Update allocations
        gibbs_Z = Gibbs_z(rho_current, z_current, p, delta)
        z_current = numpy.copy(gibbs_Z[0])
        
        # Update alpha
        MHa = MH_alpha(gamma, float(alpha_current), float(beta_current), rho_current, sigma_a)
        alpha_current = numpy.copy(MHa[0])
        accepted_alpha += MHa[1]
        
        # Update beta (Gibbs)
        beta_current = numpy.random.gamma(4*k*alpha_current + phi, 1/(numpy.sum(rho_current) + lamb))
    
        #Update rho
        MH = MH_rho(rho_current, z_current, p, alpha_current, beta_current, perc_var)
        rho_current = numpy.copy(MH[0])
        accepted_rho = accepted_rho + MH[1]
            
        if (step - burn_in)%thin ==0:
            z_chain[int((step)/thin), :] = z_current
            rho_chain[:,:,int((step)/thin)] = rho_current
            alpha_chain[int((step)/thin)] = alpha_current
            beta_chain[int((step)/thin)] = beta_current
            probs_array[int((step)/thin),:,:] = gibbs_Z[1]
            
        if (step % 10000) == 0:
            print("Step: ", step)
        step += 1
    
    results = [rho_chain, z_chain, alpha_chain, beta_chain,
               accepted_rho/chain_size, accepted_alpha/chain_size, probs_array]                         
    
    return results





