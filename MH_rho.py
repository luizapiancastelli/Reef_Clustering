import numpy
import math
from numba import jit

@jit(nopython=True)
def empty_vector(n_row):
    """ Creates an empty vector of n_row floats compatible with Numba. """
    a = numpy.empty(n_row, numpy.float64) 
    return(a)

@jit(nopython=True)
def empty(n_row, n_col):
    """Creates an empty matrix of floats compatible with Numba."""
    a = numpy.empty((n_row,n_col), numpy.float64) 
    return(a)
        
@jit(nopython=True)
def sig_root(rho,target_var):
    """Calculates expression (7), the log-Normal sigma
       parameter for the desired target variance. """
    k = target_var
    return numpy.sqrt( numpy.log(( numpy.sqrt(4*k*rho**2 + rho**4) + rho**2)/(2*rho**2)))

@jit(nopython=True)
def lognorm(x, mean, sigma):
    """ Metropolis-Hastings Log-Normal jumping distribution. """
    return -(1/(2*sigma**2))*(numpy.log(x) - mean)**2 - numpy.log(x) - numpy.log(sigma)

@jit(nopython=True)
def proposal_sigma(rho, perc_var):
    """ Returns expression (7) for a given p_var parameter. """
    target_var = rho*perc_var
    sigma = sig_root(rho, target_var)
    return(sigma)

@jit(nopython=True)
def loglik_dirichlet_j_group(j, p, rho_l):
    
    #j: observation
    cat = len(rho_l)
    p_j = p[j,:]
  
    rho_lgamma = empty_vector(cat)
    for i in range(cat):
        rho_lgamma[i] = math.lgamma(rho_l[i])  
                
    l = numpy.sum((rho_l - 1)*numpy.log(p_j))
    
    loglik = l + math.lgamma(numpy.sum(rho_l)) - numpy.sum(rho_lgamma)
        
    return loglik


@jit(nopython=True)
def rho_conditional(l, rho_l, z, p, alpha, beta):
    """ Conditional distribution of rho. 
    l: cluster index 1,..., K.
    """

    p_l = p[z==l,:]
    nrow = numpy.shape(p_l)[0]
    
    if nrow == 0:
        likelihood = 0.0
    else:
        likelihood_i = numpy.zeros(nrow)
        for j in range(nrow):
            likelihood_i[j] = loglik_dirichlet_j_group(j,p_l,rho_l)
        likelihood = numpy.sum(likelihood_i)

    prior = (alpha-1)*numpy.sum(numpy.log(rho_l)) - beta*numpy.sum(rho_l)
    
    return likelihood + prior


@jit(nopython=True)
def MH_rho(rho, z, p, alpha, beta, perc_var):
    """ Metropolis-Hastings update for rho matrix.
    rho: current rho matrix (K x cat)
    z: current cluster allocation vector
    p: data set of compositions
    alpha: current alpha parameter value
    beta: current beta parameter value
    perc_var: value in [0,1] of p_var parameter (Section 4.4)
    """
    
    cat = numpy.shape(rho)[1] #Lenght of compositions vector
    K = numpy.shape(rho)[0]   
    accept = numpy.zeros((K,cat))
    rho_output = empty(K,cat)

    rho_current = numpy.copy(rho) 
    
    l = 1 # group index 
    while l <= K:
        
        rho_l = rho_current[l-1,:]
        
        i = 0  # proportion index
        while i <= (cat-1):
        
            sigma_t = proposal_sigma(rho_l[i], perc_var)
            
            rho_l_new = numpy.copy(rho_l)
            rho_l_new[i] = numpy.random.lognormal(numpy.log(rho_l[i]),sigma_t)
            
            sigma_star =  proposal_sigma(rho_l_new[i], perc_var)
            
            r =  rho_conditional(l, rho_l_new, z, p, alpha, beta) - \
                 rho_conditional(l, rho_l, z, p, alpha, beta) + \
                lognorm(rho_l[i], numpy.log(rho_l_new[i]), sigma_star) - \
                lognorm(rho_l_new[i], numpy.log(rho_l[i]), sigma_t)
                
            if math.log(numpy.random.uniform(0.0, 1.0)) < r:
                rho_output[l-1, i] = rho_l_new[i]
                accept[l-1, i] += 1.0
                rho_l = numpy.copy(rho_l_new) 
            else:
                rho_output[l-1, i] = rho_l[i]
                
            i += 1
            
        l += 1
    
    return rho_output, accept


# Metropolis-Hastings for alpha parameter
@jit(nopython=True)
def logtarget_alpha(alpha, rho, beta, gamma):
    """Calculates expression (4), the target alpha posterior. """
    
    cat = numpy.shape(rho)[1]
    k = numpy.shape(rho)[0]
    
    target = (cat*k*alpha)*numpy.log(beta) - (cat*k)*math.lgamma(alpha) + \
    (alpha-1)*numpy.sum(numpy.log(rho)) - alpha*gamma
    
    return target


@jit(nopython=True)
def MH_alpha(gamma, alpha_current, beta_current, rho_current, sigma):
    """Metropolis-Hastings step for alpha.
    gamma: hyperparameter of alpha prior
    alpha_current: current alpha parameter value
    beta_current: current beta parameter value
    rho_current: current rho matrix
    sigma: log-Normal proposal variance
    """
    
    alpha_prop = numpy.random.lognormal(numpy.log(alpha_current), sigma)

    r = logtarget_alpha(alpha_prop, rho_current, beta_current, gamma) - \
        logtarget_alpha(alpha_current, rho_current, beta_current, gamma) + \
        numpy.log(alpha_prop) - numpy.log(alpha_current)
        
    if math.log(numpy.random.uniform(0.0, 1.0)) < r:
        alpha_accepted = alpha_prop
        accepted = 1.0
    else:
        alpha_accepted = alpha_current
        accepted = 0.0

    return [alpha_accepted, accepted]





