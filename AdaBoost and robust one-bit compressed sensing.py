# -*- coding: utf-8 -*-
"""
AdaBoost and robust one-bit compressed sensing

Geoffrey Chinot, Felix Kuchelmeister, Matthias Löffler and Sara van de Geer
"""

###############################################################################
# Packages
###############################################################################

import numpy as np
from scipy.optimize import linprog

###############################################################################
# Functions
###############################################################################

#creates a random s-sparse vector with dimension p and length 1
def sparse_rademacher_prior_beta(s,p):
    tmp = np.random.randint(0,2,s)*2-1 #Rademacher sample
    tmp = tmp/np.sqrt(s) #normalize
    tmp = np.append(tmp,[0]*(p-s))
    np.random.shuffle(tmp)
    return(tmp)

def adversary(X,beta,corr): #corr is the number of corrupted observations
    #find elements with largest influence
    candidates = abs(np.dot(X,beta))
    vandalize = np.argsort(-candidates)[0:corr] #find indices to be manipulated
    
    tmp = np.repeat(1,n)
    tmp[vandalize] = -1
    
    return(tmp)

#how p is calculated
def f_1(n):
    return(2*int(n**1.1))

#how T is calculated
def f_T(n,p,s,corr,eps):
    return(2*int(np.log(p)**(4/3)*n**(2/3)*(s+corr)**(1/3)/eps))

def max_margin(n,p,yX):
    C = np.concatenate((-yX,yX),axis = 1) 
    d = np.repeat(-1,n)
    c = np.repeat(1,2*p)
    res = linprog(c, A_ub= C , b_ub= d, bounds = (0,None), method='interior-point', options = {'maxiter': 500000})
    hat_beta = res.x[0:p]-res.x[p:2*p]
    print("interpolation:", all(np.dot(yX,hat_beta)>=-1e-11))
    return(res)

def max_margin_beta(res,p):
    tmp = res.x[0:p] - res.x[p:2*p]
    tmp = tmp/np.linalg.norm(tmp,2)
    return(tmp)

#use this, as to obtain the margin, we do not divide by the ell_2_norm.
def max_margin_margin(res,p):
    tmp = res.x[0:p] - res.x[p:2*p]
    return(1/np.linalg.norm(tmp,1))

def Ada(n,p,s,yX,T,eps): #T is run time, eps is learning rate
    #initialize
    bet_tilde = np.zeros(p, dtype = float) 
    W = np.zeros(n, dtype = float)
    
    #rescale
    X_infty = np.max(np.abs(yX))
    yX_r = yX/X_infty
        
    for t in range(0,T):
        #calculate weights
        tmp = np.exp(-np.dot(yX_r,bet_tilde)) #auxiliary function
        if(any(tmp == np.inf)): #test if any value is infinite
            tmp_2 = np.zeros(n)
            tmp_2[tmp == np.inf] = 1

        W = tmp/np.sum(tmp)
    
        #find update direction
        candidate_index = 0
        candidate_value = 0
        for v in range(0,p):
            tmp = np.dot(W,yX_r[:,v])
            if(np.abs(tmp) > np.abs(candidate_value)):
                candidate_value = tmp
                candidate_index = v
                
        #update
        direction = np.zeros(p, dtype = float)
        direction[candidate_index] = 1
        bet_tilde = bet_tilde + eps*candidate_value*direction

    print("interpolation:", all(np.dot(yX,bet_tilde)>=-1e-11))
    return(bet_tilde/np.linalg.norm(bet_tilde,2))

###############################################################################
# Plot 1: Distance to \beta^* as number of observations n grows - Adversarial
###############################################################################

DISTANCE = np.ones([10], dtype = float) 
DISTANCE_ADA = np.ones([10], dtype = float) 

s = 5  # sparsity
eps = 0.2 #step size adaboost

np.random.seed(0) 

for i in range(0,10):
    print('Iteration:', i+1)
    
    n = 100*i+100 # number of observations
    
    p = f_1(n) # dimension of observations
    beta = sparse_rademacher_prior_beta(s, p) # generating \beta^*
    corr = 40 # number of corrupted observations
    X = np.random.normal(size = (n,p)) # design matrix
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*adversary(X,beta,corr) # adversarial corruption of Y
    yX = np.dot(np.diag(Y),X) 
    
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    DISTANCE[i] = np.linalg.norm(beta_max_margin-beta,2) # calculating distance
    
    T = f_T(n,p,s,corr,eps) # number of steps for adaboost
    beta_Ada = Ada(n,p,s,yX,T,eps) # calculating \tilde{\beta}_T
    DISTANCE_ADA[i] = np.linalg.norm(beta_Ada-beta,2) # calculating distance
    
###############################################################################
# Plot 2: Distance to \beta^* as contamination |O| grows - Adversarial
###############################################################################

DISTANCE = np.ones([10], dtype = float)
DISTANCE_ADA = np.ones([10], dtype = float)

n = 500 # number of observations
p = f_1(n) # dimension of observations
s = 5 # sparsity
eps = 0.2 # step size adaboost

np.random.seed(0)    
beta = sparse_rademacher_prior_beta(s, p) # generating \beta^*
X = np.random.normal(size = (n,p)) # design matrix 
    
for i in range(0,10):
    print('Iteration:', i+1)
    
    corr = 5*i+5  # number of corrupted observations
        
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*adversary(X,beta,corr) # adversarial corruption of Y
    yX = np.dot(np.diag(Y),X) 
    
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    DISTANCE[i] = np.linalg.norm(beta_max_margin-beta,2) # calculating distance
        
    T = f_T(n,p,s,corr,eps) # number of steps for adaboost
    beta_Ada = Ada(n,p,s,yX,T,eps) # calculating \tilde{\beta}_T
    DISTANCE_ADA[i] = np.linalg.norm(beta_Ada-beta,2) # calculating distance

###############################################################################
# Plot 3: Distance to \beta^* as number of observations n grows - Noiseless
###############################################################################

DISTANCE = np.ones([10], dtype = float) 
DISTANCE_ADA = np.ones([10], dtype = float) 

s = 5  # sparsity
eps = 0.2 # step size adaboost

np.random.seed(0)

for i in range(0,10):
    print('Iteration:', i+1)
    
    n = 100*i+100 # number of observations
    
    p = f_1(n) # dimension of observations
    beta = sparse_rademacher_prior_beta(s, p) # generating \beta^*
    X = np.random.normal(size = (n,p)) # design matrix
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    yX = np.dot(np.diag(Y),X) 
    
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    DISTANCE[i] = np.linalg.norm(beta_max_margin-beta,2) # calculating distance
    
    T = f_T(n,p,s,corr,eps) # number of steps for adaboost
    beta_Ada = Ada(n,p,s,yX,T,eps) # calculating \tilde{\beta}_T
    DISTANCE_ADA[i] = np.linalg.norm(beta_Ada-beta,2) # calculating distance

###############################################################################
# Plot 4: Margin as number of observations n grows - Noiseless
###############################################################################

MARGIN = np.ones([10], dtype = float) #initialize
MARGIN_ADA = np.ones([10], dtype = float) #initialize

s = 5  # Sparsity
eps = 0.2 #step size adaboost

np.random.seed(0)

for i in range(0,10):
    print('Iteration:', i+1)

    n = 100*i+100 # number of observations
    
    p = f_1(n) # dimension of observations
    beta = sparse_rademacher_prior_beta(s, p) # generating \beta^*
    
    X = np.random.normal(size = (n,p)) # design matrix
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    yX = np.dot(np.diag(Y),X) 
    
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    MARGIN[i] = max_margin_margin(sol_max_margin,p) # extracting \gamma

    T = f_T(n,p,s,corr,eps) # number of steps for adaboost
    beta_Ada = Ada(n,p,s,yX,T,eps) # calculating \tilde{\beta}_T
    tmp = np.dot(yX, beta_Ada)/np.linalg.norm(beta_Ada,1)
    MARGIN_ADA[i] = np.min(tmp) # calculating \tilde{\gamma}_T



