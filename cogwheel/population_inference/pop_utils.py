'''
functions that are useful for defining population models
'''

import numpy as np


def normalized_powerlaw_distribution(x, alpha, x_min, x_max):
    '''
    Returns f(x| alpha, x_min, x_max) = A * x^alpha 
    where A = (1+alpha)/(x_max^(1+alpha) - x_min^(1+alpha))
    '''
    mask_zero = np.where(np.logical_or(x<x_min, x>x_max))[0]
    
    A = (1+alpha)/(x_max**(1+alpha) - x_min**(1+alpha))
    f_x = x**alpha
    normed_f_x = A*f_x
    normed_f_x[mask_zero] = 1e-10
    
    return normed_f_x

# def normalized_truncated_gaussian_distribution(x, mu, sigma, x_min, x_max):
#     '''
#     Returns f(x|mu, sigma, x_min, x_max) = A * exp(-(x-mu)^2 / 2*sigma^2)
#     '''
    

def smoothing_function(x, x_min, delta_x):
    '''
    Returns f(x| x_min, delta_x) = 
        0                                for (x < x_min)
        [f(x-x_min, delta_x) + 1]^-1     for (x_min <= x < x_min+delta_x)
        1                                for (x >= x_min + delta_x)
        
    where f(x', delta_x) = exp(delta_x/x' + delta_x/(x'-delta_x))
    '''
    if x<x_min:
        S_x = 0
    elif x>= x_min+delta_x:
        S_x = 1
    elif x>=x_min and x<(x_min+delta_x):
        x_prime = x-x_min
        f_x = np.exp(delta_x/x_prime + delta_x/(x_prime - delta_x))
        S_x = (f_x + 1)**-1
    return S_x

