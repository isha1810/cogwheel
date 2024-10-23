'''
functions that are useful for defining population models
'''

import numpy as np
from scipy.special import erf
# from scipy.integrate import quad, trapezoid


def normalized_powerlaw_distribution(x, alpha, x_min, x_max):
    '''
    Returns f(x| alpha, x_min, x_max) = A * x^alpha 
    where A = (1+alpha)/(x_max^(1+alpha) - x_min^(1+alpha))
    '''
    mask_zero = np.where(np.logical_or(x<x_min, x>x_max))[0]

    A = (1+alpha)/(np.power(x_max, 1+alpha) - np.power(x_min, 1+alpha))
    f_x = np.power(x, alpha)
    normed_f_x = A*f_x
    normed_f_x[mask_zero] = 1e-10
    
    return normed_f_x
    

def smoothing_function(x, x_min, delta_x):
    '''
    Returns f(x| x_min, delta_x) = 
        0                                for (x < x_min)
        [f(x-x_min, delta_x) + 1]^-1     for (x_min <= x < x_min+delta_x)
        1                                for (x >= x_min + delta_x)
        
    where f(x', delta_x) = exp(delta_x/x' + delta_x/(x'-delta_x))
    '''
    # Initialize the result with zeros
    S_x = np.zeros_like(x, dtype=float)
    
    # Define the range where x_min <= x < x_min + delta_x
    mask_middle = (x >= x_min) & (x < x_min + delta_x)
    
    # Define x_prime for the middle range
    x_prime = x[mask_middle] - x_min

    # Calculate f(x_prime, delta_x) and the smoothing function in the middle range
    if x_prime.size > 0:  # Ensure there is data in the middle range
        f_x = np.exp(delta_x / x_prime + delta_x / (x_prime - delta_x))
        S_x[mask_middle] = np.power(f_x+1, -1)
    
    # Define the range where x >= x_min + delta_x
    mask_one = (x >= x_min + delta_x)
    
    # Set the smoothing function to 1 for the x >= x_min + delta_x range
    S_x[mask_one] = 1

    return S_x