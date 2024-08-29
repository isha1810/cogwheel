'''
functions that are useful for defining population models
'''

import numpy as np
from scipy.special import erf


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
#     where A is the normalization
#     '''
#     mask_zero = np.where(np.logical_or(x<x_min, x>x_max))[0]
    
#     A = (np.sqrt(np.pi/(2*sigma**2)) * 
#          (erf(np.sqrt(sigma**2/2)*(x_max-mu)) - erf(np.sqrt(sigma**2/2)*(x_min-mu))))**(-1)
#     f_x = np.exp(-(x-mu)**2/(2*sigma**2))
#     normed_f_x = A*f_x
#     normed_f_x[mask_zero] = 1e-10

#     return normed_f_x

# def smoothing_function(x, x_min, delta_x):
#     '''
#     Returns f(x| x_min, delta_x) = 
#         0                                for (x < x_min)
#         [f(x-x_min, delta_x) + 1]^-1     for (x_min <= x < x_min+delta_x)
#         1                                for (x >= x_min + delta_x)
        
#     where f(x', delta_x) = exp(delta_x/x' + delta_x/(x'-delta_x))
#     '''
#     mask_zero = np.where(x<x_min)[0]
#     mask_one = np.where(x>=x_min+delta_x)[0]

#     x_prime = x-x_min
#     f_x = np.exp(delta_x/x_prime + delta_x/(x_prime - delta_x))
#     S_x = (f_x + 1)**-1
    
#     S_x[mask_zero] = 0
#     S_x[mask_one] = 1

#     return S_x

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
        S_x[mask_middle] = (f_x + 1) ** -1
    
    # Define the range where x >= x_min + delta_x
    mask_one = (x >= x_min + delta_x)
    
    # Set the smoothing function to 1 for the x >= x_min + delta_x range
    S_x[mask_one] = 1

    return S_x
