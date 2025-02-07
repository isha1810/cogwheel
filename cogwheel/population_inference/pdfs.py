"""
Useful PDFs.
"""
import numpy as np
from scipy.special import erf

def powerlaw(x, alpha, x_min, x_max):
    """
    Returns f(x| alpha, x_min, x_max) = A * x^alpha 
    where A = (1+alpha)/(x_max^(1+alpha) - x_min^(1+alpha)).
    """
    mask_zero = np.where(np.logical_or(x<x_min, x>x_max))[0]
    log_x_max_alpha = (1+alpha)*np.log(x_max)
    log_x_min_alpha = (1+alpha)*np.log(x_min)

    if alpha==-1.0:
        A = 1/np.log(np.abs(x_max/x_min))
    else:
        A = (1+alpha)/(np.exp(log_x_max_alpha) - np.exp(log_x_min_alpha))
    log_x_alpha = alpha*np.log(x)
    f_x = np.exp(log_x_alpha)
    normed_f_x = A*f_x
    normed_f_x[mask_zero] = np.float32(1e-10)
    
    return normed_f_x
    

def smoothing_function(x, x_min, delta_x):
    """
    Returns f(x| x_min, delta_x) = 
        0                                for (x < x_min)
        [f(x-x_min, delta_x) + 1]^-1     for (x_min <= x < x_min+delta_x)
        1                                for (x >= x_min + delta_x)
        
    where f(x', delta_x) = exp(delta_x/x' + delta_x/(x'-delta_x)).
    """
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


def smoothed_uniform(x, x_min, x_max, sigma_x):
    """
    Returns f(x| x_min, x_max, sigma_x) = 
        0                                              for (x < x_min+sigma_x, x > x_max+sigma_x)
        1/2 [1 - cos(\pi (np.abs(x)-x_min/max)/sigma_x)]     for (x_min <= x < x_min+sigma_x, x_max <= x < x_max+sigma_x)
        1                                              for (x_min < x <= x_max)
        and f is normalized to integrate to 1.
        normalization factor A = 1/(sigma_x + (x_max-x_min)).
    """
    f_x = np.zeros_like(x, dtype=float)
    mask_ones = (x >= x_min) & (x<= x_max)
    mask_smoothed_near_min = (x>=x_min-sigma_x) & (x<x_min)
    mask_smoothed_near_max = (x>x_max) & (x<=x_max+sigma_x)

    if np.sum(mask_ones)>0:
        f_x[mask_ones] = 1.0
    if np.sum(mask_smoothed_near_min)>0:
        x_shifted = np.abs(x)-x_min
        g_x = 0.5*(1 + np.cos(np.pi*(x_shifted)/sigma_x))
        f_x[mask_smoothed_near_min] = g_x[mask_smoothed_near_min]
    if np.sum(mask_smoothed_near_max)>0:
        x_shifted = np.abs(x)-x_max
        g_x = 0.5*(1 + np.cos(np.pi*(x_shifted)/sigma_x))
        f_x[mask_smoothed_near_max] = g_x[mask_smoothed_near_max]
        
    A = 1/(sigma_x+(x_max-x_min))
    return A*f_x


def truncated_gaussian(x, x_min, x_max, mean, std):
    """
    evaluates the gaussian pdf described by mean, std, x_min, x_max
    at array of values x.
    f_x  = A * exp(-(x-mean)/std**2),
        A = [sqrt(2*pi*sigma**2) * (Phi((x_max-mu)/sqrt(2)*sigma) - Phi((x_min-mu)/sqrt(2)*sigma)]^-1
    where Phi(xi) = 0.5*(1 + erf(xi/sqrt(2))).
    """
    xi_max = (x_max-mean)/std
    xi_min = (x_min-mean)/std
    Phi_x_max = 0.5*(1+erf(xi_max/np.sqrt(2)))
    Phi_x_min = 0.5*(1+erf(xi_min/np.sqrt(2)))
    
    A = (np.sqrt(2*np.pi*std**2)*(Phi_x_max-Phi_x_min))**-1
    f_x = A*np.exp(-(x-mean)**2/(2*std**2))
    
    mask_zero = np.logical_or(x<x_min, x>x_max)
    f_x[mask_zero] = 0
    
    return f_x