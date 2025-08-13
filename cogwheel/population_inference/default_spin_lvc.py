"""
PriorRatio class for default spin model
Eq. B19 and B20 in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.13.011048
"""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)
from cogwheel.prior import IdentityTransformMixin, Prior

from .base_prior_ratio import PriorRatio
from .jacobians import m1_m2_to_m1s_q
from .pdfs import (powerlaw,
                    smoothing_function,
                    truncated_gaussian)
from .pe_priors import (powerlaw_peak_primary_mass_lnp,
                        powerlaw_primary_mass_lnp,
                        powerlaw_mass_ratio_lnp,
                        smoothed_powerlaw_peak_primary_mass_lnp,
                        smoothed_powerlaw_mass_ratio_lnp,
                        lvc_mass_lnp)

def alpha_beta_from_mu_var(mu, var):
    """
    returns the alpha and beta for a beta 
    distribution given mean and variance of the 
    distribution
    """
    omicron = (1-mu)*mu/var - 1

    alpha = mu*omicron
    beta = (1-mu)*omicron
    return alpha, beta

def cartesian_to_spin_mag_costilt(x, y, z):
    """
    returns (r, costheta) given (x, y, z)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    costheta = z/r
    return r, costheta

def spin_tilt_mixture_to_lvc_ratio(z, zeta, sigma_z):
    """
    returns the log prior ratio of the default spin tilt
    distribution to lvc spin tilt distribution
    """
    lvc_tilt_lnp = np.log(0.5)

    # Gaussian centered at z=1 (aligned)
    z_min = -1
    z_max = 1
    z_mean = 1
    gaussian_spin_tilt_prior = truncated_gaussian(
        z, z_min, z_max, z_mean, sigma_z)
    isotropic_spin_tilt_prior = 0.5
    default_spin_tilt_lnp = np.log(
        zeta*gaussian_spin_tilt_prior +
        (1-zeta)*isotropic_spin_tilt_prior)
    
    return default_spin_tilt_lnp-lvc_tilt_lnp

def beta_spin_magnitudes_to_lvc_lnp(chi, alpha_chi, beta_chi):
    """
    returns the log prior ratio of the spin magnitude beta
    distirbution to the lvc uniform magnitude distribution
    """
    if np.logical_or(alpha_chi<=1, beta_chi<=1):
        spin_magnitude_lnp = -np.inf
    else:
        spin_magnitude_lnp = stats.beta.logpdf(chi, alpha_chi, beta_chi)
    lvc_spin_magnitude_lnp = np.log(1)
    return spin_magnitude_lnp-lvc_spin_magnitude_lnp

class PLPDefaultSpinsToLVCPriorRatio(PriorRatio):
    numerator = 'PLPDefaultSpins'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity', 's1x', 's1y', 's1z',
                      's2x', 's2y', 's2z']
    derived_quantities = ['chi1', 'z1', 'chi2', 'z2', 'z',
                          'lvc_mass_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'zeta', 'mu_chi',
                  'var_chi', 'sigma_z']

    def compute_auxiliary_quantities(self, samples):
        samples['chi1'], samples['z1'] = cartesian_to_spin_mag_costilt(
            samples['s1x'], samples['s1y'], samples['s1z'])
        samples['chi2'], samples['z2'] = cartesian_to_spin_mag_costilt(
            samples['s2x'], samples['s2y'], samples['s2z'])
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['lvc_mass_lnp'] = (lvc_mass_lnp()
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chi1, z1, chi2, z2,
                      z, lvc_mass_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta,
                      zeta, mu_chi, var_chi, sigma_z):
        
        alpha_chi, beta_chi = alpha_beta_from_mu_var(mu_chi, var_chi)
        pop_spin_to_lvc_spin_lnp = (beta_spin_magnitudes_to_lvc_lnp(chi1, alpha_chi, beta_chi)
                                    + spin_tilt_mixture_to_lvc_ratio(z1, zeta, sigma_z)
                                    + beta_spin_magnitudes_to_lvc_lnp(chi2, alpha_chi, beta_chi)
                                    + spin_tilt_mixture_to_lvc_ratio(z2, zeta, sigma_z))
        
        pop_mass_to_lvc_mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min)
                                   - lvc_mass_lnp)

        # time_dilation = - np.log(1+z)
        pop_to_lvc_lnp = (pop_mass_to_lvc_mass_lnp
                          + pop_spin_to_lvc_spin_lnp
                          # + time_dilation
                         )
        
        return pop_to_lvc_lnp

class SmoothedPLPDefaultSpinsToLVCPriorRatio(PriorRatio):
    numerator = 'SmoothedPLPDefaultSpins'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity', 's1x', 's1y', 's1z',
                      's2x', 's2y', 's2z']
    derived_quantities = ['chi1', 'z1', 'chi2', 'z2', 'z',
                          'lvc_mass_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'delta_m', 'zeta', 'mu_chi',
                  'var_chi', 'sigma_z']

    def compute_auxiliary_quantities(self, samples):
        samples['chi1'], samples['z1'] = cartesian_to_spin_mag_costilt(
            samples['s1x'], samples['s1y'], samples['s1z'])
        samples['chi2'], samples['z2'] = cartesian_to_spin_mag_costilt(
            samples['s2x'], samples['s2y'], samples['s2z'])
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['lvc_mass_lnp'] = (lvc_mass_lnp()
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chi1, z1, chi2, z2,
                      z, lvc_mass_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, delta_m,
                      zeta, mu_chi, var_chi, sigma_z):
        
        alpha_chi, beta_chi = alpha_beta_from_mu_var(mu_chi, var_chi)
        pop_spin_to_lvc_spin_lnp = (beta_spin_magnitudes_to_lvc_lnp(chi1, alpha_chi, beta_chi)
                                    + spin_tilt_mixture_to_lvc_ratio(z1, zeta, sigma_z)
                                    + beta_spin_magnitudes_to_lvc_lnp(chi2, alpha_chi, beta_chi)
                                    + spin_tilt_mixture_to_lvc_ratio(z2, zeta, sigma_z))
        
        pop_mass_to_lvc_mass_lnp = (smoothed_powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std, delta_m)
                         + smoothed_powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min, m_max, delta_m)
                                   - lvc_mass_lnp)

        # time_dilation = - np.log(1+z)
        pop_to_lvc_lnp = (pop_mass_to_lvc_mass_lnp
                          + pop_spin_to_lvc_spin_lnp
                          # + time_dilation
                         )
        
        return pop_to_lvc_lnp