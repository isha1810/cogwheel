"""
PriorRatio classes for mass models 
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

# ----------------------------------------------------------------------
# Powerlaw+peak mass Prior Ratio - no smoothing

class PLPToLVCPriorRatio(PriorRatio):
    numerator = 'PLP'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'lvc_mass_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['lvc_mass_lnp'] = (lvc_mass_lnp()
                                   + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, z, lvc_mass_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta):
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))

        pop_mass_lnp = mass_lnp
        return pop_mass_lnp-lvc_mass_lnp

# Powerlaw+peak mass Prior Ratio - with smoothing
class SmoothedPLPToLVCPriorRatio(PriorRatio):
    numerator = 'SmoothedPLP'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'lvc_mass_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'delta_m']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['lvc_mass_lnp'] = (lvc_mass_lnp()
                                   + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, z, lvc_mass_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, delta_m):
        
        mass_lnp = (smoothed_powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std, delta_m)
                         + smoothed_powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min, m_max,
                                                            delta_m))

        pop_mass_lnp = mass_lnp
        return pop_mass_lnp-lvc_mass_lnp

# ----------------------------------------------------------------------
# Powerlaw mass Prior Ratio - no smoothing

class PowerlawToLVCPriorRatio(PriorRatio):
    numerator = 'Powerlaw'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'lvc_mass_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        
        samples['lvc_mass_lnp'] = (lvc_mass_lnp()
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, z, lvc_mass_lnp,
                      alpha, m_min, m_max, beta):
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))

        pop_mass_lnp = mass_lnp
        return pop_mass_lnp-lvc_mass_lnp
