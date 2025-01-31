"""Define prior ratios for lvc injection prior."""
import numpy as np
import pandas as pd
from scipy import stats

from .base_prior_ratio import PriorRatio
from .pop_utils import normalized_powerlaw_distribution
from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio

# *************************************************
# LVC Injection Prior and IAS PE Prior 
# *************************************************

class LVCInjectionPriorToIASPriorRatio(PriorRatio):
    '''
    Ratio between the LVC Injection Prior and the IASPrior
    '''
    numerator = 'LVCInjectionPrior'
    denominator = 'IASPrior'
    params = ['m1_source', 'q', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] = \
                comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])
            
    def lnprior_ratio(self, m1_source, q, s1x, s1y, s1z, s2x, s2y, s2z,
                      z, comoving_to_luminosity):
        # define constants
        alpha1=-2.35
        alpha2=1.0
        mmin=2.
        mmax=100.
        max_spin=0.998

        injection_mass_jacobian = np.log(m1_source)
        injection_spin_jacobian = np.log(1+q)-np.log(q)

        log_m1_source_norm = - np.log(np.power(mmax, alpha1+1)/(alpha1+1) - np.power(mmin, alpha1+1)/(alpha1+1))
        log_m2_source_norm = - np.log(np.power(m1_source,alpha2+1)/(alpha2+1) - np.power(mmin,alpha2+1)/(alpha2+1))
        
        injection_mass_lnp = (alpha1*np.log(m1_source) + alpha2*np.log(q*m1_source) + 
                               log_m1_source_norm + log_m2_source_norm +
                               injection_mass_jacobian)
        injection_spin_lnp = (-np.log(4*np.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin) -
                              np.log(4*np.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin) + 
                              injection_spin_jacobian)
        injection_distance_lnp = np.log(comoving_to_luminosity) + np.log(1+z)
        
        injection_lnp = injection_mass_lnp + injection_spin_lnp + injection_distance_lnp
        
        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        ias_spin_lnp = np.log(0.5)
        ias_lnp = ias_mass_lnp + ias_spin_lnp 
        
        return (injection_lnp - ias_lnp)

class IASPriorToLVCInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the IASPrior and the LVC Injection Prior
    '''
    numerator = 'IASPrior'
    denominator = 'LVCInjectionPrior'
    params = ['m1_source', 'q', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] = \
                comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])

    def lnprior_ratio(self, m1_source, q, s1x, s1y, s1z, s2x, s2y, s2z,
                      z, comoving_to_luminosity):
        # define constants
        alpha1=-2.35
        alpha2=1.0
        mmin=2.
        mmax=100.
        max_spin=0.998

        injection_mass_jacobian = np.log(m1_source)
        injection_spin_jacobian = np.log(1+q)-np.log(q)

        log_m1_source_norm = - np.log(np.power(mmax, alpha1+1)/(alpha1+1) - np.power(mmin, alpha1+1)/(alpha1+1))
        log_m2_source_norm = - np.log(np.power(m1_source,alpha2+1)/(alpha2+1) - np.power(mmin,alpha2+1)/(alpha2+1))
        
        injection_mass_lnp = (alpha1*np.log(m1_source) + alpha2*np.log(q*m1_source) + 
                               log_m1_source_norm + log_m2_source_norm +
                               injection_mass_jacobian)
        injection_spin_lnp = (-np.log(4*np.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin) -
                              np.log(4*np.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin) + 
                              injection_spin_jacobian)
        injection_distance_lnp = np.log(comoving_to_luminosity) + np.log(1+z)
        
        injection_lnp = injection_mass_lnp + injection_spin_lnp + injection_distance_lnp
        
        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        ias_spin_lnp = np.log(0.5)
        ias_lnp = ias_mass_lnp + ias_spin_lnp 
        
        return (ias_lnp - injection_lnp)

# *************************************************
# LVC Injection Prior and LVC PE Prior 
# *************************************************

class LVCInjectionPriorToLVCPriorRatio(PriorRatio):
    '''
    Ratio between the LVC Injection Prior and the LVC PE Prior
    '''
    numerator = 'LVCInjectionPrior'
    denominator = 'LVCPrior'
    params = ['m1_source', 'q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'injection_lnp', 'lvc_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
            
        if 'injection_lnp' not in samples.keys():
            # define constants
            alpha1 = -2.35
            alpha2 = 1.0
            mmin = 2.
            mmax = 100.
            max_spin=0.998
            
            m1_source = samples['m1_source'].values
            q = samples['q'].values
            injection_mass_jacobian = np.log(m1_source)
            log_m1_source_lnp = np.log(normalized_powerlaw_distribution(m1_source, alpha1, mmin, mmax))
            log_m2_source_lnp = np.log(normalized_powerlaw_distribution(m1_source*q, alpha2, mmin, m1_source))
            injection_mass_lnp = (log_m1_source_lnp + log_m2_source_lnp
                                   + injection_mass_jacobian)
            injection_lnp = injection_mass_lnp
            samples['injection_lnp'] = injection_lnp
            
        if 'lvc_lnp' not in samples.keys():
            m1_source = samples['m1_source'].values
            z = samples['z']
            lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
            lvc_mass_lnp = lvc_mass_jacobian
            lvc_lnp = lvc_mass_lnp
            samples['lvc_lnp'] = lvc_lnp

    def lnprior_ratio(self, m1_source, q, z, injection_lnp, lvc_lnp):
        return (injection_lnp - lvc_lnp)
        

class LVCPriorToLVCInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the LVC PE Prior and the LVC Injection Prior
    '''
    numerator = 'LVCPrior'
    denominator = 'LVCInjectionPrior'
    params = ['m1_source', 'q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'injection_lnp', 'lvc_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
            
        if 'injection_lnp' not in samples.keys():
            # define constants
            alpha1 = -2.35
            alpha2 = 1.0
            mmin = 2.
            mmax = 100.
            max_spin=0.998
            
            m1_source = samples['m1_source'].values
            q = samples['q'].values
            injection_mass_jacobian = np.log(m1_source)
            log_m1_source_lnp = np.log(normalized_powerlaw_distribution(m1_source, alpha1, mmin, mmax))
            log_m2_source_lnp = np.log(normalized_powerlaw_distribution(m1_source*q, alpha2, mmin, m1_source))
            injection_mass_lnp = (log_m1_source_lnp + log_m2_source_lnp
                                   + injection_mass_jacobian)
            injection_lnp = injection_mass_lnp
            samples['injection_lnp'] = injection_lnp
            
        if 'lvc_lnp' not in samples.keys():
            m1_source = samples['m1_source'].values
            z = samples['z'].values
            lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
            lvc_mass_lnp = lvc_mass_jacobian
            lvc_lnp = lvc_mass_lnp
            samples['lvc_lnp'] = lvc_lnp

    def lnprior_ratio(self, m1_source, q, z, injection_lnp, lvc_lnp):
        return (lvc_lnp - injection_lnp)
