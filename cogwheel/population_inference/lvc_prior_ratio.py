"""Define prior ratios for lvc injection prior."""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio

from .base_prior_ratio import PriorRatio
from .jacobians import m1_m2_to_m1s_m2s
from .pe_priors import (lvc_mass_lnp,
                        lvc_injection_mass_lnp)

# ----------------------------------------------------------------------
# PriorRatios involving LVC Injection Prior and LVC PE Prior 

class LVCInjectionPriorToLVCPriorRatio(PriorRatio):
    '''
    Ratio between the LVC Injection Prior
    and the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume) 
    '''
    numerator = 'LVCInjectionPrior'
    denominator = 'LVCPrior(cosmo)'
    params = []
    base_quantities = ['m1_source', 'm2_source','d_luminosity']
    derived_quantities = ['z', 'lvc_injection_to_lvc_pe_mass_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'lvc_injection_to_lvc_pe_mass_lnp' not in samples.keys():
            samples['lvc_injection_to_lvc_pe_mass_lnp'] = (
                lvc_injection_mass_lnp(samples['m1_source'],
                                       samples['m2_source'])
                - (lvc_mass_lnp() +
                    m1_m2_to_m1s_m2s(samples['z']))
            )

    def lnprior_ratio(self, z, lvc_injection_to_lvc_pe_mass_lnp):
        return (lvc_injection_to_lvc_pe_mass_lnp)
        

class LVCPriorToLVCInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume) 
    and the LVC Injection Prior
    '''
    numerator = 'LVCPrior(cosmo)'
    denominator = 'LVCInjectionPrior'
    params = []
    base_quantities = ['m1_source', 'm2_source','d_luminosity']
    derived_quantities = ['z', 'lvc_pe_to_lvc_injection_mass_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'lvc_pe_to_lvc_injection_mass_lnp' not in samples.keys():
            samples['lvc_pe_to_lvc_injection_mass_lnp'] = (
                (lvc_mass_lnp() +
                    m1_m2_to_m1s_m2s(samples['z']))
                - lvc_injection_mass_lnp(samples['m1_source'],
                                         samples['m2_source'])
            )

    def lnprior_ratio(self, z, lvc_pe_to_lvc_injection_mass_lnp):
        return (lvc_pe_to_lvc_injection_mass_lnp)

