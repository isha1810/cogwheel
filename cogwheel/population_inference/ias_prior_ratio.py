"""Define prior ratios for ias injection prior."""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio

from .base_prior_ratio import PriorRatio
from .jacobians import m1_m2_to_m1s_q
from .pe_priors import (lvc_mass_lnp,
                        ias_o1o2_injection_lnp,
                        isotropic_spins_marginal_chieff_lnp)

# ----------------------------------------------------------------------
# PriorRatios involving IAS Injection Prior and LVC PE Prior 

class IASInjectionPriorToLVCPriorRatio(PriorRatio):
    '''
    Ratio between the IAS Injection Prior
    and the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume) 
    '''
    numerator = 'IASInjectionPrior'
    denominator = 'LVCPrior(cosmo)'
    params = []
    base_quantities = ['m1_source', 'q','d_luminosity']
    derived_quantities = ['z', 'ias_injection_to_lvc_pe_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'ias_injection_to_lvc_pe_lnp' not in samples.keys():
            lvc_pe_lnp = (lvc_mass_lnp() 
                          + m1_m2_to_m1s_q(samples['m1_source'],
                                           samples['z'])
                          + isotropic_spins_marginal_chieff_lnp(
                              samples['chieff'], samples['q']))
            samples['ias_injection_to_lvc_pe_lnp'] = (
                ias_o1o2_injection_lnp(samples['m1_source'],
                                       samples['q'],
                                       samples['chieff']) 
                - lvc_pe_lnp)

    def lnprior_ratio(self, z, ias_injection_to_lvc_pe_lnp):
        return (ias_injection_to_lvc_pe_lnp)
        

class LVCPriorToIASInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume)
    and the IAS Injection Prior
    '''
    numerator = 'LVCPrior(cosmo)'
    denominator = 'IASInjectionPrior'
    params = []
    base_quantities = ['m1_source', 'q','d_luminosity']
    derived_quantities = ['z', 'lvc_pe_to_ias_injection_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'lvc_pe_to_ias_injection_lnp' not in samples.keys():
            lvc_pe_lnp = (lvc_mass_lnp() 
                          + m1_m2_to_m1s_q(samples['m1_source'],
                                           samples['z'])
                          + isotropic_spins_marginal_chieff_lnp(
                              samples['chieff'], samples['q']))
            samples['lvc_pe_to_ias_injection_lnp'] = (
                lvc_pe_lnp
                - ias_o1o2_injection_lnp(samples['m1_source'],
                                       samples['q'],
                                       samples['chieff'])
            )

    def lnprior_ratio(self, z, lvc_pe_to_ias_injection_lnp):
        return (lvc_pe_to_ias_injection_lnp)