"""Define prior ratios for lvc injection prior."""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio
from astropy.cosmology import FlatwCDM
from astropy import units as u

from .base_prior_ratio import PriorRatio
from .jacobians import m1_m2_to_m1s_m2s
from .pe_priors import (lvc_mass_lnp,
                        lvc_spin_lnp,
                        lvc_injection_mass_lnp, 
                        lvc_redshift_lnp)

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
        time_dilation = np.log(1+z)
        lvc_injection_to_lvc_pe_mass_lnp += time_dilation
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
        time_dilation = np.log(1+z)
        # lvc_pe_to_lvc_injection_mass_lnp -= time_dilation
        return (lvc_pe_to_lvc_injection_mass_lnp)

# ----------------------------------------------------------------------
# PriorRatios involving Combined LVC Injection Prior and LVC PE Prior 
# I am using the O1 and O2 injection priors from zenodo since I am unable
# to reproduce the values myself
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
def f_z(z, pow_z):
    dVc_dz = cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value * 4 * np.pi
    return (1 + z)**(pow_z - 1)* dVc_dz

class CombinedLVCInjectionPriorToLVCPriorRatio(PriorRatio):
    '''
    Ratio between the LVC Injection Prior
    and the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume) 
    '''
    numerator = 'CombinedLVCInjectionPrior'
    denominator = 'LVCPrior(cosmo)'
    params = []
    base_quantities = ['m1_source', 'm2_source','d_luminosity']
    derived_quantities = ['z', 'lvc_injection_to_lvc_pe_mass_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'lvc_injection_to_lvc_pe_mass_lnp' not in samples.keys():
            if 'sampling_pdf' in samples.keys():
                lvc_injection_lnp = np.log(samples['sampling_pdf'])
                lvc_pe_lnp = (lvc_mass_lnp()
                              + m1_m2_to_m1s_m2s(samples['z'])
                              + lvc_spin_lnp(samples['s1x'], samples['s1y'], samples['s1z'],
                                            samples['s2x'], samples['s2y'], samples['s2z'])
                              + np.log(f_z(samples['z'], 0))
                              )
            else:
                lvc_injection_lnp = lvc_injection_mass_lnp(samples['m1_source'],
                                       samples['m2_source'])
                lvc_pe_lnp = (lvc_mass_lnp() 
                              + m1_m2_to_m1s_m2s(samples['z']))
            samples['lvc_injection_to_lvc_pe_mass_lnp'] = (
                lvc_injection_lnp - lvc_pe_lnp)

    def lnprior_ratio(self, z, lvc_injection_to_lvc_pe_mass_lnp):
        return (lvc_injection_to_lvc_pe_mass_lnp)
    
class LVCPriorToCombinedLVCInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the LVC(cosmo) PE Prior (cosmo=>uniform in comoving volume)
    and the LVC Injection Prior
    '''
    numerator = 'LVCPrior(cosmo)'
    denominator = 'CombinedLVCInjectionPrior'
    params = []
    base_quantities = ['m1_source', 'm2_source','d_luminosity']
    derived_quantities = ['z', 'lvc_pe_to_lvc_injection_lnp']
    hyperparams = []

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

        if 'lvc_pe_to_lvc_injection_lnp' not in samples.keys():
            lvc_pe_lnp = (lvc_mass_lnp()
                          + m1_m2_to_m1s_m2s(samples['z'])
                          + lvc_spin_lnp(samples['s1x'], samples['s1y'], samples['s1z'],
                                            samples['s2x'], samples['s2y'], samples['s2z'])
                          + np.log(f_z(samples['z'], 0))
                         )
            lvc_injection_lnp = np.log(samples['sampling_pdf'])
            lvc_pe_to_lvc_injection_lnp = (lvc_pe_lnp
                                           - lvc_injection_lnp)
            samples['lvc_pe_to_lvc_injection_lnp'] = lvc_pe_to_lvc_injection_lnp

    def lnprior_ratio(self, z, lvc_pe_to_lvc_injection_lnp):
        return (lvc_pe_to_lvc_injection_lnp)