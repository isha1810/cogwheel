"""
PriorRatio classes for spin models 
"""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)
from cogwheel.prior import IdentityTransformMixin, Prior

from .base_prior_ratio import PriorRatio
from .jacobians import (m1_m2_to_m1s_q,
                        chieff_cumchidiff_to_s1z_s2z)
from .pdfs import (powerlaw,
                    smoothing_function,
                    truncated_gaussian)
from .pe_priors import (powerlaw_peak_primary_mass_lnp, powerlaw_primary_mass_lnp,
                        powerlaw_mass_ratio_lnp,
                        uniform_chieff_cartesian_spins_lnp,
                        uniform_chieff_lnp,
                        natally_spinning_prior, tidally_locked_secondary_spin_prior,
                        tidally_torqued_secondary_spin_prior,
                        lvc_lnp_cosmo)

# ----------------------------------------------------------------------
# Prior Ratio classes involving spin A2 model

class PLPA2ToLVCPriorRatio(PriorRatio):
    numerator = 'PLPA2'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp


class PowerlawA2ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA2'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio classes involving spin A4 model

class PLPA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PLPA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp', 
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))

    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp
    
# ----------------------------------------------------------------------
# Prior Ratio classes involving only natal spins

class PLPNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PLPNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'sigma_chi']
    
    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        spin_lnp = (natal_spins_chieff_lnp 
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp', 
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        spin_lnp = (natal_spins_chieff_lnp
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp    
# ----------------------------------------------------------------------
# Prior Ratio classes involving uniform+delta function spin model

class DeltaUniformQChieffToLVCPriorRatio(PriorRatio):
    numerator = 'DeltaUniformQChieff'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'truncnorm_chieff_zero',
                          'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 'm_mean', 'm_std', 
                   'beta', 'f']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['truncnorm_chieff_zero'] = truncated_gaussian(samples['chieff'],
                                            -1.0, 1.0, 0, 0.05)
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, truncnorm_chieff_zero, 
                      uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std,
                      beta, f):
        max_spin=0.998
        q_min = m_min/m1_source
        truncnorm_q_one = truncated_gaussian(q, q_min.values, 1.0, 0.05)
        spin_lnp = (np.log(f*truncnorm_chieff_zero*truncnorm_q_one + 
                           (1-f)*powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min)*0.5/max_spin)
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
                           
        mass_lnp = powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                        m_min, m_max, m_mean, m_std)
        
        mass_spin_lnp = mass_lnp+spin_lnp
        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)

        return pop_lnp-lvc_lnp

